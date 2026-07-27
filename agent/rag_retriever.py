import os
import time
import threading
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types

logger = logging.getLogger("hr_agent")

# ==============================
# CONFIG: RAG / CHROMADB
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hr_policies")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

logger.info("RAG CONFIG: CHROMA_DB_PATH=%s (abs=%s)", CHROMA_DB_PATH, os.path.abspath(CHROMA_DB_PATH))
logger.info("RAG CONFIG: CHROMA_COLLECTION_NAME=%s", CHROMA_COLLECTION_NAME)
logger.info("RAG CONFIG: GEMINI_API_KEY present=%s", bool(GEMINI_API_KEY))

# ==============================
# GEMINI RATE LIMITER
# ==============================
GEMINI_TPM_LIMIT = 30000
GEMINI_RPM_LIMIT = 100
GEMINI_RPD_LIMIT = 1000
GEMINI_TPM_SAFETY_MARGIN = 0.15
GEMINI_RPM_SAFETY_MARGIN = 0.20


class GeminiRateLimiter:
    """Conservative token-based rate limiter for Gemini Embedding API."""
    def __init__(self):
        self._lock = threading.Lock()
        self._minute_tokens = 0
        self._minute_requests = 0
        self._day_tokens = 0
        self._day_requests = 0
        self._minute_start = time.time()
        self._day_start = time.time()
        self._last_request_time = 0

    def _reset_windows(self):
        now = time.time()
        if now - self._minute_start >= 60:
            self._minute_tokens = 0
            self._minute_requests = 0
            self._minute_start = now
        if now - self._day_start >= 86400:
            self._day_tokens = 0
            self._day_requests = 0
            self._day_start = now

    def acquire(self, estimated_tokens: int = 1000):
        """Block until it's safe to make a request."""
        with self._lock:
            self._reset_windows()
            now = time.time()

            # Daily hard stops
            if self._day_requests >= int(GEMINI_RPD_LIMIT * 0.25):
                raise RuntimeError(f"Gemini daily request limit approaching: {self._day_requests}/{GEMINI_RPD_LIMIT}")
            if self._day_tokens >= int(GEMINI_TPM_LIMIT * 24 * 0.10):
                raise RuntimeError("Gemini daily token limit approaching")

            safe_tpm = int(GEMINI_TPM_LIMIT * GEMINI_TPM_SAFETY_MARGIN)
            safe_rpm = int(GEMINI_RPM_LIMIT * GEMINI_RPM_SAFETY_MARGIN)

            # RPM check
            if self._minute_requests >= safe_rpm:
                wait_time = 60 - (now - self._minute_start) + 1
                logger.warning("Gemini RPM limit reached (%d/%d), waiting %.1fs",
                             self._minute_requests, safe_rpm, wait_time)
                self._lock.release()
                time.sleep(max(wait_time, 0))
                self._lock.acquire()
                self._reset_windows()
                if self._minute_requests >= safe_rpm:
                    raise RuntimeError("Still at RPM limit after waiting")

            # TPM check
            if self._minute_tokens + estimated_tokens > safe_tpm:
                wait_time = 60 - (now - self._minute_start) + 1
                logger.warning("Gemini TPM would exceed (%d + %d > %d), waiting %.1fs",
                             self._minute_tokens, estimated_tokens, safe_tpm, wait_time)
                self._lock.release()
                time.sleep(max(wait_time, 0))
                self._lock.acquire()
                self._reset_windows()

            # Min spacing between requests
            min_spacing = 60.0 / safe_rpm
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < min_spacing and self._last_request_time > 0:
                sleep_time = min_spacing - elapsed_since_last
                self._lock.release()
                time.sleep(sleep_time)
                self._lock.acquire()

            self._minute_tokens += estimated_tokens
            self._minute_requests += 1
            self._day_tokens += estimated_tokens
            self._day_requests += 1
            self._last_request_time = time.time()


gemini_limiter = GeminiRateLimiter()


# ==============================
# RAG RETRIEVER CLASS
# ==============================
class RAGRetriever:
    """Retrieves HR policy docs from ChromaDB using Gemini embeddings."""
    def __init__(self):
        self.client = None
        self.collection = None
        self.gemini_client = None
        self._initialized = False
        self._init_error = None

    def initialize(self):
        if self._initialized:
            return True
        if self._init_error:
            return False
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — RAG retriever disabled")
            self._init_error = "No GEMINI_API_KEY"
            return False
        try:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            try:
                self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
                count = self.collection.count()
                logger.info("RAG: Connected to '%s' with %d documents", CHROMA_COLLECTION_NAME, count)
            except Exception as e:
                logger.warning("RAG: Collection '%s' not found: %s", CHROMA_COLLECTION_NAME, e)
                self.collection = None
            self._initialized = True
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.error("RAG initialization failed: %s", e)
            return False

    def _get_embedding(self, text: str) -> List[float]:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")
        estimated_tokens = max(len(text) // 4, 100)
        gemini_limiter.acquire(estimated_tokens)
        result = self.gemini_client.models.embed_content(
            model="models/gemini-embedding-2",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return result.embeddings[0].values

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self.initialize():
            return []
        if self.collection is None:
            return []
        top_k = top_k or RAG_TOP_K
        
        try:
            query_embedding = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            chunks = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                relevance = 1.0 - distance
                logger.debug("Chunk %d: distance=%.4f, relevance=%.4f", i, distance, relevance)
                
                chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": distance,
                    "relevance_score": round(relevance, 3)
                })
            logger.info("RAG: Retrieved %d chunks", len(chunks))
            return chunks
        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            return []

    def is_available(self) -> bool:
        if not self.initialize():
            return False
        if self.collection is None:
            return False
        try:
            return self.collection.count() > 0
        except:
            return False


rag_retriever = RAGRetriever()


def retrieve_policy_context(query: str, top_k: int = None) -> str:
    """Retrieve HR policy documents and format as context string."""
    if not rag_retriever.is_available():
        return ""
    chunks = rag_retriever.retrieve(query, top_k=top_k)
    if not chunks:
        return ""
    lines = ["\n=== HR POLICY DOCUMENTS (Relevant Chunks) ==="]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown document")
        page = meta.get("page", "N/A")
        score = chunk.get("relevance_score", 0)
        lines.append(f"\n[Chunk {i}] Source: {source} | Page: {page} | Relevance: {score}")
        lines.append(chunk["text"])
    lines.append("\n=== END HR POLICY DOCUMENTS ===\n")
    return "\n".join(lines)