"""
LlamaIndex + ChromaDB setup for HR policy document RAG.
Uses free Gemini embedding-2. No paid LLM required for ingestion.
"""

import os
import re
import sys
from pathlib import Path

# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from ingestion.gemini_embedder import GeminiGenAIEmbedder 

# ── Paths ──────────────────────────────────────────────────────────────
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", r"C:\Users\USER\Documents\hrms-api\data\chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hr_policies")

# ── Gemini Embedding ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Set GEMINI_API_KEY env var (free tier from Google AI Studio)."
    )

embed_model = GeminiGenAIEmbedder(
    api_key=GEMINI_API_KEY,
    model_name="gemini-embedding-2", 
    max_tpm=25000,
)

# Global settings so all LlamaIndex ops use this embedder
Settings.embed_model = embed_model

# ── Text Cleaning ───────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """Remove PDF extraction artifacts, footers, and normalize whitespace."""
    # Remove control chars except \n, \r, \t
    text = ''.join(c for c in text if c in '\n\r\t' or ord(c) >= 32 and ord(c) != 127)
    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _strip_footers(text: str) -> str:
    """Remove repeated footer/header lines common in HR policy PDFs."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line_stripped = line.strip()
        # Skip common footer/header patterns
        if re.search(r'(?i)sample document.*malaysia hr forum', line_stripped):
            continue
        if re.search(r'(?i)facebook\.com/groups/MalaysiaHRForum', line_stripped):
            continue
        if re.match(r'^\s*Page\s+\d+\s+of\s+\d+\s*$', line_stripped, re.I):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def _contextualize_chunk(text: str, doc_title: str, section: str = "General", page: str = "N/A") -> str:
    """Prepend document context so embeddings capture structure."""
    return f"Document: {doc_title} | Section: {section} | Page: {page}\n\n{text}"

def _detect_section(text: str) -> str:
    """Detect section heading from first meaningful line of text.

    Matches multi-level numbered headings like:
      1.0. Introduction
      6.1.1. Commenting on appearance...
      2.2. The Company does not tolerate...
    """
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match numbered headings: 1.0.  6.1.1.  2.2.  etc.
        if re.match(r'^\d+(\.\d+)*\.?\s+', line) and len(line) < 100:
            return line[:80]
        # Match ALL CAPS headings (short, no lowercase)
        if len(line) < 60 and line.isupper() and len(line) > 3:
            return line[:60]
    return "General"

# ── Chroma Client ──────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(name=COLLECTION_NAME)

# ── Index Builders ─────────────────────────────────────────────────────
def build_index_from_documents(documents: list) -> VectorStoreIndex:
    """Ingest documents into ChromaDB using direct insertion with contextual chunks."""
    client = get_chroma_client()
    
    # Fresh collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[OK] Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    
    # Larger chunks for policy docs — keeps multi-paragraph sections intact
    node_parser = SentenceSplitter(
        chunk_size=1200,       # ← tuned for dense HR policy pages (~2-3k chars/page)
        chunk_overlap=150,     # ← ~12% overlap for clause continuity
        paragraph_separator="\n\n",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if n.text and n.text.strip()]
    print(f"[i] {len(nodes)} non-empty nodes from {len(documents)} documents")
    
    # Extract document title from metadata
    doc_title = documents[0].metadata.get("source_file", "Handbook") if documents else "Handbook"
    doc_title = doc_title.replace("-", " ").replace("_", " ").title()
    
    # Embed and insert in batches
    batch_size = 10
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        
        # Clean and contextualize each chunk
        texts = []
        for n in batch:
            clean = _clean_text(n.text)
            clean = _strip_footers(clean)
            section = _detect_section(clean)
            page = n.metadata.get("page_label", "N/A")
            contextualized = _contextualize_chunk(clean, doc_title, section, page)
            texts.append(contextualized)
        
        # Get embeddings
        embeddings = embed_model._get_text_embeddings(texts)
        
        # Prepare ChromaDB data
        ids = [n.node_id for n in batch]
        metadatas = []
        for n in batch:
            clean = _clean_text(n.text)
            clean = _strip_footers(clean)
            meta = {
                "source": n.metadata.get("source_file", n.metadata.get("file_name", "unknown")),
                "page": n.metadata.get("page_label", "N/A"),
                "section": _detect_section(clean),
            }
            meta = {k: str(v) for k, v in meta.items()}
            metadatas.append(meta)
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[i] Inserted batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
    
    print(f"[OK] Indexed into '{COLLECTION_NAME}'.")
    print(f"[OK] Collection now has {collection.count()} total chunks.")
    
    # Return a VectorStoreIndex wrapper for compatibility
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def load_existing_index() -> VectorStoreIndex:
    """Load index from existing ChromaDB collection for querying."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        raise ValueError(f"Collection '{COLLECTION_NAME}' is empty. Run ingestion first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )