"""
Custom Gemini embedder using the new google-genai SDK.
With TPM (tokens-per-minute) rate limiting for free tier compliance.
"""

import os
import time
from typing import List

from google import genai
from google.genai import errors as genai_errors
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field


class GeminiGenAIEmbedder(BaseEmbedding):
    """LlamaIndex-compatible embedder using google-genai (free tier)."""
    
    api_key: str = Field(default="")
    model_name: str = Field(default="gemini-embedding-2")
    client: genai.Client = Field(default=None, exclude=True)
    max_tpm: int = Field(default=25000)
    last_request_time: float = Field(default=0.0)
    tokens_this_minute: int = Field(default=0)
    minute_start: float = Field(default=0.0)
    requests_this_minute: int = Field(default=0)
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-embedding-2", 
                 max_tpm: int = 25000, **kwargs):
        api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        super().__init__(
            api_key=api_key, 
            model_name=model_name, 
            max_tpm=max_tpm,
            **kwargs
        )
        self.client = genai.Client(api_key=api_key)
        self.minute_start = time.time()
    
    def _rate_limit(self, token_count: int):
        """Enforce TPM limit — wait if we'd exceed."""
        now = time.time()
        
        if now - self.minute_start >= 60:
            self.tokens_this_minute = 0
            self.requests_this_minute = 0
            self.minute_start = now
        
        if self.tokens_this_minute + token_count > self.max_tpm:
            wait = 60 - (now - self.minute_start) + 1
            print(f"[!] TPM limit approaching ({self.tokens_this_minute}/{self.max_tpm}). "
                  f"Waiting {wait:.0f}s for next minute...")
            time.sleep(wait)
            self.tokens_this_minute = 0
            self.requests_this_minute = 0
            self.minute_start = time.time()
        
        self.tokens_this_minute += token_count
        self.requests_this_minute += 1
    
    def _embed_with_retry(self, texts: List[str], max_retries: int = 2) -> List[List[float]]:
        """Embed one text at a time to avoid google-genai batching issues."""
        embeddings = []
        
        for text in texts:
            estimated_tokens = max(len(text) // 4, 100)
            self._rate_limit(estimated_tokens)
            
            for attempt in range(max_retries):
                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                    )
                    embeddings.append(result.embeddings[0].values)
                    break
                except genai_errors.ClientError as e:
                    if e.code == 429 and attempt == 0:
                        print(f"[!] Rate limited (429). Waiting 60s for quota reset...")
                        time.sleep(60)
                        self.tokens_this_minute = 0
                        self.requests_this_minute = 0
                        self.minute_start = time.time()
                    else:
                        raise
        return embeddings
    
    def _get_embedding(self, text: str) -> List[float]:
        return self._embed_with_retry([text])[0]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_with_retry(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)