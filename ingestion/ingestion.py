r"""
Per-file ingestion script for HR policy documents.
Run manually for each file:  python ingestion.py "path\to\policy.pdf"
Uses PyMuPDF for clean text extraction.
"""

import sys
import argparse
import re
from pathlib import Path

# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

import fitz  # PyMuPDF
from llama_index.core import Document

from index import build_index_from_documents, _strip_footers

# ── Config ─────────────────────────────────────────────────────────────
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
SUPPORTED_EXTS = [".pdf", ".docx", ".txt", ".md"]


def _clean_text(text: str) -> str:
    """Remove PDF extraction artifacts and normalize whitespace."""
    # Remove control chars except \n, \r, \t
    text = ''.join(c for c in text if c in '\n\r\t' or ord(c) >= 32 and ord(c) != 127)
    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def resolve_file_path(file_arg: str) -> Path:
    """Resolve file path from argument (absolute, relative, or just filename)."""
    path = Path(file_arg)

    if not path.is_absolute() and len(path.parts) == 1:
        path = RAW_DOCS_DIR / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTS)}"
        )

    return path


def ingest_file(file_path: Path) -> None:
    """Ingest a single PDF into ChromaDB using PyMuPDF."""
    print(f"[i] Loading: {file_path.name}")

    doc = fitz.open(str(file_path))
    pages_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text and text.strip():
            pages_text.append(text)

    doc.close()

    full_text = "\n\n".join(pages_text)
    clean_text = _clean_text(full_text)

    # Create LlamaIndex Document
    documents = [Document(
        text=clean_text,
        metadata={
            "source_file": file_path.name,
            "ingested_at": str(Path(__file__).stat().st_mtime),
            "total_pages": str(len(pages_text)),
        }
    )]

    print(f"[i] Extracted {len(pages_text)} pages, {len(clean_text)} chars")

    build_index_from_documents(documents)

    print(f"[OK] '{file_path.name}' ingested successfully.")


def main():
    parser = argparse.ArgumentParser(description="Ingest a single HR policy document")
    parser.add_argument("file", help="Filename (in raw_docs) or full path to document")
    args = parser.parse_args()

    file_path = resolve_file_path(args.file)
    ingest_file(file_path)


if __name__ == "__main__":
    main()