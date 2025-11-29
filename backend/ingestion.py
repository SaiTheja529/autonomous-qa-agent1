# backend/ingestion.py
"""
Utilities to ingest support documents and build a Chroma vector store.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# NOTE: heavy libs (chromadb, fitz, sentence-transformers) are imported lazily
# to avoid high memory usage at import time on small instances (Render 512MB).
# See _embedding_function(), _client(), and _extract_text() for lazy imports.

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "vector_store"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CHECKOUT_PATH = BASE_DIR / "checkout" / "checkout.html"
COLLECTION_NAME = "testing_brain"


@dataclass
class IngestResult:
    docs_ingested: int
    chunks_added: int
    sources: List[str]
    html_saved: bool


def _ensure_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHECKOUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --- Lightweight local embedding fallback (no heavy model) ---
class LocalHashEmbeddingFunction:
    """
    Lightweight embedding function that does not require any external
    model downloads. It uses scikit-learn's HashingVectorizer to turn
    text into numeric vectors deterministically.

    Provides the minimal interface Chroma expects:
      - callable to produce embeddings
      - name() -> str
      - get_config() -> dict (optional)
    """

    def __init__(self, n_features: int = 512):
        # Local import to keep top-level import light
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )
        self._n_features = n_features

    def __call__(self, input):
        if not input:
            return []
        matrix = self.vectorizer.transform(list(input))
        # Convert sparse matrix to a list of dense float lists
        return matrix.toarray().astype("float32").tolist()

    def name(self) -> str:
        # Chroma expects an embedding function to expose a name()
        return "local-hash"

    def get_config(self) -> dict:
        # Optional: provide a tiny config so chroma can compare embedding funcs
        return {"n_features": self._n_features}


def _embedding_function():
    """
    Returns the embedding function used by Chroma.

    By default, a local hashing-based embedding is used so the app works
    even in environments without network access. To use a Hugging Face
    sentence-transformer instead, set:

        EMBEDDING_BACKEND=sentence-transformer
        EMBEDDING_MODEL=all-MiniLM-L6-v2  # or another model
    """
    backend = os.getenv("EMBEDDING_BACKEND", "local-hash").lower()
    if backend == "sentence-transformer":
        # Lazy imports: only import chromadb embedding helper + model name when necessary
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("sentence-transformer embedding backend requested but chromadb utils are not available") from exc
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingFunction(model_name=model_name)
    return LocalHashEmbeddingFunction()


# Lazy client factory
def _client():
    """
    Create or return a chromadb PersistentClient. Lazy-import chromadb so that
    the module import does not allocate memory unless vector store is used.
    """
    _ensure_dirs()
    try:
        import chromadb  # type: ignore
    except Exception as exc:
        raise RuntimeError("chromadb is required for vector store operations but is not installed") from exc

    # Use PersistentClient instead of top-level PersistentClient import to avoid import-time work
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(reset: bool = False):
    """
    Returns the Chroma collection, optionally resetting it first.
    """
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Collection may not exist on first run; ignore.
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Paragraph-aware splitter that keeps chunks semantically tighter while
    staying within the configured size budget.
    """
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph is larger than the chunk size, fall back to
        # a sliding character window for that paragraph to avoid losing content.
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            length = len(para)
            while start < length:
                end = min(length, start + chunk_size)
                segment = para[start:end]
                chunks.append(segment)
                if end >= length:
                    break
                start = max(0, end - overlap)
            continue

        if not current:
            current = para
            continue

        candidate = current + "\n\n" + para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def _extract_text(path: Path) -> str:
    """
    Extract text from various document types.
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    if suffix == ".html" or suffix == ".htm":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        # Lazy import fitz (PyMuPDF) only when needed
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF (fitz) is required to parse PDF files but is not installed") from exc
        text_parts: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)
    # Fallback: best-effort UTF-8 decode
    return path.read_text(encoding="utf-8", errors="ignore")


def save_checkout_html(content: bytes, filename: str = "checkout.html") -> Path:
    """
    Persist the uploaded checkout HTML so the agent can reuse it for Selenium generation.
    """
    _ensure_dirs()
    CHECKOUT_PATH.write_bytes(content)
    return CHECKOUT_PATH


def ingest_files(file_paths: Iterable[Path], reset: bool = False) -> IngestResult:
    """
    Ingests a list of files into Chroma, chunking content and attaching metadata.
    """
    collection = get_collection(reset=reset)
    docs_ingested = 0
    chunks_added = 0
    sources: List[str] = []

    for path in file_paths:
        text = _extract_text(path)
        chunks = _chunk_text(text)
        if not chunks:
            continue

        # Lightweight console logging so you can see chunking behavior
        try:
            print(f"[INGEST] {path.name}: {len(chunks)} chunks")
            for idx, chunk in enumerate(chunks):
                preview = chunk.replace("\n", " ")[:120]
                print(f"[INGEST] {path.name} chunk {idx}: {len(chunk)} chars | {preview!r}")
        except Exception:
            # Logging should never break ingestion
            pass

        ids: List[str] = []
        metadatas: List[dict] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{path.name}-{uuid.uuid4()}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source": path.name,
                    "chunk": idx,
                    "filename": path.name,
                    "path": str(path),
                }
            )

        # Chroma accepts a plain list of strings; avoid instantiating the typing alias.
        collection.add(documents=list(chunks), metadatas=metadatas, ids=ids)
        docs_ingested += 1
        chunks_added += len(chunks)
        sources.append(path.name)

    return IngestResult(
        docs_ingested=docs_ingested,
        chunks_added=chunks_added,
        sources=sources,
        html_saved=False,
    )


def load_checkout_html() -> str:
    """
    Return the persisted checkout HTML as text. Raises FileNotFoundError if missing.
    """
    if not CHECKOUT_PATH.exists():
        raise FileNotFoundError("checkout.html not found. Upload it before generating scripts.")
    return CHECKOUT_PATH.read_text(encoding="utf-8")


def load_checkout_excerpt(max_chars: int = 1200) -> str:
    """
    Shortened version of the checkout HTML to echo back in the API.
    """
    html = load_checkout_html()
    return html[:max_chars]
