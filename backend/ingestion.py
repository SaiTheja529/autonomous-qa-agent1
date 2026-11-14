# backend/ingestion.py
"""
Utilities to ingest support documents and build a Chroma vector store.
"""
from __future__ import annotations

import hashlib
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
PAGES_DIR = BASE_DIR / "pages"
COLLECTION_PREFIX = "testing_brain"
DEFAULT_PROJECT = "default"


def collection_name(project_id: str = DEFAULT_PROJECT) -> str:
    """Per-project Chroma collection name: ``testing_brain_{project_id}``."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (project_id or DEFAULT_PROJECT))
    return f"{COLLECTION_PREFIX}_{safe}"


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
    PAGES_DIR.mkdir(parents=True, exist_ok=True)


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

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        # Chroma may pass a single string for queries
        if isinstance(input, str):
            input = [input]
        return self(input)

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


def get_collection(reset: bool = False, project_id: str = DEFAULT_PROJECT):
    """
    Returns the Chroma collection for ``project_id``, optionally resetting it first.
    """
    name = collection_name(project_id)
    client = _client()
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            # Collection may not exist on first run; ignore.
            pass
    collection = client.get_or_create_collection(
        name=name,
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


# --- Page storage (uploads + crawls) -----------------------------------

def _page_dir(project_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (project_id or DEFAULT_PROJECT))
    path = PAGES_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_page_html(html: str, project_id: str = DEFAULT_PROJECT, page_key: Optional[str] = None) -> tuple[Path, str]:
    """
    Write page HTML into ``backend/pages/{project_id}/`` and return
    ``(absolute_path, path_relative_to_backend)``.
    """
    _ensure_dirs()
    key = page_key or uuid.uuid4().hex
    key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key) or uuid.uuid4().hex
    if key.startswith("."):
        key = "_" + key
    dest = _page_dir(project_id) / f"{key}.html"
    dest.write_text(html, encoding="utf-8")
    rel = str(dest.relative_to(BASE_DIR))
    return dest, rel


def load_page_file(rel_path: str) -> str:
    """Read a stored page file given its path relative to ``backend/``."""
    path = (BASE_DIR / rel_path).resolve()
    if BASE_DIR.resolve() not in path.parents or not path.exists():
        raise FileNotFoundError(f"Page file not found: {rel_path}")
    return path.read_text(encoding="utf-8")


def crawl_page(url: str, project_id: str = DEFAULT_PROJECT, timeout: int = 20) -> dict:
    """
    Fetch ``url``, parse it with BeautifulSoup, and return a DOM snapshot
    (prettified HTML) plus metadata for creating a Page row.
    """
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "autonomous-qa-agent/0.1 (+page-crawler)"},
        allow_redirects=True,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    snapshot = soup.prettify()
    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    return {
        "url": resp.url,
        "name": title or resp.url,
        "dom_snapshot": snapshot,
        "status_code": resp.status_code,
    }


def ingest_files(
    file_paths: Iterable[Path],
    reset: bool = False,
    chunk_size: int = 800,
    overlap: int = 150,
    top_k: Optional[int] = None,
    project_id: str = DEFAULT_PROJECT,
) -> IngestResult:
    """
    Ingests a list of files into Chroma, chunking content and attaching metadata.

    ``chunk_size``/``overlap`` control the text splitter; ``top_k`` (when
    provided) is persisted as the default retrieval depth for generation.
    """
    collection = get_collection(reset=reset, project_id=project_id)
    docs_ingested = 0
    chunks_added = 0
    sources: List[str] = []

    for path in file_paths:
        text = _extract_text(path)
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
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

        # Record the document in the relational store for traceability/coverage.
        try:
            import db  # lazy import: keeps module import light

            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            db.upsert_document(path.name, sha256, project_id=project_id)
        except Exception as exc:
            # Document bookkeeping should never break ingestion
            print(f"[INGEST] failed to record document {path.name}: {exc}")

        docs_ingested += 1
        chunks_added += len(chunks)
        sources.append(path.name)

    if top_k is not None:
        try:
            import db  # lazy import: keeps module import light

            db.set_setting("top_k", str(max(1, int(top_k))), project_id=project_id)
        except Exception as exc:
            print(f"[INGEST] failed to persist top_k setting: {exc}")

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


def page_html(page) -> str:
    """
    Return the HTML/DOM snapshot for a Page row: prefer the stored file,
    falling back to the dom_snapshot column.
    """
    if page.file_path:
        try:
            return load_page_file(page.file_path)
        except FileNotFoundError:
            pass
    if page.dom_snapshot:
        return page.dom_snapshot
    raise FileNotFoundError(f"Page #{page.id} has no stored HTML or DOM snapshot.")


def page_excerpt(page, max_chars: int = 1200) -> str:
    return page_html(page)[:max_chars]


def selected_page_html(page_id: Optional[int] = None, project_id: str = DEFAULT_PROJECT) -> Optional[str]:
    """
    HTML for the page selected for serving at /checkout: explicit ``page_id``,
    else the latest page for ``project_id``. Returns None when no page exists.
    """
    import db  # lazy import to keep module import light

    page = db.get_page(page_id) if page_id is not None else db.latest_page(project_id)
    if page is None:
        return None
    return page_html(page)


def default_page_base_url(project_id: str = DEFAULT_PROJECT, backend_url: str = "http://localhost:8000") -> str:
    """Base URL that serves the project's selected page via /checkout."""
    return f"{backend_url}/checkout/?project_id={project_id}"
