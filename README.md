# Autonomous QA Agent

An end-to-end QA assistant that ingests project documentation and web pages, builds a per-project knowledge base, generates grounded test cases, and emits runnable Python Selenium scripts — including remote execution against a Selenium Grid with screenshots and a coverage/traceability matrix.

The backend is FastAPI with a Chroma vector store; the UI is Streamlit. All generated artifacts (documents, test cases, scripts, runs) are persisted in SQLite.

## Features

- **Document ingestion** — upload Markdown, TXT, JSON, and PDF files; they are parsed, chunked, embedded, and stored in a Chroma vector store
- **Multi-page support** — upload HTML files or crawl a URL (BeautifulSoup DOM snapshot) as test targets, served back via `/checkout/*`
- **Per-project isolation** — every dataset is scoped by `project_id`, with a dedicated Chroma collection `testing_brain_{project_id}` per project
- **Grounded test case generation** — Gemini produces structured JSON test cases (feature, scenario, steps, expected, grounded_in) retrieved from your docs
- **Stale detection** — re-ingesting changed documents automatically marks affected test cases as `stale`
- **Selenium script generation** — emits runnable Python scripts against the selected page, with an optional LLM **review** pass that critiques the script
- **Streaming generation** — `POST /generate-script/stream` streams the script token-by-token
- **Test execution engine** — runs stored scripts via subprocess against a remote Selenium Grid (`REMOTE_WEBDRIVER_URL`), capturing logs and failure screenshots
- **Coverage matrix** — document ↔ test-case traceability via `GET /coverage`
- **Exports** — export test cases as CSV or a pytest suite (`GET /testcases/export`)
- **API-key auth** — optional `X-API-Key` header protection on all endpoints except `/health` and `/checkout*`
- **Docker Compose** — one-command stack: backend + Streamlit frontend + Selenium Chrome Grid

## Stack

- FastAPI + Uvicorn for the API
- Streamlit for the UI
- ChromaDB + sentence-transformers for embeddings and retrieval
- Google Gemini (`google-genai`) for generation (set `GEMINI_API_KEY`)
- SQLAlchemy + SQLite for persistence (documents, test cases, scripts, runs, pages)
- Selenium 4 for emitted scripts and the execution engine
- BeautifulSoup4 for page crawling/DOM snapshots

## Repository Layout

- `backend/` — FastAPI app, ingestion + RAG utilities, execution engine
  - `main.py` — API endpoints
  - `ingestion.py` — parsing/chunking/embeddings, page storage
  - `rag.py` — LLM calls (test cases, scripts, review, streaming)
  - `executor.py` — Selenium run execution against remote WebDriver
  - `db.py` — SQLite persistence layer
  - `auth.py` — API-key auth dependency
  - `models.py` — Pydantic models
  - `checkout/checkout.html` — sample target UI
  - `storage/` — Chroma persistence + uploaded docs + `app.db` SQLite database + screenshots (created at runtime)
- `frontend/app.py` — Streamlit UI
- `support_docs/` — example documents used to seed the knowledge base
- `docs/ROADMAP.md` — multi-phase feature roadmap and status
- `Dockerfile`, `docker-compose.yml` — containerized deployment
- `requirements.txt` — Python dependencies

## Setup

1. Python 3.10+ recommended.
2. Install dependencies (use your virtualenv):

   ```bash
   pip install -r requirements.txt
   ```

3. Environment variables:

   | Variable | Required | Default | Description |
   |---|---|---|---|
   | `GEMINI_API_KEY` | Yes (for generation) | — | Google Gemini API key |
   | `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
   | `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | sentence-transformers model |
   | `BACKEND_URL` | No | `http://localhost:8000` | Backend URL used by Streamlit |
   | `DATABASE_URL` | No | `sqlite:///backend/storage/app.db` | SQLAlchemy database URL |
   | `API_KEYS` / `API_KEY` | No | unset (auth disabled) | Comma-separated keys accepted via `X-API-Key` header |
   | `PROJECT_ID` / `API_KEY` | No | — | Streamlit sidebar defaults |
   | `BACKEND_PUBLIC_URL` | No | `http://localhost:8000` | Used to build page base URLs in generated scripts |
   | `REMOTE_WEBDRIVER_URL` | No | `http://localhost:4444/wd/hub` | Selenium Grid endpoint for script execution |
   | `RUN_TIMEOUT_SECONDS` | No | `300` | Timeout for a single test-case run |

## Running

Open two terminals:

```bash
# Terminal 1: FastAPI
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Streamlit UI
streamlit run frontend/app.py
```

### Docker Compose

```bash
docker compose up --build
```

This starts the backend (`:8000`), Streamlit UI (`:8501`), and a Selenium Chrome standalone container (`:4444`). Set `GEMINI_API_KEY` in your environment or a `.env` file before running. Data persists in the `backend-storage` volume.

## Usage Flow

1. In Streamlit, set a `project_id` in the sidebar, upload support docs (MD/TXT/JSON/PDF) and page HTML files (or crawl a URL), then click **Build Knowledge Base** (optionally reset existing vectors). Re-ingesting changed docs marks affected test cases stale.
2. Enter a request such as "Generate positive and negative test cases for the discount code feature." Click **Generate Test Cases** to get structured test cases (JSON) grounded in the uploaded docs. Generated cases are persisted to SQLite and shown as a selectable table.
3. On the **Test Cases** page you can load previously saved cases, filter by status (`active`, `draft`, `archived`, `stale`), update status, delete them, or export to CSV/pytest.
4. On **Selenium Scripts**, select a persisted test case (or paste one manually) and click **Generate Selenium Script** — optionally enable review mode or stream the output. The backend fetches the page HTML + context and returns a runnable Python Selenium script.
5. Execute a test case via `POST /testcases/{id}/run` (or the UI run action): the stored script runs against the remote WebDriver, and the run record captures status, log, and a failure screenshot retrievable via `GET /runs/{id}/screenshot`.
6. Copy the script and run it against a served copy of the page (adjust `base_url` inside the script as needed).

## API Quick Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Status check (unauthenticated) |
| `/ingest` | POST | Multipart upload: `docs` files[], optional `checkout`, `pages` files[]; form fields `reset`, `project_id`, `chunk_size`, `overlap`, `top_k` |
| `/generate-testcases` | POST | JSON `{ "query": str, "top_k": int, "project_id": str }` |
| `/generate-script` | POST | JSON `{ "test_case": str, "top_k": int, "page_id": int, "project_id": str, "base_url": str, "review": bool }` |
| `/generate-script/stream` | POST | Same payload; streams the generated script |
| `/testcases` | GET | List persisted test cases (`?status=`, `?project_id=` filters) |
| `/testcases/export` | GET | Export test cases (`?format=csv` or `pytest`, `?project_id=`) |
| `/testcases/{id}` | PATCH | Partially update a test case (`feature`, `scenario`, `steps`, `expected`, `grounded_in`, `status`) |
| `/testcases/{id}` | DELETE | Delete a test case |
| `/testcases/{id}/run` | GET/POST | Generate/accept a script and execute it |
| `/runs/{id}` | GET | Run status and log |
| `/runs/{id}/screenshot` | GET | Run failure screenshot |
| `/coverage` | GET | Document↔test-case traceability matrix (`?project_id=`) |
| `/pages/upload` | POST | Multipart `files[]` HTML uploads + `project_id` form field |
| `/pages/crawl` | POST | JSON `{ "url": str, "project_id": str, "name": str }`; fetches the URL and stores a DOM snapshot |
| `/pages` | GET | List pages (`?project_id=`) |
| `/pages/{id}` | GET/DELETE | Get or delete a page |
| `/checkout-html` | GET | HTML/DOM snapshot of the selected page (`?page_id=`, `?project_id=`) |
| `/checkout/` | GET | Serves the selected page's HTML; `/checkout/page/{id}` serves a specific page; other `/checkout/*` paths fall through to the static mount |

All endpoints except `/health` and `/checkout*` require the `X-API-Key` header when `API_KEYS`/`API_KEY` is configured.

## Projects & Pages

- Every table (`documents`, `pages`, `test_cases`, `scripts`, `runs`, `settings`) carries a `project_id` column; pass `project_id` on requests to scope data.
- Each project gets its own Chroma collection `testing_brain_{project_id}`.
- Pages are stored as HTML files under `backend/pages/{project_id}/` with a DOM snapshot in the `pages` table; crawled pages are fetched and serialized with BeautifulSoup.

## Provided Assets

- `support_docs/product_specs.md` — pricing, discount, validation rules
- `support_docs/ui_ux_guide.txt` — UI/UX and accessibility guidelines
- `support_docs/api_endpoints.json` — sample API contracts
- `backend/checkout/checkout.html` — target E-Shop checkout page (products, cart, discount, validation, payment success messaging)

## Notes

- The backend persists vectors under `backend/storage/vector_store` and relational data in `backend/storage/app.db` (SQLite). Use the "Reset" option in the UI to rebuild vectors from scratch; test cases can be managed via the API/UI.
- If you run offline, ensure the embedding model is cached locally or swap `EMBEDDING_MODEL` to one you have available.
- No automated tests are included; run a manual smoke test by ingesting the bundled support docs + checkout HTML and generating one test case and script.
- See `docs/ROADMAP.md` for the phased feature plan (persistence, execution engine, self-healing locators, reporting, CI integration).
