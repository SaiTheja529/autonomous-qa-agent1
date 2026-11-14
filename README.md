# Autonomous QA Agent

An end-to-end QA assistant that ingests project documentation and checkout HTML, builds a knowledge base, generates grounded test cases, and emits runnable Python Selenium scripts. The backend is FastAPI with a Chroma vector store; the UI is Streamlit.

## Stack
- FastAPI + Uvicorn for the API
- Streamlit for the UI
- ChromaDB + sentence-transformers for embeddings and retrieval
- Google Gemini (`google-genai`) for generation (set `GEMINI_API_KEY`)
- SQLAlchemy + SQLite for persistence (documents, test cases, scripts, runs)
- Selenium 4 for emitted scripts

## Repository Layout
- `backend/` — FastAPI app, ingestion + RAG utilities, checkout asset
  - `main.py` (API), `ingestion.py` (parsing/chunking/embeddings), `rag.py` (LLM calls), `models.py` (Pydantic), `db.py` (SQLite persistence)
  - `checkout/checkout.html` — sample target UI
  - `storage/` — Chroma persistence + uploaded docs + `app.db` SQLite database (created at runtime)
- `frontend/app.py` — Streamlit UI
- `support_docs/` — Example documents used to seed the knowledge base
- `docs/ROADMAP.md` — multi-phase feature roadmap and status
- `requirements.txt` — Python dependencies

## Setup
1) Python 3.10+ recommended.  
2) Install dependencies (use your virtualenv):
```bash
pip install -r requirements.txt
```
3) Environment variables:
- `GEMINI_API_KEY` (required for generation)
- `GEMINI_MODEL` (optional, default `gemini-2.5-flash`)
- `EMBEDDING_MODEL` (optional, default `all-MiniLM-L6-v2`)
- `BACKEND_URL` (optional for Streamlit, default `http://localhost:8000`)
- `DATABASE_URL` (optional, default `sqlite:///backend/storage/app.db`)
- `API_KEYS` or `API_KEY` (optional; comma-separated keys accepted via the `X-API-Key` header. When unset, auth is disabled)
- `PROJECT_ID` / `API_KEY` (optional Streamlit defaults for the sidebar inputs)
- `BACKEND_PUBLIC_URL` (optional, default `http://localhost:8000`; used to build page base URLs in generated scripts)

## Running
Open two terminals:
```bash
# Terminal 1: FastAPI
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Streamlit UI
streamlit run frontend/app.py
```

## Usage Flow
1. In Streamlit, upload support docs (MD/TXT/JSON/PDF) and the `checkout.html` file, then click **Build Knowledge Base** (optionally reset existing vectors).  
2. Enter a request such as “Generate positive and negative test cases for the discount code feature.” Click **Generate Test Cases** to get structured test cases (JSON) grounded in the uploaded docs. Generated cases are persisted to SQLite and shown as a selectable table.  
3. On the **Test Cases** page you can also load previously saved cases, filter by status, update status, or delete them. On **Selenium Scripts**, select a persisted test case (or paste one manually) and click **Generate Selenium Script**. The backend fetches the checkout HTML + context and returns a runnable Python Selenium script.  
4. Copy the script and run it against a served copy of `backend/checkout/checkout.html` (adjust `base_url` inside the script as needed).

## API Quick Reference
- `GET /health` — status check (unauthenticated)  
- `POST /ingest` — multipart upload (`docs` files[], optional `checkout`, `pages` files[] for multiple HTML pages), form fields `reset`, `project_id`, `chunk_size`, `overlap`, `top_k`  
- `POST /generate-testcases` — JSON `{ "query": str, "top_k": int, "project_id": str }`  
- `POST /generate-script` — JSON `{ "test_case": str, "top_k": int, "page_id": int, "project_id": str, "base_url": str, "review": bool }`  
- `POST /generate-script/stream` — same payload as `/generate-script`, returns a streamed script  
- `GET /testcases` — list persisted test cases (optional `?status=`, `?project_id=` filters)  
- `PATCH /testcases/{id}` — partially update a test case (`feature`, `scenario`, `steps`, `expected`, `grounded_in`, `status`)  
- `DELETE /testcases/{id}` — delete a test case  
- `GET /testcases/{id}/run` → `POST /testcases/{id}/run` — generate/accept a script and execute it  
- `GET /runs/{id}` / `GET /runs/{id}/screenshot` — run status, log, screenshot  
- `GET /coverage` — document↔test-case traceability matrix (optional `?project_id=`)  
- `POST /pages/upload` — multipart `files[]` HTML uploads + `project_id` form field  
- `POST /pages/crawl` — JSON `{ "url": str, "project_id": str, "name": str }`; fetches the URL and stores a BeautifulSoup DOM snapshot  
- `GET /pages` / `GET /pages/{id}` / `DELETE /pages/{id}` — page management  
- `GET /checkout-html` — HTML/DOM snapshot of the selected page (optional `?page_id=`, `?project_id=`)  
- `GET /checkout/` — serves the selected page's HTML (`?page_id=` or latest for `?project_id=`); `GET /checkout/page/{id}` serves a specific page; other `/checkout/*` paths fall through to the static mount

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
- The backend persists vectors under `backend/storage/vector_store` and relational data in `backend/storage/app.db` (SQLite). Use the “Reset” option in the UI to rebuild vectors from scratch; test cases can be managed via the API/UI.
- If you run offline, ensure the embedding model is cached locally or swap `EMBEDDING_MODEL` to one you have available.
- No automated tests are included; run a manual smoke test by ingesting the bundled support docs + checkout HTML and generating one test case and script.
