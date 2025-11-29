# Autonomous QA Agent

An end-to-end QA assistant that ingests project documentation and checkout HTML, builds a knowledge base, generates grounded test cases, and emits runnable Python Selenium scripts. The backend is FastAPI with a Chroma vector store; the UI is Streamlit.

## Stack
- FastAPI + Uvicorn for the API
- Streamlit for the UI
- ChromaDB + sentence-transformers for embeddings and retrieval
- OpenAI chat models for generation (set `OPENAI_API_KEY`)
- Selenium 4 for emitted scripts

## Repository Layout
- `backend/` — FastAPI app, ingestion + RAG utilities, checkout asset
  - `main.py` (API), `ingestion.py` (parsing/chunking/embeddings), `rag.py` (LLM calls), `models.py` (Pydantic)
  - `checkout/checkout.html` — sample target UI
  - `storage/` — Chroma persistence + uploaded docs (created at runtime)
- `frontend/app.py` — Streamlit UI
- `support_docs/` — Example documents used to seed the knowledge base
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
2. Enter a request such as “Generate positive and negative test cases for the discount code feature.” Click **Generate Test Cases** to get a Markdown table grounded in the uploaded docs.  
3. Select a test case from the rendered table (or paste one manually) and click **Generate Selenium Script**. The backend fetches the checkout HTML + context and returns a runnable Python Selenium script.  
4. Copy the script and run it against a served copy of `backend/checkout/checkout.html` (adjust `base_url` inside the script as needed).

## API Quick Reference
- `GET /health` — status check  
- `POST /ingest` — multipart upload (`docs` files[], optional `checkout`), form field `reset` (bool)  
- `POST /generate-testcases` — JSON `{ "query": str, "top_k": int }`  
- `POST /generate-script` — JSON `{ "test_case": str, "top_k": int }`  
- `GET /checkout-html` — returns stored checkout HTML

## Provided Assets
- `support_docs/product_specs.md` — pricing, discount, validation rules
- `support_docs/ui_ux_guide.txt` — UI/UX and accessibility guidelines
- `support_docs/api_endpoints.json` — sample API contracts
- `backend/checkout/checkout.html` — target E-Shop checkout page (products, cart, discount, validation, payment success messaging)

## Notes
- The backend persists vectors under `backend/storage/vector_store`. Use the “Reset” option in the UI to rebuild from scratch.
- If you run offline, ensure the embedding model is cached locally or swap `EMBEDDING_MODEL` to one you have available.
- No automated tests are included; run a manual smoke test by ingesting the bundled support docs + checkout HTML and generating one test case and script.
