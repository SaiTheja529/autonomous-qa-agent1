from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
import executor
from ingestion import (
    UPLOAD_DIR,
    CHECKOUT_PATH,
    ingest_files,
    save_checkout_html,
    load_checkout_excerpt,
    load_checkout_html,
)
from models import (
    GenerateScriptRequest,
    GenerateScriptResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
    IngestResponse,
    RunRecord,
    RunTestCaseRequest,
    TestCaseRecord,
    TestCaseUpdate,
)
from rag import generate_selenium_script, generate_test_cases

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Autonomous QA Agent Backend",
    version="0.1.0",
    description="FastAPI backend powering ingestion, test case generation, and Selenium scripts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve checkout HTML statically at /checkout/...
app.mount(
    "/checkout",
    StaticFiles(directory=str(CHECKOUT_PATH.parent), html=True),
    name="checkout",
)


def _normalize_ingest_result(result: Any) -> dict:
    """
    Convert whatever ingest_files returned into a dict with keys:
      - docs_ingested (int)
      - chunks_added (int)
      - sources (list)
    Safe: tolerates dict, object with attributes, or None.
    """
    out = {"docs_ingested": 0, "chunks_added": 0, "sources": []}
    if result is None:
        return out
    # dict-like
    if isinstance(result, dict):
        out["docs_ingested"] = int(result.get("docs_ingested") or result.get("docs") or 0)
        out["chunks_added"] = int(result.get("chunks_added") or result.get("chunks") or 0)
        out["sources"] = list(result.get("sources") or result.get("src") or out["sources"])
        return out
    # object-like
    docs = getattr(result, "docs_ingested", None)
    if docs is None:
        docs = getattr(result, "docs", None)
    chunks = getattr(result, "chunks_added", None)
    if chunks is None:
        chunks = getattr(result, "chunks", None)
    sources = getattr(result, "sources", None) or getattr(result, "src", None) or []
    try:
        out["docs_ingested"] = int(docs) if docs is not None else 0
    except Exception:
        out["docs_ingested"] = 0
    try:
        out["chunks_added"] = int(chunks) if chunks is not None else 0
    except Exception:
        out["chunks_added"] = 0
    try:
        out["sources"] = list(sources)
    except Exception:
        out["sources"] = []
    return out


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    reset: bool = Form(False),
    docs: List[UploadFile] = File(default_factory=list),
    checkout: Optional[UploadFile] = File(None),
):
    """
    Upload support docs + checkout HTML and build the knowledge base.
    This handler is robust to ingest_files returning either a dict, an object with
    attributes, or None.
    """
    if not docs and checkout is None:
        raise HTTPException(status_code=400, detail="Upload at least one document or checkout.html.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for file in docs:
        content = await file.read()
        dest = UPLOAD_DIR / file.filename
        dest.write_bytes(content)
        saved_paths.append(dest)
        try:
            db.upsert_document(file.filename, hashlib.sha256(content).hexdigest())
        except Exception as exc:
            logger.warning("Failed to record document %s in DB: %s", file.filename, exc)

    html_saved = False
    if checkout:
        content = await checkout.read()
        save_checkout_html(content, filename=checkout.filename)
        html_saved = True

    # Call existing ingest_files (could return object or dict)
    try:
        result = ingest_files(saved_paths, reset=reset)
    except Exception as exc:
        logger.exception("Ingestion failed (exception in ingest_files): %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    norm = _normalize_ingest_result(result)

    return IngestResponse(
        docs_ingested=norm["docs_ingested"],
        chunks_added=norm["chunks_added"],
        sources=norm["sources"],
        html_saved=html_saved,
    )


@app.post("/generate-testcases", response_model=GenerateTestCasesResponse)
def generate_testcases_endpoint(request: GenerateTestCasesRequest):
    return generate_test_cases(request.query, top_k=request.top_k)


@app.post("/generate-script", response_model=GenerateScriptResponse)
def generate_script_endpoint(request: GenerateScriptRequest):
    return generate_selenium_script(
        request.test_case,
        top_k=request.top_k,
        base_url=request.base_url,
    )


@app.get("/testcases", response_model=List[TestCaseRecord])
def list_testcases(status: Optional[str] = None):
    """List persisted test cases, optionally filtered by status."""
    return [db.test_case_to_record(row) for row in db.list_test_cases(status=status)]


@app.patch("/testcases/{test_case_id}", response_model=TestCaseRecord)
def patch_testcase(test_case_id: int, update: TestCaseUpdate):
    """Partially update a persisted test case."""
    row = db.update_test_case(test_case_id, update.model_dump(exclude_unset=True))
    if row is None:
        raise HTTPException(status_code=404, detail="Test case not found.")
    return db.test_case_to_record(row)


@app.delete("/testcases/{test_case_id}", status_code=204)
def delete_testcase(test_case_id: int):
    """Delete a persisted test case."""
    if not db.delete_test_case(test_case_id):
        raise HTTPException(status_code=404, detail="Test case not found.")


def _strip_code_fence(code: str) -> str:
    """Remove a leading/trailing markdown code fence if Gemini wrapped the script."""
    text = (code or "").strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _case_to_prompt_text(case) -> str:
    return (
        f"{case.feature} - {case.scenario}. "
        f"Steps: {case.steps or ''}. Expected: {case.expected or ''}"
    )


@app.post("/testcases/{test_case_id}/run", response_model=RunRecord)
async def run_testcase(test_case_id: int, request: Optional[RunTestCaseRequest] = None):
    """
    Execute a test case: generate or accept a Selenium script, persist it,
    create a runs row, and execute asynchronously in the background.
    """
    case = db.get_test_case(test_case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Test case not found.")

    request = request or RunTestCaseRequest()

    if request.script_code and request.script_code.strip():
        code = _strip_code_fence(request.script_code)
        base_url = request.base_url or ""
    else:
        generated = generate_selenium_script(
            _case_to_prompt_text(case),
            top_k=request.top_k,
            base_url=request.base_url or "http://localhost:8000/checkout/checkout.html",
        )
        code = _strip_code_fence(generated.script)
        base_url = request.base_url or "http://localhost:8000/checkout/checkout.html"

    script = db.create_script(code=code, test_case_id=test_case_id, base_url=base_url)
    run = db.create_run(script_id=script.id)
    asyncio.create_task(executor.execute_run(run.id, script.id))
    return db.run_to_record(run)


@app.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: int):
    """Get run status, full log, log tail, and screenshot URL."""
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return db.run_to_record(row)


@app.get("/runs/{run_id}/screenshot")
def get_run_screenshot(run_id: int):
    """Serve the screenshot captured for a run."""
    row = db.get_run(run_id)
    if row is None or not row.screenshot_path:
        raise HTTPException(status_code=404, detail="Screenshot not found.")
    path = (db.STORAGE_DIR / row.screenshot_path).resolve()
    if not path.exists() or db.STORAGE_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Screenshot not found.")
    return FileResponse(path)


@app.get("/checkout-html")
def checkout_html():
    """
    Convenience endpoint to fetch the current checkout HTML.
    """
    return {"html": load_checkout_html(), "excerpt": load_checkout_excerpt()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
