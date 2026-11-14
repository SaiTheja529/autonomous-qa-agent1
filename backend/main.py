from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import db
import executor
from auth import require_api_key
from ingestion import (
    UPLOAD_DIR,
    CHECKOUT_PATH,
    crawl_page,
    ingest_files,
    save_checkout_html,
    load_checkout_excerpt,
    load_checkout_html,
    page_html,
    selected_page_html,
    store_page_html,
)
from models import (
    CoverageDocument,
    CoverageResponse,
    CoverageTestCase,
    CrawlPageRequest,
    GenerateScriptRequest,
    GenerateScriptResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
    IngestResponse,
    PageRecord,
    RunRecord,
    RunTestCaseRequest,
    TestCaseRecord,
    TestCaseUpdate,
)
from rag import generate_selenium_script, generate_test_cases, iter_script_stream

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

# NOTE: the /checkout static mount is registered near the bottom of this
# module, AFTER the /checkout/ and /checkout/page/{id} routes, so those
# routes take precedence and can serve the selected page dynamically.


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


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_endpoint(
    reset: bool = Form(False),
    project_id: str = Form("default"),
    docs: List[UploadFile] = File(default_factory=list),
    checkout: Optional[UploadFile] = File(None),
    pages: List[UploadFile] = File(default_factory=list),
    chunk_size: int = Form(800),
    overlap: int = Form(150),
    top_k: int = Form(5),
):
    """
    Upload support docs + page HTML and build the project's knowledge base.
    ``pages`` accepts multiple HTML files; each becomes a Page row whose DOM
    snapshot can be served at /checkout and targeted by script generation.
    This handler is robust to ingest_files returning either a dict, an object with
    attributes, or None.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk settings: require chunk_size > 0 and 0 <= overlap < chunk_size.",
        )
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1.")
    if not docs and checkout is None and not pages:
        raise HTTPException(status_code=400, detail="Upload at least one document or page HTML.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    changed_docs: List[str] = []
    for file in docs:
        content = await file.read()
        dest = UPLOAD_DIR / file.filename
        dest.write_bytes(content)
        saved_paths.append(dest)
        try:
            sha = hashlib.sha256(content).hexdigest()
            existing = db.get_document(file.filename, project_id=project_id)
            if existing is not None and existing.sha256 != sha:
                changed_docs.append(file.filename)
            db.upsert_document(file.filename, sha, project_id=project_id)
        except Exception as exc:
            logger.warning("Failed to record document %s in DB: %s", file.filename, exc)

    html_saved = False
    page_records: List[PageRecord] = []
    if checkout:
        content = await checkout.read()
        save_checkout_html(content, filename=checkout.filename)
        html_saved = True
        try:
            _, rel = store_page_html(
                content.decode("utf-8", errors="replace"),
                project_id=project_id,
                page_key=f"checkout-{checkout.filename or 'page'}",
            )
            row = db.create_page(
                project_id=project_id,
                name=checkout.filename or "checkout.html",
                source_type="upload",
                file_path=rel,
                dom_snapshot=content.decode("utf-8", errors="replace"),
            )
            page_records.append(db.page_to_record(row))
        except Exception as exc:
            logger.warning("Failed to register checkout page: %s", exc)

    for page_file in pages:
        try:
            content = await page_file.read()
            html_text = content.decode("utf-8", errors="replace")
            _, rel = store_page_html(
                html_text, project_id=project_id, page_key=f"upload-{page_file.filename or 'page'}"
            )
            row = db.create_page(
                project_id=project_id,
                name=page_file.filename or "page.html",
                source_type="upload",
                file_path=rel,
                dom_snapshot=html_text,
            )
            page_records.append(db.page_to_record(row))
        except Exception as exc:
            logger.warning("Failed to store page upload %s: %s", page_file.filename, exc)

    # Call existing ingest_files (could return object or dict)
    try:
        result = ingest_files(
            saved_paths,
            reset=reset,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
            project_id=project_id,
        )
    except Exception as exc:
        logger.exception("Ingestion failed (exception in ingest_files): %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    norm = _normalize_ingest_result(result)

    stale_marked = 0
    if changed_docs:
        try:
            stale_marked = db.mark_test_cases_stale(changed_docs, project_id=project_id)
        except Exception as exc:
            logger.warning("Failed to mark stale test cases: %s", exc)

    return IngestResponse(
        docs_ingested=norm["docs_ingested"],
        chunks_added=norm["chunks_added"],
        sources=norm["sources"],
        html_saved=html_saved,
        changed_docs=changed_docs,
        stale_marked=stale_marked,
        pages_added=len(page_records),
        pages=page_records,
    )


@app.post("/generate-testcases", response_model=GenerateTestCasesResponse, dependencies=[Depends(require_api_key)])
def generate_testcases_endpoint(request: GenerateTestCasesRequest):
    return generate_test_cases(request.query, top_k=request.top_k, project_id=request.project_id)


@app.post("/generate-script", response_model=GenerateScriptResponse, dependencies=[Depends(require_api_key)])
def generate_script_endpoint(request: GenerateScriptRequest, review: bool = False):
    return generate_selenium_script(
        request.test_case,
        top_k=request.top_k,
        base_url=request.base_url,
        review=request.review or review,
        page_id=request.page_id,
        project_id=request.project_id,
    )


@app.post("/generate-script/stream", dependencies=[Depends(require_api_key)])
def generate_script_stream_endpoint(request: GenerateScriptRequest):
    """
    Stream a Selenium script (and optional review) as plain text.

    First chunk is a ``__META__<json>`` header line with contexts/excerpt;
    script chunks follow, then a ``--- REVIEW ---`` section when review=true.
    """
    stream = iter_script_stream(
        request.test_case,
        top_k=request.top_k,
        base_url=request.base_url,
        review=request.review,
        page_id=request.page_id,
        project_id=request.project_id,
    )
    return StreamingResponse(stream, media_type="text/plain; charset=utf-8")


@app.get("/testcases", response_model=List[TestCaseRecord], dependencies=[Depends(require_api_key)])
def list_testcases(status: Optional[str] = None, project_id: Optional[str] = None):
    """List persisted test cases, optionally filtered by status and project."""
    return [
        db.test_case_to_record(row)
        for row in db.list_test_cases(status=status, project_id=project_id)
    ]


@app.patch("/testcases/{test_case_id}", response_model=TestCaseRecord, dependencies=[Depends(require_api_key)])
def patch_testcase(test_case_id: int, update: TestCaseUpdate):
    """Partially update a persisted test case."""
    row = db.update_test_case(test_case_id, update.model_dump(exclude_unset=True))
    if row is None:
        raise HTTPException(status_code=404, detail="Test case not found.")
    return db.test_case_to_record(row)


@app.delete("/testcases/{test_case_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_testcase(test_case_id: int):
    """Delete a persisted test case."""
    if not db.delete_test_case(test_case_id):
        raise HTTPException(status_code=404, detail="Test case not found.")


# --- Test case export ----------------------------------------------------

_CSV_FIELDS = ["id", "feature", "scenario", "steps", "expected", "grounded_in", "status", "created_at"]

_PYTEST_HEADER = '''"""Auto-generated pytest suite for autonomous QA test cases."""
import os

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


@pytest.fixture
def driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Remote(
        command_executor=os.environ.get("REMOTE_WEBDRIVER_URL", "http://localhost:4444/wd/hub"),
        options=options,
    )
    yield driver
    driver.quit()

'''

# Matches the standard driver/options bootstrap block emitted by the generator.
_DRIVER_BLOCK_RE = re.compile(
    r"(?ms)^[ \t]*options\s*=\s*Options\(\)[ \t]*\n"
    r"(?:[ \t]*options\.add_argument\([^\n]*\)[ \t]*\n)*"
    r"[ \t]*driver\s*=\s*webdriver\.Remote\([\s\S]*?^[ \t]*\)[ \t]*\n"
)
# Fallback: remove just the webdriver.Remote(...) assignment.
_REMOTE_ONLY_RE = re.compile(
    r"(?ms)^[ \t]*driver\s*=\s*webdriver\.Remote\([\s\S]*?^[ \t]*\)[ \t]*\n"
)
# Single-line variant: driver = webdriver.Remote(...)
_REMOTE_INLINE_RE = re.compile(
    r"(?m)^[ \t]*driver\s*=\s*webdriver\.Remote\([^\n]*\)[ \t]*\n"
)
_MAIN_DEF_RE = re.compile(
    r"(?ms)^[ \t]*def main\(\)\s*:[ \t]*\n(?P<body>(?:[ \t]+[^\n]*\n?|[ \t]*\n)*)"
)
_GUARD_RE = re.compile(
    r"(?ms)^[ \t]*if\s+__name__\s*==\s*[\"']__main__[\"']\s*:[ \t]*\n(?P<body>(?:[ \t]+[^\n]*\n?|[ \t]*\n)*)"
)
_QUIT_LINE_RE = re.compile(r"(?m)^([ \t]*)driver\.quit\(\)[ \t]*$")


def _dedent_block(block: str, spaces: int = 4) -> str:
    """Remove up to ``spaces`` leading spaces from each non-empty line."""
    out = []
    for line in block.splitlines():
        if line.strip() and line.startswith(" " * spaces):
            out.append(line[spaces:])
        elif line.strip():
            out.append(line.lstrip())
        else:
            out.append("")
    return "\n".join(out)


def _script_body_for_pytest(code: str) -> str:
    """
    Convert a generated standalone script into a function body that uses the
    ``driver`` pytest fixture: strip the driver bootstrap, inline ``main()``
    and the ``__main__`` guard, and neutralize ``driver.quit()`` calls (the
    fixture handles teardown).
    """
    text = _strip_code_fence(code or "")
    text = _DRIVER_BLOCK_RE.sub("", text)
    text = _REMOTE_ONLY_RE.sub("", text)
    text = _REMOTE_INLINE_RE.sub("", text)

    main_def = _MAIN_DEF_RE.search(text)
    if main_def:
        inlined = _dedent_block(main_def.group("body"))
        text = text[: main_def.start()] + inlined + "\n" + text[main_def.end() :]

    guard = _GUARD_RE.search(text)
    if guard:
        calls = [
            ln.strip()
            for ln in guard.group("body").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if calls == ["main()"] and main_def:
            replacement = ""  # main() body was already inlined above
        elif calls == ["main()"]:
            replacement = ""  # nothing else to run; avoid a NameError
        else:
            replacement = _dedent_block(guard.group("body"))
        text = text[: guard.start()] + replacement + text[guard.end() :]

    text = _QUIT_LINE_RE.sub(r"\1pass", text)
    return text.strip("\n")


def _build_pytest_export(cases: List) -> str:
    parts = [_PYTEST_HEADER]
    for case in cases:
        func_name = f"test_case_{case.id}"
        doc = (case.scenario or "").replace('"""', "'''").strip()
        parts.append(f'\n\ndef {func_name}(driver):\n    """{doc}"""\n')
        script = db.get_latest_script_for_test_case(case.id)
        body = _script_body_for_pytest(script.code) if script else ""
        if not body:
            parts.append("    pytest.skip('No Selenium script available for this test case')\n")
            continue
        for line in body.splitlines():
            parts.append(f"    {line}\n" if line.strip() else "\n")
    return "".join(parts)


def _build_csv_export(cases: List) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for case in cases:
        writer.writerow(
            {
                "id": case.id,
                "feature": case.feature,
                "scenario": case.scenario,
                "steps": case.steps or "",
                "expected": case.expected or "",
                "grounded_in": case.grounded_in or "",
                "status": case.status or "",
                "created_at": case.created_at.isoformat() if case.created_at else "",
            }
        )
    return buf.getvalue()


@app.get("/testcases/export", dependencies=[Depends(require_api_key)])
def export_testcases(format: str = "csv", project_id: Optional[str] = None):
    """Export test cases as CSV or a runnable pytest suite."""
    fmt = (format or "csv").lower()
    if fmt not in {"csv", "pytest"}:
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'pytest'.")
    cases = db.list_test_cases(project_id=project_id)
    if fmt == "csv":
        return Response(
            content=_build_csv_export(cases),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="testcases.csv"'},
        )
    return Response(
        content=_build_pytest_export(cases),
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="testcases_pytest.py"'},
    )


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


@app.post("/testcases/{test_case_id}/run", response_model=RunRecord, dependencies=[Depends(require_api_key)])
async def run_testcase(test_case_id: int, request: Optional[RunTestCaseRequest] = None):
    """
    Execute a test case: generate or accept a Selenium script, persist it,
    create a runs row, and execute asynchronously in the background.
    """
    case = db.get_test_case(test_case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Test case not found.")

    request = request or RunTestCaseRequest()
    project_id = request.project_id or case.project_id or "default"
    base_url = request.base_url or ""

    if request.script_code and request.script_code.strip():
        code = _strip_code_fence(request.script_code)
    else:
        generated = generate_selenium_script(
            _case_to_prompt_text(case),
            top_k=request.top_k,
            base_url=request.base_url,
            project_id=project_id,
        )
        code = _strip_code_fence(generated.script)

    if not base_url and not request.script_code:
        from ingestion import default_page_base_url
        from rag import BACKEND_PUBLIC_URL

        base_url = default_page_base_url(project_id, backend_url=BACKEND_PUBLIC_URL)
    base_url = base_url or ""

    script = db.create_script(
        code=code, test_case_id=test_case_id, base_url=base_url, project_id=project_id
    )
    run = db.create_run(script_id=script.id, project_id=project_id)
    asyncio.create_task(executor.execute_run(run.id, script.id))
    return db.run_to_record(run)


@app.get("/runs/{run_id}", response_model=RunRecord, dependencies=[Depends(require_api_key)])
def get_run(run_id: int):
    """Get run status, full log, log tail, and screenshot URL."""
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return db.run_to_record(row)


@app.get("/runs/{run_id}/screenshot", dependencies=[Depends(require_api_key)])
def get_run_screenshot(run_id: int):
    """Serve the screenshot captured for a run."""
    row = db.get_run(run_id)
    if row is None or not row.screenshot_path:
        raise HTTPException(status_code=404, detail="Screenshot not found.")
    path = (db.STORAGE_DIR / row.screenshot_path).resolve()
    if not path.exists() or db.STORAGE_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Screenshot not found.")
    return FileResponse(path)


@app.get("/coverage", response_model=CoverageResponse, dependencies=[Depends(require_api_key)])
def coverage(project_id: Optional[str] = None):
    """
    Coverage/traceability matrix: for each ingested document, report whether
    any test case is grounded in it (via test_cases.grounded_in) and list the
    covering test cases.
    """
    report = db.get_coverage_report(project_id=project_id)
    documents: List[CoverageDocument] = []
    for entry in report:
        doc = entry["document"]
        cases = [
            CoverageTestCase(
                id=c["id"],
                feature=c["feature"],
                scenario=c["scenario"],
                grounded_in=c["grounded_in"],
                status=c["status"],
            )
            for c in entry["test_cases"]
        ]
        documents.append(
            CoverageDocument(
                id=doc["id"],
                filename=doc["filename"],
                sha256=doc["sha256"],
                ingested_at=doc["ingested_at"],
                status="covered" if cases else "uncovered",
                test_cases=cases,
            )
        )
    covered = sum(1 for d in documents if d.status == "covered")
    return CoverageResponse(
        documents=documents,
        total_documents=len(documents),
        covered_documents=covered,
        uncovered_documents=len(documents) - covered,
    )


@app.get("/checkout-html", dependencies=[Depends(require_api_key)])
def checkout_html(page_id: Optional[int] = None, project_id: str = "default"):
    """
    Convenience endpoint to fetch the HTML/DOM snapshot of the selected page
    (explicit ``page_id`` or the project's latest page), falling back to the
    legacy checkout.html.
    """
    try:
        page = db.get_page(page_id) if page_id is not None else db.latest_page(project_id)
    except Exception:
        page = None
    if page is not None:
        return {
            "page_id": page.id,
            "project_id": page.project_id,
            "name": page.name,
            "url": page.url,
            "source_type": page.source_type,
            "html": page_html(page),
            "excerpt": page_html(page)[:1200],
        }
    return {"html": load_checkout_html(), "excerpt": load_checkout_excerpt()}


# --- Pages: uploads + crawls --------------------------------------------

@app.post("/pages/upload", response_model=List[PageRecord], dependencies=[Depends(require_api_key)])
async def upload_pages(
    files: List[UploadFile] = File(default_factory=list),
    project_id: str = Form("default"),
):
    """Upload one or more HTML files; each is stored as a page with a DOM snapshot."""
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one HTML file.")
    records: List[PageRecord] = []
    for file in files:
        content = await file.read()
        html_text = content.decode("utf-8", errors="replace")
        _, rel = store_page_html(
            html_text, project_id=project_id, page_key=f"upload-{file.filename or 'page'}"
        )
        row = db.create_page(
            project_id=project_id,
            name=file.filename or "page.html",
            source_type="upload",
            file_path=rel,
            dom_snapshot=html_text,
        )
        records.append(db.page_to_record(row))
    return records


@app.post("/pages/crawl", response_model=PageRecord, dependencies=[Depends(require_api_key)])
def crawl_page_endpoint(request: CrawlPageRequest):
    """Fetch a URL, snapshot its DOM via BeautifulSoup, and store it as a page."""
    try:
        snapshot = crawl_page(request.url, project_id=request.project_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to crawl {request.url}: {exc}")
    _, rel = store_page_html(snapshot["dom_snapshot"], project_id=request.project_id)
    row = db.create_page(
        project_id=request.project_id,
        name=request.name or snapshot["name"],
        url=snapshot["url"],
        source_type="crawl",
        file_path=rel,
        dom_snapshot=snapshot["dom_snapshot"],
    )
    return db.page_to_record(row)


@app.get("/pages", response_model=List[PageRecord], dependencies=[Depends(require_api_key)])
def list_pages(project_id: Optional[str] = None):
    return [db.page_to_record(row) for row in db.list_pages(project_id=project_id)]


@app.get("/pages/{page_id}", response_model=PageRecord, dependencies=[Depends(require_api_key)])
def get_page(page_id: int):
    row = db.get_page(page_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    return db.page_to_record(row)


@app.delete("/pages/{page_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_page(page_id: int):
    row = db.get_page(page_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    if row.file_path:
        try:
            path = (db.BASE_DIR / row.file_path).resolve()
            if db.BASE_DIR.resolve() in path.parents and path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to delete page file %s: %s", row.file_path, exc)
    db.delete_page(page_id)


# --- /checkout: serve the selected page ---------------------------------
# These routes are defined before the StaticFiles mount below so they take
# precedence; the mount remains as a fallback for legacy files like
# /checkout/checkout.html.

@app.get("/checkout/", include_in_schema=False)
def serve_selected_page(page_id: Optional[int] = None, project_id: str = "default"):
    """Serve the selected page's HTML: explicit page_id or the project's latest page."""
    try:
        html = selected_page_html(page_id=page_id, project_id=project_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if html is None:
        if CHECKOUT_PATH.exists():
            html = CHECKOUT_PATH.read_text(encoding="utf-8")
        else:
            raise HTTPException(status_code=404, detail="No page uploaded yet.")
    return HTMLResponse(content=html)


@app.get("/checkout/page/{page_id}", include_in_schema=False)
def serve_page_by_id(page_id: int):
    """Serve a specific stored page by ID."""
    row = db.get_page(page_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    try:
        return HTMLResponse(content=page_html(row))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Serve remaining /checkout/... paths (e.g. legacy checkout.html) statically.
app.mount(
    "/checkout",
    StaticFiles(directory=str(CHECKOUT_PATH.parent), html=True),
    name="checkout",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
