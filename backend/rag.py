# backend/rag.py
"""
Retrieval-augmented generation utilities for test case and Selenium script creation.
Now powered by Gemini via google-genai (optional).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from fastapi import HTTPException

import db
from ingestion import (
    DEFAULT_PROJECT,
    default_page_base_url,
    get_collection,
    load_checkout_html,
    page_html,
)
from models import (
    ContextChunk,
    GenerateScriptResponse,
    GenerateTestCasesResponse,
    TestCase,
)

logger = logging.getLogger("uvicorn.error")


def _ensure_genai_client():
    """
    Lazy import of the Gemini client, validating the API key.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your environment to enable generation.",
        )
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"google-genai package is not installed: {exc}",
        ) from exc

    return genai.Client(api_key=api_key)


BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000")


def retrieve_context(query: str, top_k: int = 5, project_id: str = DEFAULT_PROJECT) -> List[ContextChunk]:
    """
    Retrieve the most relevant chunks for a query from the project's Chroma collection.
    """
    collection = get_collection(reset=False, project_id=project_id)
    # defensive: collection.query returns dict with lists
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector DB query failed: {exc}") from exc

    docs = results.get("documents", [[]])[0] if isinstance(results.get("documents", [[]]), list) else []
    metas = results.get("metadatas", [[]])[0] if isinstance(results.get("metadatas", [[]]), list) else []
    ids = results.get("ids", [[]])[0] if isinstance(results.get("ids", [[]]), list) else []

    contexts: List[ContextChunk] = []
    for text, meta, cid in zip(docs, metas, ids):
        contexts.append(
            ContextChunk(
                text=text,
                source=meta.get("source", meta.get("filename", "unknown")),
                chunk_id=str(cid),
            )
        )
    return contexts


def _context_block(contexts: List[ContextChunk]) -> str:
    lines = []
    for ctx in contexts:
        lines.append(f"[{ctx.source}] {ctx.text}")
    return "\n\n".join(lines)


def _iter_gemini(client, model: str, prompt: str, temperature: float = 0.2):
    """
    Stream Gemini generate_content_stream chunks, yielding text deltas.
    """
    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini generation failed: {exc}") from exc


def _generate_with_gemini(client, model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    Collect a full Gemini response via streaming and return the text.
    """
    return "".join(_iter_gemini(client, model, prompt, temperature=temperature))


def _default_top_k(project_id: str = DEFAULT_PROJECT) -> int:
    """Retrieval depth persisted by /ingest for the project, falling back to 5."""
    try:
        value = db.get_setting("top_k", project_id=project_id)
        return max(1, int(value)) if value else 5
    except Exception:
        return 5


META_PREFIX = "__META__"
REVIEW_MARKER = "\n\n--- REVIEW ---\n\n"


def _parse_cases_json(content: str) -> List[TestCase]:
    """
    Extract a JSON array of test cases from a Gemini response.
    Tolerates markdown code fences and surrounding prose.
    """
    text = (content or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini test case JSON: %.200s", text)
        return []
    cases: List[TestCase] = []
    if isinstance(data, list):
        for item in data:
            try:
                cases.append(TestCase(**item))
            except Exception:
                logger.warning("Skipping malformed test case entry: %r", item)
    return cases


def generate_test_cases(
    query: str, top_k: Optional[int] = None, project_id: str = DEFAULT_PROJECT
) -> GenerateTestCasesResponse:
    """
    Use retrieved context + Gemini to synthesize structured test cases.
    Generated cases are persisted to the database and returned as records.
    """
    top_k = top_k or _default_top_k(project_id)
    contexts = retrieve_context(query, top_k=top_k, project_id=project_id)
    if not contexts:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Ingest support documents first.",
        )

    client = _ensure_genai_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"""
You are a QA test designer. Using only the provided context, create concise, structured test cases for the request below.
- Respond with ONLY a JSON array (no markdown, no prose). Each element must be an object with keys:
  "feature" (string), "scenario" (string), "steps" (string, semicolon-separated steps),
  "expected" (string), "grounded_in" (string, brief citation of the source document name).
- Include both positive and negative cases where relevant.
- Ground every Expected result with a brief citation from the source document name.
- Do not invent features; rely strictly on context.

User Request:
{query}

Context:
{_context_block(contexts)}
    """.strip()

    content = _generate_with_gemini(client, model, prompt, temperature=0.2)

    cases = _parse_cases_json(content)
    records = []
    for case in cases:
        try:
            row = db.create_test_case(
                feature=case.feature,
                scenario=case.scenario,
                steps=case.steps,
                expected=case.expected,
                grounded_in=case.grounded_in,
                project_id=project_id,
            )
            records.append(db.test_case_to_record(row))
        except Exception as exc:
            logger.exception("Failed to persist generated test case: %s", exc)

    if not records:
        logger.warning("Gemini returned no parseable test cases for query: %s", query)

    return GenerateTestCasesResponse(
        query=query,
        contexts=contexts,
        test_cases=content,
        cases=records,
    )


def _build_script_prompt(
    test_case: str,
    contexts: List[ContextChunk],
    page_html_text: str,
    base_url: str,
    page_name: str = "",
) -> str:
    page_label = page_name or "the page under test"
    return f"""
You are a senior QA automation engineer writing Selenium (Python) scripts.
Use the page HTML/DOM snapshot and documentation context to generate an executable script that automates the test case against {page_label}.

Requirements:
- Use Selenium 4 Python style with a REMOTE WebDriver. Connect via:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import os
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Remote(
        command_executor=os.environ.get("REMOTE_WEBDRIVER_URL", "http://localhost:4444/wd/hub"),
        options=options,
    )
- Use WebDriverWait with expected_conditions for element lookups.
- Prefer IDs or names for selectors; fall back to CSS selectors if needed.
- Set `base_url = "{base_url}"` at the top of the script so it runs against the served page.
- Apply discount codes, select shipping/payment, and validate inline error or success messages when applicable.
- Include assertions for expected results described in the test case.
- Wrap the test body in try/except: on ANY exception or assertion failure, call
  `driver.save_screenshot("failure.png")` before re-raising. Always call `driver.quit()`.
- Keep the script self-contained (imports + main guard). Output only Python code.

Test Case:
{test_case}

Documentation Context:
{_context_block(contexts)}

Page HTML ({page_label}):
{page_html_text}
    """.strip()


def _review_prompt(script: str, test_case: str) -> str:
    return f"""
You are a senior QA automation reviewer. Critique the Selenium script below for the test case.

Focus on:
- Selector robustness: brittle XPath/CSS selectors, missing explicit waits, race conditions.
- Missing assertions: expected results from the test case that are never asserted.
- Error handling: screenshots on failure, cleanup via driver.quit(), swallowed exceptions.
- Flakiness: hardcoded sleeps, assumptions about element state or ordering.

Be concise. Output markdown bullets only; do not rewrite the script.

Test Case:
{test_case}

Script:
{script}
    """.strip()


def _resolve_page(page_id: Optional[int], project_id: str):
    """
    Resolve the page under test: explicit ``page_id`` first, then the project's
    latest page, then the legacy checkout.html fallback (returned as None).
    Returns ``(page_row_or_None, html_text)``.
    """
    page = None
    if page_id is not None:
        page = db.get_page(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page #{page_id} not found.")
    else:
        page = db.latest_page(project_id)

    if page is not None:
        try:
            return page, page_html(page)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return None, load_checkout_html()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _default_base_url(page, project_id: str, base_url: Optional[str]) -> str:
    if base_url:
        return base_url
    if page is not None:
        return f"{BACKEND_PUBLIC_URL}/checkout/page/{page.id}"
    return default_page_base_url(project_id, backend_url=BACKEND_PUBLIC_URL)


def _prepare_script_generation(
    test_case: str,
    top_k: Optional[int],
    base_url: Optional[str],
    page_id: Optional[int] = None,
    project_id: str = DEFAULT_PROJECT,
):
    """
    Shared setup for script generation. Returns (contexts, html_excerpt, client, model, prompt).
    Raises HTTPException eagerly so callers can return proper error responses.
    """
    top_k = top_k or _default_top_k(project_id)
    contexts = retrieve_context(test_case, top_k=top_k, project_id=project_id)

    page, html = _resolve_page(page_id, project_id)
    html_excerpt = html[:1200]
    effective_base_url = _default_base_url(page, project_id, base_url)
    page_name = (page.name or page.url) if page is not None else "checkout.html"

    client = _ensure_genai_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = _build_script_prompt(test_case, contexts, html, effective_base_url, page_name=page_name)
    return contexts, html_excerpt, client, model, prompt


def generate_selenium_script(
    test_case: str,
    top_k: Optional[int] = None,
    base_url: Optional[str] = None,
    review: bool = False,
    page_id: Optional[int] = None,
    project_id: str = DEFAULT_PROJECT,
) -> GenerateScriptResponse:
    """
    Generate a runnable Selenium Python script for the provided test case.
    ``page_id`` selects which stored page the script targets; ``project_id``
    selects the knowledge base and default page. When ``review`` is true, a
    second Gemini pass critiques the script.
    """
    contexts, html_excerpt, client, model, prompt = _prepare_script_generation(
        test_case, top_k, base_url, page_id=page_id, project_id=project_id
    )

    script = _generate_with_gemini(client, model, prompt, temperature=0.2)
    review_text = ""
    if review:
        review_text = _generate_with_gemini(
            client, model, _review_prompt(script, test_case), temperature=0.4
        )
    return GenerateScriptResponse(
        test_case=test_case,
        contexts=contexts,
        checkout_excerpt=html_excerpt,
        script=script,
        review=review_text,
    )


def iter_script_stream(
    test_case: str,
    top_k: Optional[int] = None,
    base_url: Optional[str] = None,
    review: bool = False,
    page_id: Optional[int] = None,
    project_id: str = DEFAULT_PROJECT,
):
    """
    Prepare a streaming script generation and return an iterator of text chunks.

    The first yielded chunk is a ``__META__<json>`` header line carrying the
    retrieval contexts and checkout excerpt; subsequent chunks are the script
    text. When ``review`` is true, a ``--- REVIEW ---`` marker followed by the
    critique text is appended after the script.

    This is a regular function (not a generator) so setup errors raise
    HTTPException eagerly, before StreamingResponse begins.
    """
    contexts, html_excerpt, client, model, prompt = _prepare_script_generation(
        test_case, top_k, base_url, page_id=page_id, project_id=project_id
    )
    meta = {
        "contexts": [c.model_dump() for c in contexts],
        "checkout_excerpt": html_excerpt,
    }

    def _gen():
        yield META_PREFIX + json.dumps(meta) + "\n"
        script_parts = []
        for chunk in _iter_gemini(client, model, prompt, temperature=0.2):
            script_parts.append(chunk)
            yield chunk
        if review:
            yield REVIEW_MARKER
            yield from _iter_gemini(
                client, model, _review_prompt("".join(script_parts), test_case), temperature=0.4
            )

    return _gen()
