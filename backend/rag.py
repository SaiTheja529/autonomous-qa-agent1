# backend/rag.py
"""
Retrieval-augmented generation utilities for test case and Selenium script creation.
Now powered by Gemini via google-genai (optional).
"""
from __future__ import annotations

import os
from typing import List

from fastapi import HTTPException

from ingestion import get_collection, load_checkout_excerpt, load_checkout_html
from models import ContextChunk, GenerateScriptResponse, GenerateTestCasesResponse


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


def retrieve_context(query: str, top_k: int = 5) -> List[ContextChunk]:
    """
    Retrieve the most relevant chunks for a query from Chroma.
    """
    collection = get_collection(reset=False)
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


def _generate_with_gemini(client, model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    Call Gemini generate_content with a minimal generation config and return text.
    """
    try:
        completion = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini generation failed: {exc}") from exc
    # defensive: some clients return different shapes; try to extract text
    try:
        return getattr(completion, "text", "") or completion.get("text", "") or ""
    except Exception:
        return ""


def generate_test_cases(query: str, top_k: int = 5) -> GenerateTestCasesResponse:
    """
    Use retrieved context + Gemini to synthesize structured test cases.
    """
    contexts = retrieve_context(query, top_k=top_k)
    if not contexts:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Ingest support documents first.",
        )

    client = _ensure_genai_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"""
You are a QA test designer. Using only the provided context, create concise, structured test cases for the request below.
- Use Markdown table with columns: Test_ID, Feature, Test_Scenario, Steps, Expected_Result, Grounded_In.
- Include both positive and negative cases where relevant.
- Ground every Expected_Result with a brief citation from the source document name.
- Do not invent features; rely strictly on context.

User Request:
{query}

Context:
{_context_block(contexts)}
    """.strip()

    content = _generate_with_gemini(client, model, prompt, temperature=0.2)
    return GenerateTestCasesResponse(
        query=query,
        contexts=contexts,
        test_cases=content,
    )


def generate_selenium_script(test_case: str, top_k: int = 5, base_url: str = "http://localhost:8000/checkout/checkout.html") -> GenerateScriptResponse:
    """
    Generate a runnable Selenium Python script for the provided test case.
    """
    contexts = retrieve_context(test_case, top_k=top_k)
    try:
        html = load_checkout_html()
        html_excerpt = load_checkout_excerpt()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    client = _ensure_genai_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = f"""
You are a senior QA automation engineer writing Selenium (Python) scripts.
Use the checkout HTML and documentation context to generate an executable script that automates the test case.

Requirements:
- Use Selenium 4 Python style: webdriver.Chrome(), WebDriverWait with expected_conditions.
- Prefer IDs or names for selectors; fall back to CSS selectors if needed.
- Set `base_url = "{base_url}"` at the top of the script so it runs against the served checkout page.
- Apply discount codes, select shipping/payment, and validate inline error or success messages when applicable.
- Include assertions for expected results described in the test case.
- Keep the script self-contained (imports + main guard).

Test Case:
{test_case}

Documentation Context:
{_context_block(contexts)}

Checkout HTML:
{html}
    """.strip()

    script = _generate_with_gemini(client, model, prompt, temperature=0.2)
    return GenerateScriptResponse(
        test_case=test_case,
        contexts=contexts,
        checkout_excerpt=html_excerpt,
        script=script,
    )
