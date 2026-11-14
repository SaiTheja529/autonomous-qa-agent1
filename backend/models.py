# backend/models.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    text: str
    source: str
    chunk_id: str


class IngestResponse(BaseModel):
    docs_ingested: int
    chunks_added: int
    sources: List[str] = Field(default_factory=list)
    html_saved: bool
    changed_docs: List[str] = Field(default_factory=list)
    stale_marked: int = 0


class GenerateTestCasesRequest(BaseModel):
    query: str = Field(..., description="User prompt describing which test cases to generate")
    top_k: Optional[int] = Field(None, description="How many context chunks to retrieve (defaults to the value set via /ingest)")


class TestCase(BaseModel):
    """A single structured test case produced by the generator."""

    feature: str
    scenario: str
    steps: str = ""
    expected: str = ""
    grounded_in: str = ""


class TestCaseRecord(BaseModel):
    """A persisted test case as returned by the API."""

    id: int
    feature: str
    scenario: str
    steps: str = ""
    expected: str = ""
    grounded_in: str = ""
    status: str = "active"
    created_at: Optional[datetime] = None


class TestCaseUpdate(BaseModel):
    """Partial update payload for PATCH /testcases/{id}."""

    feature: Optional[str] = None
    scenario: Optional[str] = None
    steps: Optional[str] = None
    expected: Optional[str] = None
    grounded_in: Optional[str] = None
    status: Optional[str] = None


class GenerateTestCasesResponse(BaseModel):
    query: str
    contexts: List[ContextChunk]
    test_cases: str
    cases: List[TestCaseRecord] = Field(default_factory=list)


class GenerateScriptRequest(BaseModel):
    test_case: str = Field(..., description="Single test case description to automate")
    top_k: Optional[int] = Field(None, description="How many context chunks to retrieve (defaults to the value set via /ingest)")
    base_url: str = Field(
        "http://localhost:8000/checkout/checkout.html",
        description="Base URL where checkout.html is served",
    )
    review: bool = Field(False, description="Run a second Gemini pass critiquing selector robustness and missing assertions")


class GenerateScriptResponse(BaseModel):
    test_case: str
    contexts: List[ContextChunk]
    checkout_excerpt: str
    script: str
    review: str = ""


class RunTestCaseRequest(BaseModel):
    """Optional overrides for POST /testcases/{id}/run."""

    script_code: Optional[str] = Field(
        None, description="Script to run; if omitted, one is generated from the test case."
    )
    base_url: Optional[str] = Field(None, description="Base URL override for script generation.")
    top_k: Optional[int] = Field(None, description="Context chunks to retrieve when generating a script.")


class RunRecord(BaseModel):
    """A persisted test execution run as returned by the API."""

    id: int
    script_id: Optional[int] = None
    status: str = "pending"
    log: str = ""
    log_tail: str = ""
    screenshot_path: str = ""
    screenshot_url: Optional[str] = None
    healed: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class CoverageTestCase(BaseModel):
    """A test case that covers (is grounded in) a document."""

    id: int
    feature: str
    scenario: str
    grounded_in: str = ""
    status: str = "active"


class CoverageDocument(BaseModel):
    """A document's coverage status and the test cases covering it."""

    id: int
    filename: str
    sha256: str
    ingested_at: Optional[datetime] = None
    status: str = "uncovered"
    test_cases: List[CoverageTestCase] = Field(default_factory=list)


class CoverageResponse(BaseModel):
    """Coverage/traceability matrix: documents x covering test cases."""

    documents: List[CoverageDocument] = Field(default_factory=list)
    total_documents: int = 0
    covered_documents: int = 0
    uncovered_documents: int = 0


class ScriptRecord(BaseModel):
    """A persisted Selenium script version."""

    id: int
    test_case_id: Optional[int] = None
    version: int = 1
    base_url: str = ""
    created_at: Optional[datetime] = None
