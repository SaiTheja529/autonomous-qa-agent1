# backend/models.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    text: str
    source: str
    chunk_id: str


class PageRecord(BaseModel):
    """A persisted page under test (uploaded HTML or crawled DOM snapshot)."""

    id: int
    project_id: str = "default"
    name: str = ""
    url: str = ""
    source_type: str = "upload"
    file_path: str = ""
    dom_snapshot: str = ""
    created_at: Optional[datetime] = None


class IngestResponse(BaseModel):
    docs_ingested: int
    chunks_added: int
    sources: List[str] = Field(default_factory=list)
    html_saved: bool
    changed_docs: List[str] = Field(default_factory=list)
    stale_marked: int = 0
    pages_added: int = 0
    pages: List[PageRecord] = Field(default_factory=list)


class CrawlPageRequest(BaseModel):
    """Request to crawl a URL and store its DOM snapshot as a page."""

    url: str = Field(..., description="URL to fetch and snapshot")
    project_id: str = Field("default", description="Project this page belongs to")
    name: Optional[str] = Field(None, description="Display name (defaults to page title/URL)")


class GenerateTestCasesRequest(BaseModel):
    query: str = Field(..., description="User prompt describing which test cases to generate")
    top_k: Optional[int] = Field(None, description="How many context chunks to retrieve (defaults to the value set via /ingest)")
    project_id: str = Field("default", description="Project whose knowledge base to query")


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
    project_id: str = "default"
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
    page_id: Optional[int] = Field(None, description="Page under test; defaults to the project's selected page")
    project_id: str = Field("default", description="Project whose knowledge base and pages to use")
    base_url: Optional[str] = Field(
        None,
        description="Base URL where the page is served (auto-derived from page_id/project_id when omitted)",
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
    project_id: Optional[str] = Field(None, description="Project override (defaults to the test case's project).")


class RunRecord(BaseModel):
    """A persisted test execution run as returned by the API."""

    id: int
    project_id: str = "default"
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
