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


class GenerateTestCasesRequest(BaseModel):
    query: str = Field(..., description="User prompt describing which test cases to generate")
    top_k: int = Field(5, description="How many context chunks to retrieve")


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
    top_k: int = Field(5, description="How many context chunks to retrieve")
    base_url: str = Field(
        "http://localhost:8000/checkout/checkout.html",
        description="Base URL where checkout.html is served",
    )


class GenerateScriptResponse(BaseModel):
    test_case: str
    contexts: List[ContextChunk]
    checkout_excerpt: str
    script: str
