# backend/models.py
from __future__ import annotations

from typing import List

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


class GenerateTestCasesResponse(BaseModel):
    query: str
    contexts: List[ContextChunk]
    test_cases: str


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
