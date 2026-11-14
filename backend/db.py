# backend/db.py
"""
SQLite persistence layer for the Autonomous QA Agent.

Tables:
  - documents: ingested support files (filename + sha256 dedupe)
  - test_cases: generated/curated test cases
  - scripts: versioned Selenium scripts linked to test cases
  - runs: execution records for scripts
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
DEFAULT_DB_PATH = STORAGE_DIR / "app.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")


def _ensure_db_dir() -> None:
    if DATABASE_URL.startswith("sqlite:///"):
        db_file = DATABASE_URL.removeprefix("sqlite:///")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    sha256: Mapped[str] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TestCaseRow(Base):
    __tablename__ = "test_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature: Mapped[str] = mapped_column(String(512))
    scenario: Mapped[str] = mapped_column(Text)
    steps: Mapped[str] = mapped_column(Text, default="")
    expected: Mapped[str] = mapped_column(Text, default="")
    grounded_in: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Script(Base):
    __tablename__ = "scripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_case_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("test_cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    code: Mapped[str] = mapped_column(Text)
    base_url: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    script_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("scripts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    log: Mapped[str] = mapped_column(Text, default="")
    screenshot_path: Mapped[str] = mapped_column(String(1024), default="")
    healed: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


_ensure_db_dir()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


_db_ready = False


def init_db() -> None:
    """Create all tables if they do not exist."""
    global _db_ready
    Base.metadata.create_all(engine)
    _db_ready = True


def ensure_db() -> None:
    """Idempotent table creation, safe to call from any entry point."""
    if not _db_ready:
        init_db()


# --- Document helpers -------------------------------------------------

def upsert_document(filename: str, sha256: str) -> Document:
    """Insert or update a document row keyed by filename."""
    ensure_db()
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.filename == filename))
        if doc is None:
            doc = Document(filename=filename, sha256=sha256)
            session.add(doc)
        else:
            doc.sha256 = sha256
            doc.ingested_at = _utcnow()
        session.commit()
        session.refresh(doc)
        return doc


# --- Test case helpers -------------------------------------------------

def create_test_case(
    feature: str,
    scenario: str,
    steps: str = "",
    expected: str = "",
    grounded_in: str = "",
    status: str = "active",
) -> TestCaseRow:
    ensure_db()
    with SessionLocal() as session:
        row = TestCaseRow(
            feature=feature,
            scenario=scenario,
            steps=steps,
            expected=expected,
            grounded_in=grounded_in,
            status=status,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row


def list_test_cases(status: Optional[str] = None) -> list[TestCaseRow]:
    ensure_db()
    with SessionLocal() as session:
        stmt = select(TestCaseRow).order_by(TestCaseRow.id)
        if status:
            stmt = stmt.where(TestCaseRow.status == status)
        return list(session.scalars(stmt))


def get_test_case(test_case_id: int) -> Optional[TestCaseRow]:
    ensure_db()
    with SessionLocal() as session:
        return session.get(TestCaseRow, test_case_id)


def update_test_case(test_case_id: int, fields: dict) -> Optional[TestCaseRow]:
    allowed = {"feature", "scenario", "steps", "expected", "grounded_in", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_test_case(test_case_id)
    ensure_db()
    with SessionLocal() as session:
        row = session.get(TestCaseRow, test_case_id)
        if row is None:
            return None
        for key, value in updates.items():
            setattr(row, key, value)
        session.commit()
        session.refresh(row)
        return row


def delete_test_case(test_case_id: int) -> bool:
    ensure_db()
    with SessionLocal() as session:
        row = session.get(TestCaseRow, test_case_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


def test_case_to_record(row: TestCaseRow):
    """Convert a TestCaseRow ORM object into the API TestCaseRecord model."""
    from models import TestCaseRecord

    return TestCaseRecord(
        id=row.id,
        feature=row.feature,
        scenario=row.scenario,
        steps=row.steps or "",
        expected=row.expected or "",
        grounded_in=row.grounded_in or "",
        status=row.status or "active",
        created_at=row.created_at,
    )
