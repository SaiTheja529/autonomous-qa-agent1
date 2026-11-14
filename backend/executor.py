# backend/executor.py
"""
Test execution engine.

Runs a stored Selenium script in a temporary working directory via subprocess.
The script is expected to connect to a remote WebDriver at REMOTE_WEBDRIVER_URL
and to save a screenshot (e.g. ``failure.png``) on failure.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import db

logger = logging.getLogger("uvicorn.error")

REMOTE_WEBDRIVER_URL = os.getenv("REMOTE_WEBDRIVER_URL", "http://localhost:4444/wd/hub")
RUN_TIMEOUT_SECONDS = int(os.getenv("RUN_TIMEOUT_SECONDS", "300"))
SCREENSHOT_DIR = db.STORAGE_DIR / "screenshots"
SCREENSHOT_GLOBS = ("failure*.png", "screenshot*.png", "*.png")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _find_screenshot(workdir: Path) -> Path | None:
    """Return the first screenshot the script wrote into its workdir."""
    for pattern in SCREENSHOT_GLOBS:
        matches = sorted(workdir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _run_subprocess(script_path: Path, workdir: Path, env: dict) -> tuple[int, str, str]:
    """Blocking subprocess call; intended to run inside asyncio.to_thread."""
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


async def execute_run(run_id: int, script_id: int) -> None:
    """
    Execute a stored script and update its runs row.

    Safe to call as a fire-and-forget asyncio task; all errors are captured
    into the run's log/status rather than raised.
    """
    run = db.get_run(run_id)
    script = db.get_script(script_id)
    if run is None or script is None:
        logger.error("execute_run: missing run=%s script=%s", run_id, script_id)
        return

    db.update_run(run_id, status="running", started_at=_utcnow())

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"qa_run_{run_id}_"))
    script_path = workdir / "script.py"
    script_path.write_text(script.code or "", encoding="utf-8")

    env = dict(os.environ)
    env["REMOTE_WEBDRIVER_URL"] = REMOTE_WEBDRIVER_URL

    stdout = stderr = ""
    exit_code: int | None = None
    status = "failed"
    try:
        exit_code, stdout, stderr = await asyncio.to_thread(
            _run_subprocess, script_path, workdir, env
        )
        status = "passed" if exit_code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = (exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")) or ""
        stderr += f"\nTimed out after {RUN_TIMEOUT_SECONDS}s"
        status = "error"
    except Exception as exc:  # pragma: no cover - defensive
        stderr = f"Executor error: {exc}"
        status = "error"

    screenshot_rel = ""
    shot = _find_screenshot(workdir)
    if shot is not None:
        dest = SCREENSHOT_DIR / f"run_{run_id}.png"
        try:
            shutil.copyfile(shot, dest)
            screenshot_rel = str(dest.relative_to(db.STORAGE_DIR))
        except Exception as exc:
            logger.warning("Failed to persist screenshot for run %s: %s", run_id, exc)

    log = stdout
    if stderr:
        log = f"{stdout}\n--- stderr ---\n{stderr}" if stdout else stderr
    if exit_code is not None:
        log = f"{log}\n--- exit code: {exit_code} ---".strip()

    db.update_run(
        run_id,
        status=status,
        log=log,
        screenshot_path=screenshot_rel,
        finished_at=_utcnow(),
    )

    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("Run %s finished with status=%s", run_id, status)
