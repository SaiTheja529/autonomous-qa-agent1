# Autonomous QA Agent — Feature Roadmap

Tracking document for the multi-phase roadmap. Each phase is implemented in a
separate PR/branch; later phases build on the persistence schema introduced in
Phase 1.

## Phase 1 — Persistence layer (this PR)

Goal: everything the agent produces is stored in SQLite so later phases can
build on durable state.

- [x] `backend/db.py` — SQLAlchemy + SQLite schema:
  - `documents(id, filename, sha256, ingested_at)`
  - `test_cases(id, feature, scenario, steps, expected, grounded_in, status, created_at)`
  - `scripts(id, test_case_id, version, code, base_url)`
  - `runs(id, script_id, status, log, screenshot_path, healed, started_at, finished_at)`
- [x] `generate_test_cases` requests JSON from Gemini instead of a Markdown table
- [x] `TestCase` Pydantic model in `backend/models.py`
- [x] Generated test cases are persisted on creation
- [x] `GET /testcases`, `PATCH /testcases/{id}`, `DELETE /testcases/{id}`
- [x] Frontend renders test cases as a selectable dataframe from JSON;
      `parse_markdown_table` removed
- [x] All existing endpoints remain backward compatible

Status: **In review (PR open)**

## Phase 2 — Script generation persistence & versioning

- [ ] `POST /generate-script` persists each generated script into `scripts`
      with an incrementing `version` per `test_case_id`
- [ ] Response includes `script_id` and `version`
- [ ] `GET /scripts`, `GET /scripts/{id}`, `GET /testcases/{id}/scripts`
- [ ] Frontend: script history per test case, version selector

Status: Not started

## Phase 3 — Test execution runner

- [ ] `POST /runs` — execute a stored script via Selenium against `base_url`
- [ ] `runs` rows capture `status`, `log`, `screenshot_path`, timestamps
- [ ] `GET /runs`, `GET /runs/{id}`; artifacts stored under `backend/storage/runs/`
- [ ] Frontend: "Run" button on scripts, run history view

Status: Not started

## Phase 4 — Self-healing locators

- [ ] On locator failure, attempt repair (fallback selectors, retry strategies)
- [ ] Persist healed scripts as new `scripts` versions; set `runs.healed = true`
- [ ] Surface healing diff/log in the UI

Status: Not started

## Phase 5 — Reporting & analytics

- [ ] Aggregate pass/fail rates, flaky detection, coverage-by-feature views
- [ ] Export run reports (HTML/JSON)
- [ ] Dashboard page in Streamlit

Status: Not started

## Phase 6 — CI integration & scheduling

- [ ] CLI/entrypoint to run suites headlessly
- [ ] Optional scheduled runs + webhook/Slack notifications
- [ ] Docker Compose for backend + frontend + runner

Status: Not started
