import json
import os
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _headers() -> dict:
    """X-API-Key header for backend auth (env API_KEY or sidebar input)."""
    key = os.getenv("API_KEY") or st.session_state.get("api_key", "")
    return {"X-API-Key": key} if key else {}


def _project_id() -> str:
    """Active project id (sidebar input, env PROJECT_ID, or 'default')."""
    return st.session_state.get("project_id") or os.getenv("PROJECT_ID", "default")


def check_backend() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.ok
    except Exception:
        return False


def ingest_documents(docs: List, checkout_html: Optional, page_htmls: List, reset: bool, chunk_size: int, overlap: int, top_k: int):
    files = []
    for doc in docs:
        files.append(("docs", (doc.name, doc.getvalue(), doc.type)))
    if checkout_html:
        files.append(("checkout", (checkout_html.name, checkout_html.getvalue(), checkout_html.type)))
    for page in page_htmls:
        files.append(("pages", (page.name, page.getvalue(), page.type)))
    data = {
        "reset": str(reset).lower(),
        "project_id": _project_id(),
        "chunk_size": str(chunk_size),
        "overlap": str(overlap),
        "top_k": str(top_k),
    }
    # Allow extra time for first-time model downloads/embeddings
    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, data=data, headers=_headers(), timeout=240)
    resp.raise_for_status()
    return resp.json()


def generate_test_cases(query: str, top_k: int):
    payload = {"query": query, "top_k": top_k, "project_id": _project_id()}
    resp = requests.post(f"{BACKEND_URL}/generate-testcases", json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    return resp.json()


def list_pages():
    resp = requests.get(f"{BACKEND_URL}/pages", params={"project_id": _project_id()}, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def upload_pages(page_files: List):
    files = [("files", (f.name, f.getvalue(), f.type or "text/html")) for f in page_files]
    resp = requests.post(
        f"{BACKEND_URL}/pages/upload",
        files=files,
        data={"project_id": _project_id()},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def crawl_page(url: str, name: Optional[str] = None):
    payload = {"url": url, "project_id": _project_id()}
    if name:
        payload["name"] = name
    resp = requests.post(f"{BACKEND_URL}/pages/crawl", json=payload, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def delete_page(page_id: int):
    resp = requests.delete(f"{BACKEND_URL}/pages/{page_id}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.status_code


def generate_script(test_case: str, top_k: int, base_url: Optional[str], review: bool = False, page_id: Optional[int] = None):
    payload = {"test_case": test_case, "top_k": top_k, "project_id": _project_id(), "review": review}
    if page_id is not None:
        payload["page_id"] = page_id
    if base_url:
        payload["base_url"] = base_url
    resp = requests.post(f"{BACKEND_URL}/generate-script", json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    return resp.json()


def generate_script_stream(test_case: str, top_k: int, base_url: Optional[str], review: bool = False, page_id: Optional[int] = None):
    """
    POST /generate-script/stream and return (metadata, chunk_iterator).

    The stream begins with a ``__META__<json>`` header line (contexts +
    checkout excerpt); subsequent chunks are the script text followed by an
    optional ``--- REVIEW ---`` section when review=True.
    """
    payload = {"test_case": test_case, "top_k": top_k, "project_id": _project_id(), "review": review}
    if page_id is not None:
        payload["page_id"] = page_id
    if base_url:
        payload["base_url"] = base_url
    resp = requests.post(f"{BACKEND_URL}/generate-script/stream", json=payload, headers=_headers(), stream=True, timeout=300)
    resp.raise_for_status()
    chunks = resp.iter_content(chunk_size=1024, decode_unicode=True)

    buf = ""
    while "\n" not in buf:
        piece = next(chunks, "")
        if piece == "":
            break
        buf += piece

    meta: dict = {}
    rest = buf
    if buf.startswith("__META__"):
        line, _, rest = buf.partition("\n")
        try:
            meta = json.loads(line[len("__META__"):])
        except Exception:
            meta = {}

    def gen():
        try:
            if rest:
                yield rest
            for piece in chunks:
                yield piece
        finally:
            resp.close()

    return meta, gen()


def export_test_cases(fmt: str):
    resp = requests.get(
        f"{BACKEND_URL}/testcases/export",
        params={"format": fmt, "project_id": _project_id()},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def get_test_cases(status: Optional[str] = None):
    params = {"project_id": _project_id()}
    if status:
        params["status"] = status
    resp = requests.get(f"{BACKEND_URL}/testcases", params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def update_test_case(test_case_id: int, fields: dict):
    resp = requests.patch(f"{BACKEND_URL}/testcases/{test_case_id}", json=fields, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def delete_test_case(test_case_id: int):
    resp = requests.delete(f"{BACKEND_URL}/testcases/{test_case_id}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.status_code


def run_test_case(test_case_id: int, script_code: Optional[str] = None, base_url: Optional[str] = None):
    payload = {"project_id": _project_id()}
    if script_code:
        payload["script_code"] = script_code
    if base_url:
        payload["base_url"] = base_url
    resp = requests.post(f"{BACKEND_URL}/testcases/{test_case_id}/run", json=payload, headers=_headers(), timeout=180)
    resp.raise_for_status()
    return resp.json()


def get_coverage():
    resp = requests.get(f"{BACKEND_URL}/coverage", params={"project_id": _project_id()}, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_run(run_id: int):
    resp = requests.get(f"{BACKEND_URL}/runs/{run_id}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_run_screenshot(run_id: int):
    resp = requests.get(f"{BACKEND_URL}/runs/{run_id}/screenshot", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.content


def case_label(case: dict) -> str:
    cid = case.get("id")
    scenario = case.get("scenario", "")
    return f"#{cid}: {scenario}" if cid is not None else scenario


def case_to_text(case: dict) -> str:
    return (
        f"{case.get('feature', '')} - {case.get('scenario', '')}. "
        f"Steps: {case.get('steps', '')}. Expected: {case.get('expected', '')}"
    )


st.set_page_config(page_title="Autonomous QA Agent", layout="wide")

# --- Theming ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Inter', system-ui, -apple-system, sans-serif;
        background: radial-gradient(circle at 10% 20%, #f1f8ff 0, #f7fef7 30%, #ffffff 60%);
        color: #0f172a;
    }
    .glass {
        background: rgba(255,255,255,0.82);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
    }
    .hero {
        background: linear-gradient(120deg, #0ea5e9, #16a34a);
        color: #fff;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 18px 46px rgba(14,165,233,0.35);
    }
    .nav {
        display: flex;
        gap: 14px;
        align-items: center;
        padding: 12px 16px;
        background: rgba(255,255,255,0.6);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        backdrop-filter: blur(6px);
        margin-bottom: 12px;
    }
    .nav a {
        padding: 8px 12px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        color: #0f172a;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .nav a:hover {
        background: #e0f2fe;
        border-color: #bae6fd;
    }
    footer {
        margin-top: 24px;
        padding: 12px 16px;
        border-radius: 14px;
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
        color: #fff;
        text-align: center;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: auth + project ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "API Key (X-API-Key)",
        value=os.getenv("API_KEY", ""),
        type="password",
        help="Required when the backend sets API_KEYS/API_KEY.",
    )
    st.session_state["api_key"] = api_key
    project_id = st.text_input(
        "Project ID",
        value=os.getenv("PROJECT_ID", "default"),
        help="Selects the per-project knowledge base, pages, and test cases.",
    )
    st.session_state["project_id"] = project_id or "default"
    st.caption(f"Backend: {BACKEND_URL}")

# --- Simple page state + Navbar ---
PAGES = {
    "kb": "Knowledge Base",
    "cases": "Test Cases",
    "scripts": "Selenium Scripts",
    "coverage": "Coverage",
}
if "page" not in st.session_state:
    st.session_state["page"] = "kb"

nav_cols = st.columns(len(PAGES))
for idx, (key, label) in enumerate(PAGES.items()):
    with nav_cols[idx]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state["page"] == key else "secondary"):
            st.session_state["page"] = key

# --- Hero ---
with st.container():
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown(
            """
            <div class="hero">
              <h1 style="margin:0 0 6px 0;">Autonomous QA Agent</h1>
              <p style="font-size:16px; line-height:1.5; margin-bottom:10px;">
                Build a testing brain from your docs, generate grounded test cases,
                and emit Selenium scripts that match your checkout UI.
              </p>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span style="padding:6px 10px; border-radius:10px; background:rgba(255,255,255,0.15); font-weight:700;">Docs ➜ KB</span>
                <span style="padding:6px 10px; border-radius:10px; background:rgba(255,255,255,0.15); font-weight:700;">KB ➜ Test Cases</span>
                <span style="padding:6px 10px; border-radius:10px; background:rgba(255,255,255,0.15); font-weight:700;">Cases ➜ Selenium</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        backend_up = check_backend()
        status = "✅ Backend reachable" if backend_up else "⚠️ Backend not reachable"
        st.markdown(
            f"""
            <div class="glass">
                <div style="font-weight:700; font-size:16px;">Status</div>
                <div style="font-size:14px;">{status}</div>
                <div style="color:#475569; font-size:13px;">BACKEND_URL={BACKEND_URL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

page = st.session_state["page"]

# --- Knowledge Base Page ---
if page == "kb":
    st.markdown("### 📚 Upload & Build Knowledge Base")
    with st.container():
        with st.form("kb_form"):
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            docs = st.file_uploader(
                "Upload support docs (MD, TXT, JSON, PDF)", type=["md", "txt", "json", "pdf"], accept_multiple_files=True
            )
            checkout_html = st.file_uploader("Upload checkout.html", type=["html", "htm"])
            page_htmls = st.file_uploader(
                "Upload page HTML files (multiple supported)", type=["html", "htm"], accept_multiple_files=True
            )
            reset = st.checkbox("Reset existing knowledge base", value=False)
            chunk_size = st.slider("Chunk size (chars)", min_value=200, max_value=2000, value=800, step=50, key="ingest_chunk_size")
            overlap = st.slider("Chunk overlap (chars)", min_value=0, max_value=min(500, chunk_size - 50), value=min(150, chunk_size - 50), step=25, key="ingest_overlap")
            ingest_top_k = st.slider("Default top_k for generation", min_value=3, max_value=12, value=6, step=1, key="ingest_top_k")
            submitted = st.form_submit_button("Build Knowledge Base", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not docs and not checkout_html and not page_htmls:
            st.error("Please upload at least one document or page HTML.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    result = ingest_documents(
                        docs, checkout_html, page_htmls or [], reset, chunk_size, overlap, ingest_top_k
                    )
                    stale_note = ""
                    if result.get("changed_docs"):
                        stale_note = (
                            f" Changed docs: {', '.join(result['changed_docs'])}; "
                            f"marked {result.get('stale_marked', 0)} test case(s) stale."
                        )
                    pages_note = f" Pages added: {result.get('pages_added', 0)}." if result.get("pages_added") else ""
                    st.success(
                        f"Ingested {result['docs_ingested']} docs with {result['chunks_added']} chunks. "
                        f"Checkout saved: {result['html_saved']}.{stale_note}{pages_note}"
                    )
                    st.json(result)
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    # --- Pages under test ---
    st.markdown("### 🌐 Pages under test")
    st.caption("Uploaded HTML files and crawled URLs become pages that scripts can target.")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        page_uploads = st.file_uploader(
            "Upload HTML pages", type=["html", "htm"], accept_multiple_files=True, key="pages_upload"
        )
        if st.button("Add pages", use_container_width=True, disabled=not page_uploads):
            try:
                added = upload_pages(page_uploads)
                st.success(f"Added {len(added)} page(s).")
                st.session_state["pages_cache"] = None
            except Exception as exc:
                st.error(f"Page upload failed: {exc}")
    with pcol2:
        crawl_url = st.text_input("Or crawl a URL", key="crawl_url", placeholder="https://example.com")
        crawl_name = st.text_input("Page name (optional)", key="crawl_name")
        if st.button("Crawl page", use_container_width=True, disabled=not crawl_url.strip()):
            try:
                page = crawl_page(crawl_url.strip(), name=crawl_name.strip() or None)
                st.success(f"Crawled page #{page['id']}: {page.get('name') or page['url']}")
                st.session_state["pages_cache"] = None
            except Exception as exc:
                st.error(f"Crawl failed: {exc}")

    try:
        pages_list = st.session_state.get("pages_cache")
        if pages_list is None:
            pages_list = list_pages()
            st.session_state["pages_cache"] = pages_list
        if pages_list:
            pdf = pd.DataFrame(
                [
                    {
                        "id": p["id"],
                        "name": p.get("name", ""),
                        "url": p.get("url", ""),
                        "source": p.get("source_type", ""),
                        "project": p.get("project_id", ""),
                    }
                    for p in pages_list
                ]
            )
            st.dataframe(pdf, use_container_width=True, hide_index=True)
            del_col, _, _ = st.columns([1, 2, 5])
            with del_col:
                del_id = st.selectbox(
                    "Delete page", options=[p["id"] for p in pages_list],
                    format_func=lambda i: f"#{i}", key="pages_delete_select",
                )
                if st.button("Delete", use_container_width=True):
                    try:
                        delete_page(del_id)
                        st.session_state["pages_cache"] = None
                        st.success(f"Page #{del_id} deleted.")
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
        else:
            st.info("No pages yet for this project.")
    except Exception as exc:
        st.error(f"Failed to load pages: {exc}")

# --- Test Cases Page ---
elif page == "cases":
    st.markdown("### 🧾 Generate Test Cases")
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        default_prompt = "Generate positive and negative test cases for the discount code and checkout validation."
        query = st.text_area("Test case request", value=default_prompt, height=120, key="cases_query")
        top_k = st.slider("Context chunks to retrieve", min_value=3, max_value=12, value=6, step=1, key="cases_top_k")
        if st.button("Generate Test Cases", use_container_width=True):
            with st.spinner("Generating test cases..."):
                try:
                    result = generate_test_cases(query, top_k)
                    st.session_state["last_test_cases"] = result.get("cases", [])
                    st.session_state["last_contexts"] = result["contexts"]
                    st.session_state["saved_cases"] = None
                    st.success("Test cases generated and saved to the database.")
                except Exception as exc:
                    st.error(f"Failed to generate test cases: {exc}")

        # Always show the last generated test cases, even after navigation
        last_cases = st.session_state.get("last_test_cases")
        last_contexts = st.session_state.get("last_contexts", [])
        if last_cases:
            df = pd.DataFrame(last_cases)
            if "created_at" in df.columns:
                df = df.drop(columns=["created_at"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            if last_contexts:
                with st.expander("Retrieved context (last run)"):
                    for ctx in last_contexts:
                        st.markdown(f"**{ctx['source']}** — `{ctx['chunk_id']}`")
                        st.write(ctx["text"])

        # --- Saved test case management ---
        st.markdown("#### Saved test cases")
        col_load, col_status_filter = st.columns([1, 1])
        with col_load:
            if st.button("Load saved test cases", use_container_width=True):
                try:
                    st.session_state["saved_cases"] = get_test_cases()
                except Exception as exc:
                    st.error(f"Failed to load test cases: {exc}")
        with col_status_filter:
            status_filter = st.selectbox(
                "Filter by status",
                options=["", "active", "draft", "archived", "stale"],
                format_func=lambda s: s or "all",
                key="cases_status_filter",
            )
            if st.button("Apply filter", use_container_width=True):
                try:
                    st.session_state["saved_cases"] = get_test_cases(status=status_filter or None)
                except Exception as exc:
                    st.error(f"Failed to load test cases: {exc}")

        saved_cases = st.session_state.get("saved_cases")
        if saved_cases:
            sdf = pd.DataFrame(saved_cases)
            if "created_at" in sdf.columns:
                sdf = sdf.drop(columns=["created_at"])
            st.dataframe(sdf, use_container_width=True, hide_index=True)

            edit_col, status_col, btn_col = st.columns([2, 1, 1])
            with edit_col:
                edit_options = [case_label(c) for c in saved_cases]
                edit_choice = st.selectbox("Select a case to manage", edit_options, key="cases_manage_select")
            selected = saved_cases[edit_options.index(edit_choice)] if edit_options else None
            if selected:
                with status_col:
                    new_status = st.selectbox(
                        "Status",
                        options=["active", "draft", "archived", "stale"],
                        index=["active", "draft", "archived", "stale"].index(selected.get("status", "active"))
                        if selected.get("status", "active") in ["active", "draft", "archived", "stale"]
                        else 0,
                        key="cases_status_select",
                    )
                with btn_col:
                    if st.button("Save status", use_container_width=True):
                        try:
                            update_test_case(selected["id"], {"status": new_status})
                            st.session_state["saved_cases"] = get_test_cases(status=status_filter or None)
                            st.success(f"Test case #{selected['id']} updated.")
                        except Exception as exc:
                            st.error(f"Update failed: {exc}")
                    if st.button("Delete", use_container_width=True):
                        try:
                            delete_test_case(selected["id"])
                            st.session_state["saved_cases"] = get_test_cases(status=status_filter or None)
                            st.success(f"Test case #{selected['id']} deleted.")
                        except Exception as exc:
                            st.error(f"Delete failed: {exc}")
                    if st.button("Run test", use_container_width=True):
                        try:
                            run = run_test_case(selected["id"])
                            st.session_state["active_run_id"] = run["id"]
                            st.success(f"Run #{run['id']} started for test case #{selected['id']}.")
                        except Exception as exc:
                            st.error(f"Failed to start run: {exc}")

        # --- Export ---
        st.markdown("#### Export test cases")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Prepare CSV export", use_container_width=True):
                try:
                    st.session_state["export_csv"] = export_test_cases("csv")
                    st.success("CSV export ready.")
                except Exception as exc:
                    st.error(f"CSV export failed: {exc}")
        with exp_col2:
            if st.button("Prepare pytest export", use_container_width=True):
                try:
                    st.session_state["export_pytest"] = export_test_cases("pytest")
                    st.success("pytest export ready.")
                except Exception as exc:
                    st.error(f"pytest export failed: {exc}")
        if st.session_state.get("export_csv"):
            st.download_button(
                "Download testcases.csv",
                data=st.session_state["export_csv"],
                file_name="testcases.csv",
                mime="text/csv",
                key="dl_csv",
            )
        if st.session_state.get("export_pytest"):
            st.download_button(
                "Download testcases_pytest.py",
                data=st.session_state["export_pytest"],
                file_name="testcases_pytest.py",
                mime="text/x-python",
                key="dl_pytest",
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Run results panel ---
    active_run_id = st.session_state.get("active_run_id")
    if active_run_id:
        st.markdown("#### Latest run")
        rcol, rbtn = st.columns([3, 1])
        with rbtn:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        try:
            run = get_run(active_run_id)
        except Exception as exc:
            st.error(f"Failed to fetch run #{active_run_id}: {exc}")
        else:
            status = run.get("status", "unknown")
            badge = {"passed": "✅", "failed": "❌", "error": "💥", "running": "⏳"}.get(status, "•")
            st.markdown(f"**Run #{run['id']}** — {badge} `{status}`  \nScript #{run.get('script_id')}")
            if run.get("log_tail"):
                with st.expander("Log tail", expanded=True):
                    st.code(run["log_tail"], language="text")
            if run.get("screenshot_url"):
                try:
                    img = get_run_screenshot(run["id"])
                    st.image(img, caption=f"Screenshot for run #{run['id']}")
                except Exception as exc:
                    st.warning(f"Screenshot unavailable: {exc}")

# --- Selenium Scripts Page ---
elif page == "scripts":
    st.markdown("### 🛠️ Generate Selenium Script")
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        # Collect selectable test cases: persisted saved cases first, then last generated batch
        candidate_cases: List[dict] = []
        saved_cases = st.session_state.get("saved_cases") or []
        last_cases = st.session_state.get("last_test_cases") or []
        seen_ids = set()
        for c in list(saved_cases) + list(last_cases):
            if not isinstance(c, dict):
                continue
            key = c.get("id")
            if key is not None:
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            candidate_cases.append(c)

        selected_case_text = ""
        if candidate_cases:
            options = [case_label(c) for c in candidate_cases]
            choice = st.selectbox("Select a test case", options, key="scripts_case_select")
            selected_case_text = case_to_text(candidate_cases[options.index(choice)])

        manual_input = st.text_area(
            "Or paste a test case description to automate",
            value=selected_case_text,
            height=140,
            placeholder="Paste a single test case (ID, scenario, expected result)...",
        )
        script_top_k = st.slider(
            "Context chunks for Selenium prompt", min_value=3, max_value=12, value=6, step=1, key="script_top_k"
        )
        # Page under test: explicit selection, or the project's latest page
        selected_page_id: Optional[int] = None
        try:
            script_pages = list_pages()
        except Exception:
            script_pages = []
        if script_pages:
            page_options = ["(project default / latest page)"] + [
                f"#{p['id']}: {p.get('name') or p.get('url') or 'page'}" for p in script_pages
            ]
            page_choice = st.selectbox("Page under test", page_options, key="scripts_page_select")
            if page_choice != page_options[0]:
                selected_page_id = script_pages[page_options.index(page_choice) - 1]["id"]
        base_url = st.text_input(
            "Base URL override (optional)",
            value="",
            placeholder="Auto: /checkout serves the selected page",
            help="Leave blank to auto-derive from the selected page / project.",
        )
        review_flag = st.checkbox(
            "Review script (second Gemini pass: selector robustness + missing assertions)",
            value=False,
            key="script_review",
        )
        if st.button("Generate Selenium Script", use_container_width=True):
            if not manual_input.strip():
                st.error("Provide a test case to automate.")
            else:
                try:
                    meta, stream = generate_script_stream(
                        manual_input.strip(),
                        script_top_k,
                        base_url.strip() or None,
                        review=review_flag,
                        page_id=selected_page_id,
                    )
                    accumulated: List[str] = []

                    def tapped():
                        for chunk in stream:
                            accumulated.append(chunk)
                            yield chunk

                    st.write_stream(tapped())
                    full_text = "".join(accumulated)
                    script_text, sep, review_text = full_text.partition("--- REVIEW ---")
                    st.session_state["last_script"] = script_text.strip()
                    st.session_state["last_script_contexts"] = meta.get("contexts", [])
                    st.session_state["last_review"] = review_text.strip() if sep else ""
                    st.success("Selenium script generated and cached.")
                except Exception as exc:
                    st.error(f"Failed to generate Selenium script: {exc}")

        # Always show the last generated script, even after navigation
        last_script = st.session_state.get("last_script")
        last_script_contexts = st.session_state.get("last_script_contexts", [])
        last_review = st.session_state.get("last_review", "")
        if last_script:
            if last_script_contexts:
                with st.expander("Retrieved context (last run)"):
                    for ctx in last_script_contexts:
                        st.markdown(f"**{ctx['source']}** — `{ctx['chunk_id']}`")
                        st.write(ctx["text"])
            st.code(last_script, language="python")
        if last_review:
            with st.expander("Gemini review"):
                st.markdown(last_review)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Coverage / Traceability Page ---
elif page == "coverage":
    st.markdown("### 🧭 Coverage & Traceability")
    st.markdown(
        "Matrix of ingested documents vs. test cases. A cell is marked when the "
        "test case's `grounded_in` citation references that document."
    )
    try:
        coverage = get_coverage()
    except Exception as exc:
        st.error(f"Failed to load coverage: {exc}")
    else:
        docs = coverage.get("documents", [])
        m1, m2, m3 = st.columns(3)
        m1.metric("Documents", coverage.get("total_documents", 0))
        m2.metric("Covered", coverage.get("covered_documents", 0))
        m3.metric("Uncovered", coverage.get("uncovered_documents", 0))

        if not docs:
            st.info("No documents ingested yet. Upload support docs in the Knowledge Base tab.")
        else:
            # Collect every covering test case across all documents for matrix columns
            case_labels: dict = {}
            case_meta: dict = {}
            for d in docs:
                for tc in d.get("test_cases", []):
                    cid = tc.get("id")
                    case_labels[cid] = f"#{cid}: {tc.get('scenario', '')}"
                    case_meta[cid] = tc

            # doc x test-case matrix
            matrix = {}
            for d in docs:
                covered_ids = {tc.get("id") for tc in d.get("test_cases", [])}
                matrix[d["filename"]] = {
                    case_labels[cid]: ("✅" if cid in covered_ids else "")
                    for cid in case_labels
                }
            if case_labels:
                df = pd.DataFrame(matrix).T
                df.insert(0, "status", ["covered" if d["status"] == "covered" else "uncovered" for d in docs])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No test cases are grounded in any document yet. Generate test cases first.")
                status_df = pd.DataFrame(
                    [{"filename": d["filename"], "status": d["status"]} for d in docs]
                )
                st.dataframe(status_df, use_container_width=True, hide_index=True)

            # Per-document detail: covering test cases
            st.markdown("#### Covering test cases per document")
            for d in docs:
                badge = "✅ covered" if d["status"] == "covered" else "⬜ uncovered"
                with st.expander(f"{d['filename']} — {badge} ({len(d.get('test_cases', []))} cases)"):
                    st.caption(f"sha256: `{d.get('sha256', '')[:16]}…`")
                    for tc in d.get("test_cases", []):
                        st.markdown(f"**#{tc['id']}** {tc.get('feature', '')} — {tc.get('scenario', '')}")
                        st.caption(f"grounded_in: {tc.get('grounded_in', '')}")

# --- Footer ---
st.markdown(
    """
    <footer>
      Developer Sai Theja
    </footer>
    """,
    unsafe_allow_html=True,
)
