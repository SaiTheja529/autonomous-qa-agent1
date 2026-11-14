import json
import os
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_backend() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.ok
    except Exception:
        return False


def ingest_documents(docs: List, checkout_html: Optional, reset: bool):
    files = []
    for doc in docs:
        files.append(("docs", (doc.name, doc.getvalue(), doc.type)))
    if checkout_html:
        files.append(("checkout", (checkout_html.name, checkout_html.getvalue(), checkout_html.type)))
    data = {"reset": str(reset).lower()}
    # Allow extra time for first-time model downloads/embeddings
    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, data=data, timeout=240)
    resp.raise_for_status()
    return resp.json()


def generate_test_cases(query: str, top_k: int):
    payload = {"query": query, "top_k": top_k}
    resp = requests.post(f"{BACKEND_URL}/generate-testcases", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def generate_script(test_case: str, top_k: int, base_url: str):
    payload = {"test_case": test_case, "top_k": top_k, "base_url": base_url}
    resp = requests.post(f"{BACKEND_URL}/generate-script", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_test_cases(status: Optional[str] = None):
    params = {}
    if status:
        params["status"] = status
    resp = requests.get(f"{BACKEND_URL}/testcases", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def update_test_case(test_case_id: int, fields: dict):
    resp = requests.patch(f"{BACKEND_URL}/testcases/{test_case_id}", json=fields, timeout=30)
    resp.raise_for_status()
    return resp.json()


def delete_test_case(test_case_id: int):
    resp = requests.delete(f"{BACKEND_URL}/testcases/{test_case_id}", timeout=30)
    resp.raise_for_status()
    return resp.status_code


def run_test_case(test_case_id: int, script_code: Optional[str] = None, base_url: Optional[str] = None):
    payload = {}
    if script_code:
        payload["script_code"] = script_code
    if base_url:
        payload["base_url"] = base_url
    resp = requests.post(f"{BACKEND_URL}/testcases/{test_case_id}/run", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def get_run(run_id: int):
    resp = requests.get(f"{BACKEND_URL}/runs/{run_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_run_screenshot(run_id: int):
    resp = requests.get(f"{BACKEND_URL}/runs/{run_id}/screenshot", timeout=30)
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

# --- Simple page state + Navbar ---
PAGES = {
    "kb": "Knowledge Base",
    "cases": "Test Cases",
    "scripts": "Selenium Scripts",
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
            reset = st.checkbox("Reset existing knowledge base", value=False)
            submitted = st.form_submit_button("Build Knowledge Base", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not docs and not checkout_html:
            st.error("Please upload at least one document or checkout.html.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    result = ingest_documents(docs, checkout_html, reset)
                    st.success(
                        f"Ingested {result['docs_ingested']} docs with {result['chunks_added']} chunks. "
                        f"Checkout saved: {result['html_saved']}."
                    )
                    st.json(result)
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

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
                options=["", "active", "draft", "archived"],
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
                        options=["active", "draft", "archived"],
                        index=["active", "draft", "archived"].index(selected.get("status", "active"))
                        if selected.get("status", "active") in ["active", "draft", "archived"]
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
        base_url = st.text_input(
            "Base URL where checkout.html is served",
            value="http://localhost:8000/checkout/checkout.html",
            help="The generated script will set base_url to this value.",
        )
        if st.button("Generate Selenium Script", use_container_width=True):
            if not manual_input.strip():
                st.error("Provide a test case to automate.")
            else:
                with st.spinner("Generating Selenium script..."):
                    try:
                        result = generate_script(manual_input.strip(), script_top_k, base_url.strip())
                        st.session_state["last_script"] = result["script"]
                        st.session_state["last_script_contexts"] = result["contexts"]
                        st.success("Selenium script generated and cached.")
                    except Exception as exc:
                        st.error(f"Failed to generate Selenium script: {exc}")

        # Always show the last generated script, even after navigation
        last_script = st.session_state.get("last_script")
        last_script_contexts = st.session_state.get("last_script_contexts", [])
        if last_script:
            if last_script_contexts:
                with st.expander("Retrieved context (last run)"):
                    for ctx in last_script_contexts:
                        st.markdown(f"**{ctx['source']}** — `{ctx['chunk_id']}`")
                        st.write(ctx["text"])
            st.code(last_script, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    """
    <footer>
      Developer Sai Theja
    </footer>
    """,
    unsafe_allow_html=True,
)
