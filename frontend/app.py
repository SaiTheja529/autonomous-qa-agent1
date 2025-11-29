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


def parse_markdown_table(md: str):
    """
    Parse simple Markdown table rows into dictionaries so the user can pick a case.
    """
    rows = []
    for line in md.splitlines():
        if not line.strip().startswith("|"):
            continue
        if set(line.strip()) <= {"|", "-"}:
            continue  # skip separator rows
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "id": parts[0],
                "feature": parts[1],
                "scenario": parts[2],
                "steps": parts[3] if len(parts) > 3 else "",
                "expected": parts[4] if len(parts) > 4 else "",
            }
        )
    return rows


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
                    st.session_state["last_test_cases"] = result["test_cases"]
                    st.session_state["last_contexts"] = result["contexts"]
                    st.success("Test cases generated and cached.")
                except Exception as exc:
                    st.error(f"Failed to generate test cases: {exc}")

        # Always show the last generated test cases, even after navigation
        last_cases = st.session_state.get("last_test_cases")
        last_contexts = st.session_state.get("last_contexts", [])
        if last_cases:
            parsed_rows = parse_markdown_table(last_cases)
            if parsed_rows:
                df = pd.DataFrame(parsed_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown(last_cases)
            if last_contexts:
                with st.expander("Retrieved context (last run)"):
                    for ctx in last_contexts:
                        st.markdown(f"**{ctx['source']}** — `{ctx['chunk_id']}`")
                        st.write(ctx["text"])
        st.markdown('</div>', unsafe_allow_html=True)

# --- Selenium Scripts Page ---
elif page == "scripts":
    st.markdown("### 🛠️ Generate Selenium Script")
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        last_cases = st.session_state.get("last_test_cases", "")
        parsed_rows = parse_markdown_table(last_cases) if last_cases else []

        selected_case_text = ""
        if parsed_rows:
            options = [f"{row['id']}: {row['scenario']}" for row in parsed_rows]
            choice = st.selectbox("Select a test case from the table above", options)
            selected_row = parsed_rows[options.index(choice)]
            selected_case_text = (
                f"{selected_row['id']} - {selected_row['feature']} - {selected_row['scenario']}. "
                f"Steps: {selected_row['steps']}. Expected: {selected_row['expected']}"
            )

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
