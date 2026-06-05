'''
AI Document Analyst - Streamlit UI.

This module is the entrypoint for the Streamlit app (and for `streamlit run`).
It is a thin presentation layer on top of Agent.DocumentAnalystAgent - all
extraction, analysis, and LLM calls live there.

Deployment notes (Streamlit Cloud):
    - Set the API key in the app's Secrets manager as OPENCODE_API_KEY
      (or OPENCODE_ZEN_API_KEY). The agent will pick it up automatically.
    - `packages.txt` installs tesseract for OCR on the cloud image.
    - `requirements.txt` pins Python deps; this file just imports them.
'''

import os
import re
import sys
import time
import uuid

# Mark this process as a Streamlit run so Agent.py stays quiet.
os.environ["STREAMLIT_RUN"] = "1"

# ---------------------------------------------------------------------------
# Upload path safety (ISSUES.md #2)
# ---------------------------------------------------------------------------
# The original Agent.py used `temp_<uploaded_file.name>` as a relative path,
# which is a path-traversal sink on Streamlit Cloud (CWD is the repo root).
# `werkzeug.utils.secure_filename` is the canonical fix, but werkzeug is not
# in our dependency tree, so we replicate its behavior with a strict
# allowlist + basename. The on-disk path is also keyed on a UUID and lives
# in a dedicated directory, so the user's filename never appears in the
# filesystem path at all.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TEMP_UPLOAD_DIR = "temp_uploads"


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe basename derived from `name`.

    Strips directory components, replaces any non `[A-Za-z0-9._-]` run with
    a single underscore, and falls back to a UUID if the result is empty.
    This mirrors werkzeug's `secure_filename` semantics without adding a
    dependency.
    """
    # 1. Drop any path components (handles `..`, `/`, `\\`).
    base = os.path.basename(name)
    # 2. Replace every unsafe run with a single underscore.
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._-")
    # 3. Empty result (e.g. name was all metacharacters) -> random fallback.
    return cleaned or uuid.uuid4().hex

import streamlit as st

# Local import: Agent.py is the engine, this file is the UI.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Agent import DocumentAnalystAgent, _get_api_key, DEFAULT_MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# Model catalogue (sourced live from OpenCode Zen)
# ---------------------------------------------------------------------------
# Keep this small and curated - the model picker is a UI element, not a
# reference doc. The "free" tier is surfaced first so users on a budget land
# on something that won't burn credits.
AVAILABLE_MODELS = {
    "minimax-m3-free": {
        "name": "minimax-m3-free (default)",
        "description": "Free tier MiniMax model on OpenCode Zen.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "mimo-v2.5-free": {
        "name": "MiMo v2.5 Free",
        "description": "Free tier MiMo, good general chat.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "qwen3.6-plus-free": {
        "name": "Qwen 3.6 Plus Free",
        "description": "Free tier Qwen, solid reasoning.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "deepseek-v4-flash-free": {
        "name": "DeepSeek v4 Flash Free",
        "description": "Free tier DeepSeek, fast responses.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "nemotron-3-ultra-free": {
        "name": "Nemotron 3 Ultra Free",
        "description": "Free tier Nemotron, large context.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "minimax-m2.7": {
        "name": "minimax m2.7",
        "description": "Latest paid MiniMax model.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐",
    },
    "minimax-m2.5": {
        "name": "minimax m2.5",
        "description": "Previous-generation MiniMax.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐",
    },
    "gpt-5": {
        "name": "GPT-5",
        "description": "OpenAI GPT-5 via Zen.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐⭐",
    },
    "claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6",
        "description": "Anthropic Sonnet via Zen.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐⭐",
    },
    "gemini-3.1-pro": {
        "name": "Gemini 3.1 Pro",
        "description": "Google Gemini Pro via Zen.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐⭐",
    },
}


# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------
# Compact replacement for the original Agent.py CSS block. The original
# was ~700 lines of `!important` arms-race styling; this is a focused,
# scoped version that still hits the major elements but won't dominate
# the file.

DARK_CSS = """
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #262730;
    --bg-tertiary: #3d3d3d;
    --text-primary: #fafafa;
    --text-secondary: #e0e0e0;
    --accent-color: #667eea;
    --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    --border-color: #555555;
    --card-bg: #262730;
    --upload-bg: #1e1e2e;
    --tab-bg: #262730;
    --tab-selected: #667eea;
}
.stApp { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox > div > div { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"] { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"]:hover { background-color: var(--accent-color) !important; color: white !important; }
.streamlit-expanderHeader, .streamlit-expanderContent { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stMetric, [data-testid="metric-container"] { background-color: var(--card-bg) !important; color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab"] { background-color: var(--tab-bg) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { background-color: var(--tab-selected) !important; color: white !important; }
.stButton > button { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; border-color: var(--border-color) !important; }
.stButton > button:hover { background-color: var(--accent-color) !important; color: white !important; border-color: var(--accent-color) !important; }
.stFileUploader, .stDataFrame { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stProgress > div > div > div > div { background-color: var(--accent-color) !important; }
.stSlider label, .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label { color: var(--text-primary) !important; }
h1, h2, h3, h4, h5, h6, p, div, span, label, li, strong, em, a { color: var(--text-primary) !important; }
.stMarkdown, .stMarkdown * { color: var(--text-primary) !important; }
.main-header { background: var(--accent-gradient); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.feature-card { background: var(--card-bg); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--accent-color); margin: 1rem 0; color: var(--text-primary); }
.upload-zone { border: 2px dashed var(--accent-color); border-radius: 10px; padding: 2rem; text-align: center; background: var(--upload-bg); color: var(--text-primary); margin: 1rem 0; }
.chat-container { background: var(--card-bg); border-radius: 10px; padding: 1rem; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
"""

LIGHT_CSS = """
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-tertiary: #e9ecef;
    --text-primary: #212529;
    --accent-color: #667eea;
    --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    --border-color: #dee2e6;
    --card-bg: #f8f9fa;
    --upload-bg: #f8f9fa;
    --tab-bg: #f0f2f6;
    --tab-selected: #667eea;
}
.stApp { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea { background-color: white !important; color: var(--text-primary) !important; }
.stSelectbox > div > div { background-color: white !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
.stSelectbox [role="option"] { background-color: white !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"]:hover { background-color: var(--accent-color) !important; color: white !important; }
.streamlit-expanderHeader, .streamlit-expanderContent { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stMetric, [data-testid="metric-container"] { background-color: var(--card-bg) !important; color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab"] { background-color: var(--tab-bg) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { background-color: var(--tab-selected) !important; color: white !important; }
.stButton > button { background-color: white !important; color: var(--text-primary) !important; border-color: var(--border-color) !important; }
.stButton > button:hover { background-color: var(--accent-color) !important; color: white !important; }
.main-header { background: var(--accent-gradient); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.feature-card { background: var(--card-bg); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--accent-color); margin: 1rem 0; color: var(--text-primary); }
.upload-zone { border: 2px dashed var(--accent-color); border-radius: 10px; padding: 2rem; text-align: center; background: var(--upload-bg); color: var(--text-primary); margin: 1rem 0; }
.chat-container { background: var(--card-bg); border-radius: 10px; padding: 1rem; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
"""


def _inject_css(theme_mode: str) -> None:
    css = DARK_CSS if theme_mode == "dark" else LIGHT_CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AI Document Analyst",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/DevanshSrajput/Dataa_Analyst_Agent",
            "Report a bug": "mailto:dksdevansh@gmail.com",
            "About": "AI-powered document analysis tool by Devansh Singh",
        },
    )

    # --- session state defaults ---
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 500
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = 3

    _inject_css(st.session_state.theme_mode)

    # --- header ---
    # SECURITY (ISSUES.md #1): the only interpolated values are
    # `theme_indicator` (a hard-coded emoji literal) and
    # `theme_mode.title()` (constrained to "Dark" or "Light"), but we
    # still avoid unsafe_allow_html and let the CSS `.main-header` class
    # style the markdown. Streamlit sanitizes by default.
    theme_indicator = "🌙" if st.session_state.theme_mode == "dark" else "☀️"
    st.markdown(
        f"# 🤖 AI Document Analyst {theme_indicator}\n\n"
        f"_Transform your documents into actionable insights with AI_  \n"
        f"Built by Devansh Singh · Powered by OpenCode Zen · "
        f"{st.session_state.theme_mode.title()} Mode",
    )

    # --- agent init ---
    api_key = _get_api_key()
    if not api_key:
        st.error("🔐 **API Key Required!**")
        with st.expander("🔧 How to set up your OpenCode Zen API key"):
            st.markdown(
                """
                **Locally (development):**
                1. Get an API key at [opencode.ai/zen](https://opencode.ai/zen)
                2. Add it to `.env`: `OPENCODE_API_KEY=your_key_here`

                **On Streamlit Cloud (deployment):**
                1. Open your app in the Streamlit Cloud dashboard
                2. Go to **Settings → Secrets**
                3. Paste:
                   ```
                   OPENCODE_API_KEY = "your_key_here"
                   ```
                4. Save — the app will reboot automatically.
                """
            )
        st.stop()

    if "agent" not in st.session_state:
        with st.spinner("🚀 Initializing AI Agent..."):
            try:
                st.session_state.agent = DocumentAnalystAgent(
                    api_key=api_key,
                    model=st.session_state.get("model", DEFAULT_MODEL),
                )
                st.success("✅ AI Agent ready for action!")
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.stop()

    agent = st.session_state.agent

    # --- sidebar ---
    with st.sidebar:
        st.markdown("### 🎨 Theme")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "☀️ Light",
                type="primary" if st.session_state.theme_mode == "light" else "secondary",
                use_container_width=True,
                key="light_theme",
            ):
                st.session_state.theme_mode = "light"
                st.rerun()
        with col2:
            if st.button(
                "🌙 Dark",
                type="primary" if st.session_state.theme_mode == "dark" else "secondary",
                use_container_width=True,
                key="dark_theme",
            ):
                st.session_state.theme_mode = "dark"
                st.rerun()

        st.markdown("---")
        st.markdown("### 🛠️ Control Panel")
        if agent.document_content:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Files", len(agent.document_content))
            with col2:
                st.metric("📊 Datasets", len(agent.data_frames))

        with st.expander("🔑 API Configuration"):
            if api_key:
                st.success("✅ API key loaded")
                st.text(f"Key: {api_key[:8]}...")
                st.text(f"Model: {agent.model}")
            else:
                st.warning("⚠️ No API key found")

        if agent.document_content:
            st.markdown("### 📋 Processed Files")
            for file_name in agent.document_content.keys():
                file_type = agent.document_content[file_name]["file_type"]
                icon = "📊" if file_type in ["csv", "xlsx", "xls"] else "📄"
                st.text(f"{icon} {file_name}")

        if agent.document_content:
            if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
                agent.document_content.clear()
                agent.data_frames.clear()
                agent.analysis_results.clear()
                agent.conversation_history.clear()
                st.success("✨ All files cleared!")
                st.rerun()

        st.markdown("---")
        st.markdown("**🤖 AI Document Analyst v3.0**")
        st.markdown("Built by Devansh Singh")
        st.markdown("Powered by OpenCode Zen")

    # --- main tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏠 Home", "📤 Upload & Process", "💬 AI Chat", "📊 Analytics", "⚙️ Settings"]
    )

    # ---- TAB 1: HOME ----
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("## 🎯 What can I do for you?")
            features = [
                ("📄 Multi-Format Support", "PDF, DOCX, TXT, CSV, Excel, Images - all handled."),
                ("🧠 AI-Powered Analysis", "Smart summaries and insights via OpenCode Zen."),
                ("📊 Data Visualization", "Automatic charts, graphs, statistical analysis."),
                ("💬 Conversational Q&A", "Ask questions in natural language."),
                ("📈 Comprehensive Reports", "Executive summaries with key findings."),
                ("⚡ Real-time Processing", "Fast document processing with live progress."),
            ]
            for title, desc in features:
                # SECURITY (ISSUES.md #1): developer-controlled strings,
                # but rendered without unsafe_allow_html so any future
                # change to the list can't introduce HTML injection.
                st.markdown(f"### {title}\n\n{desc}")
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown(
                """
                1. **📤 Upload** documents in the Upload tab
                2. **🔄 Process** files to extract insights
                3. **💬 Chat** with your documents using AI
                4. **📊 Analyze** data with visualizations
                5. **📈 Export** reports
                """
            )
            if not agent.document_content:
                st.info("👆 Go to **Upload & Process** tab to start!")
            else:
                st.success(f"🎉 {len(agent.document_content)} files ready for analysis!")

    # ---- TAB 2: UPLOAD ----
    with tab2:
        st.markdown("## 📤 Document Upload & Processing")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📁 Upload Your Documents")
            uploaded_files = st.file_uploader(
                "Choose files to analyze",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "csv", "xlsx", "jpg", "jpeg", "png", "xls"],
                help="Supported formats: PDF, DOCX, TXT, CSV, Excel, Images (JPG, PNG)",
                key="main_uploader",
            )
            st.markdown(
                """
                **📋 Instructions:**
                - **Drag & drop** files or **click** to browse
                - **Multiple files** can be uploaded at once
                - **Supported formats**: PDF, Word, Text, CSV, Excel, Images
                - **File size limit**: 200MB per file
                """
            )

        with col2:
            st.markdown("### 📊 Upload Stats")
            if agent.document_content:
                st.metric("📄 Total Files", len(agent.document_content))
                st.metric("📊 Data Files", len(agent.data_frames))
                st.metric("💬 Conversations", len(agent.conversation_history))
            else:
                st.info("No files uploaded yet")

        if uploaded_files:
            st.markdown("### 🔄 Processing Documents...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.name in agent.document_content:
                    status_text.text(f"✅ {uploaded_file.name} already processed")
                    continue

                status_text.text(f"🔄 Processing {uploaded_file.name}...")

                # Persist a temp file the extractors can read from a path.
                # SECURITY (ISSUES.md #2): the on-disk path must not be
                # derived from the user-controlled filename. We write to
                # `temp_uploads/<uuid>.<safe_ext>` so path-traversal
                # payloads in the original name can never escape the
                # uploads directory. The agent still receives the user's
                # original name for display and dict-keying.
                safe_name = _safe_filename(uploaded_file.name)
                # Preserve a single dot-extension if the original had one.
                _, orig_ext = os.path.splitext(uploaded_file.name)
                safe_ext = _SAFE_FILENAME_RE.sub("", orig_ext)[:10]
                upload_dir = _TEMP_UPLOAD_DIR
                os.makedirs(upload_dir, exist_ok=True)
                temp_filename = f"{uuid.uuid4().hex}{('.' + safe_ext) if safe_ext else ''}"
                temp_path = os.path.join(upload_dir, temp_filename)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    result = agent.process_document(temp_path, uploaded_file.name)

                    # ISSUES.md #1 (extraction-errors): short-circuit the
                    # render path on failure. We MUST NOT call
                    # `result["summary"]` or hand `result["content"]` to
                    # the LLM — the agent guarantees those are empty
                    # on failure, but a defensive UI also won't render
                    # them.
                    if not result.get("success"):
                        st.error(
                            f"❌ Could not process {uploaded_file.name}: "
                            f"{result.get('error') or 'unknown error'}"
                        )
                        # Clean up any half-stored data frame so chat /
                        # analytics don't see a phantom key.
                        agent.data_frames.pop(uploaded_file.name, None)
                        agent.document_content.pop(uploaded_file.name, None)
                    else:
                        with st.expander(
                            f"✅ {uploaded_file.name} - Processed Successfully",
                            expanded=True,
                        ):
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                st.markdown("**📝 AI Summary:**")
                                st.write(result["summary"])
                            with c2:
                                st.markdown("**📄 File Info:**")
                                st.text(f"Type: {result['file_type'].upper()}")
                                st.text(f"Size: {len(result['content'])} chars")
                                if result["data_frame"] is not None:
                                    df = result["data_frame"]
                                    st.text(f"Rows: {df.shape[0]}")
                                    st.text(f"Columns: {df.shape[1]}")

                    if uploaded_file.name in agent.data_frames:
                        df = agent.data_frames[uploaded_file.name]
                        with st.expander(f"📊 Data Preview - {uploaded_file.name}"):
                            st.markdown("**First 10 rows:**")
                            st.dataframe(df.head(10), use_container_width=True)
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("📏 Rows", df.shape[0])
                            with c2:
                                st.metric("📋 Columns", df.shape[1])
                            with c3:
                                st.metric(
                                    "🔢 Numeric",
                                    len(df.select_dtypes(include=["number"]).columns),
                                )
                            with c4:
                                st.metric(
                                    "📝 Text",
                                    len(df.select_dtypes(include=["object"]).columns),
                                )

                        with st.spinner("🎨 Creating visualizations..."):
                            viz_paths = agent.create_visualizations(df, uploaded_file.name)
                            if viz_paths:
                                with st.expander(f"📈 Auto-Generated Charts - {uploaded_file.name}"):
                                    viz_cols = st.columns(2)
                                    for idx, viz_path in enumerate(viz_paths):
                                        if os.path.exists(viz_path):
                                            with viz_cols[idx % 2]:
                                                chart_name = (
                                                    os.path.basename(viz_path)
                                                    .replace(".png", "")
                                                    .replace("_", " ")
                                                    .title()
                                                )
                                                st.markdown(f"**{chart_name}**")
                                                st.image(viz_path, use_container_width=True)

                except Exception as e:
                    # This `except` is now a safety net for unexpected
                    # failures (e.g. filesystem errors before extraction
                    # begins). Extraction failures are handled inside
                    # the `if not result.get("success")` branch above.
                    st.error(f"❌ Unexpected error processing {uploaded_file.name}: {e}")
                finally:
                    # SECURITY (ISSUES.md #2): clean up the UUID-keyed
                    # temp file we just wrote. Resolved against
                    # `temp_path` (which is now inside temp_uploads/), so
                    # this can never reach outside the uploads directory.
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("✅ All files processed successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        else:
            # SECURITY (ISSUES.md #1): no interpolation, no need for
            # unsafe_allow_html. Streamlit sanitizes by default.
            st.markdown(
                "### 📁 Drop your documents here!\n\n"
                "Supported formats: PDF, DOCX, TXT, CSV, Excel, Images\n\n"
                "Use the file uploader above to get started ⬆️"
            )

    # ---- TAB 3: CHAT ----
    with tab3:
        st.markdown("## 💬 Chat with Your Documents")
        if not agent.document_content:
            st.info("📤 Upload documents first to start chatting!")
            st.markdown(
                """
                ### 🎯 What you can ask:
                - "What are the key insights from this data?"
                - "Summarize the main points of this document"
                - "What patterns do you see in the numbers?"
                - "What are the most important findings?"
                - "Show me correlations between variables"
                """
            )
        else:
            st.markdown("### 🗨️ Ask anything about your documents")
            user_question = st.text_area(
                "Your Question:",
                placeholder="Ask anything about your uploaded documents...",
                height=100,
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                ask_button = st.button(
                    "🔍 Get AI Answer", type="primary", use_container_width=True
                )

            if ask_button and user_question.strip():
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        answer = agent.answer_question(user_question)
                        # SECURITY (ISSUES.md #1): never interpolate LLM
                        # output or user input into raw HTML. st.chat_message
                        # renders the text with markdown sanitization on, and
                        # avoids the unsafe_allow_html sink entirely.
                        st.markdown("### 💡 AI Response:")
                        with st.chat_message("user"):
                            st.markdown(user_question)
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")
            elif ask_button:
                st.warning("Please enter a question first!")

            if agent.data_frames:
                st.markdown("### 🚀 Quick Questions")
                quick_questions = [
                    ("📊 Key Statistics", "What are the key statistics and summary of this dataset?"),
                    ("📈 Trends & Patterns", "What trends and patterns do you see in this data?"),
                    ("🔍 Data Quality", "Are there any missing values or data quality issues?"),
                    ("💡 Key Insights", "What are the most important insights from this data?"),
                    ("🔗 Correlations", "What correlations exist between different variables?"),
                    ("📋 Executive Summary", "Provide an executive summary of the findings"),
                ]
                for i in range(0, len(quick_questions), 2):
                    c1, c2 = st.columns(2)
                    with c1:
                        if i < len(quick_questions):
                            title, question = quick_questions[i]
                            if st.button(title, key=f"quick_{i}", use_container_width=True):
                                with st.spinner("🤖 Analyzing..."):
                                    try:
                                        answer = agent.answer_question(question)
                                        st.success("💡 **Answer:**")
                                        st.write(answer)
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                    with c2:
                        if i + 1 < len(quick_questions):
                            title, question = quick_questions[i + 1]
                            if st.button(
                                title, key=f"quick_{i + 1}", use_container_width=True
                            ):
                                with st.spinner("🤖 Analyzing..."):
                                    try:
                                        answer = agent.answer_question(question)
                                        st.success("💡 **Answer:**")
                                        st.write(answer)
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")

            if agent.conversation_history:
                st.markdown("### 💭 Recent Conversations")
                for i, item in enumerate(reversed(agent.conversation_history[-3:])):
                    with st.expander(
                        f"💬 Q{len(agent.conversation_history) - i}: {item['question'][:60]}..."
                    ):
                        st.markdown(f"**❓ Question:** {item['question']}")
                        st.markdown(f"**🤖 Answer:** {item['answer']}")

    # ---- TAB 4: ANALYTICS ----
    with tab4:
        st.markdown("## 📊 Analytics Dashboard")
        if not agent.data_frames:
            st.info("📈 Upload CSV or Excel files to see analytics!")
            st.markdown(
                """
                ### 📊 Available Analytics:
                - **Statistical Summary**: Mean, median, mode, standard deviation
                - **Data Quality Check**: Missing values, duplicates, outliers
                - **Correlation Analysis**: Relationships between variables
                - **Distribution Plots**: Histograms, box plots, scatter plots
                - **Trend Analysis**: Time series and pattern recognition
                """
            )
        else:
            for file_name, df in agent.data_frames.items():
                with st.expander(f"📊 Analytics: {file_name}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("📏 Rows", df.shape[0])
                    with c2:
                        st.metric("📋 Columns", df.shape[1])
                    with c3:
                        missing_pct = (
                            df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                        )
                        st.metric("❌ Missing %", f"{missing_pct:.1f}%")
                    with c4:
                        numeric_cols = len(df.select_dtypes(include=["number"]).columns)
                        st.metric("🔢 Numeric", numeric_cols)

                    numeric_df = df.select_dtypes(include=["number"])
                    if not numeric_df.empty:
                        st.markdown("**📈 Statistical Summary:**")
                        st.dataframe(numeric_df.describe(), use_container_width=True)

                    viz_dir = f"visualizations_{file_name.replace('.', '_')}"
                    if os.path.exists(viz_dir):
                        viz_files = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
                        if viz_files:
                            st.markdown("**📊 Visualizations:**")
                            for viz_file in viz_files:
                                viz_path = os.path.join(viz_dir, viz_file)
                                chart_name = (
                                    viz_file.replace(".png", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                st.markdown(f"*{chart_name}*")
                                st.image(viz_path, use_container_width=True)

    # ---- TAB 5: SETTINGS ----
    with tab5:
        st.markdown("## ⚙️ Application Settings")

        # API key status
        st.markdown("### 🔑 API Key")
        st.success("✅ API key loaded") if api_key else st.error("❌ No API key found")
        st.markdown(
            "Keys are read in this order: `st.secrets → OPENCODE_API_KEY env var → TOGETHER_API_KEY env var`."
        )

        st.markdown("---")

        # Model
        st.markdown("### 🤖 AI Model Configuration")
        st.markdown(f"**Current Model:** `{agent.model}`")
        model_names = list(AVAILABLE_MODELS.keys())
        try:
            current_index = model_names.index(agent.model)
        except ValueError:
            current_index = 0
        selected_model = st.selectbox(
            "Select AI Model:",
            options=model_names,
            format_func=lambda x: AVAILABLE_MODELS[x]["name"],
            index=current_index,
        )
        model_info = AVAILABLE_MODELS[selected_model]
        st.markdown(
            f"""
            **Model Details:**
            - **Description:** {model_info['description']}
            - **Tier:** {model_info['tier']}
            - **Performance:** {model_info['performance']}
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Apply Model Change", type="primary", use_container_width=True):
                if selected_model != agent.model:
                    agent.model = selected_model
                    st.success(f"✅ Model changed to: {model_info['name']}")
                    st.rerun()
                else:
                    st.info("Model is already selected")
        with c2:
            if st.button("🧪 Test Selected Model", use_container_width=True):
                old_model = agent.model
                agent.model = selected_model
                try:
                    test_response = agent._make_api_call_with_retry(
                        "Respond with the single word: OK", max_tokens=10
                    )
                    if "error" not in test_response.lower():
                        st.success(f"✅ Model responded: {test_response[:80]}")
                    else:
                        st.error(f"❌ {test_response}")
                except Exception as e:
                    st.error(f"❌ {e}")
                finally:
                    agent.model = old_model

        st.markdown("---")

        # Processing
        st.markdown("### 🎛️ Processing Settings")
        max_tokens = st.slider(
            "Max Response Tokens:", 100, 2000, st.session_state.max_tokens, 50
        )
        temperature = st.slider(
            "AI Creativity (Temperature):",
            0.0,
            1.0,
            st.session_state.temperature,
            0.1,
        )
        max_retries = st.slider(
            "Max Retry Attempts:", 1, 5, st.session_state.max_retries
        )
        if st.button("💾 Save Processing Settings", use_container_width=True):
            st.session_state.max_tokens = max_tokens
            st.session_state.temperature = temperature
            st.session_state.max_retries = max_retries
            st.success("✅ Processing settings saved!")

        st.markdown("---")
        st.markdown("### 📈 Session Information")
        st.metric("📄 Processed Files", len(agent.document_content))
        st.metric("📊 Datasets Loaded", len(agent.data_frames))
        st.metric("💬 Conversations", len(agent.conversation_history))

        if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Session reset!")
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            ### ℹ️ About
            **📊 AI Document Analyst v3.0**
            - Built by Devansh Singh
            - Powered by [OpenCode Zen](https://opencode.ai/zen)
            - Default model: `minimax-m3-free`
            """
        )


if __name__ == "__main__":
    main()
