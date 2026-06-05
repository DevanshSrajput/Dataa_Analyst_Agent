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
import sys

# Mark this process as a Streamlit run so Agent.py stays quiet.
os.environ["STREAMLIT_RUN"] = "1"

import streamlit as st

# Local import: Agent.py is the engine, this file is the UI.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Agent import DocumentAnalystAgent, _get_api_key, DEFAULT_MODEL  # noqa: E402

# Re-exports: keep the legacy `from app import _safe_filename` import
# path working for tests and any external callers, while the source of
# truth lives in app_helpers.py / theme.py.
from app_helpers import _safe_filename, _TEMP_UPLOAD_DIR  # noqa: E402
from app_helpers import AVAILABLE_MODELS, DEFAULT_MODEL_ID, list_model_choices  # noqa: E402
from theme import css_for_theme  # noqa: E402


def _inject_css(theme_mode: str) -> None:
    """Render the active theme's CSS into the Streamlit page.

    Thin wrapper around `theme.css_for_theme` kept here so the call
    site in `main()` doesn't have to know about the theme module.
    """
    st.markdown(f"<style>{css_for_theme(theme_mode)}</style>", unsafe_allow_html=True)


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
                # ISSUES.md #2 + #1: free the per-agent temp viz dir,
                # the in-memory viz bytes cache, AND the BM25 chunk
                # cache so memory doesn't leak across resets.
                agent.clear_caches()
                agent.clear_visualizations()
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
                # ISSUES.md: derive the uploader allowlist from
                # `Agent._SUPPORTED_EXTENSIONS` so the upload widget
                # and the extractor can never disagree. If you add a
                # new type, add it to that frozenset — the widget
                # picks it up automatically.
                type=sorted(Agent._SUPPORTED_EXTENSIONS),
                help=(
                    "Supported formats: PDF, DOCX, TXT, CSV, Excel "
                    "(XLSX/XLS), Images (JPG, JPEG, PNG, TIFF, BMP)"
                ),
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
                            # ISSUES.md #2: agent now returns (label, bytes)
                            # pairs; render straight from memory rather than
                            # reading back from `visualizations_<name>/` in CWD.
                            viz_pairs = agent.create_visualizations(df, uploaded_file.name)
                            if viz_pairs:
                                with st.expander(f"📈 Auto-Generated Charts - {uploaded_file.name}"):
                                    viz_cols = st.columns(2)
                                    for idx, (chart_name, png_bytes) in enumerate(viz_pairs):
                                        with viz_cols[idx % 2]:
                                            st.markdown(f"**{chart_name}**")
                                            st.image(png_bytes, use_container_width=True)

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
                # SECURITY (ISSUES.md #1): never interpolate LLM
                # output or user input into raw HTML. st.chat_message
                # renders the text with markdown sanitization on, and
                # avoids the unsafe_allow_html sink entirely.
                # UX (ISSUES.md #1 — streaming): render the user turn
                # first so the exchange is visible, then stream the
                # assistant turn token-by-token via st.write_stream.
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.chat_message("assistant"):
                    try:
                        # UX (ISSUES.md #1 — streaming): hand the
                        # generator straight to st.write_stream, which
                        # keeps the assistant bubble in a "thinking"
                        # state until the first token arrives and then
                        # appends each chunk in place. Persistence to
                        # conversation_history happens inside
                        # stream_answer when the generator is fully
                        # drained, so the side-effect ordering matches
                        # answer_question exactly.
                        st.write_stream(agent.stream_answer(user_question))
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
                                try:
                                    # Streaming: same path as the main
                                    # chat input. write_stream handles
                                    # the "thinking" state for us.
                                    st.markdown(f"**💡 {title}:**")
                                    st.write_stream(agent.stream_answer(question))
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                    with c2:
                        if i + 1 < len(quick_questions):
                            title, question = quick_questions[i + 1]
                            if st.button(
                                title, key=f"quick_{i + 1}", use_container_width=True
                            ):
                                try:
                                    st.markdown(f"**💡 {title}:**")
                                    st.write_stream(agent.stream_answer(question))
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

                    # ISSUES.md #2: read viz bytes from the in-memory
                    # cache (`agent.visualizations`) instead of listing
                    # `visualizations_<file_name>/` in the CWD. The
                    # agent populates the cache when create_visualizations
                    # is called (during upload), and the bytes survive
                    # Streamlit reruns because they live on the agent
                    # instance inside `st.session_state`.
                    viz_pairs = agent.visualizations.get(file_name, [])
                    if viz_pairs:
                        st.markdown("**📊 Visualizations:**")
                        for chart_name, png_bytes in viz_pairs:
                            st.markdown(f"*{chart_name}*")
                            st.image(png_bytes, use_container_width=True)

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
        # Curated list from app_helpers — free tier first, then paid.
        # If the agent's current model is no longer in the catalogue
        # (e.g. an id was removed), fall back to the default rather
        # than crashing the selectbox.
        model_choices = list_model_choices()
        model_ids = [mid for mid, _name in model_choices]
        try:
            current_index = model_ids.index(agent.model)
        except ValueError:
            current_index = 0
        selected_model = st.selectbox(
            "Select AI Model:",
            options=model_ids,
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
            # Wipe the per-agent caches (in-memory + on-disk SQLite)
            # BEFORE we start touching session_state. If we did it
            # in the other order, a stale agent reference could be
            # used by the next render before the DB was cleared.
            try:
                agent.clear_caches()
                agent.clear_visualizations()
            except Exception:
                # Even if the agent is in a bad state, the reset must
                # still complete — never let cleanup errors block a
                # user's "start over" action.
                pass

            # Pop every key one at a time. Using .pop(..., None) is
            # safe across Streamlit versions: it never raises on a
            # missing key and never mutates the dict while iterating
            # it (we capture keys into a list first, like the
            # original code did, but pop is the documented public
            # API for session_state and tolerates the rare race
            # where Streamlit adds a key between snapshot and pop).
            for key in list(st.session_state.keys()):
                st.session_state.pop(key, None)

            # Drop Streamlit's own caches so any @st.cache_data
            # resources tied to the old session are released.
            try:
                st.cache_data.clear()
            except Exception:
                pass

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
