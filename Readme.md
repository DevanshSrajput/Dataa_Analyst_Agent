# 📊 AI Document Analyst v3.0

> _"Because reading your own documents is so 2022. Let the AI do the heavy lifting while you take all the credit. Now with 100% more dark mode and a backend that actually works!"_

<a id="top"></a>

---

## 🧭 Table of Contents

- [😏 What Is This Masterpiece?](#-what-is-this-masterpiece)
- [🕰️ What Was New in v2.0 (The Glow-Up Edition)](#-what-was-new-in-v20-the-glow-up-edition)
- [📸 Screenshots](#-screenshots)
- [🚀 Features](#-features-because-youre-too-busy-to-read-the-code)
  - [🏠 Home Tab](#-home-tab)
  - [📤 Upload & Process Tab](#-upload--process-tab)
  - [💬 AI Chat Tab](#-ai-chat-tab)
  - [📊 Analytics Tab](#-analytics-tab)
  - [⚙️ Settings Tab](#-settings-tab)
- [🛠️ How to Run](#-how-to-run-because-reading-instructions-is-actually-important)
  - [☁️ Deploy to Streamlit Cloud](#-deploy-to-streamlit-cloud-one-click)
- [🧪 Running the Tests](#-running-the-tests)
- [🗂️ Project Structure](#-project-structure)
- [🔑 API Key (OpenCode Zen)](#-api-key-opencode-zen)
- [💾 Session Persistence](#-session-persistence)
  - [Local Development](#local-development)
  - [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
  - [Available AI Models](#available-ai-models)
- [🤖 How It Works](#-how-it-works-magic-but-with-science)
- [🎯 Why Use This?](#-why-use-this-besides-my-incredible-ego)
- [🔧 Installation Troubleshooting](#-installation-troubleshooting)
- [📝 Credits & Thanks](#-credits--thanks)
- [⚠️ Disclaimer](#-disclaimer)
- [💌 Feedback & Support](#-feedback--support)
- [🎉 Recent Changes (v3.1)](#-recent-changes-v31)
  - [Security & Correctness](#security--correctness)
  - [Retrieval & UX](#retrieval--ux)
  - [Test Coverage](#test-coverage)
  - [Project Structure](#project-structure)
  - [What Changed (v3.0 → v3.1)](#what-changed-v30--v31)
- [🧭 Project History](#-project-history)
- [👤 Maintainer](#-maintainer)

---

## 😏 What Is This Masterpiece?

Welcome to **AI Document Analyst v3.0** – the tool you never knew you desperately needed, now with a complete backend overhaul and a cleaner architecture. Built by _Devansh Singh_ (yes, I made this, and yes, I'm still waiting for my Nobel Prize).

This Python-powered, AI-infused, theme-switching, sarcasm-enabled agent will:

- **Read** your PDFs, DOCX, TXT, CSV, Excel (XLSX/XLS), and images (JPG, JPEG, PNG, TIFF, BMP — OCR, because why not?).
- **Summarize** them via **OpenCode Zen** — OpenAI-compatible chat completions, with `minimax-m3-free` as the default model.
- **Stream** answers back token-by-token via `st.write_stream` — first words in <1 s, full reply in 1–3 s.
- **Analyze** your data with pandas wizardry (the Avengers of data science).
- **Visualize** trends and patterns with auto-generated charts (because you love pretty colors).
- **Chat** with your documents like they're your best friend (spoiler: they're more reliable).
- **Switch** between light and dark themes (because your eyes deserve options).
- **Generate Reports** that sound like you spent hours on them (you didn't).
- **Deploy** to Streamlit Cloud with one click (yes, really).

All this, wrapped in a gorgeous Streamlit UI with tabs, themes, and more bells and whistles than a marching band.

[⬆ Back to top](#top)

---

## 🕰️ What Was New in v2.0 (The Glow-Up Edition)

<details>
<summary>📜 Click to expand the v2.0 changelog (historical context)</summary>

A historical changelog entry — v2.0 was the previous major version
before the 2026-06-05 reset to v3.0. Kept here for readers who arrive
via the v2.0 screenshots below.

- **🌙 Dark/Light Mode**: Toggle between themes like a pro. Your retinas will thank you.
- **📑 Tabbed Interface**: Home, Upload, Chat, Analytics, and Settings tabs. Organization is sexy.
- **🎯 Enhanced Upload**: Drag & drop files with style. Progress bars included (because waiting is fun).
- **💬 Interactive Chat**: Ask your documents anything. They actually respond now.
- **⚙️ Settings Panel**: Configure everything from AI models to themes. Power user vibes.
- **📊 Advanced Analytics**: Beautiful charts, stats, and insights that'll make Excel cry.
- **🎪 Better UI**: Modern gradients, cards, and animations. Instagram-worthy data analysis.

</details>

[⬆ Back to top](#top)

---

## 📸 Screenshots

> Glimpses! So you know it actually works 😁

<div align="center">
<table>
<tr>
<td align="center"><img src="Assests/v2/Home.png" width="400"><br><b>🏠 Home Dashboard</b></td>
<td align="center"><img src="Assests/v2/U&P.png" width="400"><br><b>📤 Upload & Process</b></td>
</tr>
<tr>
<td align="center"><img src="Assests/v2/U&P-2.png" width="400"><br><b>📊 File Processing</b></td>
<td align="center"><img src="Assests/v2/chat.png" width="400"><br><b>💬 AI Chat Interface</b></td>
</tr>
<tr>
<td align="center"><img src="Assests/v2/chat-2.png" width="400"><br><b>🤖 Chat Conversation</b></td>
<td align="center"><img src="Assests/v2/Analytics.png" width="400"><br><b>📊 Analytics Dashboard</b></td>
</tr>
<tr>
<td align="center"><img src="Assests/v2/Settings.png" width="400"><br><b>⚙️ Settings Panel</b></td>
<td align="center"><img src="Assests/v2/settings-2.png" width="400"><br><b>🌙 Dark Mode Settings</b></td>
</tr>
</table>
</div>

> The UI may differ slightly if I decided to tweak it and forgot to update screenshots. JK! (But seriously, it might.)

[⬆ Back to top](#top)

---

## 🚀 Features (Because You're Too Busy to Read the Code)

### 🏠 **Home Tab**

- Welcome dashboard with feature overview
- Quick start guide (for the impatient)
- Status indicators (so you know things are working)

### 📤 **Upload & Process Tab**

- **Multi-format Support:** PDF, DOCX, TXT, CSV, Excel (XLSX/XLS), and images (JPG, JPEG, PNG, TIFF, BMP)
- **Drag & Drop Interface:** Because clicking is so 2010
- **Real-time Processing:** Watch your files get analyzed in real-time
- **Progress Tracking:** Know exactly what's happening (transparency is key)
- **Auto-Visualization:** Charts generate themselves (like magic, but with code)

### 💬 **AI Chat Tab**

- **Conversational Q&A:** Ask anything about your documents
- **Context Awareness:** Remembers your conversation (better than most humans)
- **Quick Questions:** Pre-built buttons for instant insights
- **Smart Responses:** Powered by OpenCode Zen (OpenAI-compatible chat completions)
- **Token-by-token streaming:** Answers render chunk-by-chunk via
  `st.write_stream`, so the first words appear in <1 s instead of
  waiting for the full 1–3 s completion

### 📊 **Analytics Tab**

- **Statistical Summaries:** Mean, median, mode, and other math-y things
- **Data Quality Checks:** Missing values, duplicates, outliers
- **Correlation Analysis:** Find relationships you never knew existed
- **Auto-Generated Charts:** Histograms, heatmaps, box plots, and more

### ⚙️ **Settings Tab**

- **API Key Management:** Built-in key configuration (no more .env hunting)
- **Model Selection:** Choose from multiple AI models
- **Theme Switching:** Light/Dark mode toggle
- **Processing Settings:** Customize AI behavior
- **Session Management:** Reset everything when you mess up

[⬆ Back to top](#top)

---

## 🛠️ How to Run (Because Reading Instructions Is Actually Important)

### 1. **Install Requirements:**

```sh
pip install -r requirements.txt
```

(Or just install everything you see in the imports. I believe in your package management skills.)

### 2. **Run the App:**

```sh
streamlit run app.py
```

> **Note:** the entrypoint is `app.py`, not `Agent.py`. `Agent.py` is
> the engine module; `app.py` is the Streamlit UI. The old
> `Data_Analyst_Agent.py` / `Agent.py` monolith is gone.

If `streamlit` is on your `PATH` you can also do `python -m streamlit run app.py`.

### 3. **Open Your Browser:**

- The app opens at `http://localhost:8501` (Streamlit's default)
- If it doesn't, manually navigate there (I can't click for you)

### 4. **Start Analyzing:**

- Set your `OPENCODE_API_KEY` in `.env` (or paste it into the app's
  **Settings** tab once it's running)
- Upload your files and start chatting — the default model is
  `minimax-m3-free` (no credit card burn)

### ☁️ **Deploy to Streamlit Cloud (one click):**

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), click **New app**,
   select the repo, and set the main file path to `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   OPENCODE_API_KEY = "your_key_here"
   ```
4. Click **Deploy**. The first build will pull `tesseract` from
   `packages.txt` and Python deps from `requirements.txt`.

> `packages.txt` in the repo root installs the system `tesseract` binary
> on the Cloud image so image OCR works.

[⬆ Back to top](#top)

---

## 🧪 Running the Tests

The repo ships with a 41-test suite under `tests/` that covers extraction
failures, BM25 retrieval, the SSRF policy, the path-traversal-safe
filename helper, and the extension allowlist. Run it with either:

```sh
# stdlib only — works out of the box, no install step
python -m unittest discover -s tests -v

# or, if you've installed pytest as a dev dep:
python -m pytest tests -v
```

The suite finishes in ~100 ms because it stubs the LLM layer and never
hits the network. Heavy visualization deps (matplotlib, seaborn, PIL,
pytesseract) are not imported by the tests.

To install pytest (optional):

```sh
pip install pytest
```

[⬆ Back to top](#top)

---

## 🗂️ Project Structure

```
Dataa_Analyst_Agent/
├── app.py              # Streamlit UI — the entrypoint for `streamlit run`
├── Agent.py            # Engine: extractors, BM25 retriever, OpenCode Zen client
├── ISSUES.md           # Open audit findings (6 items open as of v3.1)
├── Readme.md           # You are here
├── requirements.txt    # Python runtime deps
├── packages.txt        # System deps (tesseract for OCR on Streamlit Cloud)
├── pyproject.toml      # [tool.pytest.ini_options] for the test suite
├── .streamlit/
│   └── config.toml     # Cloud-friendly Streamlit defaults
├── tests/
│   ├── __init__.py
│   ├── conftest.py     # Shared fixtures (works under pytest OR unittest)
│   └── test_agent.py   # 41 tests covering extraction, retrieval, SSRF, persistence, reset path, streaming
└── venv/               # Local virtualenv (not committed)
```

**At runtime, not in the repo:**

- `tempfile.gettempdir()/dataa_analyst_state_<uuid>.db` — per-agent
  SQLite store for session persistence. See [Session
  Persistence](#-session-persistence) above.
- `tempfile.gettempdir()/dataa_analyst_viz_<uuid>/` — per-agent
  directory holding chart PNGs.

[⬆ Back to top](#top)

---

## 🔑 API Key (OpenCode Zen)

The app talks to **[OpenCode Zen](https://opencode.ai/zen)** — a single
endpoint that fronts multiple model providers behind an
OpenAI-compatible chat-completions API. Sign up there, paste a credit
card (the free tier is enough for most use), and grab an API key.

**Resolution order** (the app tries these in sequence):

1. `st.secrets["OPENCODE_API_KEY"]` — used when deployed on Streamlit Cloud
2. `OPENCODE_API_KEY` environment variable — used for local dev via `.env`
3. `TOGETHER_API_KEY` environment variable — legacy fallback from v2.0

### Local development

```sh
echo "OPENCODE_API_KEY=your_key_here" > .env
```

### Streamlit Cloud deployment

In the app dashboard, go to **Settings → Secrets** and paste:

```toml
OPENCODE_API_KEY = "your_key_here"
```

Save. The app will reboot and pick the key up automatically — no
code change needed.

[⬆ Back to top](#top)

### Available AI Models

<details>
<summary>🤖 Click to see all 10 supported models (default is <code>minimax-m3-free</code>)</summary>

Default model is `minimax-m3-free` (free tier). You can swap to any of
these from the in-app **Settings** tab:

- `minimax-m3-free` — default, free tier
- `mimo-v2.5-free` — free tier
- `qwen3.6-plus-free` — free tier
- `deepseek-v4-flash-free` — free tier
- `nemotron-3-ultra-free` — free tier
- `minimax-m2.7` — paid, latest MiniMax
- `minimax-m2.5` — paid, previous MiniMax
- `gpt-5` — paid, via Zen
- `claude-sonnet-4-6` — paid, via Zen
- `gemini-3.1-pro` — paid, via Zen

The full live catalog is at `https://opencode.ai/zen/v1/models`.

</details>

---

## 💾 Session Persistence

Uploads, conversation history, analysis results, BM25 chunk caches,
and chart bytes are persisted to a **SQLite database** under
`tempfile.gettempdir()`. The DB survives Streamlit container
recycles on Cloud but doesn't pollute the repo, and the agent
hydrates from it on init so a fresh container picks up exactly
where the previous one left off.

- **DB path:** `tempfile.gettempdir()/dataa_analyst_state_<uuid>.db`
  — one file per agent instance, UUID-keyed.
- **What's stored:** documents (content + summary), DataFrames
  (parquet blobs), analyses (JSON), conversation history (Q&A),
  BM25 chunk cache, and `(label, png_bytes)` viz pairs.
- **Cleared by:** the in-app **"Clear All Files"** button (which
  calls `agent.clear_caches()`, including the store) and the
  **"Reset Session"** button on the Settings tab.
- **No new dependencies** — `sqlite3` is in the Python stdlib.

If you want a clean slate, hit **"Reset Session"** in the Settings
tab; on the next render the agent will be re-instantiated and a
new DB will be created.

[⬆ Back to top](#top)

---

## 🤖 How It Works (Magic, But With Science)

1. **🏠 Start at Home:** Overview of features and quick start guide
2. **📤 Upload Files:** Drag & drop your documents in the Upload tab
3. **🔄 Auto-Processing:** Text extraction, OCR, data loading - all automatic
4. **📊 Get Analytics:** Instant stats, charts, and insights in the Analytics tab
5. **💬 Chat Away:** Ask questions in the Chat tab - get smart answers
6. **⚙️ Customize:** Tweak settings, change themes, swap AI models
7. **📈 Export Results:** Screenshots, insights, whatever you need

[⬆ Back to top](#top)

---

## 🎯 Why Use This? (Besides My Incredible Ego)

- **Zero Setup Hassle:** API key included, just run and go
- **Beautiful UI:** Dark mode, themes, modern design
- **Actually Smart:** Real AI analysis, not just fancy buttons
- **Multiple File Types:** PDF, Excel, images - it reads everything
- **Conversation Memory:** Ask follow-up questions like a normal human
- **Free to Use:** No hidden costs, no subscription nonsense
- **Regular Updates:** I actually maintain this thing

[⬆ Back to top](#top)

---

## 🔧 Installation Troubleshooting

If you encounter any errors (because software is never perfect):

### Numpy/Pandas Issues:

```bash
pip uninstall numpy pandas -y
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install -r requirements.txt
```

### Streamlit Issues:

```bash
pip install --upgrade streamlit
```

### API Issues:

- Get a key from [OpenCode Zen](https://opencode.ai/zen)
- Add it in the **Settings** tab of the app, in `.env` (local), or in
  Streamlit Cloud → Settings → Secrets (deployment)
- The app will pick it up on next reload

[⬆ Back to top](#top)

---

## 📝 Credits & Thanks

Made with excessive amounts of coffee, determination, and a healthy dose of sarcasm by **Devansh Singh**.

Special thanks to:

- **OpenCode Zen** for their OpenAI-compatible chat API
- **Streamlit** for making beautiful UIs possible
- **Meta** for Llama models that actually work
- **You** for using this instead of doing manual analysis

[⬆ Back to top](#top)

---

## ⚠️ Disclaimer

- This tool is for educational and productivity purposes
- AI responses are smart but not infallible (unlike me)
- No documents were harmed in the making of this agent
- Dark mode may cause addiction to superior UI experiences
- Free API usage is subject to reasonable limits (don't abuse it)

[⬆ Back to top](#top)

---

## 💌 Feedback & Support

- Open an issue on GitHub (I actually read them)
- Email: dksdevansh@gmail.com (for serious stuff)
- Or just scream into the void (therapeutic but less helpful)

[⬆ Back to top](#top)

---

## 🎉 Recent Changes (v3.1)

The v3.0 cutover (June 2026) shipped the OpenCode Zen migration and the
`Agent.py` / `app.py` split. **v3.1** layers a security + correctness
pass on top of that, plus the retrieval upgrade and a real test suite.

<details>
<summary><h3 style="display:inline">Security &amp; correctness</h3></summary>

- **XSS in chat output fixed.** The `unsafe_allow_html=True` blocks
  that interpolated LLM output into HTML (10+ call sites in `app.py`)
  are gone. The chat response now uses `st.chat_message`, which
  sanitizes by default. The only remaining `unsafe_allow_html=True`
  is the CSS-injection at `app.py:217`, which has to be raw HTML for
  theming to work.
- **Path-traversal via uploaded filename fixed.** The original
  `temp_<uploaded_file.name>` sink is gone. Uploads now go to
  `temp_uploads/<uuid>.<safe_ext>` via `app._safe_filename()` (a
  werkzeug-free sanitizer: basename + allowlist regex + UUID fallback).
  All 18 adversarial inputs (path traversal, shell metacharacters,
  Windows backslashes, 300-char overflow) resolve to paths strictly
  inside `temp_uploads/`.
- **SSRF chokepoint added.** `Agent.safe_fetch_url()` is now the only
  path any future URL fetcher should use. Validates scheme
  (http/https), blocks 17 private/loopback/link-local IPv4 + IPv6
  ranges, normalises IPv4-mapped IPv6, and re-checks DNS + redirects
  at every hop to defeat DNS rebinding. A `SECURITY:` comment at the
  `import requests` line points future contributors to it.
- **Extraction errors no longer fed to the LLM.** The four
  `extract_*` helpers now raise instead of returning an error string.
  `process_document` returns `success: bool` + `error: str | None`; on
  failure `content` and `summary` are **empty** and the file is **not**
  added to `document_content` or `data_frames`. The UI short-circuits
  the render + viz pipeline and shows `st.error` with the actual
  message. `load_structured_data` no longer silently returns an empty
  DataFrame on failure (silent data loss bug).
- **Extension allowlist unified.** `Agent._SUPPORTED_EXTENSIONS` is
  the single source of truth. The `st.file_uploader` `type=` list is
  derived from it, so adding an extension once extends both layers.
  Side effect: `tiff` and `bmp` uploads now actually reach the OCR
  extractor (they were silently dropped by the uploader filter before).

</details>

<details>
<summary><h3 style="display:inline">Retrieval &amp; UX</h3></summary>

- **BM25 retrieval replaces blind truncation.** `answer_question` used
  to send only the first `[:1500]` chars per doc and the first
  `[:4000]` of the assembled context to the LLM. A 50-page PDF or a
  10-section CSV left the model blind past the opening pages. Now
  `process_document` chunks text at extraction time
  (paragraph-aware, 800/150), and `answer_question` scores chunks
  with a stdlib BM25-lite ranker and sends the top-4 per doc into a
  12k-char context budget. Pure stdlib — no sentence-transformers,
  no chromadb. A follow-up could swap the ranker for embedding-based
  cosine similarity without changing the chunker or the budget.
- **Visualisations no longer leak into CWD.** The old
  `visualizations_<file_name>/` in the repo root is gone. Charts now
  go to a per-agent UUID-keyed subdir of `tempfile.gettempdir()` and
  are also kept in memory as `(label, png_bytes)` pairs on the agent
  (consumed by the analytics tab on subsequent reruns).
  `clear_visualizations()` is wired to the "Clear All Files" button.

</details>

<details>
<summary><h3 style="display:inline">Test coverage</h3></summary>

- **41 tests** in `tests/test_agent.py` (one new optional dep:
  `httpx`, used lazily for streaming; runs under stdlib `unittest` or
  `pytest`).
- Coverage: extension detection + allowlist, `_safe_filename` for
  path-traversal safety, `process_document` happy + failure paths,
  BM25 retrieval surfaces the right chunk, `safe_fetch_url` blocks
  unsafe schemes + private IP ranges, BM25 ranker correctness,
  SQLite persistence across container recycle, Reset Session
  handler (no mid-iteration `del`, wipes on-disk store), and
  token-by-token streaming (SSE parser, error surfacing, full-text
  persistence).
- No network calls, no heavy-dep imports in the test path. Full
  suite finishes in ~100 ms.

</details>

<details>
<summary><h3 style="display:inline">Project structure</h3></summary>

- `app.py` — Streamlit UI (the entrypoint for `streamlit run`).
- `Agent.py` — engine: extractors, BM25 retriever, OpenCode Zen
  client, `safe_fetch_url` SSRF chokepoint.
- `tests/` — 41 tests + shared fixtures (`conftest.py`).
- `pyproject.toml` — `[tool.pytest.ini_options]` for the test suite.
- `ISSUES.md` — personal-tracked audit; updated as each fix lands.
- `.streamlit/config.toml` — Cloud-friendly defaults (port 8501, headless).

</details>

<details>
<summary><h3 style="display:inline">What Changed (v3.0 → v3.1)</h3></summary>

1. **XSS in chat output** (10+ `unsafe_allow_html` sites) → `st.chat_message`
2. **Path-traversal sink** (`temp_<uploaded_name>`) → UUID-keyed `temp_uploads/`
3. **SSRF TODO** → working `safe_fetch_url()` chokepoint
4. **Extraction errors fed to LLM** → `success: bool` + UI short-circuit
5. **Blind [:4000] truncation** → BM25 retrieval over pre-chunked text
6. **`visualizations_*` dirs in CWD** → in-memory bytes + `tempfile.gettempdir()`
7. **Hardcoded extension list** → `Agent._SUPPORTED_EXTENSIONS` (single source of truth)
8. **No tests** → 41-test suite under `tests/`
9. **In-memory state lost on container recycle** → SQLite store under
   `tempfile.gettempdir()`, hydrated on init, write-through on every
   mutation. No new deps.
10. **`del st.session_state[key]` mid-iteration in Reset Session** →
    `st.session_state.pop(key, None)` plus `agent.clear_caches()` +
    `agent.clear_visualizations()` so the on-disk store is wiped too.
11. **Synchronous HTTP, no token-by-token feedback** → `httpx`
    `AsyncClient.stream` + `st.write_stream` so the chat bubble
    appears in <1 s and tokens arrive in place. The new
    `stream_answer` mirrors `answer_question`'s side effects
    (conversation history, on-disk store) so a streamed answer
    and a non-streamed answer see the same persistence path.

</details>

[⬆ Back to top](#top)

---

## 🧭 Project History

This repo went through a one-time history reset on **2026-06-05**.

Prior to that date, the `main` branch carried 26 commits of v2.0 history
that mixed the agent, the Streamlit UI, the launcher, and ~700 lines of
theme CSS in a single 2,197-line `Agent.py` file, with a real
`TOGETHER_API_KEY` committed to `.env`. The reset replaced that history
with the v3.0 structure described above.

If you're looking at an old clone and `git pull` shows nothing, that's
why — please re-clone.

[⬆ Back to top](#top)

---

## 👤 Maintainer

This project is maintained solely by **[@DevanshSrajput](https://github.com/DevanshSrajput)** (Devansh Singh).

As of **2026-06-05**, collaborator write access for **@aditya-ig10** has
been revoked. The repo is now single-maintainer; PRs from other
contributors are not accepted and pushes from outside the maintainer
account will be force-reverted.

For issues, suggestions, or security reports, contact
`dksdevansh@gmail.com`.

[⬆ Back to top](#top)

---

Enjoy the new and improved document analysis experience! (Or don't, but at least it looks pretty now) 🚀
