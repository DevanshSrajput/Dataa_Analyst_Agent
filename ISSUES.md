# 🐞 Project Audit — Open Issues

Findings from a fresh read of the current `Agent.py` and `app.py`. The
original audit (35 items, kept in git history) has been reduced to the
issues that are still open. Resolved items have been removed.

> **Legend**
> 🔴 Security · 🟠 Correctness / data loss · 🟡 Reliability / robustness · 🔵 Performance / scaling · ⚪ Style / maintainability
> 🌱 = **Good first issue** — small, well-scoped, no deep codebase knowledge required.

---

## 🌱 Quick wins (good first issues)

These are deliberately small, have a clear "done" state, and don't require
knowing the whole codebase. Pick any of them as your first contribution.

### G1. Dead variable in upload handler — `app.py:458`
- `safe_name = _safe_filename(uploaded_file.name)` is computed but never
  used. The real path is built from `safe_ext` + a UUID on the next line.
- **Fix:** delete the line, or use it (e.g. include it in a log line).
- **Skill:** reading, small edit.
- **File:** `app.py`, ~line 458.

### G2. `import streamlit` is below a function definition — `app.py:53`
- `import streamlit as st` is sandwiched between the `_safe_filename`
  helper and the `AVAILABLE_MODELS` dict. It works (the function isn't
  called at import time), but it's ugly and trips linters.
- **Fix:** move the import to the top of the file with the other stdlib
  imports.
- **Skill:** refactor, import hygiene.
- **File:** `app.py`, line 53.

### G3. Pin versions in `requirements.txt`
- `numpy`, `matplotlib`, `seaborn`, `PyPDF2`, `python-docx`, `Pillow`,
  `pytesseract`, `requests`, `streamlit`, `python-dotenv` are unpinned.
  `pandas` is pinned to `2.2.3`. Reproducible builds need every line
  pinned.
- **Fix:** pick a tested set (e.g. `numpy==1.26.4`, `matplotlib==3.8.2`,
  `streamlit==1.32.0`, `PyPDF2==3.0.1`, `python-docx==1.1.0`,
  `Pillow==10.2.0`, `pytesseract==0.3.10`, `requests==2.31.0`,
  `python-dotenv==1.0.1`, `seaborn==0.13.2`) and run the test checklist
  in `Verification` below.
- **Skill:** dependency management.
- **File:** `requirements.txt`.

### G4. README hasn't been updated for the OpenCode Zen migration
- `Readme.md` still describes Together AI, the `TOGETHER_API_KEY` flow,
  and the free-tier notes. The codebase now uses OpenCode Zen and
  `OPENCODE_API_KEY`.
- **Fix:** rewrite the setup section to match `app.py:268-285` (the
  in-app help text), and update any screenshots.
- **Skill:** docs.
- **File:** `Readme.md`.

### G5. `Readme.md` vs `readme.md` casing
- The actual file is `Readme.md`. On case-sensitive filesystems (Linux,
  the Streamlit Cloud image), links that say `readme.md` break.
- **Fix:** decide on a casing (recommend `README.md`, the GitHub norm)
  and rename.
- **Skill:** trivial.
- **File:** repo root.

### G6. `plt.style.use('default')` and `sns.set_palette` are global side effects — `Agent.py:194-195`
- Both run in `DocumentAnalystAgent.__init__`, mutating matplotlib's
  global rcParams. Re-instantiation resets the user's style. Other
  matplotlib consumers in the same process are also affected.
- **Fix:** use `with plt.style.context(...)` around the savefig blocks in
  `create_visualizations`, or pass style explicitly per chart.
- **Skill:** matplotlib, scoping.
- **File:** `Agent.py`, `__init__` and `create_visualizations`.

### G7. Rename `Readme.md` to `README.md` and add a `LICENSE` file
- Repo has no `LICENSE`. The README references the author but the legal
  terms aren't stated.
- **Fix:** add an MIT or Apache-2.0 `LICENSE` file with the right year
  and name, and rename `Readme.md` → `README.md` (links in the file
  itself may need updating).
- **Skill:** trivial + legal boilerplate.
- **File:** repo root.

---

## 🟠 Correctness / data loss

### 1. Truncation in chat context is lossy — `Agent.py:466, 483`
- **Where:** `answer_question`: `content_preview = doc_info['content'][:1500]`
  and `context[:4000]`.
- **Issue:** 50-page PDF → only the first ~10 KB reaches the model. No
  chunking, no embeddings, no retrieval.
- **Fix:** out of scope for a first issue, but a great follow-up. RAG
  over a vector store (e.g. `chromadb`) with sentence-transformer
  embeddings, returning top-k chunks per question.
- **Skill:** embeddings, RAG.

### 2. `process_document` extension detection is fragile — `Agent.py:257`
- **Where:** `file_extension = file_name.lower().split('.')[-1]`.
- **Issue:** a file named `archive.tar.gz` → extension `gz` (we don't
  support that). A file with no extension → `''` (silently no-op).
  `report.PDF` works but `Report` (no extension) doesn't.
- **Fix:** use `os.path.splitext`, decide on a fallback policy
  (sniff the magic bytes, or reject unknown extensions up front in
  the upload handler).
- **Skill:** file-type detection.

---

## 🟡 Reliability / robustness

### 3. No tests — repo root
- **Issue:** no `tests/`, no `pytest`, no CI. Every change is a leap of
  faith.
- **Fix:** add a small `tests/test_agent.py` covering `process_document`
  with a fixture file, and a mocked `_chat_complete` for
  `answer_question`. (See G8 for the minimum scaffolding.)
- **Skill:** pytest, mocking.
- **File:** new `tests/` directory.

### 4. Add minimal test scaffolding — new file
- **Fix:** add `pytest.ini` (or `pyproject.toml [tool.pytest.ini_options]`),
  `tests/__init__.py`, `tests/conftest.py` with a small CSV fixture,
  and `tests/test_agent.py` with at least one happy-path test for
  `process_document` and one for `answer_question` (with
  `_chat_complete` monkeypatched).
- **Skill:** pytest.
- **File:** new `tests/`.

### 5. In-memory state only — `Agent.py:188-191`
- **Issue:** uploads, conversation history, analysis results live in
  `st.session_state` and on the agent instance. Refresh = total loss.
  Streamlit Cloud may recycle the container at any time.
- **Fix:** SQLite-backed session or upload-to-S3 with a session id.
- **Skill:** persistence, SQLite/S3.

### 6. "Reset Session" deletes keys mid-render — `app.py:793-797`
- **Where:** the loop `for key in list(st.session_state.keys()): del st.session_state[key]`
  then calls `st.rerun()`. Streamlit 1.30+ tolerates it, but earlier
  versions raise `RuntimeError: dictionary changed size during
  iteration`.
- **Fix:** gate the loop on a Streamlit version check, or just call
  `st.cache_data.clear()` + `st.rerun()` and let the next render
  re-initialise from the defaults in `main()`.
- **Skill:** Streamlit internals.

---

## 🔵 Performance / scaling

### 7. Synchronous HTTP from Streamlit — `Agent.py:144`
- **Issue:** a 1k-token completion takes 1–3 s synchronously. No
  streaming, no async. Streamlit reruns the whole script on every
  widget interaction, so the perceived latency compounds.
- **Fix:** switch to `httpx.AsyncClient` and `st.write_stream` for
  token-by-token rendering of chat responses.
- **Skill:** async, streaming.

### 8. `create_visualizations` always renders 4 charts — `Agent.py:332-415`
- **Issue:** a 3-row CSV gets the full treatment including a
  correlation heatmap with a 1×1 matrix that seaborn happily renders
  and annotates. `numeric_columns` may also exceed 4 — only the first
  4 are shown, silently.
- **Fix:** early return on `df.empty`; cap charts by row count; warn
  when truncating columns.
- **Skill:** matplotlib, defensive UI.

### 9. `df.to_string()` is stored in agent state — `Agent.py:277`
- **Issue:** a 100k-row CSV is converted to a 10 MB+ string and stored
  in `document_content`, then truncated to 1500 chars at Q&A time. The
  truncation hides the loss, but the memory cost is paid up front.
- **Fix:** store the `DataFrame` separately and only stringify-on-demand
  for the first N rows. (Already half-done via `data_frames`, but the
  string is also stored in `document_content[file_name]["content"]`.)
- **Skill:** memory profiling.

---

## ⚪ Style / maintainability

### 10. `app.py` mixes UI, theming, business logic, and helpers
- The 50-line CSS block (`DARK_CSS`, `LIGHT_CSS`) could live in a
  `theme.py` or in `static/`. The `_safe_filename` helper and
  `AVAILABLE_MODELS` dict could move to `app_helpers.py`. `app.py`
  would shrink to pure UI orchestration.
- **Skill:** refactoring, separation of concerns.

### 11. Model catalogue is hard-coded and partially fictional
- `AVAILABLE_MODELS` in `app.py:66-127` lists `mimo-v2.5-free`,
  `qwen3.6-plus-free`, `deepseek-v4-flash-free`, `nemotron-3-ultra-free`,
  `gemini-3.1-pro`, `gpt-5`, `claude-sonnet-4-6`, `minimax-m2.7` —
  check these against OpenCode Zen's actual catalogue and remove the
  ones that 404.
- **Fix:** either replace with a live fetch from a Zen models endpoint,
  or trim the dict to a verified subset.
- **Skill:** API integration, data hygiene.

---

## Summary table

| # | Severity | Area | One-liner |
|---|---|---|---|
| 1 | 🟠 | Correctness | 1500-char truncation, no RAG |
| 2 | 🟠 | Correctness | `split('.')[-1]` extension detection breaks on multi-dot / extensionless files |
| 3 | 🟡 | Quality | No tests |
| 4 | 🟡 | Quality | Add minimal pytest scaffolding |
| 5 | 🟡 | State | All state is in `st.session_state` |
| 6 | 🟡 | Stability | `del st.session_state[key]` mid-iteration |
| 7 | 🔵 | UX | No streaming; every rerun re-pays latency |
| 8 | 🔵 | Noise | Charts always render, even for 3-row data |
| 9 | 🔵 | Memory | `df.to_string()` stored in agent state |
| 10 | ⚪ | Structure | `app.py` still mixes UI + theming + helpers |
| 11 | ⚪ | Data | Model catalogue may include fictional entries |

---

## Good first issues (TL;DR)

🌱 **G1** — remove dead `safe_name` variable (`app.py:458`)
🌱 **G2** — move `import streamlit` to the top of `app.py`
🌱 **G3** — pin versions in `requirements.txt`
🌱 **G4** — update `Readme.md` for OpenCode Zen
🌱 **G5** — rename `Readme.md` → `README.md` (or pick one casing)
🌱 **G6** — scope `plt.style.use` and `sns.set_palette` instead of mutating globals
🌱 **G7** — add a `LICENSE` file
🌱 **G8** — add minimal pytest scaffolding (pair with #7)
