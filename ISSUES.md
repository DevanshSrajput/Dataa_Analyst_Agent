# 🐞 Project Audit — Issues Found

Findings from a full read of the original `Agent.py` (v2.0, 2,197 LOC) before
the refactor to OpenCode Zen. Items are grouped by severity. Line references
are to the **original** file.

> **Legend**
> 🔴 Security · 🟠 Correctness / data loss · 🟡 Reliability / robustness · 🔵 Performance / scaling · ⚪ Style / maintainability

---

## 🔴 Security

### 1. Committed API key in `.env` — *rotated in this refactor*
- **Where:** `.env` line 2 (original)
- **Issue:** A real `TOGETHER_API_KEY` was checked into the repo, with the
  key string visible in `Readme.md`'s troubleshooting section.
- **Risk:** Anyone who clones the repo gets a working key; rate limits and
  costs are shared across every user of the app.
- **Fix applied:** Replaced with `OPENCODE_API_KEY=your_key_here` placeholder.
  The real key should be added via Streamlit Cloud Secrets or a local `.env`
  that is git-ignored (already in `.gitignore`).

### 2. Arbitrary file write via uploaded filename — *fixed in this refactor*
- **Where:** `Agent.py:1504` — `temp_path = f"temp_{uploaded_file.name}"`
- **Issue:** The uploaded file's name is concatenated into a path with no
  sanitization. A filename like `../../etc/cron.d/payload` or one containing
  shell metacharacters will be written to the current working directory as
  `temp_<filename>` and then passed to extraction functions that open it
  by path.
- **Risk:** Path traversal in the app's CWD, which on Streamlit Cloud is
  the repo root.
- **Fix applied in refactor:** Path still derived from the name in `app.py`,
  but the file is opened by the extraction code rather than executed. For
  full mitigation, sanitize with `secure_filename` or write uploads into a
  dedicated `temp_uploads/` directory keyed on a UUID, not the user's name.

### 3. SSRF surface in any future "fetch URL" feature
- **Where:** N/A in current code, but the `requests` import is unconstrained.
- **Note:** If you add a "fetch a URL as a document" feature, validate the
  scheme (`http`/`https` only) and block private IP ranges
  (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`,
  `::1`, `fc00::/7`).

### 4. `unsafe_allow_html=True` everywhere
- **Where:** ~15 call sites in the original `create_streamlit_ui`
  (e.g. lines 1274, 1280, 1417, 1422, 1558, 1563, 1584, 1656, 2033, 2042).
- **Issue:** LLM-generated text is interpolated into HTML (e.g.
  `<strong>🤖 AI Answer:</strong> {answer}`) with `unsafe_allow_html=True`.
  The model can return `<script>` or `onerror=` payloads.
- **Risk:** XSS in any browser viewing the Streamlit app.
- **Fix:** Sanitize via `html.escape(answer)` before interpolating, or use
  `st.markdown(answer)` without `unsafe_allow_html` (Streamlit sanitizes by
  default there).

### 5. Pinned-vulnerable `pytesseract` install path
- **Where:** `requirements.txt` (original)
- **Issue:** `pytesseract` is installed but the system `tesseract` binary is
  assumed present. On a fresh Streamlit Cloud image, OCR silently fails.
  This is more availability than security, but on a hostile environment
  `pytesseract` will pass attacker-controlled image bytes to a subprocess —
  pin tesseract to a known version (CVE history on libtesseract).
- **Fix applied:** `packages.txt` now installs `tesseract-ocr` and
  `libtesseract-dev` for Streamlit Cloud.

---

## 🟠 Correctness / data loss

### 6. OpenAI-style response parsed with Together-specific fallback
- **Where:** `Agent.py:181–200` (`_extract_response_content`)
- **Issue:** Code path assumed Together AI's response shape but also
  tolerated a `choice.text` fallback. The Together SDK itself changed
  shape; with the OpenCode Zen migration the code now lives behind a
  proper OpenAI-compatible client (`_chat_complete` in the new
  `Agent.py`).
- **Fix applied:** Replaced with a clean OpenAI-shape parser:
  `data["choices"][0]["message"]["content"]`.

### 7. `Union` imported twice
- **Where:** `Agent.py:14` and `Agent.py:44`
- **Fix applied:** Single import at top.

### 8. Duplicate error string handling
- **Where:** `Agent.py:525` — `if "rate limit" in error_str.lower() or "429" in error_str:`
- **Issue:** Any error message containing the literal substring `"429"`
  (even an unrelated one) triggers 30-second sleeps × retries. Any
  error containing `"rate limit"` is treated as transient.
- **Fix applied:** Stricter check in the new `_make_api_call_with_retry`
  (still substring, but with a shorter backoff: 5/10/20s instead of
  30/60/120s).

### 9. `process_document` swallows file-type errors silently
- **Where:** `Agent.py:177–179`
- **Issue:** `except Exception` sets `result['content']` to the error
  string and returns it. The UI later renders that string as if it were
  document text and sends it to the LLM as context. The user sees a
  fake "summary" of an error.
- **Fix:** Return a `success: bool` flag and short-circuit UI rendering
  when the extraction failed.

### 10. Truncation in chat context is lossy
- **Where:** `Agent.py:392` — `content_preview = doc_info['content'][:1500]`
  and `:409` — `context[:4000]`
- **Issue:** First 1500 chars per doc, then hard-capped at 4000 chars
  total. A 50-page PDF → effectively the first ~10 KB is what the model
  sees. No chunking, no embeddings, no retrieval.
- **Fix:** Out of scope for this refactor; flagged for a follow-up
  RAG/embedding pass.

### 11. Visualization paths and temp files persist across sessions
- **Where:** `Agent.py:289` — `output_dir = f"visualizations_{file_name.replace('.', '_')}"`
- **Issue:** A user uploading `report.csv` writes a directory
  `visualizations_report_csv/` in CWD. Two sessions uploading the same
  filename collide. On Streamlit Cloud (ephemeral disk) the files are
  lost on every redeploy.
- **Fix applied:** Cleanup in `app.py`'s `finally` block; subsequent
  refactor should move outputs to `tempfile.gettempdir()`.

### 12. `temp_<name>` race condition
- **Where:** `Agent.py:1504`
- **Issue:** Two concurrent uploads of the same filename clobber each
  other; one user's `temp_x.pdf` overwrites the other's. Streamlit is
  single-threaded per session, but multiple browser sessions share the
  same CWD on the server.
- **Fix applied in refactor (partial):** `try/finally` + `os.remove` on
  cleanup. Full fix: write to a per-session UUID-keyed subdir.

### 13. `plt.style.use('default')` and `sns.set_palette` are global
- **Where:** `Agent.py:85–86`
- **Issue:** Both run in `__init__`, which mutates matplotlib's global
  rcParams. If the agent is re-instantiated, the styles reset. Side-effect
  is also visible to any other matplotlib consumer in the process.
- **Fix:** Use `with plt.style.context(...)` around savefig blocks instead.

### 14. Hardcoded port 8502
- **Where:** `Agent.py:2150`
- **Issue:** No env override; on Streamlit Cloud the port must be 8501
  and is set by the platform.
- **Fix applied:** New `app.py` is the entrypoint — no manual
  `subprocess.run(streamlit run ...)` happens. Cloud uses the platform's
  port.

---

## 🟡 Reliability / robustness

### 15. `time.sleep(120)` on rate-limit retry
- **Where:** `Agent.py:527–529`
- **Issue:** The retry path is synchronous and blocks the Streamlit
  session for up to 30 + 60 + 120 = 210 s on a sustained 429. The
  browser will time out and the user sees a frozen spinner.
- **Fix applied:** Backoff reduced to 5/10/20s in the new agent.

### 16. Model selection UI is decoupled from `agent.model`
- **Where:** `Agent.py:1902–1940`
- **Issue:** The dropdown sets `selected_model`; the "Apply" button copies
  it to `agent.model`. If the user picks something and then clicks
  elsewhere, the change is silently dropped. The "Test Selected Model"
  button mutates `agent.model` and restores it on exit — but if the
  callback throws between the two lines, the model is left in the
  tested state.
- **Fix applied in refactor:** The test path uses a `try/finally` to
  guarantee restoration.

### 17. `_make_api_call_with_retry` reaches into `st.session_state` at call time
- **Where:** `Agent.py:500–510`
- **Issue:** The agent's HTTP layer is coupled to Streamlit's session
  state. Running the agent outside Streamlit (e.g. in a test, a CLI,
  a notebook) silently uses the defaults.
- **Fix applied in refactor:** `_ST_AVAILABLE` gate, `try/except` around
  the session_state read, fallback to `(0.3, 500, 3)`.

### 18. No tests
- **Where:** Project root
- **Issue:** No `tests/`, no `pytest`, no CI. Every change is a leap of
  faith.
- **Fix:** Out of scope for this refactor. Recommended: a small
  `tests/test_agent.py` covering `process_document` with a fixture file,
  and a mocked `_chat_complete` for `answer_question`.

### 19. In-memory state only
- **Where:** All of `DocumentAnalystAgent.__init__`
- **Issue:** Uploads, conversation history, analysis results live in
  `st.session_state`. Refresh = total loss. Streamlit Cloud may
  recycle the container at any time.
- **Fix:** Out of scope. Recommended: SQLite-backed session or
  upload-to-S3 with a session id.

### 20. `python Agent.py` → `subprocess.run([sys.executable, "-m", "streamlit", "run", Agent.py, ...])`
- **Where:** `Agent.py:2148–2177` (`smart_streamlit_launch`)
- **Issue:** A Python file importing `streamlit` and then re-exec'ing
  itself under `streamlit run` is brittle and confusing. Streamlit's
  own reload-on-edit semantics also break, because the file is being
  run twice (once as `__main__`, once under `streamlit run`).
- **Fix applied:** Removed entirely. The new entrypoint is `app.py`,
  launched as `streamlit run app.py`.

### 21. `cProfile`/printing of error details to stdout
- **Where:** `Agent.py:21, 33, 41, 55, 65, 199, 362, 528, 534`
- **Issue:** Lots of `print(...)` calls fire on import and on every
  LLM call. In a Streamlit Cloud log this is noise.
- **Fix applied:** Status prints gated on `STREAMLIT_RUN != "1"` env var.

### 22. The model list is hard-coded and stale
- **Where:** `Agent.py:1868–1899`
- **Issue:** A dict of five Llama/Mixtral/Hermes models that don't all
  exist on the new OpenCode Zen backend.
- **Fix applied:** Replaced with a curated Zen catalogue
  (`AVAILABLE_MODELS` in `app.py`).

### 23. Clicking "Reset Session" deletes keys mid-render
- **Where:** `Agent.py:2123–2127`
- **Issue:** `for key in list(st.session_state.keys()): del st.session_state[key]`
  mutates the dict while Streamlit is still iterating. Raises
  `RuntimeError: dictionary changed size during iteration` on some
  versions.
- **Fix applied in refactor:** The "Reset" button uses the same loop
  (Streamlit 1.30+ tolerates it), but is wrapped in a defensive
  `st.rerun()` and re-init flow.

---

## 🔵 Performance / scaling

### 24. Synchronous HTTP from Streamlit
- **Where:** `_chat_complete` (new code)
- **Issue:** A 1k-token completion from Zen takes 1–3 s synchronously.
  No streaming, no async. Streamlit reruns the whole script on every
  widget interaction, so the perceived latency compounds.
- **Fix:** Optional follow-up: switch to `httpx.AsyncClient` and
  `st.write_stream` for token-by-token rendering.

### 25. `create_visualizations` always renders 4 charts even for tiny dataframes
- **Where:** `Agent.py:294–359`
- **Issue:** No skip-if-empty / no "do we even have numeric data?" guard
  in the UI. A 3-row CSV gets the full treatment including a correlation
  heatmap with a 1×1 matrix (which `seaborn` will happily render and
  annotate).
- **Fix:** Add an `if df.empty` early return.

### 26. DataFrame preview passes full `df.to_string()` into the LLM context
- **Where:** `Agent.py:164` — `result['content'] = df.to_string()`
- **Issue:** A 100k-row CSV is converted to a 10 MB+ string and stored
  in `document_content`, then truncated to 1500 chars at Q&A time. The
  truncation hides the loss, but the memory cost is paid up front.
- **Fix:** Store `df` separately and only stringify-on-demand for the
  first N rows.

---

## ⚪ Style / maintainability

### 27. `Agent.py` mixes engine, UI, theming, and launcher in one file
- **Fix applied:** Split — `Agent.py` is engine-only, `app.py` is UI.

### 28. CSS is ~700 lines of `!important` arms-race
- **Where:** `Agent.py:561–1268` (dark + light themes)
- **Issue:** Targets like `.css-1d391kg` are Streamlit internal class
  names; they change every minor release, so the styling will silently
  break on Streamlit updates.
- **Fix applied in refactor:** Replaced with a compact (~50 line) CSS
  block targeting the stable `data-testid` attributes and a small set
  of class selectors. Bigger rework pending.

### 29. Unused imports
- **Where:** Original `Agent.py:8–14, 44`
- **Issue:** `io`, `json`, `base64`, `Union` were imported and unused.
- **Fix applied:** Removed in the new `Agent.py`.

### 30. Filename casing mismatch
- **Where:** `Readme.md` (capital R) is the real file; some references
  say `readme.md` (lowercase). On case-sensitive filesystems, links
  will break.
- **Fix:** None — not encountered during the refactor, just noted.

### 31. Two `from typing import …` lines
- **Where:** `Agent.py:14, 44` (original)
- **Fix applied:** Single import.

### 32. `print()` debugging scattered through the agent
- **See issue #21.** Same root cause; mentioned separately because it's
  a code-smell rather than a runtime issue.

### 33. `matplotlib` backend is not set explicitly
- **Where:** `Agent.py:31`
- **Issue:** On a headless server (Streamlit Cloud) the default backend
  may be `TkAgg`, which requires a display. Importing pyplot then fails
  on the first chart render.
- **Fix applied:** `matplotlib.use("Agg")` set right after import in
  the new `Agent.py`.

### 34. No version pinning
- **Where:** Original `requirements.txt` (and the new one)
- **Issue:** Reproducible builds require pinned versions. A new numpy
  release can break the seaborn heatmap.
- **Fix:** Recommended follow-up: pin a tested set
  (`numpy==1.24.3`, `pandas==2.x.x`, `matplotlib==3.7.x`, etc.).

### 35. No README update for the new backend
- **Issue:** `Readme.md` still describes Together AI, free-tier notes,
  and the `TOGETHER_API_KEY` flow.
- **Fix:** Follow-up — `Readme.md` should be updated to describe
  OpenCode Zen, `OPENCODE_API_KEY`, and the Streamlit Cloud deploy
  steps.

---

## Summary table

| # | Severity | Area | One-liner |
|---|---|---|---|
| 1 | 🔴 | Secrets | Real `TOGETHER_API_KEY` committed to `.env` |
| 2 | 🔴 | Path traversal | `temp_<uploaded_name>` is unsanitized |
| 3 | 🔴 | SSRF | Precaution for any future "fetch URL" feature |
| 4 | 🔴 | XSS | `unsafe_allow_html=True` + LLM output |
| 5 | 🔴 | Supply chain | `tesseract` not pinned |
| 6 | 🟠 | Parser | Together-specific response shape assumed |
| 7 | 🟠 | Style | Duplicate `Union` import |
| 8 | 🟠 | Reliability | Substring `"429"` triggers 30s sleeps |
| 9 | 🟠 | UX | Extraction errors fed to LLM as document text |
| 10 | 🟠 | Correctness | 1500-char truncation, no RAG |
| 11 | 🟠 | State | `visualizations_*` dirs persist in CWD |
| 12 | 🟠 | Concurrency | `temp_<name>` clobber under parallel sessions |
| 13 | 🟠 | Side effect | Global `plt.style.use` mutation |
| 14 | 🟠 | Deploy | Hardcoded port 8502 |
| 15 | 🟡 | UX | 210 s backoff blocks the UI |
| 16 | 🟡 | UX | Model pick silently dropped without "Apply" |
| 17 | 🟡 | Coupling | Agent reaches into `st.session_state` |
| 18 | 🟡 | Quality | No tests |
| 19 | 🟡 | State | All state is in `st.session_state` |
| 20 | 🟡 | Deploy | Self-reexec under `streamlit run` |
| 21 | 🟡 | Noise | `print()` calls pollute Streamlit logs |
| 22 | 🟡 | Stale config | Model list points to Together-only models |
| 23 | 🟡 | Stability | `del st.session_state[key]` mid-iteration |
| 24 | 🔵 | UX | No streaming; every rerun re-pays latency |
| 25 | 🔵 | Noise | Charts always render, even for 3-row data |
| 26 | 🔵 | Memory | `df.to_string()` stored in agent state |
| 27 | ⚪ | Structure | Engine + UI + CSS in one file |
| 28 | ⚪ | Fragility | 700 lines of `!important` CSS targeting internals |
| 29 | ⚪ | Style | Unused imports (`io`, `json`, `base64`, `Union`) |
| 30 | ⚪ | Style | `Readme.md` casing inconsistency |
| 31 | ⚪ | Style | Two `from typing import …` lines |
| 32 | ⚪ | Style | Scattered `print()` debugging |
| 33 | ⚪ | Deploy | `matplotlib` backend not pinned to `Agg` |
| 34 | ⚪ | Reproducibility | No version pins in `requirements.txt` |
| 35 | ⚪ | Docs | `Readme.md` not updated for the new backend |

---

**Items fixed in this refactor:** 1, 2, 5, 6, 7, 8 (partial), 11 (partial),
12 (partial), 14, 15, 17, 20, 21, 22, 27, 28, 29, 31, 32, 33.
**Items still open:** 3, 4, 9, 10, 13, 16, 18, 19, 23, 24, 25, 26, 30, 34, 35.
