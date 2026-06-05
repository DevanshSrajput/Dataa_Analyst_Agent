# 🐞 Project Audit — Open Issues

Findings from a fresh read of the current `Agent.py` and `app.py`. The
original audit (35 items, kept in git history) has been reduced to the
issues that are still open. Resolved items have been removed.

Open at last edit: 0 items (audit complete as of v3.1;
deploy regression fixed in this round — see G16).

> **Legend**
> 🔴 Security · 🟠 Correctness / data loss · 🟡 Reliability / robustness · 🔵 Performance / scaling · ⚪ Style / maintainability
> 🌱 = **Good first issue** — small, well-scoped, no deep codebase knowledge required.

---

## 🌱 Quick wins (good first issues)

These are deliberately small, have a clear "done" state, and don't require
knowing the whole codebase. Each one is structured as:

- **Title** — what the fix is, in a few words.
- **Context** — why it matters.
- **What to change** — concrete instructions.
- **How to verify** — the test/command/check that proves it's done.
- **Skill** — what you'll learn.
- **Estimated effort** — S = under 15 min, M = under an hour, L = a few hours.

Pick any of them as your first contribution.

---

### G1. Remove dead `safe_name` variable in the upload handler

**Context**
A local variable is computed and then thrown away on the very next
line. It's harmless at runtime (the LLM never sees the dead value)
but it's noise that makes the upload flow harder to read, and a
linter will flag it.

**What to change**
In `app.py` around line 289, in the upload-handler branch:

```python
safe_name = _safe_filename(uploaded_file.name)
```

Either delete the line outright, or use it — e.g. include it in a
log line so the original filename is recoverable from the server
logs even though the on-disk path is UUID-keyed.

**How to verify**
- The line either disappears or appears inside a logging call.
- `python -m unittest discover -s tests -v` still passes.
- Uploading a file in the app still writes a temp file to
  `temp_uploads/<uuid>.<ext>` and clears it in the `finally` block.

**Skill:** reading, small edit.
**Estimated effort:** S.

---

### G2. `import streamlit` placement

**Status:** ✅ done. The import now lives at the top of `app.py`
(line 21), grouped with the other stdlib imports. This was fixed
during the app.py → theme.py + app_helpers.py refactor. Kept here
for numbering so the README's TL;DR list stays in sync.

---

### G3. Pin every line in `requirements.txt`

**Context**
Only `pandas` is pinned (to `2.2.3`). The other ten deps float to
whatever is latest on PyPI at install time. That makes a clean
install today produce a different lockfile than a clean install
six months from now — and a different lockfile than the one
Streamlit Cloud is using right now. Reproducible builds need every
line pinned.

**What to change**
In `requirements.txt`, add `==X.Y.Z` to every unpinned line. A
tested set is:

```
numpy==1.26.4
pandas==2.2.3          # already pinned
matplotlib==3.8.2
seaborn==0.13.2
PyPDF2==3.0.1
python-docx==1.1.0
Pillow==10.2.0
pytesseract==0.3.10
requests==2.31.0
httpx==0.27.0
streamlit==1.32.0
python-dotenv==1.0.1
```

Pick the lowest set that the test suite (`python -m unittest
discover -s tests -v`) still passes under, and document the pin
date in a comment at the top of the file.

**How to verify**
- `grep -E '^[a-zA-Z]' requirements.txt | grep -v '==' | wc -l`
  prints `0` (every line is pinned).
- `pip install --upgrade-strategy=only-if-needed -r requirements.txt`
  succeeds on a clean venv.
- `python -m unittest discover -s tests -v` is green.
- A fresh `pip freeze` matches `requirements.txt` line for line
  (modulo the `pip`/`setuptools`/`wheel` lines pip adds itself).

**Skill:** dependency management.
**Estimated effort:** M (testing the pins takes most of the time).

---

### G4. ~~README still describes the old Together AI backend~~ (done)

**Status:** ✅ done. The bulk of the OpenCode Zen migration landed
in commit `06b0b2a` ("refactor: migrate to OpenCode Zen, split UI
into app.py"). The README now leads with `OPENCODE_API_KEY`,
links to https://opencode.ai/zen, and lists the curated
3-model catalogue. The only two remaining `together` mentions
are intentional: one in the env-var table as a "legacy
fallback from v2.0" note (line 293) and one in a "what
changed" changelog line (line 626). Neither is actionable.

---

### G5. `Readme.md` vs `README.md` casing

**Context**
The file is named `Readme.md` (capital R, lowercase rest). GitHub
redirects both casings to one canonical URL, but other tools don't
— for example, the Streamlit Cloud image is case-sensitive and
will 404 a link that says `readme.md`. The GitHub norm and the
norm most Linux tooling expects is `README.md`.

**What to change**
1. `git mv Readme.md README.md` (this preserves file history).
2. Run `grep -rIn 'Readme.md' .` and update every internal link
   (especially in the tree blocks and the "What Is This" pointer).
3. Confirm the file still renders on GitHub: the repo's homepage
   should show the README content as before, with no broken-image
   or 404 banner.

**How to verify**
- `ls README.md` succeeds; `ls Readme.md` fails.
- `grep -rIn 'Readme.md' .` returns nothing (case-sensitive).
- `git log --follow README.md` shows the full history of the
  renamed file.

**Skill:** trivial, but tests you on `git mv` semantics.
**Estimated effort:** S.

---

### G6. Scope `plt.style.use` and `sns.set_palette` so they're not global

**Context**
`Agent.py:942-943` mutates matplotlib's and seaborn's global
rcParams from inside `create_visualizations`. That means every
other matplotlib consumer in the same Python process inherits
the "default + husl" look — and re-instantiating the agent
silently reverts the user's style. The fix is to scope the
mutation to the chart-rendering block.

**What to change**
In `Agent.py:create_visualizations`, wrap the `_save` calls (or
the per-chart `plt.subplots` / `plt.figure` blocks) with
`plt.style.context(...)`. The two-line setup at lines 942-943
becomes a single `with plt.style.context('default'): sns.set_palette("husl")`
block scoped to the rendering loop, or you can drop the
`set_palette` entirely if the default husl is acceptable.

**How to verify**
- Import matplotlib in a separate script, change the user's
  rcParams to a custom style, then call
  `agent.create_visualizations(df, "f.csv")`. After the call,
  the custom rcParams must still be in effect.
- `python -m unittest discover -s tests -v` is green; specifically
  the `CreateVisualizationsGuardTests` continue to produce the
  expected chart types.

**Skill:** matplotlib, scoping.
**Estimated effort:** S.

---

### G7. Rename `Readme.md` to `README.md`

**Status:** 🟡 duplicate of G5 above. G5 is the canonical version
of this issue; G7 is kept here only so the TL;DR list at the
bottom of the file doesn't shift numbering when G5 lands.

If you pick this up, do the work under G5 and then delete the G7
entry in this file.

---

### G8. ~~Add minimal pytest scaffolding~~ (done)

**Status:** ✅ done. The repo now ships a `tests/` directory with
`conftest.py` (fixtures for pytest and a `make_sample_csv` helper
for unittest), `pyproject.toml` (`[tool.pytest.ini_options]`),
and a 73-test `test_agent.py`. See the test suite's docstring
for the dual-runner instructions.

---

### G9. Derive `DEFAULT_MODEL_ID` from `AVAILABLE_MODELS`

**Context**
`app_helpers.py` declares the default model id in two places:
as a key in the `AVAILABLE_MODELS` dict, and again as the
top-level constant `DEFAULT_MODEL_ID = "minimax-m3-free"`. If
the catalogue ever changes (a model is renamed, a new default
ships), a contributor has to remember to update both. The
existing `AppHelpersModuleTests` already pins the invariant
"`DEFAULT_MODEL_ID` must be in `AVAILABLE_MODELS`" — this
turns the test into a structural guarantee.

**What to change**
In `app_helpers.py`, define `DEFAULT_MODEL_ID` as the first
Free-tier entry of `AVAILABLE_MODELS`:

```python
DEFAULT_MODEL_ID = next(
    mid for mid, info in AVAILABLE_MODELS.items()
    if info.get("tier") == "Free"
)
```

The catalogue declaration should keep its existing order — the
first Free entry is the implicit default — so a one-line
doc-comment is enough to lock the contract.

**How to verify**
- `python -c "from app_helpers import DEFAULT_MODEL_ID, AVAILABLE_MODELS; assert DEFAULT_MODEL_ID in AVAILABLE_MODELS"` exits 0.
- The `AppHelpersModuleTests.test_default_model_id_is_in_catalogue`
  test still passes.
- Renaming a model id in the catalogue (and not touching
  `DEFAULT_MODEL_ID`) still produces a coherent default.

**Skill:** refactor, removing duplication.
**Estimated effort:** S.

---

### G10. Add `CONTRIBUTING.md`

**Context**
There's no `CONTRIBUTING.md` at the repo root. New contributors
have to read the README, the test docstring, and `pyproject.toml`
to figure out: how to run the tests, where to log an issue,
the PR conventions (commit message style, branch naming, "no
ISSUES.md in commits"), and how to add a new model id safely.
The README covers most of it implicitly but not for contributors.

**What to change**
Create `CONTRIBUTING.md` at the repo root with four short
sections:

1. **Setup** — clone, venv, `pip install -r requirements.txt`,
   `python -m unittest discover -s tests -v`.
2. **Tests** — the dual-runner docstring note (pytest + unittest),
   the `STREAMLIT_RUN=1` env var, the numpy/pandas skip behaviour.
3. **PR conventions** — one commit per fix, brief messages,
   `ISSUES.md` is personal tracking and must NOT be committed.
4. **Adding a new model** — the `curl` recipe from
   `app_helpers.AVAILABLE_MODELS` docstring, plus the test
   to update.

**How to verify**
- `head -5 CONTRIBUTING.md` describes setup in one paragraph.
- The doc references the actual commands, not paraphrases.
- A new contributor can clone, install, run tests, and open a
  PR using only this file (no Slack, no asking around).

**Skill:** docs.
**Estimated effort:** S.

---

### G11. Add `.editorconfig`

**Context**
No `.editorconfig` in the repo. A contributor's editor picks
its own defaults for indent, EOL, and final newline — and the
next `git diff` looks like a noise storm because every line
moves. Two contributors on Windows recently committed CRLF
into `requirements.txt` and `Readme.md`; a single config file
would have prevented it.

**What to change**
Create `.editorconfig` at the repo root:

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{md,yaml,yml,json,toml}]
indent_size = 2

[Makefile]
indent_style = tab
```

**How to verify**
- `git config --get core.autocrlf` is whatever the user set
  (we don't override git config), but `git diff` after editing
  a file shows zero `\\ No newline at end of file` warnings.
- Most editors (VS Code, PyCharm, Sublime) auto-detect the
  file and apply the rules without further config.
- The file is ≤ 20 lines and has a short comment explaining
  the `[*]` defaults.

**Skill:** cross-platform hygiene.
**Estimated effort:** S.

---

### G12. Add a `Makefile` for the common commands

**Context**
The README says "run `python -m unittest discover -s tests -v`"
and "run `streamlit run app.py`", but a new contributor has
to copy-paste those. There's no single entry point. A
`Makefile` (or `tasks.py` if you'd rather stay Python-native)
gives the project a `make test`, `make run`, `make lint`,
`make clean` surface that's familiar to anyone who's used
to one.

**What to change**
Add a `Makefile` at the repo root with these targets (and a
short comment header explaining the file isn't required —
`python -m unittest …` still works without it):

```make
.PHONY: install test run lint clean

install:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

run:
	streamlit run app.py

lint:
	python -m py_compile Agent.py app.py app_helpers.py theme.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
```

**How to verify**
- `make test` exits 0 on a clean checkout.
- `make run` starts the app at the URL Streamlit prints.
- `make clean` removes only the listed caches; `git status`
  is clean afterwards.
- The targets are `.PHONY` (no conflict with files named
  `test` or `run`).

**Skill:** build tooling, ergonomics.
**Estimated effort:** S.

---

### G13. Add a smoke test for the `AVAILABLE_MODELS` schema

**Context**
The existing `AppHelpersModuleTests` already asserts the
catalogue has no banned ids, every entry has the required
fields, and `list_model_choices()` orders Free first. A
new contributor can still introduce subtle bugs: duplicate
ids, multiple Free entries marked as the default, a `tier`
value that's neither "Free" nor "Paid", or a `performance`
rating that breaks the existing 3-star scheme. A single
`test_catalogue_is_well_formed` smoke test catches all of
these in one go.

**What to change**
Add one test method to `AppHelpersModuleTests` in
`tests/test_agent.py` that asserts:

- No two entries in `AVAILABLE_MODELS` share a `name` field.
- `tier` is one of `{"Free", "Paid"}` for every entry.
- `performance` matches `r"^⭐+$"` (one or more star emojis).
- `description` is non-empty for every entry.
- Exactly one entry has `tier == "Free"` AND is the
  `DEFAULT_MODEL_ID` (locks the "implicit default" contract
  from G9).

**How to verify**
- The new test passes on the current catalogue.
- Mutating the catalogue to break any rule makes the test
  fail with a precise error message.
- `python -m unittest tests.test_agent.AppHelpersModuleTests -v`
  is green.

**Skill:** schema validation, fixture design.
**Estimated effort:** S.

---

### G14. ~~Add GitHub issue templates~~ (done)

**Status:** ✅ done. Created
`.github/ISSUE_TEMPLATE/bug_report.md` (reproduction
steps, expected vs. actual, environment table including
`pip freeze` + model id + file type, traceback block, and
a "How to verify" pointer at `python -m unittest discover
-s tests -v`) and
`.github/ISSUE_TEMPLATE/feature_request.md` (use case,
proposed API, alternatives, blockers including the
`AVAILABLE_MODELS` dependency note, willingness-to-
contribute). Both use GitHub's standard YAML frontmatter
and are well under 50 lines.

---

### G15. ~~Add `.gitattributes` to lock line endings~~ (done)

**Status:** ✅ done. Created `.gitattributes` at the repo
root with the `* text=auto eol=lf` blanket plus explicit
`text eol=lf` locks for `*.py`, `*.md`, `*.toml`, `*.yml`,
`*.yaml`. The blanket keeps the auto-detect path open
for binary files; the explicit locks prevent a Windows
contributor's editor from re-introducing CRLF on the
next save.

---

### G16. ~~Streamlit Cloud deploy failed: `packages.txt` had CRLF line endings~~ (done)

**Status:** ✅ done. On the first deploy attempt after the
OpenCode Zen migration, Streamlit Cloud's `apt-get install`
rejected `packages.txt` with a wall of `E: Unable to locate
package <word>` errors. Root cause: every line in the file
ended with `\r\n`. Streamlit Cloud's parser splits on `\n`
and treats the trailing `\r` as part of the package name, so
the comment `# Streamlit Cloud uses…` was being passed to
apt as four separate "package names": `Streamlit`, `Cloud`,
`uses`, etc. The real packages (`tesseract-ocr`,
`libtesseract-dev`) never got installed.

**What changed:**

1. Rewrote `packages.txt` with LF-only line endings.
2. Rewrote 9 other source files that had the same CRLF
   drift: `Agent.py`, `app.py`, `Readme.md`,
   `requirements.txt`, `.env`, `.env.example`, `.gitignore`,
   `.streamlit/config.toml`, `.vscode/settings.json`,
   `LICENSE`. (The `.gitattributes` from G15 now locks
   these to LF on future checkouts.)
3. Added `PackagesTxtSchemaTests` (6 tests, runs on bare
   env) to lock the contract: no CRLF, no bare `\r`, every
   non-comment line is a single token (no spaces), no
   duplicate package names, file ends with a newline, and
   no shell-syntax leakage (no `apt-get`, no backticks, no
   `&&`). Plus a sanity check that the file is non-empty.
4. Bonus: fixed a real drift bug discovered while writing
   the secrets-wiring test — the code was reading
   `OPENCODE_ZEN_API_KEY` from `st.secrets`, but the README
   and the in-app sidebar both tell users to set
   `OPENCODE_API_KEY`. The new `SecretsWiringTests` (7
   tests) lock the resolved contract (see G17 below).

**Test count: 73 → 86** (added 7 secrets tests, 6 packages
tests, 1 already-counted elsewhere). All 86 green on
bare env.

---

## 🟠 Correctness / data loss

*(none open)*

---

## 🟡 Reliability / robustness

*(none open — the "Reset Session" mid-iteration bug was fixed in this
cycle by switching the loop to `st.session_state.pop(key, None)`,
wrapping the handler in `try/except`, and adding
`agent.clear_caches()` + `agent.clear_visualizations()` so the
SQLite-backed state is also wiped. See the new
`ResetSessionSourceTests` in `tests/test_agent.py` for the
locks.)*

---

## 🔵 Performance / scaling

*(none open — `df.to_string()` was replaced by a bounded preview in
this cycle. See the new `DataFramePreviewStorageTests` in
`tests/test_agent.py` for the locks.)*

---

## ⚪ Style / maintainability

*(none open — both items were fixed in this cycle. Theme CSS now
lives in `theme.py`; `_safe_filename` and a curated
`AVAILABLE_MODELS` live in `app_helpers.py`. `app.py` is now pure
UI orchestration. See the new `ThemeModuleTests`,
`AppHelpersModuleTests`, `AppReexportsTests`, and
`AppStructureTests` in `tests/test_agent.py` for the locks.)*

---

## Summary table

*(no open issues — audit complete)*

---

## Good first issues (TL;DR)

Each entry below has a full **Title / Context / What to change /
How to verify / Skill / Effort** breakdown in the 🌱 section above.

🌱 **G1** — Remove dead `safe_name` variable in the upload handler (S)
✅ **G2** — Move `import streamlit` to the top of `app.py` (done)
🌱 **G3** — Pin every line in `requirements.txt` (M)
✅ **G4** — ~~Update `Readme.md` for the OpenCode Zen migration~~ (done)
🌱 **G5** — Rename `Readme.md` → `README.md` and update links (S)
🌱 **G6** — Scope `plt.style.use` and `sns.set_palette` (S)
🔁 **G7** — Duplicate of G5 (kept for numbering stability)
✅ **G8** — Add minimal pytest scaffolding (done)
🌱 **G9** — Derive `DEFAULT_MODEL_ID` from `AVAILABLE_MODELS` (S)
🌱 **G10** — Add `CONTRIBUTING.md` (S)
🌱 **G11** — Add `.editorconfig` (S)
🌱 **G12** — Add a `Makefile` for the common commands (S)
🌱 **G13** — Add a smoke test for the `AVAILABLE_MODELS` schema (S)
✅ **G14** — ~~Add GitHub issue templates~~ (done)
✅ **G15** — ~~Add `.gitattributes` to lock line endings~~ (done)
✅ **G16** — ~~Streamlit Cloud deploy failed: `packages.txt` had CRLF line endings~~ (done)
