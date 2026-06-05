"""Tests for the engine (Agent.py) and the small in-app helpers
(safe_filename in app.py).

Run with either:
    python -m unittest discover tests
    python -m pytest tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

# Make repo root importable when this file is run directly
# (e.g. via `python tests/test_agent.py` from the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("STREAMLIT_RUN", "1")

# Import Agent.py under the test's CWD so process_document's temp-file
# cleanup can find its own files.
os.chdir(_REPO_ROOT)

# Try to import the engine + app helpers. If the optional data stack
# (numpy/pandas/PyPDF2/etc.) isn't installed on this machine, we
# still want the SOURCE-LEVEL tests below (e.g. ResetSessionSourceTests)
# to run — so the import is best-effort and Agent-touching tests are
# guarded with @unittest.skipUnless(AGENT_AVAILABLE, ...).
#
# Agent.py calls sys.exit(1) on ImportError, which raises SystemExit
# (a BaseException subclass, NOT Exception). We pre-check for the
# required modules ourselves so the import block is skipped entirely
# when the env is bare.
Agent = None  # type: ignore[assignment]
_safe_filename = None  # type: ignore[assignment]
AGENT_AVAILABLE = False
try:
    import importlib
    _missing = [m for m in ("numpy", "pandas", "matplotlib", "seaborn")
                if importlib.util.find_spec(m) is None]
    if _missing:
        raise ImportError(f"missing data deps: {_missing}")
    import Agent as _Agent  # noqa: E402
    Agent = _Agent
    from app import _safe_filename as _sf  # noqa: E402
    _safe_filename = _sf
    AGENT_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - dev-env fallback
    pass


def _require_agent():
    """Decorator: skip the test if Agent.py (and its deps) didn't import."""
    import unittest
    return unittest.skipUnless(
        AGENT_AVAILABLE,
        "Agent.py import failed (likely missing numpy/pandas/etc.)",
    )


# ---------------------------------------------------------------------------
# Extension detection (ISSUES.md #1 — extension detection)
# ---------------------------------------------------------------------------

@_require_agent()
class ExtensionDetectionTests(unittest.TestCase):
    """_extension_of and _SUPPORTED_EXTENSIONS form the allowlist for
    every extractor in process_document and for the upload widget."""

    def test_extension_of_basic(self):
        cases = {
            "report.pdf": "pdf",
            "data.CSV": "csv",
            "thing.JPG": "jpg",
            "no.dot.txt": "txt",
            "Report": "",
            "": "",
            "trailing.": "",
            "file.csv.csv": "csv",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(Agent._extension_of(name), expected)

    def test_extension_of_multi_dot(self):
        # Multi-dot names return the LAST extension, which is what
        # users mean. 'archive.tar.gz' is a gz, not a tar.
        self.assertEqual(Agent._extension_of("archive.tar.gz"), "gz")
        self.assertEqual(Agent._extension_of("weird.tar.bz2"), "bz2")

    def test_extension_of_dotfile(self):
        # os.path.splitext treats dotfiles as "name, no extension".
        # That is fine for our purposes: the allowlist rejects it.
        self.assertEqual(Agent._extension_of(".gitignore"), "")

    def test_supported_extensions_holds_known_types(self):
        for ext in ("pdf", "docx", "txt", "csv", "xlsx", "xls",
                    "jpg", "jpeg", "png", "tiff", "bmp"):
            with self.subTest(ext=ext):
                self.assertIn(ext, Agent._SUPPORTED_EXTENSIONS)

    def test_supported_extensions_rejects_unsafe(self):
        for ext in ("gz", "zip", "exe", "tar", "bz2", "md"):
            with self.subTest(ext=ext):
                self.assertNotIn(ext, Agent._SUPPORTED_EXTENSIONS)


# ---------------------------------------------------------------------------
# _safe_filename (ISSUES.md #2 — path traversal)
# ---------------------------------------------------------------------------

@_require_agent()
class SafeFilenameTests(unittest.TestCase):
    """The app-level helper that prevents uploaded filenames from
    escaping temp_uploads/."""

    def test_drops_path_components(self):
        self.assertEqual(_safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(_safe_filename("..\\..\\windows\\system32\\cmd.exe"),
                         "cmd.exe")

    def test_drops_shell_metacharacters(self):
        # No spaces, no ;, no quotes in the result.
        out = _safe_filename("foo; rm -rf bar.pdf")
        self.assertNotIn(" ", out)
        self.assertNotIn(";", out)
        self.assertNotIn("'", out)
        self.assertNotIn('"', out)

    def test_empty_input_returns_uuid(self):
        out = _safe_filename("")
        # UUID hex is 32 chars; result must be non-empty.
        self.assertTrue(out)
        self.assertGreaterEqual(len(out), 16)

    def test_all_dots_returns_uuid(self):
        out = _safe_filename("...")
        self.assertTrue(out)

    def test_normal_name_preserved(self):
        self.assertEqual(_safe_filename("normal_report.pdf"), "normal_report.pdf")
        self.assertEqual(_safe_filename("spaces in name.csv"), "spaces_in_name.csv")


# ---------------------------------------------------------------------------
# process_document (ISSUES.md #1 — extraction errors + ISSUES.md #1
# — extension detection, lossy context, viz persist)
# ---------------------------------------------------------------------------

def _make_agent():
    """A tiny factory: same kwargs as the Streamlit app uses."""
    return Agent.DocumentAnalystAgent(api_key="dummy-test-key", model="test-model")


@_require_agent()
class ProcessDocumentCsvTests(unittest.TestCase):
    """Happy path: real CSV goes in, DataFrame + content + summary come out."""

    def setUp(self):
        import tempfile
        import csv
        self.tmp = tempfile.TemporaryDirectory(prefix="dspd-")
        self.path = os.path.join(self.tmp.name, "people.csv")
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "age", "score"])
            w.writerow(["Alice", 30, 88])
            w.writerow(["Bob", 25, 92])
            w.writerow(["Carol", 35, 75])

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_happy_path(self):
        agent = _make_agent()
        with mock.patch.object(agent, "_make_api_call_with_retry",
                               return_value="[stub summary]") as stub:
            result = agent.process_document(self.path, "people.csv")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(result["file_type"], "csv")
        self.assertIsNotNone(result["data_frame"])
        self.assertEqual(result["data_frame"].shape, (3, 3))
        self.assertIn("Alice", result["content"])
        self.assertEqual(result["summary"], "[stub summary]")
        stub.assert_called_once()

    def test_csv_repopulates_data_frames(self):
        agent = _make_agent()
        with mock.patch.object(agent, "_make_api_call_with_retry",
                               return_value="[stub]"):
            result = agent.process_document(self.path, "people.csv")
        self.assertIn("people.csv", agent.data_frames)
        # Structured data is summarized, not chunked.
        self.assertNotIn("people.csv", agent._chunks)


@_require_agent()
class ProcessDocumentFailureTests(unittest.TestCase):
    """Adversarial inputs that must NOT be silently accepted."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(prefix="dspdf-")

    def tearDown(self):
        self.tmp.cleanup()

    def test_unsupported_extension_rejected(self):
        agent = _make_agent()
        # Write a real PDF-looking file but give it a .zip name.
        path = os.path.join(self.tmp.name, "fake.zip")
        with open(path, "wb") as f:
            f.write(b"%PDF-this-is-not-a-real-pdf")
        result = agent.process_document(path, "fake.zip")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["error"])
        self.assertNotIn("fake.zip", agent.document_content)

    def test_no_extension_rejected(self):
        agent = _make_agent()
        path = os.path.join(self.tmp.name, "Report")
        with open(path, "w") as f:
            f.write("hello")
        result = agent.process_document(path, "Report")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["error"])

    def test_multi_dot_name_rejected(self):
        agent = _make_agent()
        # .tar.gz is not in the allowlist; the file ends in 'gz'.
        path = os.path.join(self.tmp.name, "archive.tar.gz")
        with open(path, "wb") as f:
            f.write(b"not a real tarball")
        result = agent.process_document(path, "archive.tar.gz")
        self.assertFalse(result["success"])
        self.assertIn("gz", result["error"])

    def test_fake_pdf_returns_error_no_llm_call(self):
        agent = _make_agent()
        path = os.path.join(self.tmp.name, "bad.pdf")
        with open(path, "wb") as f:
            f.write(b"%PDF-this-is-not-a-real-pdf")
        with mock.patch.object(agent, "_make_api_call_with_retry") as stub:
            result = agent.process_document(path, "bad.pdf")
        self.assertFalse(result["success"])
        # Critical: the LLM must NOT be called with an error string.
        stub.assert_not_called()
        # The failed file must NOT linger in document_content.
        self.assertNotIn("bad.pdf", agent.document_content)
        self.assertNotIn("bad.pdf", agent._chunks)


# ---------------------------------------------------------------------------
# answer_question + BM25 retrieval (ISSUES.md #1 — lossy context)
# ---------------------------------------------------------------------------

@_require_agent()
class AnswerQuestionRetrievalTests(unittest.TestCase):
    """Pre-populate the agent's caches and confirm answer_question
    (a) surfaces the right chunk, (b) calls the LLM exactly once,
    (c) does not silently regress to [:4000] truncation."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(prefix="dsaq-")
        self.path = os.path.join(self.tmp.name, "planets.txt")
        # Multi-paragraph doc; answer is in paragraph 2.
        body = (
            "Chapter 1 introduction. The solar system consists of the Sun "
            "and the planets that orbit it. Mercury is closest, then Venus, "
            "then Earth, then Mars. Jupiter is the largest planet. "
            + ("Trivia. " * 200)
            + "\n\nChapter 2 secret ingredient. A rare isotope called "
            "unobtanium-238 was discovered on Europa. The isotope has a "
            "half-life of 412 years and forms only under intense Jovian "
            "radiation. "
            + ("Trivia. " * 200)
        )
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(body)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bm25_retrieval_surfaces_answer_chunk(self):
        agent = _make_agent()
        with mock.patch.object(agent, "_make_api_call_with_retry",
                               return_value="[stub]"):
            result = agent.process_document(self.path, "planets.txt")
        self.assertTrue(result["success"])

        # Now ask a question whose answer lives in chapter 2. The prompt
        # the LLM sees should contain the answer string.
        captured = {}

        def fake_api(self, prompt, max_tokens=500, max_retries=3):
            captured["prompt"] = prompt
            return "[stub answer]"

        with mock.patch.object(Agent.DocumentAnalystAgent,
                               "_make_api_call_with_retry", fake_api):
            answer = agent.answer_question(
                "What rare isotope was found on Europa?"
            )
        self.assertEqual(answer, "[stub answer]")
        prompt = captured["prompt"]
        self.assertIn("unobtanium-238", prompt,
                      "BM25 retrieval missed the answer chunk")

    def test_answer_question_calls_api_exactly_once(self):
        agent = _make_agent()
        with mock.patch.object(agent, "_make_api_call_with_retry",
                               return_value="[stub]") as stub:
            agent.process_document(self.path, "planets.txt")
            agent.answer_question("Anything?")
        # generate_document_summary + answer_question = 2 calls total.
        self.assertEqual(stub.call_count, 2)


# ---------------------------------------------------------------------------
# Persistence (ISSUES.md #1 — in-memory state)
# ---------------------------------------------------------------------------

@_require_agent()
class PersistenceTests(unittest.TestCase):
    """Simulate a Streamlit container recycle: agent1 writes, agent2
    reads from the same DB. No network calls."""

    def setUp(self):
        import csv
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(prefix="dspersist-")
        self.db = os.path.join(self.tmp.name, "state.db")
        self.csv_path = os.path.join(self.tmp.name, "people.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "age", "score"])
            w.writerow(["Alice", 30, 88])
            w.writerow(["Bob", 25, 92])
        # Agents created in tests; closed in tearDown so the DB file
        # can be removed on Windows.
        self.agents: list = []

    def tearDown(self):
        for a in self.agents:
            try:
                a.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def _new_agent(self) -> "Agent.DocumentAnalystAgent":
        a = Agent.DocumentAnalystAgent(api_key="k", model="m", db_path=self.db)
        self.agents.append(a)
        return a

    def test_state_survives_container_recycle(self):
        a1 = self._new_agent()
        with mock.patch.object(a1, "_make_api_call_with_retry",
                               return_value="[stub]"):
            r = a1.process_document(self.csv_path, "people.csv")
            self.assertTrue(r["success"])
            a1.perform_data_analysis(a1.data_frames["people.csv"], "people.csv")
            a1.create_visualizations(a1.data_frames["people.csv"], "people.csv")
            a1.answer_question("who scored highest?")

        # New agent on the same DB — should hydrate everything.
        a2 = self._new_agent()
        self.assertIn("people.csv", a2.document_content)
        self.assertIn("people.csv", a2.data_frames)
        self.assertEqual(a2.data_frames["people.csv"].shape, (2, 3))
        self.assertIn("people.csv", a2.analysis_results)
        self.assertIn("people.csv", a2.visualizations)
        self.assertEqual(len(a2.visualizations["people.csv"]), 4)
        self.assertEqual(len(a2.conversation_history), 1)
        self.assertEqual(a2.conversation_history[0]["question"],
                         "who scored highest?")

    def test_failed_extraction_does_not_persist(self):
        a1 = self._new_agent()
        # Write a real CSV but give it an unsupported extension.
        bad_path = os.path.join(self.tmp.name, "fake.zip")
        with open(bad_path, "wb") as f:
            f.write(b"not a real archive")
        r = a1.process_document(bad_path, "fake.zip")
        self.assertFalse(r["success"])
        # Recycle: a new agent on the same DB must NOT see the failed file.
        a2 = self._new_agent()
        self.assertNotIn("fake.zip", a2.document_content)

    def test_clear_caches_wipes_persistent_store(self):
        a1 = self._new_agent()
        with mock.patch.object(a1, "_make_api_call_with_retry",
                               return_value="[stub]"):
            a1.process_document(self.csv_path, "people.csv")
        self.assertIn("people.csv", a1.document_content)
        a1.clear_caches()
        # New agent on the same DB should see nothing.
        a2 = self._new_agent()
        self.assertEqual(a2.document_content, {})
        self.assertEqual(a2.data_frames, {})
        self.assertEqual(a2.conversation_history, [])

    def test_db_file_is_under_tempdir_by_default(self):
        # No db_path -> default location is tempfile.gettempdir().
        a = Agent.DocumentAnalystAgent(api_key="k", model="m")
        try:
            self.assertTrue(a.db_path.startswith(tempfile.gettempdir()))
        finally:
            a.close()
            try:
                os.remove(a.db_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# safe_fetch_url (ISSUES.md #1 — SSRF)
# ---------------------------------------------------------------------------

@_require_agent()
class SafeFetchUrlTests(unittest.TestCase):
    """Validate the policy layer: scheme allowlist + IP blocklist.
    No network calls are made."""

    def test_blocks_file_scheme(self):
        with self.assertRaises(ValueError) as cm:
            Agent.safe_fetch_url("file:///etc/passwd")
        self.assertIn("scheme", str(cm.exception))

    def test_blocks_ftp_scheme(self):
        with self.assertRaises(ValueError):
            Agent.safe_fetch_url("ftp://example.com/")

    def test_blocks_gopher_scheme(self):
        with self.assertRaises(ValueError):
            Agent.safe_fetch_url("gopher://example.com/")

    def test_blocks_data_uri(self):
        with self.assertRaises(ValueError):
            Agent.safe_fetch_url("data:text/html,<script>x</script>")

    def test_blocks_missing_hostname(self):
        with self.assertRaises(ValueError):
            Agent.safe_fetch_url("https://")

    def test_blocked_includes_aws_metadata(self):
        # 169.254.169.254 is the AWS/GCP metadata service. A naive
        # allowlist that only blocks RFC1918 would miss it.
        from ipaddress import ip_address
        self.assertTrue(Agent._is_blocked_ip(ip_address("169.254.169.254")))

    def test_blocked_includes_ipv4_mapped_ipv6_loopback(self):
        from ipaddress import ip_address
        # ::ffff:127.0.0.1 should unwrap to 127.0.0.1 and be blocked.
        self.assertTrue(Agent._is_blocked_ip(ip_address("::ffff:127.0.0.1")))

    def test_public_ips_allowed(self):
        from ipaddress import ip_address
        for ip in ("8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"):
            with self.subTest(ip=ip):
                self.assertFalse(Agent._is_blocked_ip(ip_address(ip)))


# ---------------------------------------------------------------------------
# Helper unit tests for the BM25 ranker (ISSUES.md #1)
# ---------------------------------------------------------------------------

@_require_agent()
class BM25RankerTests(unittest.TestCase):
    """Verify the stdlib ranker surfaces the relevant chunk and ignores noise."""

    def test_query_matches_relevant_doc(self):
        from Agent import _tokenize, _bm25_scores
        docs = [
            _tokenize("The solar system has eight planets orbiting the Sun."),
            _tokenize("A rare isotope called unobtanium was found on Europa."),
            _tokenize("Cookies are baked at 350 degrees for twelve minutes."),
        ]
        scores = _bm25_scores(_tokenize("unobtanium Europa isotope"), docs)
        # The Europa doc should clearly outscore the others.
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_empty_query_returns_zeros(self):
        from Agent import _bm25_scores
        self.assertEqual(
            _bm25_scores([], [["a", "b"], ["c", "d"]]),
            [0.0, 0.0],
        )


# ---------------------------------------------------------------------------
# ISSUES.md #1 (🟡) — "Reset Session" must not mutate session_state
# inside a `for key in st.session_state.keys()` loop. We assert on
# the source structure (no live Streamlit run needed).
# ---------------------------------------------------------------------------

class ResetSessionSourceTests(unittest.TestCase):
    """Static checks on the Reset Session handler in app.py.

    The original code was:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    which raises RuntimeError on Streamlit <1.30 and silently no-ops
    on some intermediate versions. The fix uses .pop(..., None) inside
    a try/except wrapper and also clears the agent's caches.
    """

    def setUp(self):
        import ast
        from pathlib import Path
        self.src = Path("app.py").read_text(encoding="utf-8")
        self.tree = ast.parse(self.src)

    def _reset_handler(self):
        """Return the AST node for the `if st.button("🔄 Reset Session"...):` block."""
        import ast
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Call)
                and getattr(node.test.func, "attr", "") == "button"
            ):
                # Match by the literal string in the first arg.
                if (
                    node.test.args
                    and isinstance(node.test.args[0], ast.Constant)
                    and "Reset Session" in str(node.test.args[0].value)
                ):
                    return node
        self.fail("Could not locate the '🔄 Reset Session' button handler in app.py")

    def test_no_del_inside_reset_loop(self):
        import ast
        handler = self._reset_handler()
        for sub in ast.walk(handler):
            if isinstance(sub, ast.Delete):
                targets = [
                    t.id for t in sub.targets
                    if isinstance(t, ast.Name)
                ]
                self.assertNotIn(
                    "key", targets,
                    "Reset Session must not `del st.session_state[key]`; "
                    "use .pop(key, None) instead.",
                )

    def test_reset_loop_uses_pop(self):
        import ast
        handler = self._reset_handler()
        for sub in ast.walk(handler):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "pop"
            ):
                return  # Found at least one .pop(...) call — good.
        self.fail("Reset Session must call st.session_state.pop(...) at least once")

    def test_reset_clears_agent_caches(self):
        """The handler must wipe the agent's in-memory + on-disk state."""
        import ast
        handler = self._reset_handler()
        called = set()
        for sub in ast.walk(handler):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
            ):
                called.add(sub.func.attr)
        for required in ("clear_caches", "clear_visualizations"):
            self.assertIn(
                required, called,
                f"Reset Session must call agent.{required}() to wipe state",
            )


# ---------------------------------------------------------------------------
# ISSUES.md #1 (🔵) — token-by-token streaming.
#
# We don't hit the network. Instead we install a fake httpx module
# into sys.modules with a minimal AsyncClient that yields SSE lines
# we control. The fake is restored in tearDown so other tests (and
# the import chain) keep using the real httpx if it's installed.
# ---------------------------------------------------------------------------

@_require_agent()
class StreamingTests(unittest.TestCase):
    """Validate _stream_chat_completion + stream_answer end-to-end
    with a fake SSE source. No real network calls."""

    def setUp(self):
        # Skip if httpx isn't installed — streaming is opt-in.
        try:
            import httpx  # noqa: F401
        except ImportError:
            self.skipTest("httpx not installed")

        self._saved_httpx = sys.modules.get("httpx")

    def tearDown(self):
        # Restore whatever was in sys.modules (real httpx or None).
        if self._saved_httpx is None:
            sys.modules.pop("httpx", None)
        else:
            sys.modules["httpx"] = self._saved_httpx

    def _install_fake_httpx(self, sse_lines, status_code=200, body=None):
        """Replace sys.modules["httpx"] with a fake whose AsyncClient
        yields the supplied SSE lines (or a non-2xx response)."""
        import asyncio
        import types

        sse_iter = iter(sse_lines)

        class _FakeStreamResp:
            def __init__(self, status):
                self.status_code = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aiter_lines(self):
                for line in sse_iter:
                    yield line

            async def aread(self):
                # For non-2xx the helper reads the body to extract an
                # error message. Return a JSON-decodable object so the
                # .json() call inside _stream_chat_completion works.
                if body is None:
                    return b'{"error": {"message": "synthetic failure"}}'
                return body

        class _FakeStreamCtx:
            def __init__(self, *a, **kw):
                self._resp = _FakeStreamResp(status_code)

            async def __aenter__(self):
                return self._resp

            async def __aexit__(self, *exc):
                return False

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            def stream(self, *a, **kw):
                return _FakeStreamCtx()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _Timeout:
            def __init__(self, *a, **kw):
                pass

        fake = types.ModuleType("httpx")
        fake.AsyncClient = _FakeClient  # type: ignore[attr-defined]
        fake.Timeout = _Timeout  # type: ignore[attr-defined]
        sys.modules["httpx"] = fake

        # _stream_chat_completion captures httpx at call time via
        # _httpx() — that helper does `import httpx` which will hit
        # sys.modules, so we're good.

    def test_sse_chunks_preserved_and_concatenated(self):
        """A 3-token SSE stream must yield 3 chunks in order and the
        full text must equal their concatenation."""
        sse = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":", "}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            "",
            "data: [DONE]",
            "",
        ]
        self._install_fake_httpx(sse)

        import asyncio
        pieces = []
        async def _collect():
            async for p in Agent._stream_chat_completion(
                api_key="k", model="m", messages=[{"role": "user", "content": "hi"}]
            ):
                pieces.append(p)
        asyncio.run(_collect())
        self.assertEqual(pieces, ["Hello", ", ", "world"])
        self.assertEqual("".join(pieces), "Hello, world")

    def test_done_terminates_stream(self):
        """[DONE] must stop the generator without raising."""
        sse = ['data: {"choices":[{"delta":{"content":"x"}}]}', "", "data: [DONE]"]
        self._install_fake_httpx(sse)
        import asyncio
        pieces = []
        async def _collect():
            async for p in Agent._stream_chat_completion(
                api_key="k", model="m", messages=[]
            ):
                pieces.append(p)
        asyncio.run(_collect())
        self.assertEqual(pieces, ["x"])

    def test_non_2xx_raises_runtime_error(self):
        """A 500 from the backend must surface as RuntimeError, not
        silently yield nothing."""
        self._install_fake_httpx([], status_code=500)
        import asyncio
        async def _collect():
            async for _ in Agent._stream_chat_completion(
                api_key="k", model="m", messages=[]
            ):
                pass
        with self.assertRaises(RuntimeError) as cm:
            asyncio.run(_collect())
        self.assertIn("HTTP 500", str(cm.exception))

    def test_collect_stream_drains_and_joins(self):
        """The sync wrapper used by stream_answer must concatenate
        every piece into a single string."""
        sse = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"choices":[{"delta":{"content":"B"}}]}',
            'data: {"choices":[{"delta":{"content":"C"}}]}',
            "data: [DONE]",
        ]
        self._install_fake_httpx(sse)
        import asyncio
        async def _run():
            return Agent._collect_stream(Agent._stream_chat_completion(
                api_key="k", model="m", messages=[]
            ))
        text = asyncio.run(_run())
        self.assertEqual(text, "ABC")

    def test_stream_answer_persists_full_text(self):
        """stream_answer must record the concatenated answer in
        conversation_history and the on-disk store when the stream
        is fully drained, matching answer_question's side effects."""
        import csv
        with tempfile.TemporaryDirectory(prefix="dsstream-") as tmp:
            db = os.path.join(tmp, "s.db")
            csv_path = os.path.join(tmp, "p.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "score"])
                w.writerow(["Alice", 10])
                w.writerow(["Bob", 20])
            agent = Agent.DocumentAnalystAgent(
                api_key="k", model="m", db_path=db,
            )
            try:
                with mock.patch.object(agent, "_make_api_call_with_retry",
                                       return_value="[stub]"):
                    agent.process_document(csv_path, "p.csv")

                sse = [
                    'data: {"choices":[{"delta":{"content":"Based on "}}]}',
                    'data: {"choices":[{"delta":{"content":"the data, "}}]}',
                    'data: {"choices":[{"delta":{"content":"Alice scored 10."}}]}',
                    "data: [DONE]",
                ]
                self._install_fake_httpx(sse)
                chunks = list(agent.stream_answer("who scored what?"))
                self.assertEqual(
                    "".join(chunks),
                    "Based on the data, Alice scored 10.",
                )
                # Persisted exactly once, with the concatenated text.
                self.assertEqual(len(agent.conversation_history), 1)
                self.assertEqual(
                    agent.conversation_history[0]["answer"],
                    "Based on the data, Alice scored 10.",
                )
                self.assertEqual(
                    agent.conversation_history[0]["question"],
                    "who scored what?",
                )
            finally:
                agent.close()

    def test_stream_answer_surfaces_error(self):
        """A 500 from the backend must yield an `API Error: ...`
        chunk, not silently return empty text."""
        import csv
        with tempfile.TemporaryDirectory(prefix="dsstream-") as tmp:
            db = os.path.join(tmp, "s.db")
            csv_path = os.path.join(tmp, "p.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "score"])
                w.writerow(["Alice", 10])
            agent = Agent.DocumentAnalystAgent(
                api_key="k", model="m", db_path=db,
            )
            try:
                with mock.patch.object(agent, "_make_api_call_with_retry",
                                       return_value="[stub]"):
                    agent.process_document(csv_path, "p.csv")
                self._install_fake_httpx([], status_code=500)
                chunks = list(agent.stream_answer("anything?"))
                self.assertTrue(chunks, "expected at least one chunk")
                self.assertTrue(
                    any("API Error" in c or "HTTP 500" in c for c in chunks),
                    f"expected an error chunk, got {chunks!r}",
                )
            finally:
                agent.close()


# ---------------------------------------------------------------------------
# ISSUES.md #1 (🔵) — create_visualizations must not always render 4
# charts. The new code picks chart types that match what the data
# actually supports, and surfaces column truncation in the label.
# ---------------------------------------------------------------------------

@_require_agent()
class CreateVisualizationsGuardTests(unittest.TestCase):
    """No charts should be drawn for inputs that can't support them,
    and the label should reflect any column truncation."""

    def setUp(self):
        import pandas as pd
        self.pd = pd

    def _viz(self, df, name="data.csv"):
        agent = Agent.DocumentAnalystAgent(api_key="k", model="m")
        try:
            return agent.create_visualizations(df, name)
        finally:
            agent.close()

    def test_empty_df_returns_empty(self):
        df = self.pd.DataFrame()
        out = self._viz(df)
        self.assertEqual(out, [])

    def test_zero_row_df_returns_empty(self):
        df = self.pd.DataFrame({"a": [1, 2, 3]}).iloc[0:0]
        out = self._viz(df)
        self.assertEqual(out, [])

    def test_single_row_skips_histogram_and_box(self):
        # 1 row, 1 numeric col: histogram and box plot would be
        # nonsense (no spread). Heatmap would be 1x1, also skipped.
        df = self.pd.DataFrame({"score": [42]})
        out = self._viz(df)
        labels = [label for label, _ in out]
        self.assertNotIn(
            "Distributions", labels,
            "single-row df must not produce a histogram",
        )
        self.assertNotIn(
            "Box Plots", labels,
            "single-row df must not produce a box plot",
        )
        self.assertNotIn(
            "Correlation Heatmap", labels,
            "single-column df must not produce a correlation heatmap",
        )

    def test_three_row_df_produces_no_box_plot(self):
        # 3 rows: enough for a histogram (1 bin still meaningful as a
        # "all values here" marker) but box-plot quartiles collapse.
        df = self.pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = self._viz(df)
        labels = [label for label, _ in out]
        self.assertIn("Distributions", labels)
        self.assertIn("Correlation Heatmap", labels)
        self.assertNotIn("Box Plots", labels)

    def test_single_numeric_col_skips_heatmap(self):
        df = self.pd.DataFrame({
            "score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        out = self._viz(df)
        labels = [label for label, _ in out]
        self.assertIn("Distributions", labels)
        self.assertIn("Box Plots", labels)
        self.assertNotIn(
            "Correlation Heatmap", labels,
            "single-column heatmap is a 1x1 of 1.00; skip it",
        )

    def test_column_truncation_surfaced_in_label(self):
        # 6 numeric columns, distributions capped at 4. Label must
        # say "showing 4 of 6" so the user knows data was hidden.
        df = self.pd.DataFrame({
            chr(ord("a") + i): list(range(20))
            for i in range(6)
        })
        out = self._viz(df)
        labels = [label for label, _ in out]
        dist_label = next(l for l in labels if l.startswith("Distributions"))
        self.assertIn("showing 4 of 6", dist_label)
        box_label = next(l for l in labels if l.startswith("Box Plots"))
        self.assertIn("showing 3 of 6", box_label)

    def test_high_cardinality_categorical_is_skipped(self):
        # A categorical column with 50 unique values would produce
        # a bar chart with 50 unreadable bars. The new code filters
        # those out before drawing.
        import random
        random.seed(0)
        high_card = [f"item_{i}" for i in range(50)]
        # Repeat to get 100 rows so we have data, not 50 unique rows.
        high_card = high_card * 2
        df = self.pd.DataFrame({
            "label": high_card,
            "score": [float(i) for i in range(len(high_card))],
        })
        out = self._viz(df)
        labels = [label for label, _ in out]
        self.assertNotIn("Categorical Bars", labels)

    def test_categorical_with_few_values_renders_bars(self):
        df = self.pd.DataFrame({
            "team": ["A", "B", "A", "B", "A", "B"] * 5,
            "score": list(range(30)),
        })
        out = self._viz(df)
        labels = [label for label, _ in out]
        self.assertIn("Categorical Bars", labels)


# ---------------------------------------------------------------------------
# ISSUES.md #1 (🔵) — df.to_string() in agent state.
#
# The old code persisted the full to_string() of every CSV/XLSX
# DataFrame into both document_content[file_name]["content"] and
# the SQLite documents table — a 10 MB+ string for a 100k-row CSV
# that the Q&A path never actually read. The new code persists a
# bounded preview (shape + columns + dtypes + first 20 rows). The
# full DataFrame is still in agent.data_frames[file_name].
# ---------------------------------------------------------------------------

@_require_agent()
class DataFramePreviewStorageTests(unittest.TestCase):
    """The CSV/XLSX 'content' field must be a bounded preview, not
    a full to_string() of the DataFrame."""

    def _make_csv(self, tmp, n_rows, n_cols=3):
        import csv
        path = os.path.join(tmp, "big.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"col_{i}" for i in range(n_cols)])
            for r in range(n_rows):
                w.writerow([r * 10 + i for i in range(n_cols)])
        return path

    def test_preview_is_bounded(self):
        import tempfile
        import pandas as pd
        # A 5k-row CSV — to_string() would be ~150 KB. The preview
        # should be a few KB at most.
        with tempfile.TemporaryDirectory(prefix="dsprev-") as tmp:
            path = self._make_csv(tmp, n_rows=5_000, n_cols=3)
            agent = Agent.DocumentAnalystAgent(api_key="k", model="m")
            try:
                df = agent.load_structured_data(path)
                preview = Agent._df_preview_string(df)
                self.assertLess(
                    len(preview), 5_000,
                    f"preview must be bounded; got {len(preview)} chars",
                )
                # Shape + columns + dtypes must be present.
                self.assertIn("(5000, 3)", preview)
                self.assertIn("col_0", preview)
                self.assertIn("Dtypes", preview)
            finally:
                agent.close()

    def test_process_document_does_not_inflate_content(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dsprev-") as tmp:
            path = self._make_csv(tmp, n_rows=10_000, n_cols=4)
            agent = Agent.DocumentAnalystAgent(api_key="k", model="m")
            try:
                with mock.patch.object(agent, "_make_api_call_with_retry",
                                       return_value="[stub]"):
                    result = agent.process_document(path, "big.csv")
                self.assertTrue(result["success"])
                content = agent.document_content["big.csv"]["content"]
                # Old behaviour would have been > 400 KB. New: well under 10 KB.
                self.assertLess(
                    len(content), 10_000,
                    f"document_content['content'] must be bounded; "
                    f"got {len(content)} chars",
                )
                # The full DataFrame is still in data_frames.
                self.assertEqual(
                    agent.data_frames["big.csv"].shape, (10_000, 4),
                )
            finally:
                agent.close()

    def test_preview_includes_shape_columns_dtypes(self):
        import pandas as pd
        df = pd.DataFrame({
            "name": ["Alice", "Bob"] * 5,
            "score": list(range(10)),
            "ratio": [i / 10.0 for i in range(10)],
        })
        preview = Agent._df_preview_string(df)
        self.assertIn("(10, 3)", preview)
        self.assertIn("name", preview)
        self.assertIn("score", preview)
        self.assertIn("ratio", preview)
        self.assertIn("Dtypes", preview)
        # First 20 rows are inside the preview.
        self.assertIn("First 20 rows", preview)

    def test_preview_indicates_extra_columns(self):
        import pandas as pd
        df = pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(20)})
        preview = Agent._df_preview_string(df)
        self.assertIn("Columns (20)", preview)
        # 20 > 10 -> the explicit "+10 more" suffix should appear.
        self.assertIn("+10 more", preview)

    def test_preview_handles_short_dataframe(self):
        """For a df with fewer than _DF_PREVIEW_ROWS rows, we still
        include the head — never silently empty."""
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        preview = Agent._df_preview_string(df)
        self.assertIn("(3, 1)", preview)
        self.assertIn("First 3 rows", preview)
        # The actual row values must appear in the preview.
        self.assertIn("1", preview)
        self.assertIn("2", preview)
        self.assertIn("3", preview)

    def test_preview_does_not_grow_with_row_count(self):
        """100 rows vs 100,000 rows: the preview length must be
        asymptotically constant (the row block is capped at 20)."""
        import pandas as pd
        small = pd.DataFrame({"a": list(range(100))})
        big = pd.DataFrame({"a": list(range(100_000))})
        ps = Agent._df_preview_string(small)
        pb = Agent._df_preview_string(big)
        # Allow a tiny variance for the shape/columns header (~50
        # chars), but the body should be the same size.
        self.assertLess(abs(len(ps) - len(pb)), 200)

    def test_persistence_round_trip_still_works(self):
        """A recycled agent must see the same bounded preview AND
        the full DataFrame — no regression from the persistence
        layer."""
        with tempfile.TemporaryDirectory(prefix="dsprev-") as tmp:
            db = os.path.join(tmp, "state.db")
            path = self._make_csv(tmp, n_rows=2_000, n_cols=2)
            a1 = Agent.DocumentAnalystAgent(api_key="k", model="m", db_path=db)
            try:
                with mock.patch.object(a1, "_make_api_call_with_retry",
                                       return_value="[stub]"):
                    a1.process_document(path, "big.csv")
                # Recycle.
                a2 = Agent.DocumentAnalystAgent(api_key="k", model="m", db_path=db)
                try:
                    self.assertEqual(
                        a2.data_frames["big.csv"].shape, (2_000, 2),
                        "full DataFrame must hydrate from parquet blob",
                    )
                    content = a2.document_content["big.csv"]["content"]
                    self.assertLess(len(content), 10_000)
                    self.assertIn("(2000, 2)", content)
                finally:
                    a2.close()
            finally:
                a1.close()


# ---------------------------------------------------------------------------
# ISSUES.md #1 + #2 (⚪) — split theme + helpers out of app.py and
# verify AVAILABLE_MODELS against OpenCode Zen's catalogue.
#
# These tests run WITHOUT numpy, pandas, or streamlit — they import
# the pure-Python `theme` and `app_helpers` modules directly. That
# also doubles as a sanity check that the new modules don't pull
# heavy-dep imports at module load.
# ---------------------------------------------------------------------------

class ThemeModuleTests(unittest.TestCase):
    """theme.py must export DARK_CSS / LIGHT_CSS strings and a
    `css_for_theme` selector. Both blocks must be non-empty CSS
    and must differ (otherwise the theme switch is a no-op)."""

    def setUp(self):
        from theme import DARK_CSS, LIGHT_CSS, css_for_theme
        self.DARK_CSS = DARK_CSS
        self.LIGHT_CSS = LIGHT_CSS
        self.css_for_theme = css_for_theme

    def test_dark_and_light_differ(self):
        self.assertNotEqual(
            self.DARK_CSS, self.LIGHT_CSS,
            "DARK_CSS and LIGHT_CSS must differ; the theme switch "
            "would be a no-op otherwise",
        )

    def test_both_blocks_contain_root_variables(self):
        for label, css in (("DARK_CSS", self.DARK_CSS),
                           ("LIGHT_CSS", self.LIGHT_CSS)):
            with self.subTest(css=label):
                self.assertIn(":root", css)
                self.assertIn("--bg-primary", css)
                self.assertIn("--text-primary", css)

    def test_css_for_theme_returns_correct_block(self):
        self.assertIs(self.css_for_theme("dark"), self.DARK_CSS)
        self.assertIs(self.css_for_theme("light"), self.LIGHT_CSS)

    def test_css_for_theme_falls_back_to_dark(self):
        """Unknown theme names should not crash; default to dark."""
        self.assertIs(self.css_for_theme("garbage"), self.DARK_CSS)
        self.assertIs(self.css_for_theme(""), self.DARK_CSS)


class AppHelpersModuleTests(unittest.TestCase):
    """app_helpers.py exports _safe_filename, AVAILABLE_MODELS,
    DEFAULT_MODEL_ID, and list_model_choices."""

    def setUp(self):
        from app_helpers import (
            _safe_filename, AVAILABLE_MODELS, DEFAULT_MODEL_ID,
            list_model_choices,
        )
        self._safe_filename = _safe_filename
        self.AVAILABLE_MODELS = AVAILABLE_MODELS
        self.DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
        self.list_model_choices = list_model_choices

    def test_safe_filename_matches_pre_refactor_behaviour(self):
        # Drop path components, replace shell metachars, fall back to UUID.
        self.assertEqual(self._safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(
            self._safe_filename("..\\..\\windows\\system32\\cmd.exe"),
            "cmd.exe",
        )
        out = self._safe_filename("foo; rm -rf bar.pdf")
        self.assertNotIn(" ", out)
        self.assertNotIn(";", out)
        out = self._safe_filename("")
        self.assertTrue(out)
        self.assertGreaterEqual(len(out), 16)
        self.assertEqual(
            self._safe_filename("normal_report.pdf"),
            "normal_report.pdf",
        )
        self.assertEqual(
            self._safe_filename("spaces in name.csv"),
            "spaces_in_name.csv",
        )

    def test_available_models_is_curated(self):
        """The 8-entry fictional list (mimo-v2.5-free, qwen3.6-plus-free,
        deepseek-v4-flash-free, nemotron-3-ultra-free, gemini-3.1-pro,
        gpt-5, claude-sonnet-4-6, minimax-m2.7) was mostly fictional.
        The new catalogue must NOT contain those speculative ids —
        they 404 against OpenCode Zen's live API."""
        banned = {
            "mimo-v2.5-free",
            "qwen3.6-plus-free",
            "deepseek-v4-flash-free",
            "nemotron-3-ultra-free",
            "gemini-3.1-pro",
            "gpt-5",
            "claude-sonnet-4-6",
        }
        for fake_id in banned:
            with self.subTest(model=fake_id):
                self.assertNotIn(
                    fake_id, self.AVAILABLE_MODELS,
                    f"{fake_id} is a speculative id and must not be "
                    f"in the curated catalogue",
                )

    def test_available_models_is_small(self):
        """The catalogue is a UI selector, not a reference doc. Keep
        it small so the picker stays usable."""
        self.assertLessEqual(
            len(self.AVAILABLE_MODELS), 6,
            "curated catalogue is too long; trim to a verified subset",
        )
        self.assertGreaterEqual(
            len(self.AVAILABLE_MODELS), 2,
            "need at least one free and one paid model",
        )

    def test_default_model_id_is_in_catalogue(self):
        """The default id MUST be present in AVAILABLE_MODELS, or
        the selectbox will crash on first render."""
        self.assertIn(self.DEFAULT_MODEL_ID, self.AVAILABLE_MODELS)

    def test_every_model_has_required_fields(self):
        required = {"name", "description", "tier", "performance"}
        for mid, info in self.AVAILABLE_MODELS.items():
            with self.subTest(model=mid):
                missing = required - set(info.keys())
                self.assertFalse(
                    missing,
                    f"model {mid} is missing fields: {missing}",
                )
                self.assertIn(info["tier"], ("Free", "Paid"))

    def test_list_model_choices_orders_free_first(self):
        choices = self.list_model_choices()
        self.assertGreater(len(choices), 0)
        # The first segment must be all Free, then Paid.
        seen_paid = False
        for mid, _name in choices:
            tier = self.AVAILABLE_MODELS[mid]["tier"]
            if tier == "Paid":
                seen_paid = True
            elif seen_paid and tier == "Free":
                self.fail(
                    "list_model_choices must not interleave Free "
                    "and Paid; all Free must come first",
                )

    def test_list_model_choices_ids_match_catalogue(self):
        """Every id returned by the helper must be in AVAILABLE_MODELS,
        and vice versa (no orphans)."""
        ids = [mid for mid, _ in self.list_model_choices()]
        self.assertEqual(set(ids), set(self.AVAILABLE_MODELS.keys()))


@_require_agent()
class AppReexportsTests(unittest.TestCase):
    """The legacy `from app import _safe_filename` and the
    `AVAILABLE_MODELS` / `DEFAULT_MODEL_ID` re-exports must keep
    working so external callers (and the test suite) don't break."""

    def setUp(self):
        # These imports must succeed without numpy/pandas.
        from app import _safe_filename, AVAILABLE_MODELS, DEFAULT_MODEL_ID
        self._safe_filename = _safe_filename
        self.AVAILABLE_MODELS = AVAILABLE_MODELS
        self.DEFAULT_MODEL_ID = DEFAULT_MODEL_ID

    def test_app_safe_filename_delegates_to_helper(self):
        self.assertEqual(
            self._safe_filename("../../etc/passwd"),
            "passwd",
        )
        self.assertEqual(
            self._safe_filename("spaces in name.csv"),
            "spaces_in_name.csv",
        )

    def test_app_available_models_is_same_object_as_helpers(self):
        # Same module-level dict — not a copy. Keeps mutations in
        # sync if anything ever edits the catalogue at runtime.
        from app_helpers import AVAILABLE_MODELS as helpers_dict
        self.assertIs(self.AVAILABLE_MODELS, helpers_dict)


class AppStructureTests(unittest.TestCase):
    """Static checks on app.py: the heavy CSS blocks and the
    AVAILABLE_MODELS dict must no longer live in app.py."""

    def setUp(self):
        from pathlib import Path
        self.src = Path("app.py").read_text(encoding="utf-8")

    def test_app_does_not_define_dark_css(self):
        self.assertNotIn(
            "DARK_CSS = ", self.src,
            "DARK_CSS must live in theme.py, not app.py",
        )
        self.assertNotIn(
            "LIGHT_CSS = ", self.src,
            "LIGHT_CSS must live in theme.py, not app.py",
        )

    def test_app_does_not_define_available_models_inline(self):
        # The old code had `AVAILABLE_MODELS = { ... }` as a literal
        # dict in app.py. After the refactor it must only be imported.
        self.assertNotIn(
            "AVAILABLE_MODELS = {", self.src,
            "AVAILABLE_MODELS must live in app_helpers.py, not app.py",
        )

    def test_app_does_not_define_safe_filename_inline(self):
        self.assertNotIn(
            "def _safe_filename(", self.src,
            "_safe_filename must live in app_helpers.py, not app.py",
        )

    def test_app_imports_new_modules(self):
        self.assertIn("from app_helpers import", self.src)
        self.assertIn("from theme import", self.src)


class SecretsWiringTests(unittest.TestCase):
    """Lock the API-key resolution contract.

    `_get_api_key` MUST consult three sources, in this exact order:

      1. st.secrets["OPENCODE_API_KEY"]    (Streamlit Cloud)
      2. OPENCODE_API_KEY env var          (.env / local dev)
      3. TOGETHER_API_KEY env var          (legacy v2.0 fallback)

    The README and the in-app sidebar both tell users to paste
    `OPENCODE_API_KEY = "..."` into Streamlit Cloud's Secrets
    editor. If the code ever drifts to a different key name
    (e.g. `OPENCODE_ZEN_API_KEY`), a user following the docs
    would set the key, the app would never see it, and the
    first symptom would be a confusing 401. These tests catch
    that class of bug at unit-test time.
    """

    def setUp(self):
        # Save and clear every env var the helper could read.
        # (Always runnable on bare env — does not require Agent.)
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in ("OPENCODE_API_KEY", "TOGETHER_API_KEY")
        }
        # Save whatever was in sys.modules["streamlit"] so we can
        # install a fake during the test and restore afterwards.
        self._saved_streamlit = sys.modules.get("streamlit")

    def tearDown(self):
        # Restore env vars exactly as we found them.
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Restore streamlit (or its absence).
        if self._saved_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = self._saved_streamlit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _install_fake_streamlit(self, secrets_dict):
        """Install a fake `streamlit` module in sys.modules so
        `_get_api_key` thinks it's running under Streamlit and
        reads `st.secrets` from the supplied dict. Mirrors the
        fake-httpx trick used in StreamingTests above."""
        import types

        fake = types.ModuleType("streamlit")
        fake.secrets = secrets_dict  # supports `in` and `[...]`
        sys.modules["streamlit"] = fake
        # Also drop any cached `from streamlit import secrets` in
        # Agent's namespace — but Agent reads via the module
        # attribute, so the sys.modules install is sufficient.

    def _get_api_key(self):
        """Look up the helper via the imported module so the test
        can run on bare env (skipping cleanly via the decorator)."""
        import Agent
        return Agent._get_api_key()

    # ------------------------------------------------------------------
    # 1. Source-level contract: the secrets key name must be
    #    OPENCODE_API_KEY (not OPENCODE_ZEN_API_KEY), matching the
    #    README and the in-app sidebar. Catches the silent drift
    #    bug we just fixed.
    # ------------------------------------------------------------------

    def test_secrets_key_name_is_OPENCODE_API_KEY(self):
        from pathlib import Path
        src = Path("Agent.py").read_text(encoding="utf-8")
        # Must read this exact key from st.secrets.
        self.assertIn('"OPENCODE_API_KEY"', src)
        # Must NOT read the older, drifted name from st.secrets.
        self.assertNotIn('"OPENCODE_ZEN_API_KEY"', src)

    # ------------------------------------------------------------------
    # 2. Behavioral contract: when st.secrets has the key, it's
    #    returned — regardless of the env vars.
    # ------------------------------------------------------------------

    @_require_agent()
    def test_st_secrets_wins_over_env_vars(self):
        self._install_fake_streamlit(
            {"OPENCODE_API_KEY": "from_secrets_abc"}
        )
        os.environ["OPENCODE_API_KEY"] = "from_env_xyz"
        os.environ["TOGETHER_API_KEY"] = "legacy_uvw"

        key = self._get_api_key()
        self.assertEqual(key, "from_secrets_abc")

    # ------------------------------------------------------------------
    # 3. Behavioral contract: with no secrets present, the env var
    #    is returned.
    # ------------------------------------------------------------------

    @_require_agent()
    def test_env_var_used_when_secrets_absent(self):
        # No st.secrets install → helper falls through to env.
        os.environ["OPENCODE_API_KEY"] = "from_env_xyz"
        os.environ["TOGETHER_API_KEY"] = "legacy_uvw"

        key = self._get_api_key()
        self.assertEqual(key, "from_env_xyz")

    # ------------------------------------------------------------------
    # 4. Behavioral contract: the legacy TOGETHER_API_KEY is the
    #    last-resort fallback. Only consulted when both secrets
    #    and OPENCODE_API_KEY are absent.
    # ------------------------------------------------------------------

    @_require_agent()
    def test_legacy_together_key_is_last_resort(self):
        # OPENCODE_API_KEY not set; legacy is.
        os.environ["TOGETHER_API_KEY"] = "legacy_uvw"

        key = self._get_api_key()
        self.assertEqual(key, "legacy_uvw")

    @_require_agent()
    def test_env_var_beats_legacy_together(self):
        os.environ["OPENCODE_API_KEY"] = "from_env_xyz"
        os.environ["TOGETHER_API_KEY"] = "legacy_uvw"

        key = self._get_api_key()
        # The "new" env var must win over the legacy fallback.
        self.assertEqual(key, "from_env_xyz")

    # ------------------------------------------------------------------
    # 5. Behavioral contract: with nothing set, returns None
    #    (caller raises the user-facing "API key is required" error).
    # ------------------------------------------------------------------

    @_require_agent()
    def test_returns_none_when_nothing_set(self):
        key = self._get_api_key()
        self.assertIsNone(key)

    # ------------------------------------------------------------------
    # 6. Behavioral contract: a streamlit runtime that raises
    #    when `st.secrets` is accessed (e.g. a local dev `streamlit
    #    run` with no secrets file) must NOT crash the helper —
    #    it should fall through to the env vars cleanly.
    # ------------------------------------------------------------------

    @_require_agent()
    def test_secrets_access_failure_falls_through(self):
        import types

        class _BoomOnAccess:
            """Acts like st.secrets but raises on any access,
            simulating a fresh `streamlit run` with no
            .streamlit/secrets.toml."""
            def __contains__(self, _):
                raise RuntimeError("no secrets file")
            def __getitem__(self, _):
                raise RuntimeError("no secrets file")

        fake = types.ModuleType("streamlit")
        fake.secrets = _BoomOnAccess()
        sys.modules["streamlit"] = fake

        os.environ["OPENCODE_API_KEY"] = "from_env_xyz"

        key = self._get_api_key()
        # The try/except in _get_api_key must swallow the RuntimeError
        # and fall through to the env var.
        self.assertEqual(key, "from_env_xyz")


class PackagesTxtSchemaTests(unittest.TestCase):
    """Lock the `packages.txt` contract that Streamlit Cloud's
    deploy pipeline depends on.

    Hard-won background (from the v3.1 deploy regression — see
    G16 in ISSUES.md):

    Streamlit Cloud's `packages.txt` parser splits the file on
    EVERY whitespace character (newlines, spaces, tabs) and
    passes each token to `apt-get install`. It does NOT respect
    `#` shell-style comments — every word in a comment becomes
    a "package name" in apt's eyes, and the deploy fails with
    a wall of `E: Unable to locate package <word>` errors
    before the real packages are ever installed.

    The two deploy-blockers we've hit in the field:

      1. CRLF line endings (apt treats \\r as part of the
         package name).
      2. ANY line starting with `#` — even with LF endings,
         the comment text is split word-by-word and passed
         to apt as a list of bogus package names.

    The fix is to keep the file minimal: a list of package
    names, one per line, no comments, no whitespace inside
    any line, no duplicate entries.

    These tests catch all four classes at unit-test time on
    bare env — no Agent, no streamlit, no network.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.path = Path("packages.txt")
        cls.bytes_data = cls.path.read_bytes()
        # Decode as text, but keep the raw bytes for the CRLF check.
        cls.text = cls.bytes_data.decode("utf-8")

    # ------------------------------------------------------------------
    # 1. No comments. Streamlit Cloud's parser splits on every
    #    whitespace and ignores #, so a comment line is a deploy
    #    block. This is the single most important rule.
    # ------------------------------------------------------------------

    def test_no_comment_lines(self):
        for i, raw in enumerate(self.text.splitlines(), 1):
            self.assertFalse(
                raw.lstrip().startswith("#"),
                f"Line {i!r} is a `#` comment: {raw!r}. Streamlit "
                f"Cloud's packages.txt parser splits on whitespace "
                f"and does NOT respect # comments — every word in "
                f"the comment becomes a bogus 'package name' in "
                f"apt's eyes, and the deploy fails with `E: Unable "
                f"to locate package <word>` for every word. The "
                f"file must contain ONLY package names, one per "
                f"line, no comments.",
            )
            # Also forbid inline # — `package # comment` is the
            # same disaster at the word level.
            if raw.strip():
                self.assertNotIn(
                    "#", raw,
                    f"Line {i!r} contains an inline `#`: {raw!r}. "
                    f"Inline `#` is also unsafe; remove it.",
                )

    # ------------------------------------------------------------------
    # 2. Line endings: LF only.
    # ------------------------------------------------------------------

    def test_no_crlf_line_endings(self):
        # Streamlit Cloud's apt parser is line-oriented and treats
        # the \r as part of the package name.
        self.assertNotIn(
            b"\r\n", self.bytes_data,
            "packages.txt has CRLF line endings; Streamlit Cloud's "
            "apt parser will treat \\r as part of the package name "
            "and fail with 'E: Unable to locate package' errors. "
            "Re-save the file with LF endings.",
        )
        # Belt-and-braces: also assert no bare \r anywhere.
        self.assertNotIn(
            b"\r", self.bytes_data,
            "packages.txt contains a bare CR; the entire file "
            "must be LF-only.",
        )

    def test_ends_with_newline(self):
        # Most POSIX tools (including the shell `apt` invokes)
        # expect the final line to end with a newline.
        self.assertTrue(
            self.bytes_data.endswith(b"\n"),
            "packages.txt must end with a newline.",
        )

    # ------------------------------------------------------------------
    # 3. No whitespace in any package line (no spaces, no tabs).
    # ------------------------------------------------------------------

    def test_no_whitespace_in_package_lines(self):
        for i, raw in enumerate(self.text.splitlines(), 1):
            # Every non-blank line is a single token.
            self.assertNotIn(
                " ", raw,
                f"Line {i!r} contains a space: {raw!r}. Streamlit "
                f"Cloud splits the file on every whitespace, so "
                f"spaces inside a line split the package name into "
                f"multiple bogus names.",
            )
            self.assertNotIn(
                "\t", raw,
                f"Line {i!r} contains a tab: {raw!r}. Same reason "
                f"as spaces — every tab becomes a split point.",
            )

    # ------------------------------------------------------------------
    # 4. No blank or whitespace-only package lines.
    # ------------------------------------------------------------------

    def test_no_blank_package_lines(self):
        for i, raw in enumerate(self.text.splitlines(), 1):
            if not raw.strip():
                # A line with only whitespace would be passed to
                # apt as an empty package name.
                self.assertEqual(
                    raw, "",
                    f"Line {i!r} is whitespace-only; blank lines "
                    f"must be empty (no tabs, no spaces).",
                )

    # ------------------------------------------------------------------
    # 5. No duplicate package names.
    # ------------------------------------------------------------------

    def test_no_duplicate_packages(self):
        seen = set()
        for raw in self.text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            self.assertNotIn(
                stripped, seen,
                f"Duplicate package: {stripped!r}. Remove the "
                f"duplicate entry; Streamlit Cloud installs each "
                f"package independently.",
            )
            seen.add(stripped)

    # ------------------------------------------------------------------
    # 6. Sanity: at least one package, and the file is not empty.
    # ------------------------------------------------------------------

    def test_has_at_least_one_package(self):
        pkgs = [
            line.strip() for line in self.text.splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(
            len(pkgs), 1,
            "packages.txt must declare at least one OS-level "
            "package. The app uses pytesseract for OCR, which "
            "requires the tesseract binary on the OS image.",
        )

    # ------------------------------------------------------------------
    # 7. Source-level contract: the file must not have any inline
    #    `apt-get install` or shell-style backticks (we are pinning
    #    a list of names, not a script).
    # ------------------------------------------------------------------

    def test_pure_list_not_script(self):
        for i, raw in enumerate(self.text.splitlines(), 1):
            stripped = raw.strip()
            if not stripped:
                continue
            for forbidden in ("apt-get", "apt ", "$(", "`", "&&", "||", ";"):
                self.assertNotIn(
                    forbidden, stripped,
                    f"Line {i!r} contains shell syntax ({forbidden!r}): "
                    f"{raw!r}. packages.txt is a list of package names, "
                    f"not a shell script.",
                )


if __name__ == "__main__":
    unittest.main()
