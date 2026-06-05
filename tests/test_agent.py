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


if __name__ == "__main__":
    unittest.main()
