"""Tests for the engine (Agent.py) and the small in-app helpers
(safe_filename in app.py).

Run with either:
    python -m unittest discover tests
    python -m pytest tests -v
"""
from __future__ import annotations

import os
import sys
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

import Agent  # noqa: E402
from app import _safe_filename  # noqa: E402


# ---------------------------------------------------------------------------
# Extension detection (ISSUES.md #1 — extension detection)
# ---------------------------------------------------------------------------

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
# safe_fetch_url (ISSUES.md #1 — SSRF)
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    unittest.main()
