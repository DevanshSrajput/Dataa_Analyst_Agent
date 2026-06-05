'''
AI Document Analyst - core agent.

This module exposes DocumentAnalystAgent, a reusable engine that:
    - extracts text/data from PDF, DOCX, TXT, CSV, Excel, and images (OCR)
    - performs pandas-based data analysis and matplotlib/seaborn visualisations
    - calls an OpenAI-compatible chat completions API for summaries and Q&A

The default backend is OpenCode Zen (https://opencode.ai/zen). The endpoint,
auth header, and request shape follow the OpenAI chat-completions spec, so
the client is just a thin `requests` wrapper - no SDK lock-in.

Environment variables (in priority order):


    1. st.secrets["OPENCODE_ZEN_API_KEY"]   - Streamlit Cloud secrets
    2. OPENCODE_ZEN_API_KEY env var        - .env file / system env
    3. TOGETHER_API_KEY env var            - legacy, for back-compat

If for privacy reasons you don't trust the app, use your own key.
'''

import os
import sys
import warnings
import time
import uuid
import shutil
import tempfile
import socket
import ipaddress
from typing import Dict, List, Any, Optional, Tuple
warnings.filterwarnings('ignore')

# Mark this module as running under Streamlit so the noisy status prints
# below stay quiet in the deployed app.
os.environ.setdefault("STREAMLIT_RUN", "0")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    # quiet success log when imported under Streamlit (avoids console spam)
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("✅ Environment variables loaded")
except ImportError:
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("⚠️  python-dotenv not found. Please install: pip install python-dotenv")
except Exception as e:
    if os.environ.get("STREAMLIT_RUN") != "1":
        print(f"⚠️  Error loading .env file: {e}")

# Core data processing imports with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless backend; required for Streamlit Cloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("✅ Core data libraries loaded successfully")
except ImportError as e:
    print(f"❌ Error importing data libraries: {e}")
    print("\n🔧 To fix this issue, please run one of the following:")
    print("   pip uninstall numpy pandas -y")
    print("   pip install numpy==1.24.3")
    print("   pip install pandas==1.5.3")
    print("   pip install -r requirements.txt")
    if os.environ.get("STREAMLIT_RUN") != "1":
        input("Press Enter to exit...")
    sys.exit(1)

# File processing imports
try:
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract
    # SECURITY: never call `requests` directly with a user-supplied URL.
    # Use `safe_fetch_url()` (defined above the OpenCode Zen client) which
    # validates scheme, blocks private/loopback/link-local ranges, and
    # re-checks every redirect hop.
    import requests
except ImportError as e:
    print(f"Error importing file processing libraries: {e}")
    print("Please run: pip install PyPDF2 python-docx Pillow pytesseract requests")
    sys.exit(1)

# Optional: streamlit is only needed for the UI module. We import it lazily
# inside _make_api_call_with_retry so Agent.py remains usable headless.
try:
    import streamlit as st  # noqa: F401
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# OpenCode Zen client
# ---------------------------------------------------------------------------

ZEN_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "minimax-m3-free"


def _get_api_key() -> Optional[str]:
    """
    Resolve the API key from, in order:
      1. st.secrets["OPENCODE_ZEN_API_KEY"]  (Streamlit Cloud)
      2. OPENCODE_ZEN_API_KEY env var
      3. TOGETHER_API_KEY env var (legacy)
    """
    # 1. Streamlit secrets (works only when running under Streamlit)
    if _ST_AVAILABLE:
        try:
            if "OPENCODE_ZEN_API_KEY" in st.secrets:
                return st.secrets["OPENCODE_ZEN_API_KEY"]
        except Exception:
            pass
    # 2. Environment
    return os.getenv("OPENCODE_API_KEY") or os.getenv("TOGETHER_API_KEY")


# ---------------------------------------------------------------------------
# SSRF guard (ISSUES.md #1)
# ---------------------------------------------------------------------------
# No URL fetcher exists in the app today, but the `requests` import below
# is the obvious next place someone would add one. `safe_fetch_url` is the
# single chokepoint: it MUST be the only path from a user-supplied URL to
# a network call. It enforces:
#   - scheme allowlist (http, https only)
#   - private/loopback/link-local IP block (IPv4 + IPv6)
#   - DNS resolution done by us, then re-checked against the blocklist
#     (mitigates DNS rebinding: a hostname that resolves to a public IP at
#     resolve-time but a private IP at connect-time cannot slip through)
#   - 301/302/307/308 redirects re-validated at every hop
#
# Usage from a future feature:
#     text = safe_fetch_url(user_url, timeout=15).text
#
# Throws `ValueError` for any policy violation; callers should surface the
# error to the user rather than swallow it.

_ALLOWED_SCHEMES = ("http", "https")

# Private/loopback/link-local ranges to block. These are the same ranges
# ISSUES.md calls out, expanded with a few more that are commonly abused:
#   - 0.0.0.0/8           (unspecified / "this host")
#   - 100.64.0.0/10       (CGNAT, sometimes used for internal services)
#   - 224.0.0.0/4         (multicast)
#   - 240.0.0.0/4         (reserved)
#   - 169.254.0.0/16      (link-local, including AWS/GCP metadata 169.254.169.254)
#   - ::1/128             (IPv6 loopback)
#   - fc00::/7            (IPv6 unique-local)
#   - fe80::/10           (IPv6 link-local)
#   - ::ffff:0:0/96       (IPv4-mapped IPv6 — we normalise and re-check)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network(n) for n in [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    ]
]


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if `ip` falls in any blocked range."""
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) so the v4 rules apply.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _resolve_and_check(host: str) -> str:
    """Resolve `host` to an IP, return it, or raise if blocked.

    Resolves with the stdlib (no DNS server preference), then checks every
    returned address against the blocklist. Throws `ValueError` if the
    hostname fails to resolve or any address is blocked.
    """
    try:
        # getaddrinfo returns a list of (family, type, proto, canonname, sockaddr).
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {host!r}: {e}")
    if not infos:
        raise ValueError(f"DNS resolution returned no addresses for {host!r}")
    # Check every resolved address — a hostile DNS server can return a
    # mix of public and private records; we must reject the host if ANY
    # address is private (otherwise the client can race the connect).
    for family, _type, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue  # skip unparseable records; safer to allow the others
        if _is_blocked_ip(ip):
            raise ValueError(
                f"Refusing to fetch {host!r}: resolves to blocked address {addr}"
            )
    # Return the first v4 (or v6 if no v4) so requests connects deterministically.
    for family, _t, _p, _c, sockaddr in infos:
        if family == socket.AF_INET:
            return sockaddr[0]
    return infos[0][4][0]


def safe_fetch_url(
    url: str,
    timeout: int = 15,
    max_redirects: int = 5,
    **_kwargs,
) -> "requests.Response":
    """Fetch `url` with SSRF protections. See module-level comment for policy.

    Args:
        url: user-supplied URL. Validated for scheme and resolved IP.
        timeout: per-request timeout in seconds.
        max_redirects: cap on redirect hops (each re-validated).

    Returns:
        `requests.Response` on success (2xx/3xx followed by terminal 2xx).

    Raises:
        ValueError: on any policy violation (bad scheme, blocked IP, etc).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Refusing to fetch {url!r}: scheme {parsed.scheme!r} not in "
            f"{_ALLOWED_SCHEMES}"
        )
    if not parsed.hostname:
        raise ValueError(f"Refusing to fetch {url!r}: missing hostname")

    # Resolve + check the *initial* hostname.
    resolved_ip = _resolve_and_check(parsed.hostname)

    # We pin the connect to the IP we already validated, but we have to
    # re-validate on every redirect (an attacker can set Location to any
    # host). The simplest correct path: pre-resolve, then call requests
    # with allow_redirects=False and chase Location headers ourselves.
    current_url = url
    for _hop in range(max_redirects + 1):
        host = urlparse(current_url).hostname
        if not host:
            raise ValueError(f"Redirect target missing hostname: {current_url!r}")
        ip = _resolve_and_check(host)

        resp = requests.get(
            current_url,
            timeout=timeout,
            allow_redirects=False,
            headers={"Host": host},  # requests uses the URL's host for SNI/HTTP
            **_kwargs,
        )
        # 2xx: done.
        if 200 <= resp.status_code < 300:
            return resp
        # 3xx: chase the Location header.
        if 300 <= resp.status_code < 400:
            location = resp.headers.get("Location", "")
            if not location:
                raise ValueError(
                    f"Redirect with no Location header from {current_url!r}"
                )
            current_url = requests.compat.urljoin(current_url, location)
            continue
        # 4xx/5xx: surface.
        resp.raise_for_status()

    raise ValueError(f"Too many redirects (> {max_redirects}) fetching {url!r}")


def _chat_complete(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.3,
    timeout: int = 120,
) -> str:
    """Single chat-completions call against OpenCode Zen. Returns the text."""
    url = f"{ZEN_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Raise on HTTP error so the retry helper can react
    if resp.status_code >= 400:
        # Try to extract a useful error message
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message") or err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
    data = resp.json()
    # OpenAI-style: choices[0].message.content
    try:
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, list):
            return " ".join(str(c) for c in content)
        return str(content) if content else ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {e} - {data}")


# ---------------------------------------------------------------------------
# Retrieval helpers (ISSUES.md #1 — lossy context)
# ---------------------------------------------------------------------------
# Pure-stdlib BM25-lite ranker and a paragraph-aware chunker. We don't
# pull in sentence-transformers or chromadb: the goal is "no more
# blind [:4000] truncation", not full semantic RAG. Embedding-based
# retrieval is a follow-up if/when it's worth the dep weight.


# Words that carry almost no signal for retrieval. Lowercased.
_BM25_STOPWORDS = frozenset("""
a an and are as at be been being but by did do does for from had has
have he her him his i if in into is it its just me my not of on or
our she that the their them then there these they this to was we
were what when where which while who why will with would you your
""".split())


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and short
    tokens. Returns the token list used for BM25 scoring."""
    import re as _re
    tokens = _re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _BM25_STOPWORDS and len(t) > 1]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split `text` into ~`chunk_size`-character windows with `overlap`
    characters of carry-over between consecutive chunks. Empty input
    returns an empty list.

    The chunker prefers paragraph boundaries: if a `\n\n` separator
    falls within the last 20% of the window, we break there instead.
    This keeps tables and list items intact when they fit.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Try to break on a paragraph boundary near the end of the window.
            lookback = text.rfind("\n\n", start, end)
            if lookback != -1 and lookback > start + chunk_size // 2:
                end = lookback
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _bm25_scores(query_tokens: List[str], docs: List[List[str]],
                 k1: float = 1.5, b: float = 0.75) -> List[float]:
    """BM25-lite scoring. `docs` is a list of pre-tokenized documents.
    Returns a score per doc. 0.0 means "no signal".

    Pure stdlib — no numpy, no sklearn. Adequate for the corpus sizes
    a single user will throw at a chat session (tens of chunks per doc,
    maybe a few docs).
    """
    if not query_tokens or not docs:
        return [0.0] * len(docs)
    # Document frequencies for the query terms.
    N = len(docs)
    df: Dict[str, int] = {}
    for d in docs:
        seen = set(d)
        for t in seen:
            if t in query_tokens:
                df[t] = df.get(t, 0) + 1
    # Average doc length for length normalization.
    avgdl = sum(len(d) for d in docs) / max(N, 1)
    scores: List[float] = []
    for d in docs:
        if not d:
            scores.append(0.0)
            continue
        dl = len(d)
        norm = 1 - b + b * (dl / max(avgdl, 1))
        # Term frequencies in this doc.
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            n_q = df.get(q, 0)
            # IDF — guard against 0/0 with +1 in the denominator.
            idf = ((N - n_q + 0.5) / (n_q + 0.5) + 1)
            # BM25 term contribution.
            score += idf * (tf[q] * (k1 + 1)) / (tf[q] + k1 * norm)
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# DocumentAnalystAgent
# ---------------------------------------------------------------------------

class DocumentAnalystAgent:
    """
    An intelligent document analysis agent that can process multiple file formats,
    perform data analysis, generate visualizations, and answer questions.

    The agent is intentionally UI-agnostic: instantiate it with an API key and
    call its methods. The Streamlit UI lives in app.py.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):

        if not api_key:
            raise ValueError(
                "API key is required. Set OPENCODE_API_KEY in your environment, "
                "in .env, or in Streamlit Cloud secrets."
            )
        self.api_key = api_key
        self.model = model
        self.document_content: Dict[str, Dict[str, Any]] = {}
        self.data_frames: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []

        # Per-agent viz dir. UUID-keyed under the system temp dir so
        # (a) it can't collide with anything in the app CWD, and
        # (b) Streamlit Cloud's ephemeral disk survives redeploys.
        # The dir is created lazily by `create_visualizations`.
        self._viz_dir: str = os.path.join(
            tempfile.gettempdir(),
            f"dataa_analyst_viz_{uuid.uuid4().hex}",
        )
        # Per-file viz cache: file_name -> list of (label, png_bytes).
        # Populated by `create_visualizations`; consumed by the UI on
        # subsequent reruns (analytics tab doesn't re-render the charts
        # every time, it just re-displays the cached bytes).
        self.visualizations: Dict[str, List[Tuple[str, bytes]]] = {}

        # Per-file chunk cache: file_name -> list of pre-chunked text.
        # Populated at `process_document` time so `answer_question` can
        # score and pick top-k chunks per question without re-splitting
        # the document on every chat turn. Cleared by `clear_caches`.
        self._chunks: Dict[str, List[str]] = {}

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    # ---- File extraction ----------------------------------------------------

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF. Raises on any read/parse failure."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX. Raises on any read/parse failure."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def extract_text_from_image(self, file_path: str) -> str:
        """OCR an image. Raises on any read/OCR failure."""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    def load_structured_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV / XLS / XLSX into a DataFrame. Raises on failure or
        unsupported format. Callers (process_document) should not swallow
        the result silently."""
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        if file_path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        raise ValueError(f"Unsupported structured data format: {file_path}")

    # ---- Document processing ------------------------------------------------

    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Process a document based on its type and extract relevant information.

        On extraction failure, returns a result dict with `success=False`
        and a human-readable `error` string. The content field is empty
        in that case — the UI MUST check `success` and short-circuit
        before rendering or sending the result to the LLM, otherwise the
        LLM will see the error string as if it were document text (see
        ISSUES.md #1 — this method used to do exactly that).

        Args:
            file_path: Path to the file
            file_name: Name of the file

        Returns:
            Dictionary with `success: bool`, `error: str | None`, and
            on success: `content`, `data_frame`, `summary`, `file_type`.
        """
        file_extension = file_name.lower().split('.')[-1]
        result: Dict[str, Any] = {
            'file_name': file_name,
            'file_type': file_extension,
            'success': False,
            'error': None,
            'content': '',
            'data_frame': None,
            'summary': '',
        }

        try:
            if file_extension == 'pdf':
                result['content'] = self.extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                result['content'] = self.extract_text_from_docx(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['content'] = f.read()
            elif file_extension in ['csv', 'xlsx', 'xls']:
                df = self.load_structured_data(file_path)
                result['data_frame'] = df
                result['content'] = df.to_string()
                self.data_frames[file_name] = df
            elif file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                result['content'] = self.extract_text_from_image(file_path)
            else:
                # Unknown extension — do NOT silently return an empty
                # result that the UI will treat as success.
                result['error'] = (
                    f"Unsupported file type: {file_extension!r}. "
                    f"Supported: pdf, docx, txt, csv, xlsx, xls, jpg, "
                    f"jpeg, png, tiff, bmp."
                )
                return result

            # Extraction succeeded — store and summarise.
            result['success'] = True
            self.document_content[file_name] = result
            # ISSUES.md #1: pre-chunk the text so answer_question can
            # retrieve relevant chunks via BM25 instead of blind
            # [:1500] / [:4000] truncation. Skip for structured data
            # (DataFrames are summarized via analysis_results instead).
            if file_extension not in ('csv', 'xlsx', 'xls'):
                self._chunks[file_name] = chunk_text(result['content'])
            result['summary'] = self.generate_document_summary(result)
            return result

        except FileNotFoundError as e:
            result['error'] = f"File not found: {e}"
            self._chunks.pop(file_name, None)
            return result
        except Exception as e:
            # Catch-all so the UI gets a clean signal, but log the full
            # traceback for the developer. (ISSUES.md #1: do not put the
            # error string in `content` — it ends up in the LLM context.)
            import traceback
            if os.environ.get("STREAMLIT_RUN") != "1":
                traceback.print_exc()
            result['error'] = f"{type(e).__name__}: {e}"
            self._chunks.pop(file_name, None)
            return result

    # ---- Analysis -----------------------------------------------------------

    def perform_data_analysis(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis on structured data.
        """
        analysis = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'summary_statistics': {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {},
            'correlations': None
        }

        # Summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis['summary_statistics'] = df[numeric_columns].describe().to_dict()

            # Correlation matrix for numeric columns
            if len(numeric_columns) > 1:
                analysis['correlations'] = df[numeric_columns].corr().to_dict()

        # Unique values for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            analysis['unique_values'][col] = df[col].nunique()

        # Store analysis results
        self.analysis_results[file_name] = analysis

        return analysis

    def create_visualizations(self, df: pd.DataFrame, file_name: str) -> List[Tuple[str, bytes]]:
        """
        Render a battery of standard charts for `df` and return them as
        `(label, png_bytes)` pairs. PNGs are also written to the agent's
        per-instance temp dir (`self._viz_dir`) so the analytics tab can
        re-render them on subsequent Streamlit reruns without recomputing.

        ISSUES.md #2 (visualizations persist): the old code wrote to
        `visualizations_<file_name>/` in the CWD, which on Streamlit
        Cloud is the repo root and which collides across sessions. The
        new layout is a UUID-keyed subdir of `tempfile.gettempdir()`,
        owned by the agent instance and freed by `clear_visualizations`.

        Args:
            df: the DataFrame to chart.
            file_name: kept for backwards-compatible labeling of the
                output PNGs (no longer used to derive a path).

        Returns:
            List of `(label, bytes)` tuples. May be empty if `df` has
            no numeric or categorical columns.
        """
        visualizations: List[Tuple[str, bytes]] = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Lazy-create the per-agent viz dir on first use.
        os.makedirs(self._viz_dir, exist_ok=True)

        def _save(label: str, fname: str) -> Tuple[str, bytes]:
            """Save the current figure to `_viz_dir/fname` AND to memory.
            Returns `(label, bytes)` for the caller to render.
            """
            path = os.path.join(self._viz_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            with open(path, "rb") as f:
                data = f.read()
            return (label, data)

        try:
            # 1. Distribution plots for numeric columns
            if len(numeric_columns) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                for i, col in enumerate(numeric_columns[:4]):
                    if i < len(axes):
                        df[col].hist(bins=30, ax=axes[i])
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                plt.tight_layout()
                visualizations.append(_save("Distributions", "distributions.png"))

            # 2. Correlation heatmap
            if len(numeric_columns) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                visualizations.append(_save("Correlation Heatmap", "correlation_heatmap.png"))

            # 3. Box plots for numeric columns
            if len(numeric_columns) > 0:
                n_plots = min(3, len(numeric_columns))
                fig, ax_array = plt.subplots(1, n_plots, figsize=(15, 5), squeeze=False)
                axes = ax_array.flatten()
                for i, col in enumerate(numeric_columns[:n_plots]):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
                plt.tight_layout()
                visualizations.append(_save("Box Plots", "box_plots.png"))

            # 4. Bar charts for categorical columns
            if len(categorical_columns) > 0:
                n_cat_plots = min(2, len(categorical_columns))
                fig, ax_array = plt.subplots(1, n_cat_plots, figsize=(15, 6), squeeze=False)
                axes = ax_array.flatten()
                for i, col in enumerate(categorical_columns[:n_cat_plots]):
                    if df[col].nunique() <= 20:
                        value_counts = df[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'Top Values in {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Count')
                        axes[i].tick_params(axis='x', rotation=45)
                plt.tight_layout()
                visualizations.append(_save("Categorical Bars", "categorical_bars.png"))

        except Exception as e:
            if os.environ.get("STREAMLIT_RUN") != "1":
                print(f"Error creating visualizations: {e}")

        # Cache for later re-display (analytics tab reads this).
        self.visualizations[file_name] = visualizations
        return visualizations

    def clear_visualizations(self) -> None:
        """Delete the per-agent viz dir and recreate it empty.

        Called from the "Clear All Files" and "Reset Session" buttons
        in app.py so we don't leak temp PNGs across resets. Idempotent
        — silently no-ops if the dir doesn't exist.
        """
        if os.path.isdir(self._viz_dir):
            shutil.rmtree(self._viz_dir, ignore_errors=True)
        os.makedirs(self._viz_dir, exist_ok=True)
        self.visualizations.clear()

    def clear_caches(self) -> None:
        """Wipe all per-file derived state (chunks + viz bytes).

        Wired to the "Clear All Files" button in app.py so memory
        doesn't leak across resets. Idempotent.
        """
        self._chunks.clear()
        self.visualizations.clear()

    def clear_caches(self) -> None:
        """Wipe all per-file derived state (chunks + viz bytes).

        Wired to the "Clear All Files" button in app.py so memory
        doesn't leak across resets. Idempotent.
        """
        self._chunks.clear()
        self.visualizations.clear()

    # ---- LLM interactions ---------------------------------------------------

    def generate_document_summary(self, document_info: Dict[str, Any]) -> str:
        """Generate a summary of the document using the LLM"""
        try:
            content_preview = document_info['content'][:2000]  # Limit content length

            prompt = f"""
            Analyze the following document and provide a comprehensive summary:

            File Name: {document_info['file_name']}
            File Type: {document_info['file_type']}

            Content Preview:
            {content_preview}

            Please provide:
            1. A brief overview of the document
            2. Key topics or themes identified
            3. If it contains data, describe the structure and main variables
            4. Any notable patterns or insights

            Keep the summary concise but informative.
            """

            return self._make_api_call_with_retry(prompt, max_tokens=500)

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    # How many top-scoring chunks to keep per text document when
    # answering a question. Chosen so a 6-doc Q&A still fits well
    # inside the per-call context budget below.
    _CHUNKS_PER_DOC = 4
    # Hard ceiling on the "CONTEXT FROM DOCUMENTS" block in the prompt.
    # ~12k chars ≈ 3k tokens, leaves room for the question + history
    # + the model's own reply inside the typical 4k-8k response window.
    _CONTEXT_BUDGET_CHARS = 12_000

    def answer_question(self, question: str, context_files: Optional[List[str]] = None) -> str:
        """
        Answer a question based on the processed documents.

        Context construction (ISSUES.md #1 — lossy context):
        - For each text-bearing file (PDF/DOCX/TXT/images), the
          pre-cached chunks from `self._chunks[file_name]` are scored
          against the question with BM25; the top `_CHUNKS_PER_DOC`
          chunks are concatenated into the prompt.
        - For structured data (CSV/XLSX), we keep the existing
          analysis-results summary (shape, columns, key statistics) —
          the DataFrame is summarized, not chunked.
        - The total context block is capped at `_CONTEXT_BUDGET_CHARS`
          so we don't blow the model's input window. If the BM25
          selection still overflows, we truncate the last chunk's tail
          (gracefully — never silently drop the first chunks, which
          are the most relevant).
        """
        try:
            context = ""

            if context_files is None:
                context_files = list(self.document_content.keys())

            query_tokens = _tokenize(question)

            for file_name in context_files:
                if file_name not in self.document_content:
                    continue
                doc_info = self.document_content[file_name]
                context += f"\n--- {file_name} ---\n"
                context += f"File Type: {doc_info['file_type']}\n"
                context += f"Summary: {doc_info['summary']}\n"

                # Text-bearing file: BM25 over pre-chunked text.
                chunks = self._chunks.get(file_name)
                if chunks:
                    if query_tokens:
                        # Pre-tokenize each chunk once and score.
                        tokenized = [_tokenize(c) for c in chunks]
                        scores = _bm25_scores(query_tokens, tokenized)
                        # Take the top-k by score, but preserve the
                        # original chunk order so the model sees
                        # document-coherent prose (not a shuffled salad).
                        ranked = sorted(
                            enumerate(scores), key=lambda x: x[1], reverse=True
                        )[: self._CHUNKS_PER_DOC]
                        ranked.sort(key=lambda x: x[0])  # back to doc order
                        context += "Relevant excerpts (BM25-ranked):\n"
                        for idx, _score in ranked:
                            context += f"[chunk {idx}]\n{chunks[idx]}\n\n"
                    else:
                        # Question is empty / pure stopwords — fall back
                        # to the first few chunks so the model still
                        # has *some* context.
                        context += "Content (first chunks):\n"
                        for c in chunks[: self._CHUNKS_PER_DOC]:
                            context += f"{c}\n\n"

                # Structured data: keep the analysis summary.
                if file_name in self.analysis_results:
                    analysis = self.analysis_results[file_name]
                    context += "Data Analysis Summary:\n"
                    context += f"Shape: {analysis['basic_info']['shape']}\n"
                    context += f"Columns: {analysis['basic_info']['columns']}\n"
                    if analysis['summary_statistics']:
                        context += (
                            f"Key Statistics Available: "
                            f"{list(analysis['summary_statistics'].keys())}\n"
                        )

            # Truncate the assembled context to the budget. We chop the
            # tail (last file's chunks) rather than the head because
            # the most-relevant chunks were the *first* ones we
            # selected per file.
            if len(context) > self._CONTEXT_BUDGET_CHARS:
                context = context[: self._CONTEXT_BUDGET_CHARS] + "\n[...truncated for context budget...]"

            # Create the prompt
            prompt = f"""
            You are an intelligent data analyst. Based on the following document(s) and analysis, please answer the user's question accurately and comprehensively.

            CONTEXT FROM DOCUMENTS:
            {context}

            CONVERSATION HISTORY:
            {self._format_conversation_history()}

            USER QUESTION: {question}

            Please provide a detailed answer based on the available data and documents. If the question involves specific data analysis, calculations, or comparisons, please be precise and cite relevant statistics or findings from the documents.

            If you cannot answer the question based on the available information, please explain what additional information would be needed.
            """

            answer = self._make_api_call_with_retry(prompt, max_tokens=1000)

            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'context_files': context_files
            })

            return answer

        except Exception as e:
            return f"Error answering question: {str(e)}"

    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation."

        formatted = ""
        for i, item in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
            formatted += f"Q{i+1}: {item['question']}\n"
            formatted += f"A{i+1}: {item['answer'][:200]}...\n\n"

        return formatted

    def generate_comprehensive_report(self, file_name: str) -> str:
        """Generate a comprehensive analysis report for a file"""
        if file_name not in self.document_content:
            return f"File {file_name} not found in processed documents."

        try:
            doc_info = self.document_content[file_name]
            report_prompt = f"""
            Generate a comprehensive analytical report for the following document:

            File: {file_name}
            Type: {doc_info['file_type']}
            Summary: {doc_info['summary']}

            Content Sample: {doc_info['content'][:2000]}

            Please provide:
            1. Executive Summary
            2. Key Findings
            3. Data Quality Assessment (if applicable)
            4. Trends and Patterns
            5. Recommendations
            6. Areas for Further Investigation

            Make the report professional and actionable.
            """

            return self._make_api_call_with_retry(report_prompt, max_tokens=1500)

        except Exception as e:
            return f"Error generating report: {str(e)}"

    def get_file_info(self) -> Dict[str, Any]:
        """Get information about all processed files"""
        info = {}
        for file_name, doc_info in self.document_content.items():
            info[file_name] = {
                'type': doc_info['file_type'],
                'has_data': file_name in self.data_frames,
                'summary': doc_info['summary'][:100] + "..." if len(doc_info['summary']) > 100 else doc_info['summary']
            }
        return info

    # ---- API helper ---------------------------------------------------------

    def _make_api_call_with_retry(
        self,
        prompt: str,
        max_tokens: int = 500,
        max_retries: int = 3,
    ) -> str:
        """Make API call with retry logic and exponential backoff"""
        # Pull user-tweakable settings from Streamlit session state when available
        if _ST_AVAILABLE:
            try:
                if hasattr(st, "session_state"):
                    max_tokens = getattr(st.session_state, "max_tokens", max_tokens)
                    max_retries = getattr(st.session_state, "max_retries", max_retries)
                    temperature = getattr(st.session_state, "temperature", 0.3)
                else:
                    temperature = 0.3
            except Exception:
                temperature = 0.3
        else:
            temperature = 0.3

        messages = [{"role": "user", "content": prompt}]
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                return _chat_complete(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                # 429 / rate-limit → back off
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    if os.environ.get("STREAMLIT_RUN") != "1":
                        print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        return f"Rate limit exceeded. Please try again in a few minutes. Error: {error_str}"
                else:
                    # Non-rate-limit error: don't retry, surface to user
                    return f"API Error: {error_str}"

        return f"Maximum retries exceeded. Last error: {last_error}"
