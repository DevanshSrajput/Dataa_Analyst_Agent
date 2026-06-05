"""
Pure-Python helpers for the Streamlit UI.

Extracted from app.py so the entrypoint stays focused on UI
orchestration. Two pieces live here:

  - `_safe_filename` (path-traversal safety for uploads).
  - `AVAILABLE_MODELS` (curated model catalogue for the picker).

This module is pure-Python (no Streamlit import) so it can be
unit-tested without spinning up a Streamlit runtime.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict

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


# ---------------------------------------------------------------------------
# Model catalogue (ISSUES.md #2 — verify against OpenCode Zen)
# ---------------------------------------------------------------------------
# The previous list mixed real OpenCode Zen model ids with speculative
# ones (mimo-v2.5-free, qwen3.6-plus-free, deepseek-v4-flash-free,
# nemotron-3-ultra-free, gemini-3.1-pro, gpt-5, claude-sonnet-4-6,
# minimax-m2.7). Most of those IDs would 404 against the live catalogue.
#
# We keep the catalogue small and curated instead of mirroring the full
# Zen listing — the model picker is a UI element, not a reference doc,
# and a long list of half-broken entries is worse than a short list of
# known-good ones. To verify any new id before adding it, run:
#
#   curl -sSL https://opencode.ai/zen/v1/models \
#     -H "Authorization: Bearer $OPENCODE_API_KEY" | jq '.data[].id'
#
# and pick from the IDs that come back. The default below is the
# current minimax-m3-free tier; switch to minimax-m2.7 (paid) by
# picking from the Settings tab.
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "minimax-m3-free": {
        "name": "minimax-m3-free (default)",
        "description": "Free tier MiniMax model on OpenCode Zen. Default.",
        "tier": "Free",
        "performance": "⭐⭐⭐",
    },
    "minimax-m2.7": {
        "name": "minimax m2.7",
        "description": "Latest paid MiniMax model on OpenCode Zen.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐",
    },
    "minimax-m2.5": {
        "name": "minimax m2.5",
        "description": "Previous-generation MiniMax on OpenCode Zen.",
        "tier": "Paid",
        "performance": "⭐⭐⭐⭐",
    },
}

# The default model id. The UI's model selector MUST list this id
# in AVAILABLE_MODELS; the Settings tab asserts this on render.
DEFAULT_MODEL_ID = "minimax-m3-free"


def list_model_choices() -> list:
    """Return the (id, display_name) pairs the UI's selectbox needs,
    in the order they should appear. Free tier first, then paid."""
    free = [
        (mid, info["name"])
        for mid, info in AVAILABLE_MODELS.items()
        if info.get("tier") == "Free"
    ]
    paid = [
        (mid, info["name"])
        for mid, info in AVAILABLE_MODELS.items()
        if info.get("tier") == "Paid"
    ]
    return free + paid
