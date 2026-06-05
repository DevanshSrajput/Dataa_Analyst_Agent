"""
Theme CSS for the Streamlit UI.

Extracted from app.py so the entrypoint stays focused on UI
orchestration. Two themes are defined (dark, light); the active
one is selected at runtime by `_inject_css(theme_mode)`. Both
blocks are intentionally focused and scoped — the previous
~700-line Agent.py block was overkill and made the file
unmaintainable.

This module is pure-Python (no Streamlit import) so it can be
unit-tested without spinning up a Streamlit runtime.
"""

from __future__ import annotations

# Dark mode is the default; matches session_state.theme_mode default
# in app.main().
DARK_CSS = """
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #262730;
    --bg-tertiary: #3d3d3d;
    --text-primary: #fafafa;
    --text-secondary: #e0e0e0;
    --accent-color: #667eea;
    --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    --border-color: #555555;
    --card-bg: #262730;
    --upload-bg: #1e1e2e;
    --tab-bg: #262730;
    --tab-selected: #667eea;
}
.stApp { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox > div > div { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"] { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"]:hover { background-color: var(--accent-color) !important; color: white !important; }
.streamlit-expanderHeader, .streamlit-expanderContent { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stMetric, [data-testid="metric-container"] { background-color: var(--card-bg) !important; color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab"] { background-color: var(--tab-bg) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { background-color: var(--tab-selected) !important; color: white !important; }
.stButton > button { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; border-color: var(--border-color) !important; }
.stButton > button:hover { background-color: var(--accent-color) !important; color: white !important; border-color: var(--accent-color) !important; }
.stFileUploader, .stDataFrame { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stProgress > div > div > div > div { background-color: var(--accent-color) !important; }
.stSlider label, .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label { color: var(--text-primary) !important; }
h1, h2, h3, h4, h5, h6, p, div, span, label, li, strong, em, a { color: var(--text-primary) !important; }
.stMarkdown, .stMarkdown * { color: var(--text-primary) !important; }
.main-header { background: var(--accent-gradient); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.feature-card { background: var(--card-bg); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--accent-color); margin: 1rem 0; color: var(--text-primary); }
.upload-zone { border: 2px dashed var(--accent-color); border-radius: 10px; padding: 2rem; text-align: center; background: var(--upload-bg); color: var(--text-primary); margin: 1rem 0; }
.chat-container { background: var(--card-bg); border-radius: 10px; padding: 1rem; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
"""

LIGHT_CSS = """
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-tertiary: #e9ecef;
    --text-primary: #212529;
    --accent-color: #667eea;
    --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    --border-color: #dee2e6;
    --card-bg: #f8f9fa;
    --upload-bg: #f8f9fa;
    --tab-bg: #f0f2f6;
    --tab-selected: #667eea;
}
.stApp { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea { background-color: white !important; color: var(--text-primary) !important; }
.stSelectbox > div > div { background-color: white !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
.stSelectbox [role="option"] { background-color: white !important; color: var(--text-primary) !important; }
.stSelectbox [role="option"]:hover { background-color: var(--accent-color) !important; color: white !important; }
.streamlit-expanderHeader, .streamlit-expanderContent { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.stMetric, [data-testid="metric-container"] { background-color: var(--card-bg) !important; color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab"] { background-color: var(--tab-bg) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { background-color: var(--tab-selected) !important; color: white !important; }
.stButton > button { background-color: white !important; color: var(--text-primary) !important; border-color: var(--border-color) !important; }
.stButton > button:hover { background-color: var(--accent-color) !important; color: white !important; }
.main-header { background: var(--accent-gradient); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.feature-card { background: var(--card-bg); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--accent-color); margin: 1rem 0; color: var(--text-primary); }
.upload-zone { border: 2px dashed var(--accent-color); border-radius: 10px; padding: 2rem; text-align: center; background: var(--upload-bg); color: var(--text-primary); margin: 1rem 0; }
.chat-container { background: var(--card-bg); border-radius: 10px; padding: 1rem; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
"""


def css_for_theme(theme_mode: str) -> str:
    """Return the CSS block for the given theme name.

    Unknown values fall back to dark so a misconfigured session
    state still gets a usable UI.
    """
    if theme_mode == "light":
        return LIGHT_CSS
    return DARK_CSS
