import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import html
import re

from rank_bm25 import BM25Okapi, BM25Plus
from utils import bm25okapi_search, search
import lemma
import preprocess

# Page configuration
st.set_page_config(
    page_title="Search Lyrics — Music Information Retrieval",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme & Styling
# Warm, textured palette:
# - App Background: #F8F5EE (warm natural linen)
# - Sidebar Background: #F3ECE2 (cozy parchment)
# - Cards: #FFEED6 (warm cream)
# - Inner Snippets: #FFFFFF (crisp contrast)
# - Borders: #E8A07C (warm terracotta) & #DFD5C6 (soft latte)
# - Primary Buttons & Badges: #827148 (warm olive-earth)
# - Hover/Accents: #A5AF79 & #6E5F3D
# - Text: #24211E (espresso) & #4A4038 (dark neutral)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global Reset & Base */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #F8F5EE !important;
    color: #24211E !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #F8F5EE !important;
}

/* Header Banner - Warm & Clean */
header[data-testid="stHeader"] {
    background-color: #F8F5EE !important;
    border-bottom: 1px solid #DFD5C6 !important;
}
[data-testid="stToolbar"] * {
    color: #24211E !important;
    fill: #24211E !important;
}

/* Base text contrast */
label, p, span, h1, h2, h3, h4, h5, h6 {
    color: #24211E !important;
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] * {
    color: #24211E !important;
    font-weight: 700 !important;
}

/* Main App Header */
.header-wrapper {
    padding: 0.8rem 0 1.2rem 0;
    margin-bottom: 1.4rem;
    border-bottom: 1.5px solid #DFD5C6;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #FFEED6;
    border: 1px solid #E8A07C;
    color: #827148 !important;
    font-size: 0.78rem;
    font-weight: 800 !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.header-title {
    font-size: 2.3rem;
    font-weight: 800 !important;
    color: #24211E !important;
    letter-spacing: -0.6px;
    line-height: 1.2;
    margin: 0 0 6px 0;
}
.header-subtitle {
    font-size: 1.05rem;
    color: #5A5046 !important;
    margin: 0;
    font-weight: 500 !important;
}

/* Segmented Control Style Tabs */
[data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: #EDE7DC !important;
    border: 1px solid #DFD5C6 !important;
    border-radius: 10px !important;
    padding: 5px !important;
    display: inline-flex !important;
    margin-bottom: 1.8rem !important;
}
[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 8px 22px !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="tab"] * {
    color: #5A5046 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-baseweb="tab"]:hover {
    background-color: rgba(255, 238, 214, 0.6) !important;
}
[data-baseweb="tab"]:hover * {
    color: #827148 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}
[data-baseweb="tab"][aria-selected="true"] * {
    color: #827148 !important;
    font-weight: 800 !important;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #F3ECE2 !important;
    border-right: 1.5px solid #DFD5C6 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #24211E !important;
}
.sidebar-header-box {
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #DFD5C6;
}
.sidebar-title {
    font-size: 1.45rem;
    font-weight: 800 !important;
    color: #24211E !important;
    margin: 0;
}
.sidebar-sub {
    font-size: 0.84rem;
    color: #5A5046 !important;
    font-weight: 500 !important;
    margin: 3px 0 0 0;
}
.sidebar-section-title {
    font-size: 0.84rem;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #827148 !important;
    margin: 1.2rem 0 0.5rem 0;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #DFD5C6;
    margin: 1.2rem 0;
}
.sidebar-pill {
    display: inline-block;
    background: #EBF3E6;
    border: 1px solid #A5AF79;
    color: #2D421A !important;
    font-size: 0.82rem;
    font-weight: 700 !important;
    padding: 4px 10px;
    border-radius: 12px;
    margin-top: 6px;
}
.pipeline-chip {
    background: #FFEED6;
    border: 1px solid #E8A07C;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.84rem;
    color: #3D352E !important;
    font-weight: 500;
    margin-top: 8px;
    margin-bottom: 12px;
}
.status-box {
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
}
.status-ready {
    background-color: #F1F6ED;
    border: 1.5px solid #A5AF79;
}
.status-ready .status-title {
    color: #2A4016 !important;
    font-weight: 800 !important;
    font-size: 1rem;
}
.status-ready .status-time {
    font-size: 0.88rem;
    font-weight: 700 !important;
    color: #3E5425 !important;
    margin: 4px 0 2px 0;
}
.status-ready .status-desc {
    font-size: 0.8rem;
    color: #4B6331 !important;
    font-weight: 500;
}
.status-pending {
    background-color: #FFF8EE;
    border: 1.5px dashed #E8A07C;
}
.status-pending .status-title {
    color: #827148 !important;
    font-weight: 800 !important;
    font-size: 0.95rem;
}
.status-pending .status-desc {
    font-size: 0.82rem;
    color: #5A5046 !important;
    font-weight: 500;
    margin-top: 4px;
}

/* File Uploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploadDropzone"],
section[data-testid="stFileUploadDropzone"] {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #E8A07C !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #FFEED6 !important;
    border-color: #827148 !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploader"] * {
    color: #24211E !important;
}
[data-testid="stFileUploadDropzone"] small {
    color: #6A5F54 !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #FFEED6 !important;
    border: 1px solid #E8A07C !important;
    color: #827148 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] button * {
    color: #827148 !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #827148 !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] button:hover * {
    color: #FFFFFF !important;
}

/* Selectbox Dropdown */
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #E8A07C !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    color: #24211E !important;
}
div[data-baseweb="select"] svg {
    fill: #827148 !important;
}
div[data-baseweb="popover"],
ul[role="listbox"],
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #24211E !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #FFEED6 !important;
    color: #827148 !important;
    font-weight: 700 !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label * {
    color: #24211E !important;
    font-weight: 600 !important;
}
[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #827148 !important;
}

/* Search Area & Input Alignment */
.search-label-bar {
    font-size: 0.95rem;
    font-weight: 700;
    color: #24211E !important;
    margin-bottom: 6px;
}
div[data-testid="stTextInput"] input {
    height: 48px !important;
    background-color: #FFFFFF !important;
    color: #24211E !important;
    border: 1.5px solid #E8A07C !important;
    border-radius: 8px !important;
    padding: 0 16px !important;
    font-size: 0.98rem !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #8A7E72 !important;
    font-weight: 400 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #827148 !important;
    box-shadow: 0 0 0 3px rgba(130, 113, 72, 0.22) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: transparent !important;
    border: none !important;
}

/* Number Input */
div[data-testid="stNumberInput"] input {
    height: 44px !important;
    background-color: #FFFFFF !important;
    color: #24211E !important;
    border: 1.5px solid #E8A07C !important;
    border-radius: 8px !important;
}

/* Solid Buttons - No Arch Cutout, Crisp Text */
.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
    background-color: #827148 !important;
    background: #827148 !important;
    border: 1.5px solid #6E5F3D !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 5px rgba(130, 113, 72, 0.18) !important;
    cursor: pointer !important;
    height: 48px !important;
    padding: 0 24px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}
.stButton > button *,
.stButton > button div,
.stButton > button p,
.stButton > button span {
    background-color: transparent !important;
    background: transparent !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    letter-spacing: 0.3px !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
.stButton > button:hover {
    background-color: #6E5F3D !important;
    background: #6E5F3D !important;
    border-color: #554A2E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(130, 113, 72, 0.25) !important;
}
.stButton > button:hover * {
    color: #FFFFFF !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button:disabled,
.stButton > button[disabled] {
    background-color: #DDD4C7 !important;
    background: #DDD4C7 !important;
    border-color: #C8BCAC !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}
.stButton > button:disabled * {
    color: #8A7E70 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
}

/* Search Meta Bar */
.search-meta-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background-color: #EDE7DC;
    border: 1px solid #DFD5C6;
    border-radius: 10px;
    margin-bottom: 1.4rem;
    font-size: 0.92rem;
    color: #24211E !important;
}
.search-time-badge {
    background: #FFEED6;
    border: 1px solid #E8A07C;
    color: #827148 !important;
    font-weight: 800 !important;
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 12px;
}

/* Song Result Cards */
.song-result-card {
    background-color: #FFEED6;
    border: 1.5px solid #E8A07C;
    border-radius: 14px;
    padding: 22px 24px 18px 24px;
    margin-bottom: 12px;
    box-shadow: 0 3px 10px rgba(130, 113, 72, 0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.song-result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(130, 113, 72, 0.14);
    border-color: #D88F6B;
}
.song-header-row {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 12px;
}
.song-rank-badge {
    font-size: 1.35rem;
    font-weight: 800;
    color: #827148 !important;
    background-color: #FFFFFF;
    border: 1.5px solid #E8A07C;
    padding: 6px 14px;
    border-radius: 10px;
    letter-spacing: -0.5px;
    line-height: 1;
}
.song-info-col {
    flex: 1;
}
.song-title-text {
    font-size: 1.35rem;
    font-weight: 800 !important;
    color: #24211E !important;
    margin: 0 0 4px 0;
    line-height: 1.25;
}
.song-artist-text {
    font-size: 0.95rem;
    font-weight: 600 !important;
    color: #5A5046 !important;
}
.song-preview-block {
    background-color: #FFFFFF;
    border-left: 3.5px solid #827148;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-top: 10px;
    font-size: 0.96rem;
    line-height: 1.65;
    color: #2C2523 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    white-space: pre-line;
}
.query-mark {
    background-color: #FDE68A;
    color: #1F1D19 !important;
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 800 !important;
}

/* Full Lyrics Container inside expander */
[data-testid="stExpander"] {
    background-color: #FAF7F2 !important;
    border: 1px solid #E8A07C !important;
    border-radius: 10px !important;
    margin-top: -6px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #827148 !important;
    font-size: 0.92rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #5D5032 !important;
}
.full-lyrics-box {
    max-height: 300px;
    overflow-y: auto;
    padding: 14px 16px;
    background: #FFFFFF;
    border: 1px solid #DFD5C6;
    border-radius: 8px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #24211E !important;
    white-space: pre-line;
}

/* Warm, Tactile Empty State Card */
.empty-state-card {
    text-align: center;
    padding: 2.8rem 2rem;
    background-color: #FFFDF9;
    border: 1.5px solid #E8A07C;
    border-radius: 16px;
    margin: 1.8rem 0;
    box-shadow: 0 4px 16px rgba(130, 113, 72, 0.07);
}
.empty-state-badge {
    display: inline-block;
    background-color: #FFEED6;
    border: 1px solid #E8A07C;
    color: #827148;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.empty-state-title {
    font-size: 1.35rem;
    font-weight: 800 !important;
    color: #24211E !important;
    margin: 0 0 8px 0;
}
.empty-state-desc {
    font-size: 0.96rem;
    color: #4A4038 !important;
    font-weight: 500 !important;
    max-width: 520px;
    margin: 0 auto 16px auto;
    line-height: 1.6;
}
.empty-state-chips {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.tip-chip {
    background-color: #EDE7DC;
    border: 1px solid #DFD5C6;
    color: #5A5046 !important;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
}

/* Evaluation Specific Styles */
.eval-hero {
    margin-bottom: 1.4rem;
}
.eval-title {
    font-size: 1.7rem;
    font-weight: 800 !important;
    color: #24211E !important;
    margin: 0 0 6px 0;
}
.eval-subtitle {
    font-size: 0.98rem;
    color: #5A5046 !important;
    font-weight: 500 !important;
    margin: 0;
}
.metric-card {
    background-color: #FFEED6;
    border: 1.5px solid #E8A07C;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(130, 113, 72, 0.06);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #827148 !important;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.95rem;
    font-weight: 800 !important;
    color: #24211E !important;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #5A5046 !important;
    font-weight: 500;
    margin-top: 4px;
}
.info-card {
    background-color: #FFFDF9;
    border: 1px solid #E8A07C;
    border-left: 4px solid #827148;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.92rem;
    color: #352E2A !important;
    line-height: 1.6;
    margin-top: 1.4rem;
}
</style>
""", unsafe_allow_html=True)


# Presentation Helper Functions
def highlight_query_terms(text, raw_query):
    """
    Safely highlights query terms in lyric text for presentation.
    Does NOT modify the underlying search logic or tokens.
    """
    if not text:
        return ""
    if not raw_query:
        return html.escape(str(text))
    
    # Extract query words of length > 1 for matching
    raw_words = re.findall(r'[a-zA-Z0-9]+', raw_query)
    words = [w for w in raw_words if len(w) > 1]
    if not words and raw_words:
        words = raw_words
    
    safe_text = html.escape(str(text))
    if not words:
        return safe_text
        
    words = sorted(set(words), key=len, reverse=True)
    pattern = r'(?i)\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    highlighted = re.sub(pattern, r'<mark class="query-mark">\1</mark>', safe_text)
    return highlighted


def format_lyric_preview(lyrics_text, raw_query, max_lines=3, max_chars=200):
    """
    Extracts a concise 2-3 line preview from the lyrics.
    If query words are present, centers on matching lines where practical.
    """
    if not lyrics_text:
        return "No lyrics available."
    
    raw_lines = [line.strip() for line in str(lyrics_text).splitlines() if line.strip()]
    content_lines = [l for l in raw_lines if not (l.startswith('[') and l.endswith(']'))]
    if not content_lines:
        content_lines = raw_lines
        
    query_words = [w for w in re.findall(r'[a-zA-Z0-9]+', raw_query or '') if len(w) > 1]
    start_idx = 0
    if query_words:
        pattern = r'(?i)\b(' + '|'.join(re.escape(w) for w in query_words) + r')\b'
        for i, line in enumerate(content_lines):
            if re.search(pattern, line):
                start_idx = i
                break
                
    selected = content_lines[start_idx : start_idx + max_lines]
    prefix = "... " if start_idx > 0 else ""
    suffix = "..." if (start_idx + max_lines) < len(content_lines) else ""
    preview_text = prefix + "\n".join(selected) + suffix
    
    if len(preview_text) > max_chars:
        preview_text = preview_text[:max_chars].rstrip() + "..."
        
    return highlight_query_terms(preview_text, raw_query)


# Session State Management
if 'tokenize_lyric' not in st.session_state:
    st.session_state.tokenize_lyric = None
if 'preprocesser' not in st.session_state:
    st.session_state.preprocesser = None
if 'time' not in st.session_state:
    st.session_state.time = 0.0
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_search' not in st.session_state:
    st.session_state.last_search = None
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None


# Library Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-box">
        <h2 class="sidebar-title">Library</h2>
        <p class="sidebar-sub">Dataset & Preprocessing Controls</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Upload
    st.markdown('<div class="sidebar-section-title">Music Library</div>', unsafe_allow_html=True)
    uploaded_data_file = st.file_uploader(
        "Upload dataset",
        type=["csv"],
        help="Upload the main music dataset (e.g., music_data.csv)"
    )

    if uploaded_data_file is not None:
        df = pd.read_csv(uploaded_data_file, encoding='utf-8')
        st.session_state.df = df
        st.markdown(f'<div class="sidebar-pill">✓ {len(df):,} songs loaded</div>', unsafe_allow_html=True)
    elif st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f'<div class="sidebar-pill">✓ {len(df):,} songs loaded</div>', unsafe_allow_html=True)
    else:
        df = None

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # 2. Preprocessing Configuration
    st.markdown('<div class="sidebar-section-title">Preprocessing</div>', unsafe_allow_html=True)
    normalise_text = st.selectbox(
        "Choose restoring word root form methods:",
        ("PorterStemmer", "WordNetLemmatizer", "DIY_lemmatizer", "none")
    )

    st.markdown("Select if you want to remove the stopwords:")
    remove_stopwords = st.checkbox("remove_stopword")

    pipeline = [normalise_text]
    if remove_stopwords:
        pipeline.append("remove_stopword")

    pipeline_display = ", ".join(pipeline)
    st.markdown(f'<div class="pipeline-chip">Pipeline: <strong>{pipeline_display}</strong></div>', unsafe_allow_html=True)

    pipeline.append("nltk_word_tokenizer")

    # 3. Preprocess Action
    preprocess_clicked = st.button('Preprocess', disabled=(df is None))

    if preprocess_clicked and df is not None:
        lyrics = []
        preprocesser = preprocess.Preprocessing(Pipeline=pipeline)
        process_bar = st.progress(0)
        with st.spinner("Preprocessing lyrics..."):
            start = time.time()
            total_songs = len(df.lyrics)
            for i, lyric in enumerate(df.lyrics):
                lyrics.append(preprocesser.Preprocess(lyric))
                if (i + 1) % 25 == 0 or (i + 1) == total_songs:
                    process_bar.progress((i + 1) / total_songs)
            end = time.time()

            tokenize_lyric = []
            for lyric in lyrics:
                tokenize_lyric.append(lyric.split())
            for i in range(len(df.id)):
                tokenize_lyric[i].append(f"-->{df.id[i]}")

        st.session_state.tokenize_lyric = tokenize_lyric
        st.session_state.preprocesser = preprocesser
        st.session_state.time = end - start

    # 4. Preprocessing Status
    if st.session_state.tokenize_lyric is not None:
        st.markdown(f"""
        <div class="status-box status-ready">
            <div class="status-title">✓ Ready</div>
            <div class="status-time">Execution time: {st.session_state.time:.4f}s</div>
            <div class="status-desc">Index built and ready for search & evaluation.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box status-pending">
            <div class="status-title">Index Pending</div>
            <div class="status-desc">Upload dataset and click Preprocess to initialize the BM25 index.</div>
        </div>
        """, unsafe_allow_html=True)


# Main Header
st.markdown("""
<div class="header-wrapper">
    <div class="header-badge">Music Information Retrieval</div>
    <h1 class="header-title">Search Lyrics</h1>
    <p class="header-subtitle">Find songs from the words you remember.</p>
</div>
""", unsafe_allow_html=True)


# Navigation Tabs
tab_search, tab_eval = st.tabs(["Search Engine", "Evaluation"])


# Tab 1: Search Engine
with tab_search:
    # Clean, prominent search area:
    # Search Lyrics
    # Search songs using lyrics
    # Search lyrics
    # [ text input ] [ Search ]
    st.markdown('<div class="search-label-bar">Search lyrics</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Search lyrics",
            label_visibility="collapsed",
            placeholder="Enter lyrics or keywords (e.g. 'we were both young when I first saw you')"
        )
    with col_btn:
        search_clicked = st.button("Search")

    if search_clicked:
        if df is None:
            st.warning("Please upload a music dataset in the Library sidebar first.")
        elif st.session_state.tokenize_lyric is None:
            st.error("You haven't preprocessed the data yet! Please run Preprocess first in the Library sidebar.", icon="🚨")
        elif not query.strip():
            st.info("Please enter lyrics or keywords to search.")
        else:
            preprocesser = st.session_state.preprocesser
            cleaned_query = query.replace('...', ' ')
            processed_query = preprocesser.Preprocess(cleaned_query)
            token = processed_query.split(" ")

            with st.spinner("Searching songs..."):
                tokenize_lyric = st.session_state.tokenize_lyric
                start_search = time.time()
                bm25 = BM25Okapi(tokenize_lyric)
                top_5_results = bm25okapi_search(token, bm25, tokenize_lyric, n_results=5)
                end_search = time.time()
                search_time = end_search - start_search

            st.session_state.last_search = {
                'query': query,
                'rank': top_5_results,
                'time': search_time
            }

    # Display Search Results
    if st.session_state.last_search is not None:
        last_search = st.session_state.last_search
        search_query = last_search['query']
        ranked_ids = last_search['rank']
        elapsed = last_search['time']

        st.markdown(f"""
        <div class="search-meta-bar">
            <span>Showing top <strong>{len(ranked_ids)}</strong> relevant results for <em>"{html.escape(search_query)}"</em></span>
            <span class="search-time-badge">Search time: {elapsed:.4f}s</span>
        </div>
        """, unsafe_allow_html=True)

        # Render top 5 cards preserving exact BM25 ranking
        for idx, song_id in enumerate(ranked_ids):
            match = df[df.id == song_id]
            if not match.empty:
                row = match.iloc[0]
                rank_str = f"{idx + 1:02d}"
                raw_title = str(row.get('title', 'Unknown Title')).replace('\u200b', '').strip()
                raw_artist = str(row.get('artist', 'Unknown Artist')).strip()
                raw_date = str(row.get('release_date', '')).strip()
                year = raw_date[:4] if raw_date and raw_date != 'nan' else ''
                lyrics = str(row.get('lyrics', ''))

                safe_title = html.escape(raw_title)
                safe_artist = html.escape(raw_artist)
                artist_meta = f"{safe_artist} • {year}" if year else safe_artist

                preview_html = format_lyric_preview(lyrics, search_query, max_lines=3, max_chars=200)
                full_lyrics_highlighted = highlight_query_terms(lyrics, search_query)

                # Attractive Song Result Card
                st.markdown(f"""
                <div class="song-result-card">
                    <div class="song-header-row">
                        <div class="song-rank-badge">{rank_str}</div>
                        <div class="song-info-col">
                            <h3 class="song-title-text">{safe_title}</h3>
                            <div class="song-artist-text">{artist_meta}</div>
                        </div>
                    </div>
                    <div class="song-preview-block">{preview_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Expandable interaction for full lyrics
                with st.expander(f"View Full Lyrics — {safe_title}"):
                    st.markdown(f"""
                    <div class="full-lyrics-box">
                        {full_lyrics_highlighted}
                    </div>
                    """, unsafe_allow_html=True)

    elif st.session_state.tokenize_lyric is None:
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-state-badge">🎵 Music Information Retrieval</div>
            <h3 class="empty-state-title">Music Index Not Initialized</h3>
            <p class="empty-state-desc">
                Upload <strong>music_data.csv</strong> and click <strong>Preprocess</strong> in the Library sidebar to build your song index and search engine.
            </p>
            <div class="empty-state-chips">
                <span class="tip-chip">📁 Step 1: Upload dataset</span>
                <span class="tip-chip">⚙️ Step 2: Choose normalizer</span>
                <span class="tip-chip">🚀 Step 3: Click Preprocess</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-state-badge">🎵 Ready to Search</div>
            <h3 class="empty-state-title">Find songs from the words you remember</h3>
            <p class="empty-state-desc">
                Enter any lyric fragment, phrase, or memorable hook in the search bar above to retrieve the closest matching songs.
            </p>
            <div class="empty-state-chips">
                <span class="tip-chip">💡 "we were both young"</span>
                <span class="tip-chip">💡 "breakfast at tiffany's"</span>
                <span class="tip-chip">💡 "tears left to cry"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Tab 2: Evaluation
with tab_eval:
    st.markdown("""
    <div class="eval-hero">
        <h2 class="eval-title">Model Evaluation</h2>
        <p class="eval-subtitle">Evaluate how effectively the system retrieves the relevant song for each test query.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_test_file = st.file_uploader(
        "Choose a test set file",
        type=["csv"],
        help="Upload test.csv containing queries and relevant song IDs"
    )

    if uploaded_test_file is not None:
        test_df = pd.read_csv(uploaded_test_file)

        with st.expander(f"Preview Test Set ({len(test_df)} queries)", expanded=False):
            st.dataframe(test_df, use_container_width=True)

        col_top, col_eval_btn = st.columns([3, 1])
        with col_top:
            top_n_rank = st.number_input(
                "Input number of top rank to get",
                min_value=1,
                max_value=100,
                value=5
            )
        with col_eval_btn:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            evaluate_clicked = st.button("Evaluate")

        if evaluate_clicked:
            if st.session_state.tokenize_lyric is None:
                st.error(
                    "You haven't preprocessed the data yet. Please run Preprocess first in the Library sidebar.",
                    icon="🚨"
                )
            else:
                preprocesser = st.session_state.preprocesser
                tokenize_lyric = st.session_state.tokenize_lyric

                with st.spinner("Evaluating model..."):
                    # Prepare test queries
                    test_df_eval = test_df.copy()
                    test_df_eval["qid"] = test_df_eval.index + 1

                    query_token = []
                    for _, row in test_df_eval.iterrows():
                        q = str(row["corpus"])
                        q = q.replace("...", " ")
                        q = preprocesser.Preprocess(q)
                        q = q.split()
                        query_token.append(q)

                    test_df_eval["query_token"] = query_token

                    # Create BM25 model
                    start = time.time()
                    bm25 = BM25Okapi(tokenize_lyric)
                    search_results = search(
                        test_df_eval,
                        bm25,
                        tokenize_lyric,
                        top_n_rank
                    )
                    end = time.time()

                    # Actual relevant song for every query
                    judgments = test_df_eval[["qid", "id"]].copy()
                    judgments["relevancy grade"] = 1

                    # Match retrieved results with relevant songs
                    labeled_search_results = search_results.merge(
                        judgments,
                        how="left",
                        on=["qid", "id"]
                    )
                    labeled_search_results["relevancy grade"] = (
                        labeled_search_results["relevancy grade"].fillna(0)
                    )

                    # Find rank of first relevant result
                    relevant_results = labeled_search_results[
                        labeled_search_results["relevancy grade"] == 1
                    ]
                    ranks = relevant_results.groupby("qid")["rank"].min()

                    # Queries where relevant song was not found
                    missing_queries = len(test_df_eval) - len(ranks)

                    # Reciprocal rank
                    reciprocal_ranks = 1 / ranks

                    # MRR
                    MRR = reciprocal_ranks.sum() / len(test_df_eval)

                    # Rank frequency
                    rank_counts = np.zeros(top_n_rank + 1)
                    for r in ranks:
                        if r <= top_n_rank:
                            rank_counts[int(r)] += 1
                    rank_counts[0] = missing_queries

                    st.session_state.eval_results = {
                        'MRR': MRR,
                        'execution_time': end - start,
                        'top_n_rank': top_n_rank,
                        'rank_counts': rank_counts,
                        'total_queries': len(test_df_eval),
                        'found_queries': len(ranks)
                    }

        # Render Evaluation Results
        if st.session_state.eval_results is not None:
            eval_data = st.session_state.eval_results
            mrr_val = eval_data['MRR']
            exec_time = eval_data['execution_time']
            top_n = eval_data['top_n_rank']
            counts = eval_data['rank_counts']
            total_q = eval_data['total_queries']
            found_q = eval_data['found_queries']

            st.success("Evaluation completed.")

            # Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Reciprocal Rank</div>
                    <div class="metric-value">{mrr_val:.4f}</div>
                    <div class="metric-sub">Score out of 1.0000</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Queries Resolved</div>
                    <div class="metric-value">{found_q} <span style="font-size:1.1rem; color:#7A6F65;">/ {total_q}</span></div>
                    <div class="metric-sub">Songs found in Top {top_n}</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Search Execution Time</div>
                    <div class="metric-value">{exec_time:.4f}s</div>
                    <div class="metric-sub">Total retrieval benchmark</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("") # spacing

            # Rank frequency graph
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
            fig.patch.set_facecolor('#F8F5EE')
            ax.set_facecolor('#FFFFFF')

            x_indices = np.arange(0, top_n + 1)
            bar_colors = ['#E8A07C'] + ['#827148'] * top_n

            bars = ax.bar(
                x_indices,
                counts,
                color=bar_colors,
                edgecolor='#6E5F3D',
                linewidth=0.8,
                width=0.62,
                zorder=3
            )

            # Direct data labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#24211E'
                    )

            x_labels = ['Not Found (0)'] + [f'Rank {i}' for i in range(1, top_n + 1)]
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, fontsize=9, color='#423A33')
            ax.tick_params(axis='y', colors='#423A33', labelsize=9)

            ax.set_title(
                f"Frequency of Relevant Results (Top {top_n})",
                fontsize=12, fontweight='bold', color='#24211E', pad=12
            )
            ax.set_xlabel("Rank", fontsize=10, fontweight='600', color='#423A33', labelpad=8)
            ax.set_ylabel("Number of Queries", fontsize=10, fontweight='600', color='#423A33', labelpad=8)

            ax.grid(axis='y', linestyle='--', alpha=0.35, color='#D1C7BD', zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#D1C7BD')
            ax.spines['bottom'].set_color('#D1C7BD')

            max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 10
            ax.set_ylim(0, max_count * 1.18)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Evaluation explanation
            st.markdown("""
            <div class="info-card">
                <strong>How to interpret MRR:</strong><br>
                Mean Reciprocal Rank (MRR) measures how high the first relevant song appears in the search results. 
                A relevant result at rank 1 receives a reciprocal rank of 1.0, rank 2 receives 0.5, rank 3 receives approximately 0.33, 
                and queries where the song is not found within the top rank receive 0.
            </div>
            """, unsafe_allow_html=True)
