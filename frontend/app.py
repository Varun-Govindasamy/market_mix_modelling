import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from html import escape as _esc

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MMM PIPELINE // RETRO", page_icon="�", layout="wide", initial_sidebar_state="collapsed")

# ── Retro Theme ─────────────────────────────────────────────────
COLORS = {
    "green":   "#39FF14",
    "amber":   "#FFB000",
    "cyan":    "#00FFFF",
    "magenta": "#FF00FF",
    "red":     "#FF3131",
    "pink":    "#FF6EC7",
    "blue":    "#00BFFF",
    "white":   "#E0E0E0",
    "dim":     "#507050",
    "bg":      "#0A0A0A",
    "card_bg": "rgba(57,255,20,0.03)",
    "border":  "rgba(57,255,20,0.15)",
}

CHART_PALETTE = ["#39FF14", "#FFB000", "#00FFFF", "#FF00FF", "#FF6EC7", "#00BFFF", "#FF3131", "#ADFF2F"]

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,10,0.6)",
    font=dict(family="'Courier New', 'Fira Code', monospace", color="#39FF14", size=11),
    margin=dict(l=50, r=20, t=55, b=45),
    xaxis=dict(gridcolor="rgba(57,255,20,0.08)", zerolinecolor="rgba(57,255,20,0.12)",
               tickfont=dict(color="#39FF14"), title_font=dict(color="#39FF14")),
    yaxis=dict(gridcolor="rgba(57,255,20,0.08)", zerolinecolor="rgba(57,255,20,0.12)",
               tickfont=dict(color="#39FF14"), title_font=dict(color="#39FF14")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#39FF14", size=10)),
    colorway=CHART_PALETTE,
    title_font=dict(color="#FFB000", size=14),
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&family=Press+Start+2P&display=swap');

    /* Hide sidebar */
    [data-testid="collapsedControl"] { display: none; }

    /* CRT background */
    .stApp {
        background: #0A0A0A !important;
        background-image:
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(57,255,20,0.015) 2px, rgba(57,255,20,0.015) 4px) !important;
    }

    /* Scanline overlay */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        background: repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(0,0,0,0.08) 1px, rgba(0,0,0,0.08) 2px);
        z-index: 9999;
    }

    /* Global font */
    html, body, [class*="css"], p, span, label, .stMarkdown {
        font-family: 'Share Tech Mono', 'Courier New', monospace !important;
        color: #39FF14 !important;
    }

    /* Inputs */
    .stNumberInput label, .stSlider label, .stFileUploader label {
        font-family: 'Share Tech Mono', monospace !important;
        color: #FFB000 !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    input[type="number"] {
        background: #0A0A0A !important;
        color: #39FF14 !important;
        border: 1px solid rgba(57,255,20,0.3) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: transparent !important;
        color: #39FF14 !important;
        border: 2px solid #39FF14 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: rgba(57,255,20,0.15) !important;
        box-shadow: 0 0 20px rgba(57,255,20,0.3), inset 0 0 20px rgba(57,255,20,0.05);
    }
    .stButton > button[kind="primary"] {
        border-color: #FFB000 !important;
        color: #FFB000 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: rgba(255,176,0,0.15) !important;
        box-shadow: 0 0 20px rgba(255,176,0,0.3);
    }

    /* Hero */
    .retro-hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-bottom: 1px solid rgba(57,255,20,0.15);
        margin-bottom: 1rem;
    }
    .retro-hero h1 {
        font-family: 'Press Start 2P', monospace !important;
        font-size: 1.4rem;
        color: #39FF14 !important;
        text-shadow: 0 0 10px rgba(57,255,20,0.5), 0 0 30px rgba(57,255,20,0.2);
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .retro-hero .subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #507050 !important;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
    .retro-hero .blink-cursor::after {
        content: '█';
        animation: blink 1s infinite;
        color: #39FF14;
    }
    @keyframes blink { 0%,50% { opacity: 1; } 51%,100% { opacity: 0; } }

    /* Metric cards */
    .metric-row { display: flex; gap: 0.8rem; margin: 0.8rem 0; }
    .metric-card {
        flex: 1;
        background: rgba(57,255,20,0.03);
        border: 1px solid rgba(57,255,20,0.15);
        padding: 1rem 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(57,255,20,0.4), transparent);
    }
    .metric-card .label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #507050 !important;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-family: 'VT323', monospace;
        font-size: 1.8rem;
        font-weight: 400;
        color: #39FF14 !important;
        text-shadow: 0 0 8px rgba(57,255,20,0.4);
    }
    .metric-card .value.amber { color: #FFB000 !important; text-shadow: 0 0 8px rgba(255,176,0,0.4); }
    .metric-card .value.cyan { color: #00FFFF !important; text-shadow: 0 0 8px rgba(0,255,255,0.4); }
    .metric-card .value.magenta { color: #FF00FF !important; text-shadow: 0 0 8px rgba(255,0,255,0.4); }
    .metric-card .value.red { color: #FF3131 !important; text-shadow: 0 0 8px rgba(255,49,49,0.4); }
    .metric-card .value.pink { color: #FF6EC7 !important; text-shadow: 0 0 8px rgba(255,110,199,0.4); }

    /* Section headers */
    .section-head {
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.7rem;
        color: #FFB000 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0 1rem;
        padding: 0.6rem 0;
        border-top: 1px solid rgba(255,176,0,0.2);
        border-bottom: 1px solid rgba(255,176,0,0.2);
    }
    .section-head::before { content: '> '; color: #39FF14; }

    /* Upload zone */
    .upload-zone {
        background: rgba(57,255,20,0.02);
        border: 1px dashed rgba(57,255,20,0.25);
        padding: 2.5rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
    }
    .upload-zone p { color: #39FF14 !important; }
    .upload-zone .dim { color: #507050 !important; }

    /* Divider */
    .retro-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(57,255,20,0.2), rgba(255,176,0,0.2), rgba(57,255,20,0.2), transparent);
        margin: 1.5rem 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(57,255,20,0.15);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Share Tech Mono', monospace !important;
        color: #507050 !important;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #39FF14 !important;
        border-bottom: 2px solid #39FF14 !important;
        text-shadow: 0 0 8px rgba(57,255,20,0.3);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Share Tech Mono', monospace !important;
        color: #FFB000 !important;
    }
    [data-testid="stExpander"] summary {
        gap: 0.5rem;
    }
    [data-testid="stExpander"] summary span {
        padding-left: 0.25rem;
    }

    /* Status widget */
    [data-testid="stStatusWidget"] {
        background: rgba(57,255,20,0.03) !important;
        border: 1px solid rgba(57,255,20,0.15) !important;
    }

    /* Code block for logs */
    .stCodeBlock, code {
        font-family: 'Share Tech Mono', monospace !important;
        background: rgba(57,255,20,0.03) !important;
        color: #39FF14 !important;
    }

    /* DataFrame */
    .stDataFrame { border: 1px solid rgba(57,255,20,0.15); }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #39FF14 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px solid rgba(57,255,20,0.15) !important;
    }

    /* Strategy markdown */
    .strategy-box {
        background: rgba(57,255,20,0.03);
        border: 1px solid rgba(57,255,20,0.12);
        padding: 1.5rem 2rem;
        font-family: 'Share Tech Mono', monospace;
        color: #39FF14;
        line-height: 1.7;
    }
    .strategy-box h1, .strategy-box h2, .strategy-box h3 {
        color: #FFB000 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px;
        margin-top: 1.5rem;
    }
    .strategy-box strong { color: #00FFFF !important; }
    .strategy-box li { color: #39FF14 !important; }

    /* ─── Pipeline Animation Base ─── */
    .pipeline-anim {
        background: rgba(10,10,10,0.95);
        border: 1px solid rgba(57,255,20,0.2);
        padding: 2rem 2rem 1.5rem;
        text-align: center;
        margin: 0.5rem auto;
        max-width: 720px;
        position: relative;
        overflow: hidden;
    }
    .pipeline-anim .anim-scan {
        position: absolute;
        top: 0; left: -50%;
        width: 50%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--phase-color, #39FF14), transparent);
        animation: anim-sweep 2s linear infinite;
    }
    @keyframes anim-sweep { from { left: -50%; } to { left: 100%; } }
    .pipeline-anim .anim-title {
        font-family: 'Press Start 2P', monospace;
        font-size: 0.75rem;
        letter-spacing: 3px;
        margin-bottom: 1rem;
        animation: title-pulse 2s ease-in-out infinite;
    }
    @keyframes title-pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .pipeline-anim .anim-desc {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #507050;
        margin-top: 1rem;
        letter-spacing: 1px;
    }
    .pipeline-anim.done  { border-color:rgba(57,255,20,0.4); box-shadow:0 0 30px rgba(57,255,20,0.06); }
    .pipeline-anim.done .anim-title { animation:none; }
    .pipeline-anim.error-phase { border-color:rgba(255,49,49,0.4); }

    /* ── Data Stage: Matrix rain columns ── */
    .matrix-rain {
        display: flex;
        justify-content: center;
        gap: 4px;
        height: 70px;
        margin: 0.8rem auto;
        overflow: hidden;
    }
    .matrix-col {
        display: flex;
        flex-direction: column;
        gap: 2px;
        animation: matrix-fall var(--fall-dur, 1.2s) linear infinite;
        transform: translateY(-30px);
    }
    .matrix-col span {
        font-family: 'VT323', monospace;
        font-size: 0.8rem;
        line-height: 1;
        color: var(--phase-color, #39FF14);
        opacity: 0.6;
        text-shadow: 0 0 4px var(--phase-color, #39FF14);
    }
    .matrix-col span:last-child { opacity: 1; font-size: 1rem; }
    @keyframes matrix-fall { 0% { transform:translateY(-30px); } 100% { transform:translateY(40px); } }

    /* ── Modeling/Optuna: Search grid scan ── */
    .grid-scan {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 3px;
        max-width: 260px;
        margin: 0.8rem auto;
    }
    .grid-cell {
        width: 100%; aspect-ratio: 1;
        border-radius: 2px;
        background: rgba(255,176,0,0.08);
        animation: cell-scan 2.5s ease-in-out infinite;
    }
    @keyframes cell-scan {
        0%,100% { background: rgba(255,176,0,0.06); }
        50%     { background: rgba(255,176,0,0.45); box-shadow: 0 0 6px rgba(255,176,0,0.3); }
    }

    /* ── Causal Stage: DAG constellation ── */
    .causal-dag {
        width: 260px; height: 80px;
        margin: 0.8rem auto;
        position: relative;
    }
    .dag-node {
        position: absolute;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #B388FF;
        box-shadow: 0 0 8px rgba(179,136,255,0.6);
        animation: dag-pulse 1.8s ease-in-out infinite;
    }
    .dag-edge {
        position: absolute;
        height: 1.5px;
        background: linear-gradient(90deg, rgba(179,136,255,0.1), rgba(179,136,255,0.7), rgba(179,136,255,0.1));
        transform-origin: left center;
        animation: edge-glow 2s ease-in-out infinite;
    }
    @keyframes dag-pulse {
        0%,100% { transform:scale(1); box-shadow:0 0 6px rgba(179,136,255,0.4); }
        50%     { transform:scale(1.5); box-shadow:0 0 14px rgba(179,136,255,0.8); }
    }
    @keyframes edge-glow {
        0%,100% { opacity:0.3; }
        50%     { opacity:1; }
    }

    /* ── Modeling/NUTS: MCMC chain trace ── */
    .mcmc-trace {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 70px;
        margin: 0.8rem auto;
        gap: 1px;
    }
    .mcmc-bar {
        width: 3px;
        border-radius: 1px;
        background: #FF00FF;
        animation: mcmc-walk 1.2s ease-in-out infinite alternate;
    }
    @keyframes mcmc-walk {
        0%   { height: 10px; opacity:0.3; background:#8B00FF; }
        50%  { height: 60px; opacity:1;   background:#FF00FF; }
        100% { height: 25px; opacity:0.5; background:#DA70D6; }
    }

    /* ── Simulation: Probability wave ── */
    .sim-wave {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 70px;
        margin: 0.8rem auto;
        gap: 2px;
    }
    .sim-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00FFFF;
        animation: sim-bounce 1.5s ease-in-out infinite;
        box-shadow: 0 0 6px rgba(0,255,255,0.4);
    }
    @keyframes sim-bounce {
        0%,100% { transform:translateY(0);    opacity:0.3; }
        50%     { transform:translateY(-28px); opacity:1; }
    }

    /* ── Strategy: Brain pulse rings ── */
    .brain-pulse {
        width: 80px; height: 80px;
        margin: 0.8rem auto;
        position: relative;
    }
    .brain-ring {
        position: absolute;
        border-radius: 50%;
        border: 1.5px solid #FF6EC7;
        animation: ring-expand 2s ease-out infinite;
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
    }
    .brain-core {
        position: absolute;
        width: 12px; height: 12px;
        background: #FF6EC7;
        border-radius: 50%;
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        box-shadow: 0 0 15px #FF6EC7;
        animation: core-glow 1.5s ease-in-out infinite;
    }
    @keyframes ring-expand {
        0%   { width:12px; height:12px; opacity:1; }
        100% { width:80px; height:80px; opacity:0; }
    }
    @keyframes core-glow {
        0%,100% { box-shadow:0 0 10px #FF6EC7; }
        50%     { box-shadow:0 0 25px #FF6EC7, 0 0 50px rgba(255,110,199,0.3); }
    }

    /* ── Complete: Checkmark ── */
    .check-anim {
        width: 60px; height: 60px;
        margin: 0.8rem auto;
        border-radius: 50%;
        border: 2px solid #39FF14;
        position: relative;
        animation: check-pop 0.5s ease-out;
    }
    .check-anim::after {
        content: '';
        position: absolute;
        width: 18px; height: 30px;
        border-right: 3px solid #39FF14;
        border-bottom: 3px solid #39FF14;
        top: 10px; left: 19px;
        transform: rotate(45deg);
        animation: check-draw 0.4s ease-out 0.3s both;
    }
    @keyframes check-pop  { 0%{transform:scale(0);} 100%{transform:scale(1);} }
    @keyframes check-draw  { 0%{opacity:0;} 100%{opacity:1;} }

    /* ── Error: X mark ── */
    .error-x {
        width: 60px; height: 60px;
        margin: 0.8rem auto;
        border-radius: 50%;
        border: 2px solid #FF3131;
        position: relative;
    }
    .error-x::before, .error-x::after {
        content: '';
        position: absolute;
        width: 3px; height: 30px;
        background: #FF3131;
        top: 14px; left: 28px;
    }
    .error-x::before { transform: rotate(45deg); }
    .error-x::after  { transform: rotate(-45deg); }

    /* Log terminal */
    .log-terminal {
        background: rgba(10,10,10,0.9);
        border: 1px solid rgba(57,255,20,0.1);
        margin: 0.5rem auto;
        max-width: 720px;
        font-family: 'Share Tech Mono', monospace;
        overflow: hidden;
    }
    .log-terminal .log-hdr {
        padding: 0.35rem 1rem;
        font-size: 0.6rem;
        color: #507050;
        letter-spacing: 2px;
        border-bottom: 1px solid rgba(57,255,20,0.06);
        text-transform: uppercase;
    }
    .log-terminal .log-body {
        padding: 0.7rem 1rem;
        margin: 0;
        font-size: 0.72rem;
        color: #39FF14;
        line-height: 1.7;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .log-terminal .log-body .log-cursor {
        animation: blink 1s step-end infinite;
    }

    /* Full log viewer */
    .syslog-wrap {
        background: rgba(5,5,5,0.95);
        border: 1px solid rgba(57,255,20,0.12);
        font-family: 'Share Tech Mono', monospace;
        padding: 0;
        max-height: 75vh;
        overflow-y: auto;
    }
    .syslog-wrap::-webkit-scrollbar { width: 6px; }
    .syslog-wrap::-webkit-scrollbar-track { background: #0a0a0a; }
    .syslog-wrap::-webkit-scrollbar-thumb { background: rgba(57,255,20,0.25); }
    .syslog-stage-hdr {
        position: sticky; top: 0;
        padding: 0.55rem 1rem;
        font-size: 0.7rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .syslog-stage-hdr .hdr-line {
        flex: 1;
        height: 1px;
        opacity: 0.25;
    }
    .syslog-section {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(57,255,20,0.04);
    }
    .syslog-line {
        padding: 0.15rem 1.2rem;
        font-size: 0.7rem;
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .syslog-line.sub {
        padding-left: 2rem;
        opacity: 0.82;
    }
    .syslog-line.sub2 {
        padding-left: 2.8rem;
        opacity: 0.7;
    }
    .syslog-line .val {
        color: #FFB000;
        font-weight: bold;
    }
    .syslog-line .ok  { color: #39FF14; }
    .syslog-line .wrn { color: #FFB000; }
    .syslog-line .err { color: #FF5252; }
    .syslog-summary {
        padding: 0.6rem 1rem;
        font-size: 0.65rem;
        color: #507050;
        letter-spacing: 2px;
        border-top: 1px solid rgba(57,255,20,0.08);
    }

    /* Idle radar */
    .idle-box {
        text-align: center;
        padding: 3rem 2rem;
        margin: 2rem auto;
        max-width: 600px;
        border: 1px solid rgba(57,255,20,0.1);
        background: rgba(10,10,10,0.5);
        position: relative;
        overflow: hidden;
    }
    .idle-box::before {
        content: '';
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(0deg, transparent 0px, transparent 3px,
            rgba(57,255,20,0.015) 3px, rgba(57,255,20,0.015) 4px);
        pointer-events: none;
    }
    .idle-radar {
        width: 90px; height: 90px;
        border-radius: 50%;
        border: 1px solid rgba(57,255,20,0.2);
        margin: 0 auto 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .idle-radar::before {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 50%; height: 2px;
        background: linear-gradient(90deg, #39FF14, transparent);
        transform-origin: left center;
        animation: radar 2.5s linear infinite;
    }
    .idle-radar::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #39FF14;
        box-shadow: 0 0 8px #39FF14;
    }
    @keyframes radar { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
    .idle-title {
        font-family: 'Press Start 2P', monospace;
        font-size: 0.6rem;
        color: #39FF14;
        letter-spacing: 2px;
        margin-bottom: 0.7rem;
        animation: blink 2.5s ease-in-out infinite;
    }
    .idle-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #507050;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "logs" not in st.session_state:
    st.session_state.logs = []


def metric_card(label, value, color_class=""):
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value {color_class}">{value}</div></div>'


def section_head(text):
    st.markdown(f'<div class="section-head">{text}</div>', unsafe_allow_html=True)


def apply_chart_theme(fig, height=380):
    fig.update_layout(**CHART_TEMPLATE, height=height)
    return fig


_STAGE_TO_PHASE = {
    "data_stage": "data",
    "causal_stage": "causal",
    "tuning_stage": "tuning",
    "training_stage": "training",
    "simulation_stage": "simulation",
    "forecasting_stage": "forecasting",
    "strategy_stage": "strategy",
}


def _anim_html(phase: str, detail: str = "") -> str:
    import random as _rng

    CFG = {
        "init":       ("SYSTEM BOOT",             "#39FF14", "Initializing pipeline..."),
        "data":       ("DATA STAGE",               "#39FF14", "Ingesting & transforming channels..."),
        "causal":     ("CAUSAL DISCOVERY",         "#B388FF", "Building DAG · DoWhy causal inference · deconfounding"),
        "tuning":     ("HYPERPARAMETER TUNING",    "#FFB000", "Optuna TPE · 100 trials · 5-fold CV R² scoring"),
        "training":   ("MODEL TRAINING",           "#FF00FF", "PyMC NUTS · 2 chains × 1000 draws + 500 tuning"),
        "simulation": ("SIMULATION STAGE",         "#00FFFF", "Running counterfactual simulations..."),
        "forecasting": ("FORECASTING",              "#FFD700", "STL decomposition · trend extraction · prediction intervals"),
        "strategy":   ("STRATEGY STAGE",           "#FF6EC7", "Generating AI recommendations..."),
        "complete":   ("PIPELINE COMPLETE",        "#39FF14", "All systems nominal"),
        "error":      ("SYSTEM ERROR",             "#FF3131", "Fault detected"),
    }
    title, color, desc = CFG.get(phase, ("PROCESSING", "#39FF14", "Working..."))
    if detail:
        desc = detail

    # ── Per-phase visual ──
    if phase == "data":
        # Matrix-style falling data columns
        chars = "01╳◆▪═▐░▒▓"
        cols_html = ""
        for i in range(18):
            spans = "".join(f"<span>{_rng.choice(chars)}</span>" for _ in range(5))
            dur = round(0.8 + _rng.random() * 0.8, 2)
            delay = round(_rng.random() * 0.6, 2)
            cols_html += (f'<div class="matrix-col" '
                          f'style="--fall-dur:{dur}s;animation-delay:{delay}s">{spans}</div>')
        visual = f'<div class="matrix-rain">{cols_html}</div>'

    elif phase == "causal":
        # DAG constellation — nodes with connecting edges
        import math
        node_positions = [
            (20, 10), (70, 10), (130, 10), (190, 10),   # top: channels
            (40, 70), (160, 70),                          # bottom-left/right
            (230, 40),                                    # right: sales
        ]
        nodes_html = ""
        edges_html = ""
        for idx, (x, y) in enumerate(node_positions):
            delay = round(idx * 0.25, 2)
            nodes_html += (f'<div class="dag-node" '
                           f'style="left:{x}px;top:{y}px;animation-delay:{delay}s"></div>')
        # Draw edges from channels to sales node
        sales_x, sales_y = 230, 40
        for i in range(4):
            sx, sy = node_positions[i]
            dx, dy = sales_x - sx, sales_y - sy
            length = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx))
            delay = round(i * 0.3, 2)
            edges_html += (f'<div class="dag-edge" '
                           f'style="left:{sx+5}px;top:{sy+5}px;'
                           f'width:{length:.0f}px;'
                           f'transform:rotate({angle:.1f}deg);'
                           f'animation-delay:{delay}s"></div>')
        # Confounder edges from bottom nodes
        for i, ci in [(4, 0), (4, 1), (5, 2), (5, 3)]:
            sx, sy = node_positions[i]
            tx, ty = node_positions[ci]
            dx, dy = tx - sx, ty - sy
            length = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx))
            delay = round(0.8 + i * 0.15, 2)
            edges_html += (f'<div class="dag-edge" '
                           f'style="left:{sx+5}px;top:{sy+5}px;'
                           f'width:{length:.0f}px;'
                           f'transform:rotate({angle:.1f}deg);'
                           f'animation-delay:{delay}s;'
                           f'background:linear-gradient(90deg,rgba(255,176,0,0.1),rgba(255,176,0,0.5),rgba(255,176,0,0.1))"></div>')
        visual = f'<div class="causal-dag">{nodes_html}{edges_html}</div>'

    elif phase == "tuning":
        # Grid search scan — cells light up with staggered delays
        cells = ""
        for i in range(50):
            delay = round((i % 10) * 0.25 + (i // 10) * 0.15, 2)
            cells += f'<div class="grid-cell" style="animation-delay:{delay}s"></div>'
        visual = f'<div class="grid-scan">{cells}</div>'

    elif phase == "training":
        # MCMC chain trace — bars that randomly walk
        bars = ""
        for i in range(50):
            delay = round(i * 0.04, 2)
            h = _rng.randint(15, 55)
            bars += (f'<div class="mcmc-bar" '
                     f'style="animation-delay:{delay}s;height:{h}px"></div>')
        visual = f'<div class="mcmc-trace">{bars}</div>'

    elif phase == "simulation":
        # Bouncing probability dots
        dots = ""
        for i in range(16):
            delay = round(i * 0.1, 2)
            dots += f'<div class="sim-dot" style="animation-delay:{delay}s"></div>'
        visual = f'<div class="sim-wave">{dots}</div>'

    elif phase == "forecasting":
        # Rising trend line with expanding prediction bands
        bars = ""
        for i in range(30):
            delay = round(i * 0.06, 2)
            h = 15 + int(i * 1.3) + _rng.randint(-3, 3)
            bars += (f'<div class="mcmc-bar" '
                     f'style="animation-delay:{delay}s;height:{h}px;'
                     f'background:linear-gradient(180deg,#FFD700,#FF8C00)"></div>')
        visual = f'<div class="mcmc-trace">{bars}</div>'

    elif phase == "strategy":
        # Expanding brain-pulse rings
        rings = ""
        for i in range(4):
            delay = round(i * 0.5, 1)
            rings += f'<div class="brain-ring" style="animation-delay:{delay}s"></div>'
        visual = f'<div class="brain-pulse">{rings}<div class="brain-core"></div></div>'

    elif phase == "complete":
        visual = '<div class="check-anim"></div>'

    elif phase == "error":
        visual = '<div class="error-x"></div>'

    else:
        # init / generic — small matrix rain
        chars = "◆▪░▒▓"
        cols_html = ""
        for i in range(12):
            spans = "".join(f"<span>{_rng.choice(chars)}</span>" for _ in range(4))
            dur = round(1.0 + _rng.random() * 0.5, 2)
            cols_html += f'<div class="matrix-col" style="--fall-dur:{dur}s">{spans}</div>'
        visual = f'<div class="matrix-rain">{cols_html}</div>'

    cls = "pipeline-anim"
    if phase == "complete":
        cls += " done"
    elif phase == "error":
        cls += " error-phase"

    return (
        f'<div class="{cls}" style="--phase-color:{color}">'
        f'<div class="anim-scan"></div>'
        f'<div class="anim-title" style="color:{color};text-shadow:0 0 12px {color}66;">[ {title} ]</div>'
        f'{visual}'
        f'<div class="anim-desc">{desc}</div>'
        f'</div>'
    )


def _log_html(stage_logs: list[str], total: int, max_lines: int = 14, phase_label: str = "") -> str:
    recent = stage_logs[-max_lines:]
    body = "\n".join(_esc(l) for l in recent)
    tag = _esc(phase_label).upper() if phase_label else "SYSTEM"
    return (
        '<div class="log-terminal">'
        f'<div class="log-hdr">// {tag} LOG · {len(stage_logs)} entries (total: {total})</div>'
        f'<pre class="log-body">{body}<span class="log-cursor">█</span></pre>'
        '</div>'
    )


# ── Hero Header ─────────────────────────────────────────────────
st.markdown("""
<div class="retro-hero">
    <h1>MARKET MIX MODELING PIPELINE</h1>
    <div class="subtitle">[ BAYESIAN MMM // BUDGET OPTIMIZATION // AI STRATEGY ]<span class="blink-cursor"></span></div>
</div>
""", unsafe_allow_html=True)


# ── Control Bar ─────────────────────────────────────────────────
with st.container():
    col_upload, col_budget, col_run = st.columns([3, 1.5, 1.5])

    with col_upload:
        uploaded_file = st.file_uploader("LOAD DATASET", type=["csv"])

    with col_budget:
        total_budget = st.number_input(
            "WEEKLY BUDGET (₹)", min_value=5000, max_value=500000,
            value=35000, step=1000,
        )

    with col_run:
        st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
        run_btn = st.button(">> EXECUTE PIPELINE", type="primary", use_container_width=True)

st.markdown('<div class="retro-divider"></div>', unsafe_allow_html=True)


# ── Run Pipeline ────────────────────────────────────────────────
if run_btn:
    st.session_state.results = None
    st.session_state.logs = []

    anim_slot = st.empty()
    log_slot = st.empty()
    anim_slot.markdown(_anim_html("init"), unsafe_allow_html=True)

    current_phase = "init"
    try:
        if uploaded_file is not None:
            resp = requests.post(
                f"{API_URL}/api/pipeline/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                data={"total_budget": total_budget},
                stream=True, timeout=600,
            )
        else:
            resp = requests.post(
                f"{API_URL}/api/pipeline/run",
                json={"dataset_path": "mmm_dataset.csv", "total_budget": total_budget},
                stream=True, timeout=600,
            )
        all_logs: list[str] = []     # every log across all stages
        stage_logs: list[str] = []   # logs for the *current* stage only
        phase_label = "init"

        for line in resp.iter_lines():
            if line:
                event = json.loads(line.decode())
                if event["type"] == "phase":
                    # New stage starting — clear stage-local logs
                    phase = _STAGE_TO_PHASE.get(event["stage"], current_phase)
                    if phase != current_phase:
                        current_phase = phase
                        phase_label = current_phase
                        stage_logs = []  # fresh log pane for this stage
                        anim_slot.markdown(_anim_html(current_phase), unsafe_allow_html=True)
                        log_slot.markdown(
                            _log_html(stage_logs, len(all_logs), phase_label=phase_label),
                            unsafe_allow_html=True,
                        )
                elif event["type"] == "log":
                    msg = event["message"]
                    # Append to both global + stage-local lists
                    all_logs.append(msg)
                    stage_logs.append(msg)
                    log_slot.markdown(
                        _log_html(stage_logs, len(all_logs), phase_label=phase_label),
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.15)  # drip-feed one by one
                elif event["type"] == "result":
                    st.session_state.results = event["data"]
        st.session_state.logs = all_logs
        anim_slot.markdown(_anim_html("complete"), unsafe_allow_html=True)
        time.sleep(1.5)
    except requests.ConnectionError:
        anim_slot.markdown(_anim_html("error", "Backend unreachable"), unsafe_allow_html=True)
        log_slot.empty()
        st.error("Start backend: `uv run uvicorn backend.app:app --port 8000`")
    except Exception as e:
        anim_slot.markdown(_anim_html("error", str(e)[:120]), unsafe_allow_html=True)
        log_slot.empty()
        st.error(str(e))
    st.rerun()


# ── Guard ───────────────────────────────────────────────────────
if not st.session_state.results:
    st.markdown("""
    <div class="idle-box">
        <div class="idle-radar"></div>
        <div class="idle-title">AWAITING INPUT</div>
        <div class="idle-sub">Upload a CSV dataset and execute the pipeline</div>
        <div class="idle-sub" style="margin-top:0.4rem;color:#3a5a3a;">Or run with default dataset to initialize</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

R = st.session_state.results


# ── Model Performance Banner ───────────────────────────────────
m = R["model_metrics"]
st.markdown(
    '<div class="metric-row">'
    + metric_card("R² SCORE", f"{m['r2']:.4f}", "")
    + metric_card("MAPE", f"{m['mape']:.2f}%", "amber")
    + metric_card("RMSE", f"₹{m['rmse']:,.0f}", "cyan")
    + '</div>',
    unsafe_allow_html=True,
)


# ── Tabs ────────────────────────────────────────────────────────
tab_overview, tab_causal, tab_kpi, tab_model, tab_scenarios, tab_strategy, tab_logs = st.tabs([
    "OVERVIEW", "CAUSAL", "KPIs", "MODEL // ROI", "SCENARIOS", "STRATEGY", "SYSTEM LOG",
])


# ================================================================
#  TAB 1 — Dataset Overview
# ================================================================
with tab_overview:
    summary = R["data_summary"]
    raw_df = pd.DataFrame(R["raw_data"])
    spend_cols = R["spend_columns"]

    st.markdown(
        '<div class="metric-row">'
        + metric_card("TOTAL WEEKS", summary["total_weeks"], "")
        + metric_card("CHANNELS", summary["channels"], "amber")
        + metric_card("AVG WEEKLY SALES", f"₹{summary['avg_sales']:,.0f}", "cyan")
        + metric_card("TOTAL SALES", f"₹{summary['total_sales']:,.0f}", "magenta")
        + '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("RAW DATA DUMP", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    # sales trend
    section_head("SALES SIGNAL")
    fig = px.line(raw_df, x="week", y="sales", title="SALES SIGNAL")
    fig.update_traces(line_color="#39FF14", line_width=2)
    fig.update_traces(fill="tozeroy", fillcolor="rgba(57,255,20,0.05)")
    apply_chart_theme(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

    # spend over time
    section_head("CHANNEL SPEND TRACE")
    fig = go.Figure()
    for i, col in enumerate(spend_cols):
        fig.add_trace(go.Scatter(
            x=raw_df["week"], y=raw_df[col], mode="lines",
            name=col.replace("_", " ").upper(),
            line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], width=2),
        ))
    fig.update_layout(title="CHANNEL SPEND OVER TIME", xaxis_title="WEEK", yaxis_title="SPEND (₹)")
    apply_chart_theme(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

    # spend distribution
    left, right = st.columns(2)
    with left:
        totals = {c.replace("_", " ").upper(): raw_df[c].sum() for c in spend_cols}
        fig = px.pie(names=list(totals.keys()), values=list(totals.values()),
                     title="SPEND DISTRIBUTION", color_discrete_sequence=CHART_PALETTE,
                     hole=0.55)
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        avgs = {c.replace("_", " ").upper(): raw_df[c].mean() for c in spend_cols}
        fig = px.bar(x=list(avgs.keys()), y=list(avgs.values()),
                     title="AVG WEEKLY SPEND", labels={"x": "CHANNEL", "y": "₹"},
                     color=list(avgs.keys()), color_discrete_sequence=CHART_PALETTE)
        fig.update_layout(showlegend=False)
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # correlation heatmap
    section_head("CORRELATION MATRIX")
    corr_cols = spend_cols + ["discount", "seasonality", "sales"]
    corr = raw_df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", title="CORRELATION MATRIX",
                    color_continuous_scale=[[0, "#0A0A0A"], [0.25, "#003300"], [0.5, "#006600"], [0.75, "#39FF14"], [1, "#FFB000"]],
                    zmin=-1, zmax=1, aspect="auto")
    apply_chart_theme(fig, 450)
    st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  TAB 1.5 — Causal Discovery
# ================================================================
with tab_causal:
    causal = R.get("causal_summary", {})
    if causal:
        section_head("CAUSAL DAG")

        dag_edges = causal.get("dag_edges", [])
        dag_nodes = causal.get("dag_nodes", [])
        causal_effects = causal.get("causal_effects", {})
        confounders = causal.get("confounders", {})

        st.markdown(
            '<div class="metric-row">'
            + metric_card("DAG NODES", len(dag_nodes), "")
            + metric_card("DAG EDGES", causal.get("total_edges", len(dag_edges)), "amber")
            + metric_card("VALIDATED EDGES", causal.get("validated_edges", 0), "cyan")
            + metric_card("CONFOUNDERS", len(confounders), "magenta")
            + '</div>',
            unsafe_allow_html=True,
        )

        # DAG edge list
        with st.expander("CAUSAL GRAPH EDGES", expanded=False):
            for edge in dag_edges:
                st.markdown(
                    f'<span style="color:#B388FF;font-family:\'Share Tech Mono\',monospace;">'
                    f'{edge["source"]} → {edge["target"]}</span>',
                    unsafe_allow_html=True,
                )

        # Causal effects bar chart
        section_head("AVERAGE TREATMENT EFFECTS")
        if causal_effects:
            ef_df = pd.DataFrame([
                {"channel": k, "ATE": v}
                for k, v in sorted(causal_effects.items(), key=lambda x: abs(x[1]), reverse=True)
            ])
            fig = px.bar(ef_df, x="channel", y="ATE",
                         title="CAUSAL EFFECT (ATE) PER CHANNEL",
                         color="ATE",
                         color_continuous_scale=[[0, "#3A005A"], [0.5, "#B388FF"], [1, "#FFB000"]])
            apply_chart_theme(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

        # Confounders table
        if confounders:
            section_head("IDENTIFIED CONFOUNDERS")
            for ch, conf_list in confounders.items():
                st.markdown(
                    f'<span style="color:#FFB000;font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;">'
                    f'⚠ {ch}: confounded by {", ".join(conf_list)}</span>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Causal discovery results will appear after pipeline execution.")


# ================================================================
#  TAB 1.6 — KPIs
# ================================================================
with tab_kpi:
    raw_df_kpi = pd.DataFrame(R["raw_data"])
    spend_cols_kpi = R["spend_columns"]
    metrics_kpi = R["model_metrics"]
    roi_kpi = R["roi_per_channel"]
    contrib_kpi = R["channel_contributions"]
    opt_summary = R.get("optimization_summary", {})
    forecast_kpi = R.get("forecast", [])

    # ── Totals ──────────────────────────────────────────────────
    total_sales = float(raw_df_kpi["sales"].sum())
    total_spend = sum(float(raw_df_kpi[c].sum()) for c in spend_cols_kpi)
    overall_roi = total_sales / total_spend if total_spend > 0 else 0
    cost_per_sale = total_spend / total_sales if total_sales > 0 else 0
    avg_weekly_sales = float(raw_df_kpi["sales"].mean())
    total_weeks = len(raw_df_kpi)

    section_head("FINANCIAL KPIs")
    st.markdown(
        '<div class="metric-row">'
        + metric_card("TOTAL REVENUE", f"₹{total_sales:,.0f}", "")
        + metric_card("TOTAL AD SPEND", f"₹{total_spend:,.0f}", "amber")
        + metric_card("OVERALL ROI", f"{overall_roi:.2f}×", "cyan")
        + metric_card("COST / ₹ REVENUE", f"₹{cost_per_sale:.3f}", "magenta")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Per-week averages ───────────────────────────────────────
    avg_weekly_spend = total_spend / total_weeks if total_weeks > 0 else 0
    spend_to_sales_pct = (total_spend / total_sales * 100) if total_sales > 0 else 0

    st.markdown(
        '<div class="metric-row">'
        + metric_card("AVG WEEKLY SALES", f"₹{avg_weekly_sales:,.0f}", "")
        + metric_card("AVG WEEKLY SPEND", f"₹{avg_weekly_spend:,.0f}", "amber")
        + metric_card("AD SPEND / REVENUE", f"{spend_to_sales_pct:.1f}%", "cyan")
        + metric_card("DATASET WEEKS", str(total_weeks), "magenta")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Model quality KPIs ──────────────────────────────────────
    section_head("MODEL QUALITY")
    st.markdown(
        '<div class="metric-row">'
        + metric_card("R² SCORE", f"{metrics_kpi['r2']:.4f}", "")
        + metric_card("MAPE", f"{metrics_kpi['mape']:.2f}%", "amber")
        + metric_card("RMSE", f"₹{metrics_kpi['rmse']:,.0f}", "cyan")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Channel ROI ranking ─────────────────────────────────────
    section_head("CHANNEL PERFORMANCE")
    roi_sorted = sorted(roi_kpi, key=lambda x: x["roi"], reverse=True)
    best_ch = roi_sorted[0] if roi_sorted else None
    worst_ch = roi_sorted[-1] if roi_sorted else None

    if best_ch and worst_ch:
        st.markdown(
            '<div class="metric-row">'
            + metric_card("BEST ROI CHANNEL", f"{best_ch['channel'].replace('_',' ').upper()} ({best_ch['roi']:.2f}×)", "")
            + metric_card("LOWEST ROI CHANNEL", f"{worst_ch['channel'].replace('_',' ').upper()} ({worst_ch['roi']:.2f}×)", "amber")
            + '</div>',
            unsafe_allow_html=True,
        )

    # ROI bar chart
    roi_df_kpi = pd.DataFrame(roi_sorted)
    fig = px.bar(roi_df_kpi, x="channel", y="roi",
                 title="ROI BY CHANNEL", color="roi",
                 color_continuous_scale=[[0, "#003300"], [0.5, "#39FF14"], [1, "#FFB000"]])
    apply_chart_theme(fig, 360)
    st.plotly_chart(fig, use_container_width=True, key="kpi_roi_bar")

    # ── Channel contribution breakdown ──────────────────────────
    section_head("CHANNEL CONTRIBUTION BREAKDOWN")
    contrib_sorted = sorted(contrib_kpi, key=lambda x: x["contribution_pct"], reverse=True)
    top_ch = contrib_sorted[0] if contrib_sorted else None
    if top_ch:
        st.markdown(
            '<div class="metric-row">'
            + metric_card("TOP CONTRIBUTOR", f"{top_ch['channel'].replace('_',' ').upper()}", "")
            + metric_card("CONTRIBUTION", f"{top_ch['contribution_pct']:.1f}%", "amber")
            + '</div>',
            unsafe_allow_html=True,
        )

    contrib_df_kpi = pd.DataFrame(contrib_sorted)
    fig = px.pie(contrib_df_kpi, names="channel", values="contribution_pct",
                 title="CONTRIBUTION SPLIT", color_discrete_sequence=CHART_PALETTE, hole=0.55)
    apply_chart_theme(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="kpi_contrib_pie")

    # ── Spend efficiency table ──────────────────────────────────
    section_head("SPEND EFFICIENCY TABLE")
    eff_rows = []
    for r in roi_sorted:
        eff_rows.append({
            "Channel": r["channel"].replace("_", " ").upper(),
            "Total Spend (₹)": f"₹{r['total_spend']:,.0f}",
            "Attributed Sales (₹)": f"₹{r['attributed_sales']:,.0f}",
            "ROI": f"{r['roi']:.4f}×",
            "Spend Share %": f"{r['total_spend'] / total_spend * 100:.1f}%" if total_spend > 0 else "—",
        })
    st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, hide_index=True)

    # ── Optimisation uplift (if available) ──────────────────────
    if opt_summary:
        section_head("OPTIMISATION POTENTIAL")
        uplift = opt_summary.get("uplift", 0)
        opt_pred = opt_summary.get("predicted_sales", 0)
        st.markdown(
            '<div class="metric-row">'
            + metric_card("OPTIMISED SALES", f"₹{opt_pred:,.0f}", "")
            + metric_card("PREDICTED UPLIFT", f"{uplift:+.1f}%", "cyan" if uplift >= 0 else "red")
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── Forecast summary (if available) ─────────────────────────
    if forecast_kpi:
        section_head("12-WEEK FORECAST SUMMARY")
        fc_df = pd.DataFrame(forecast_kpi)
        fc_total = float(fc_df["predicted_sales"].sum())
        fc_avg = float(fc_df["predicted_sales"].mean())
        fc_min = float(fc_df["predicted_sales"].min())
        fc_max = float(fc_df["predicted_sales"].max())
        st.markdown(
            '<div class="metric-row">'
            + metric_card("FORECAST TOTAL", f"₹{fc_total:,.0f}", "")
            + metric_card("AVG WEEKLY", f"₹{fc_avg:,.0f}", "amber")
            + metric_card("MIN WEEK", f"₹{fc_min:,.0f}", "cyan")
            + metric_card("MAX WEEK", f"₹{fc_max:,.0f}", "magenta")
            + '</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        # 95% band
        if "lower_95" in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=list(fc_df["week"]) + list(fc_df["week"][::-1]),
                y=list(fc_df["upper_95"]) + list(fc_df["lower_95"][::-1]),
                fill="toself", fillcolor="rgba(0,255,255,0.06)",
                line=dict(width=0), name="95% INTERVAL", showlegend=True))
            fig.add_trace(go.Scatter(
                x=list(fc_df["week"]) + list(fc_df["week"][::-1]),
                y=list(fc_df["upper_80"]) + list(fc_df["lower_80"][::-1]),
                fill="toself", fillcolor="rgba(0,255,255,0.12)",
                line=dict(width=0), name="80% INTERVAL", showlegend=True))
        fig.add_trace(go.Scatter(
            x=fc_df["week"], y=fc_df["predicted_sales"],
            name="POINT FORECAST", line=dict(color="#00FFFF", width=2)))
        fig.update_layout(title="12-WEEK SALES FORECAST", xaxis_title="WEEK", yaxis_title="SALES (₹)")
        apply_chart_theme(fig, 340)
        st.plotly_chart(fig, use_container_width=True, key="kpi_forecast_line")

        # Decomposition details
        fc_decomp = R.get("forecast_decomposition", {})
        if fc_decomp:
            st.markdown(
                '<div class="metric-row">'
                + metric_card("TREND SLOPE", f"{fc_decomp.get('trend_slope', 0):+.4f}/wk", "")
                + metric_card("SEASONAL PERIOD", f"{fc_decomp.get('seasonal_period', '?')} wks", "amber")
                + metric_card("RESIDUAL σ", f"₹{fc_decomp.get('residual_std', 0):,.0f}", "cyan")
                + '</div>',
                unsafe_allow_html=True,
            )


# ================================================================
#  TAB 2 — Model & ROI
# ================================================================
with tab_model:
    metrics = R["model_metrics"]

    section_head("MODEL DIAGNOSTICS")
    st.markdown(
        '<div class="metric-row">'
        + metric_card("R² SCORE", f"{metrics['r2']:.4f}", "")
        + metric_card("MAPE", f"{metrics['mape']:.2f}%", "amber")
        + metric_card("RMSE", f"₹{metrics['rmse']:,.0f}", "cyan")
        + '</div>',
        unsafe_allow_html=True,
    )

    # actual vs predicted
    section_head("ACTUAL vs PREDICTED")
    actual = R["actual_sales"]
    predicted = R["predicted_sales"]
    weeks = list(range(1, len(actual) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=actual, name="ACTUAL",
                             line=dict(color="#39FF14", width=2),
                             fill="tozeroy", fillcolor="rgba(57,255,20,0.04)"))
    fig.add_trace(go.Scatter(x=weeks, y=predicted, name="PREDICTED",
                             line=dict(color="#FFB000", width=2, dash="dash")))
    fig.update_layout(title="ACTUAL vs PREDICTED SALES", xaxis_title="WEEK", yaxis_title="SALES (₹)")
    apply_chart_theme(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

    # residuals
    residuals = [a - p for a, p in zip(actual, predicted)]
    fig = px.scatter(x=predicted, y=residuals, labels={"x": "PREDICTED", "y": "RESIDUAL"},
                     title="RESIDUAL SCATTER")
    fig.update_traces(marker=dict(color="#00FFFF", size=7, opacity=0.7,
                                  line=dict(width=1, color="rgba(0,255,255,0.3)")))
    fig.add_hline(y=0, line_dash="dash", line_color="#FF3131", line_width=1.5)
    apply_chart_theme(fig, 330)
    st.plotly_chart(fig, use_container_width=True)

    # channel contributions
    section_head("CHANNEL CONTRIBUTIONS")
    contrib = pd.DataFrame(R["channel_contributions"]).sort_values("contribution_pct", ascending=False)
    left, right = st.columns(2)
    with left:
        fig = px.bar(contrib, x="channel", y="contribution_pct",
                     title="CONTRIBUTION %", labels={"contribution_pct": "%"},
                     color="channel", color_discrete_sequence=CHART_PALETTE)
        fig.update_layout(showlegend=False)
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(contrib, names="channel", values="contribution_pct",
                     title="CONTRIBUTION SPLIT", color_discrete_sequence=CHART_PALETTE, hole=0.55)
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # ROI
    section_head("RETURN ON INVESTMENT")
    roi_df = pd.DataFrame(R["roi_per_channel"]).sort_values("roi", ascending=False)
    fig = px.bar(roi_df, x="channel", y="roi", color="roi",
                 color_continuous_scale=[[0, "#003300"], [0.5, "#39FF14"], [1, "#FFB000"]],
                 title="ROI BY CHANNEL")
    apply_chart_theme(fig, 380)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    # response curves
    if R.get("response_curves"):
        section_head("RESPONSE CURVES // DIMINISHING RETURNS")
        curves = R["response_curves"]
        cols = st.columns(2)
        for idx, (ch, data) in enumerate(curves.items()):
            with cols[idx % 2]:
                fig = px.line(x=data["spend"], y=data["predicted_sales"],
                              labels={"x": "SPEND (₹)", "y": "PREDICTED SALES (₹)"},
                              title=ch.replace("_", " ").upper())
                fig.update_traces(line_color=CHART_PALETTE[idx % len(CHART_PALETTE)], line_width=2)
                fig.update_traces(fill="tozeroy",
                                  fillcolor=f"rgba({','.join(str(int(CHART_PALETTE[idx % len(CHART_PALETTE)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.04)")
                apply_chart_theme(fig, 310)
                st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  TAB 3 — Scenarios & Optimisation
# ================================================================
with tab_scenarios:
    sim = R["simulation_results"]
    spend_cols = R["spend_columns"]
    raw_df = pd.DataFrame(R["raw_data"])
    baseline_row = sim[0] if sim else None
    baseline_sales = baseline_row["predicted_sales"] if baseline_row else 0

    # ── Interactive counterfactual controls ──────────────────────
    section_head("COUNTERFACTUAL SCENARIOS")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;color:#507050;'
        'font-size:0.78rem;margin-bottom:1rem;letter-spacing:1px;">'
        'Adjust channel spend multipliers to define scenarios. '
        '1.0 = current spend, 0.0 = shut off, 2.0 = double.</div>',
        unsafe_allow_html=True,
    )

    # Pre-built scenario presets the user can load
    PRESETS = {
        "Custom":           None,
        "Baseline":         {c: 1.0 for c in spend_cols},
        "Google Ads +30%":  dict(zip(spend_cols, [1.0, 1.3, 1.0, 1.0])),
        "TV Ads Stop":      dict(zip(spend_cols, [0.0, 1.0, 1.0, 1.0])),
        "Influencer 2×":    dict(zip(spend_cols, [1.0, 1.0, 1.0, 2.0])),
        "Meta Ads +50%":    dict(zip(spend_cols, [1.0, 1.0, 1.5, 1.0])),
        "All Digital +20%": dict(zip(spend_cols, [1.0, 1.2, 1.2, 1.2])),
        "Cut All 25%":      {c: 0.75 for c in spend_cols},
    }

    preset_choice = st.selectbox(
        "LOAD PRESET", options=list(PRESETS.keys()), index=0,
        help="Select a preset or keep 'Custom' to set your own multipliers.",
    )

    preset_vals = PRESETS[preset_choice]
    slider_cols_ui = st.columns(len(spend_cols))
    user_multipliers = {}
    for i, col in enumerate(spend_cols):
        default = preset_vals[col] if preset_vals else 1.0
        user_multipliers[col] = slider_cols_ui[i].slider(
            col.replace("_", " ").upper(),
            min_value=0.0, max_value=3.0, value=default, step=0.05,
            format="%.2f×",
            key=f"cf_{col}",
        )

    # ── Client-side prediction of the user scenario ─────────────
    has_model = R.get("best_params") and R.get("model_coef")
    user_scenario_row = None

    if has_model:
        best_params = R["best_params"]
        coef = np.array(R["model_coef"])
        intercept = R["model_intercept"]
        X_mean = np.array(R["feature_mean"])
        X_std = np.array(R["feature_std"])
        y_mean = R["target_mean"]
        y_std = R["target_std"]
        disc_med = float(raw_df["discount"].median())
        seas_med = float(raw_df["seasonality"].median())

        def _predict_spend(spend_dict):
            feats = []
            for col in spend_cols:
                alpha = best_params[f"{col}_alpha"]
                feats.append(1 - np.exp(-alpha * spend_dict[col]))
            feats += [disc_med, seas_med]
            feats = np.array(feats)
            scaled = (feats - X_mean) / X_std
            return float((intercept + scaled @ coef) * y_std + y_mean)

        avg_spend = {c: raw_df[c].mean() for c in spend_cols}
        user_spend = {c: avg_spend[c] * user_multipliers[c] for c in spend_cols}
        user_pred = _predict_spend(user_spend)
        user_delta = user_pred - baseline_sales
        user_pct = (user_delta / baseline_sales * 100) if baseline_sales else 0.0

        user_scenario_row = {
            "scenario": "YOUR SCENARIO",
            "spend": {c: round(user_spend[c], 2) for c in spend_cols},
            "predicted_sales": round(user_pred, 2),
            "delta_sales": round(user_delta, 2),
            "delta_pct": round(user_pct, 2),
        }

    # ── Build final table: backend presets + user row ───────────
    display_rows = list(sim)  # backend presets
    if user_scenario_row:
        display_rows.append(user_scenario_row)

    sim_df = pd.DataFrame(display_rows)

    # Metric cards for user scenario
    if user_scenario_row:
        delta_color = "" if user_scenario_row["delta_pct"] >= 0 else "red"
        st.markdown(
            '<div class="metric-row">'
            + metric_card("YOUR PREDICTED SALES", f"₹{user_scenario_row['predicted_sales']:,.0f}", "amber")
            + metric_card("Δ vs BASELINE", f"{user_scenario_row['delta_pct']:+.1f}%", delta_color)
            + metric_card("TOTAL SPEND", f"₹{sum(user_scenario_row['spend'].values()):,.0f}", "cyan")
            + '</div>',
            unsafe_allow_html=True,
        )

    st.dataframe(
        sim_df[["scenario", "predicted_sales", "delta_sales", "delta_pct"]],
        use_container_width=True, hide_index=True,
    )

    # horizontal bar chart
    bar_colors = []
    for r in display_rows:
        if r["scenario"] == "Baseline":
            bar_colors.append(COLORS["dim"])
        elif r["scenario"] == "YOUR SCENARIO":
            bar_colors.append(COLORS["amber"])
        elif r["delta_sales"] >= 0:
            bar_colors.append(COLORS["green"])
        else:
            bar_colors.append(COLORS["red"])

    fig = go.Figure(go.Bar(
        y=[r["scenario"] for r in display_rows],
        x=[r["predicted_sales"] for r in display_rows],
        orientation="h", marker_color=bar_colors,
        text=[f"{r['delta_pct']:+.1f}%" for r in display_rows],
        textposition="outside",
        textfont=dict(color="#39FF14", family="'Share Tech Mono', monospace", size=11),
    ))
    fig.update_layout(title="SCENARIO COMPARISON", xaxis_title="PREDICTED SALES (₹)")
    apply_chart_theme(fig, max(420, len(display_rows) * 45))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="retro-divider"></div>', unsafe_allow_html=True)

    # ── budget optimisation ─────────────────────────────────────
    section_head("OPTIMAL BUDGET ALLOCATION")
    opt = R["optimal_allocation"]
    opt_s = R["optimization_summary"]

    uplift_color = "" if opt_s["uplift"] >= 0 else "red"
    st.markdown(
        '<div class="metric-row">'
        + metric_card("CURRENT SALES", f"₹{opt_s['current_predicted_sales']:,.0f}", "cyan")
        + metric_card("OPTIMAL SALES", f"₹{opt_s['optimal_predicted_sales']:,.0f}", "amber")
        + metric_card("UPLIFT", f"{opt_s['uplift']:+.1f}%", uplift_color)
        + '</div>',
        unsafe_allow_html=True,
    )

    opt_df = pd.DataFrame(opt)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="CURRENT", x=opt_df["channel"],
                             y=opt_df["current_spend"], marker_color="rgba(57,255,20,0.2)"))
        fig.add_trace(go.Bar(name="OPTIMAL", x=opt_df["channel"],
                             y=opt_df["optimal_spend"], marker_color="#39FF14"))
        fig.update_layout(barmode="group", title="CURRENT vs OPTIMAL SPEND")
        apply_chart_theme(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(opt_df, names="channel", values="optimal_spend",
                     title="OPTIMAL BUDGET SPLIT", color_discrete_sequence=CHART_PALETTE, hole=0.55)
        apply_chart_theme(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    # ── forecast ────────────────────────────────────────────────
    if R.get("forecast"):
        st.markdown('<div class="retro-divider"></div>', unsafe_allow_html=True)
        section_head("12-WEEK FORECAST // OPTIMAL ALLOCATION")
        fc_df = pd.DataFrame(R["forecast"])
        hist_df = pd.DataFrame({"week": list(range(1, len(R["actual_sales"]) + 1)),
                                "sales": R["actual_sales"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["week"], y=hist_df["sales"],
                                 name="HISTORICAL",
                                 line=dict(color="#39FF14", width=2),
                                 fill="tozeroy", fillcolor="rgba(57,255,20,0.04)"))
        # 95% prediction band
        if "lower_95" in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=list(fc_df["week"]) + list(fc_df["week"][::-1]),
                y=list(fc_df["upper_95"]) + list(fc_df["lower_95"][::-1]),
                fill="toself", fillcolor="rgba(255,176,0,0.06)",
                line=dict(width=0), name="95% INTERVAL", showlegend=True))
            fig.add_trace(go.Scatter(
                x=list(fc_df["week"]) + list(fc_df["week"][::-1]),
                y=list(fc_df["upper_80"]) + list(fc_df["lower_80"][::-1]),
                fill="toself", fillcolor="rgba(255,176,0,0.12)",
                line=dict(width=0), name="80% INTERVAL", showlegend=True))
        fig.add_trace(go.Scatter(x=fc_df["week"], y=fc_df["predicted_sales"],
                                 name="FORECAST",
                                 line=dict(color="#FFB000", width=2, dash="dot")))
        fig.update_layout(title="12-WEEK FORECAST", xaxis_title="WEEK", yaxis_title="SALES (₹)")
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  TAB 4 — Strategy
# ================================================================
with tab_strategy:
    section_head("AI STRATEGY RECOMMENDATION")
    strategy_text = R.get("strategy_text", "_NO STRATEGY GENERATED._")
    st.markdown(f'<div class="strategy-box">{strategy_text}</div>', unsafe_allow_html=True)


# ================================================================
#  TAB 5 — Pipeline Logs
# ================================================================

_STAGE_TAGS = {
    "Data Stage":       ("📊", "DATA STAGE",            "#39FF14"),
    "Causal Stage":     ("🔬", "CAUSAL DISCOVERY",      "#B388FF"),
    "Tuning Stage":     ("🔍", "HYPERPARAMETER TUNING", "#FFB000"),
    "Training Stage":   ("🧠", "MODEL TRAINING",        "#00E5FF"),
    "Simulation Stage": ("🔮", "SIMULATION STAGE",      "#FFB000"),
    "Forecasting Stage": ("📈", "FORECASTING",           "#FFD700"),
    "Strategy Stage":   ("💡", "STRATEGY STAGE",        "#FF80AB"),
}

def _detect_stage_tag(line: str) -> str | None:
    """Return stage name if this line starts a known stage prefix."""
    for tag in _STAGE_TAGS:
        if f"[{tag}]" in line:
            return tag
    return None

def _format_log_line(line: str) -> str:
    """Return an HTML-formatted log line with colour highlights."""
    import re as _re
    esc = _esc(line)
    # Highlight numbers with units (₹, %, x, .4f etc.)
    esc = _re.sub(r'(₹[\d,]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'([\d.]+%)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(R²\s*=\s*[\d.]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(MAPE\s*=\s*[\d.]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(RMSE\s*=\s*[\d.,₹]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(ATE\(?[^)]*\)?\s*=\s*[\d.eE+-]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(pcorr=[+\-\d.]+)', r'<span class="val">\1</span>', esc)
    esc = _re.sub(r'(CV R²\s*=\s*[\d.]+)', r'<span class="val">\1</span>', esc)
    # Highlight ✓ / ✗ / ⚠ markers
    esc = esc.replace('✓', '<span class="ok">✓</span>')
    esc = esc.replace('✗', '<span class="err">✗</span>')
    esc = esc.replace('⚠', '<span class="wrn">⚠</span>')
    # Determine indent class
    cls = 'syslog-line'
    if line.startswith('    '):
        cls += ' sub2'
    elif line.startswith('  '):
        cls += ' sub'
    return f'<div class="{cls}">{esc}</div>'

def _build_full_log_html(logs: list[str]) -> str:
    """Build the complete formatted log HTML grouped by stage."""
    sections: list[str] = []
    current_stage: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal buf, current_stage
        if not buf:
            return
        icon, label, color = _STAGE_TAGS.get(
            current_stage or "", ("", "SYSTEM", "#39FF14")
        )
        header = (
            f'<div class="syslog-stage-hdr" style="color:{color};border-color:{color};background:rgba(0,0,0,0.85);'
            f'border-bottom-color:{color};">'
            f'{icon} {label}'
            f'<span class="hdr-line" style="background:{color};"></span>'
            f'<span style="font-size:0.55rem;opacity:0.6;">{len(buf)} entries</span>'
            f'</div>'
        )
        body = '\n'.join(buf)
        sections.append(
            f'<div class="syslog-section">{header}{body}</div>'
        )
        buf = []

    for line in logs:
        stage = _detect_stage_tag(line)
        if stage and stage != current_stage:
            flush()
            current_stage = stage
        buf.append(_format_log_line(line))
    flush()

    summary = (
        f'<div class="syslog-summary">'
        f'// TOTAL: {len(logs)} LOG ENTRIES · 7 STAGES'
        f'</div>'
    )
    return f'<div class="syslog-wrap">{"" .join(sections)}{summary}</div>'


with tab_logs:
    section_head("SYSTEM EXECUTION LOG")
    logs = st.session_state.logs
    if logs:
        st.markdown(_build_full_log_html(logs), unsafe_allow_html=True)
    else:
        st.info("Execute the pipeline to generate system logs.")
