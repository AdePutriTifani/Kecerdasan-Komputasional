"""app.py - SiKuDara | Sistem Prediksi Kualitas Udara"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fuzzy_engine import infer_category, get_all_memberships_curve, CATEGORY_CONFIG
import requests
import folium
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox

import sys
from train_model import ANNModel

# Daftarkan kelas ANNModel agar dikenali oleh joblib/pickle sebagai bagian dari __main__
sys.modules['__main__'].ANNModel = ANNModel

st.set_page_config(
    page_title="SiKuDara - Prediksi Kualitas Udara",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:       #06060c;
  --bg2:      #0b0b16;
  --bg3:      #101026;
  --surface:  rgba(255,255,255,0.03);
  --surface-glass: rgba(16,16,32,0.65);
  --border:   rgba(255,255,255,0.06);
  --border2:  rgba(255,255,255,0.12);
  --accent:   #6366f1;
  --accent-cyan: #06b6d4;
  --accent-purple: #8b5cf6;
  --accent3:  #a5b4fc;
  --text:     #f8fafc;
  --muted:    rgba(241,245,249,0.5);
  --muted2:   rgba(241,245,249,0.25);
  --good:     #10b981;
  --moderate: #f59e0b;
  --bad:      #f97316;
  --danger:   #ef4444;
  --title-gradient: linear-gradient(120deg, #ffffff, #c7d2fe);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg:       #f8fafc;
    --bg2:      #ffffff;
    --bg3:      #f1f5f9;
    --surface:  rgba(15,23,42,0.03);
    --surface-glass: rgba(255,255,255,0.7);
    --border:   rgba(15,23,42,0.08);
    --border2:  rgba(15,23,42,0.15);
    --accent:   #4f46e5;
    --accent-cyan: #0891b2;
    --accent-purple: #7c3aed;
    --accent3:  #4f46e5;
    --text:     #0f172a;
    --muted:    rgba(15,23,42,0.6);
    --muted2:   rgba(15,23,42,0.35);
    --title-gradient: linear-gradient(120deg, #0f172a, #4f46e5);
  }
}

*, html, body {
  font-family: 'Plus Jakarta Sans', sans-serif;
  box-sizing: border-box;
}

/* ── App Background ── */
.stApp {
  background: var(--bg);
  background-image:
    radial-gradient(circle at 50% -10%, rgba(99,102,241,0.15) 0%, transparent 60%),
    radial-gradient(circle at 15% 30%, rgba(6,182,212,0.07) 0%, transparent 40%),
    radial-gradient(circle at 85% 75%, rgba(139,92,246,0.06) 0%, transparent 45%);
  background-attachment: fixed;
}


/* ── Main container ── */
.block-container {
  padding: 7.5rem 3rem 3rem 3rem !important;
  max-width: 1440px !important;
}

/* ── Glassmorphism on native Streamlit containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface-glass) !important;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.6rem !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
  transition: all 0.3s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(99,102,241,0.25) !important;
  box-shadow: 0 12px 40px 0 rgba(99,102,241,0.05) !important;
}


/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] .block-container {
  padding: 2rem 1.3rem !important;
}

/* ── Radio nav ── */
div[data-testid="stRadio"] > div {
  gap: 4px !important;
}
div[data-testid="stRadio"] label {
  display: flex !important;
  align-items: center;
  padding: 0.7rem 1rem !important;
  border-radius: 10px;
  color: var(--muted) !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stRadio"] label:hover {
  background: var(--surface) !important;
  color: var(--text) !important;
  transform: translateX(2px);
}
div[data-testid="stRadio"] label[data-testid*="selected"],
div[data-testid="stRadio"] label:has(input:checked) {
  background: rgba(99,102,241,0.15) !important;
  color: var(--text) !important;
  border-color: rgba(99,102,241,0.3) !important;
  box-shadow: 0 4px 15px rgba(99,102,241,0.08);
}

/* ── Sliders ── */
.stSlider label { 
  font-size: 0.82rem !important; 
  color: var(--text) !important; 
  font-weight: 600; 
  letter-spacing: 0.2px;
}
div[data-testid="stWidgetLabel"] p {
  color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.7rem 1.6rem !important;
  letter-spacing: 0.5px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
  width: 100%;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  color: white !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  box-shadow: none !important;
  color: var(--text) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(99,102,241,0.3) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.01) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px 12px 0 0 !important;
  padding: 0.35rem 0.35rem 0 !important;
  gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  background: transparent !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 0.6rem 1.2rem !important;
  border-radius: 8px 8px 0 0 !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  background: var(--bg3) !important;
  border-bottom: 2px solid var(--accent) !important;
  box-shadow: 0 -4px 12px rgba(99,102,241,0.1) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 16px 16px !important;
  padding: 1.8rem !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25) !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background: var(--surface-glass) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1.1rem !important;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  transition: all 0.25s ease;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: var(--text) !important;
}
div[data-testid="stMetric"]:hover {
  border-color: rgba(99,102,241,0.2) !important;
  transform: translateY(-2px);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Custom components ── */
.page-header {
  margin-bottom: 2rem;
  padding-bottom: 1.4rem;
  border-bottom: 1px solid var(--border);
}
.page-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.5px;
  margin: 0 0 0.4rem;
  background: var(--title-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.page-sub {
  font-size: 0.92rem;
  color: var(--muted);
  margin: 0;
}

.card {
  background: var(--surface-glass);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
}
.card:hover {
  border-color: rgba(99,102,241,0.25);
  box-shadow: 0 12px 40px 0 rgba(99,102,241,0.05);
}

.card-sm {
  background: rgba(255,255,255,0.02);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  transition: all 0.3s ease;
}
.card-sm:hover {
  border-color: rgba(99,102,241,0.2);
}

.stat-card {
  background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.03));
  border: 1px solid rgba(99,102,241,0.15);
  border-radius: 14px;
  padding: 1.4rem;
  text-align: center;
  transition: all 0.3s ease;
}
.stat-card:hover {
  transform: translateY(-3px);
  border-color: rgba(99,102,241,0.3);
  box-shadow: 0 8px 24px rgba(99,102,241,0.1);
}
.stat-label {
  font-size: 0.72rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent3);
  font-weight: 700;
  margin-bottom: 0.5rem;
}
.stat-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1;
}
.stat-unit { font-size: 0.78rem; color: var(--muted); margin-top: 0.4rem; opacity: 0.7; }

.result-wrapper {
  border-radius: 16px;
  padding: 2.2rem 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-wrapper::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 70%);
  z-index: 0;
}
.glow-baik {
  background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.02)) !important;
  border: 1px solid rgba(16,185,129,0.4) !important;
  box-shadow: 0 8px 32px rgba(16,185,129,0.18), inset 0 0 20px rgba(16,185,129,0.05) !important;
}
.glow-baik:hover {
  box-shadow: 0 12px 48px rgba(16,185,129,0.32), inset 0 0 24px rgba(16,185,129,0.08) !important;
  border-color: rgba(16,185,129,0.65) !important;
}
.glow-sedang {
  background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.02)) !important;
  border: 1px solid rgba(245,158,11,0.4) !important;
  box-shadow: 0 8px 32px rgba(245,158,11,0.18), inset 0 0 20px rgba(245,158,11,0.05) !important;
}
.glow-sedang:hover {
  box-shadow: 0 12px 48px rgba(245,158,11,0.32), inset 0 0 24px rgba(245,158,11,0.08) !important;
  border-color: rgba(245,158,11,0.65) !important;
}
.glow-tidak-sehat {
  background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(249,115,22,0.02)) !important;
  border: 1px solid rgba(249,115,22,0.4) !important;
  box-shadow: 0 8px 32px rgba(249,115,22,0.18), inset 0 0 20px rgba(249,115,22,0.05) !important;
}
.glow-tidak-sehat:hover {
  box-shadow: 0 12px 48px rgba(249,115,22,0.32), inset 0 0 24px rgba(249,115,22,0.08) !important;
  border-color: rgba(249,115,22,0.65) !important;
}
.glow-berbahaya {
  background: linear-gradient(135deg, rgba(239,110,110,0.12), rgba(239,110,110,0.02)) !important;
  border: 1px solid rgba(239,110,110,0.4) !important;
  box-shadow: 0 8px 32px rgba(239,110,110,0.22), inset 0 0 24px rgba(239,110,110,0.08) !important;
}
.glow-berbahaya:hover {
  box-shadow: 0 12px 48px rgba(239,110,110,0.35), inset 0 0 30px rgba(239,110,110,0.12) !important;
  border-color: rgba(239,110,110,0.7) !important;
}

.result-badge {
  display: inline-block;
  padding: 0.3rem 0.9rem;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  border: 1px solid currentColor;
  margin-bottom: 1.2rem;
  position: relative;
  z-index: 1;
}
.result-number {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 5rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1;
  letter-spacing: -2px;
  position: relative;
  z-index: 1;
}
.result-label { font-size: 0.85rem; color: var(--muted); margin-top: 0.4rem; position: relative; z-index: 1; }

.saran-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.3rem;
  margin-top: 1.2rem;
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.7;
}
.saran-title {
  font-size: 0.72rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--accent3);
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.mf-row { margin-bottom: 12px; }
.mf-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
.mf-name { font-size: 0.8rem; font-weight: 700; color: var(--text); }
.mf-value { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--muted); opacity: 0.8; }
.mf-track { height: 6px; background: var(--surface); border-radius: 4px; overflow: hidden; }
.mf-fill  { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }

.ref-tbl { width: 100%; border-collapse: collapse; font-size: 0.86rem; border-radius: 8px; overflow: hidden; }
.ref-tbl thead th {
  background: var(--surface);
  color: var(--muted);
  padding: 0.8rem 1.1rem;
  text-align: left;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
.ref-tbl tbody td {
  padding: 0.8rem 1.1rem;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}
.ref-tbl tbody tr:last-child td { border-bottom: none; }
.ref-tbl tbody tr:hover td { background: var(--surface); }

.method-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem;
  height: 100%;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.method-card:hover { 
  border-color: rgba(99,102,241,0.25);
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(99,102,241,0.05);
}
.method-num {
  font-size: 0.7rem;
  letter-spacing: 2px;
  font-weight: 800;
  color: var(--accent-cyan);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.method-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.8rem;
}
.method-desc { font-size: 0.84rem; color: var(--muted); line-height: 1.75; }

.arch-flow {
  display: flex;
  align-items: stretch;
  gap: 0;
  margin: 1.5rem 0;
}
.arch-node {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1rem;
  text-align: center;
  transition: all 0.3s ease;
}
.arch-node:hover {
  border-color: var(--border2);
  transform: translateY(-2px);
}
.arch-arrow {
  display: flex;
  align-items: center;
  padding: 0 0.5rem;
  color: var(--muted);
  opacity: 0.4;
  font-size: 1.2rem;
  font-weight: 300;
}
.arch-tag {
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--accent3);
  margin-bottom: 0.4rem;
}
.arch-label {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text);
}
.arch-sub {
  font-size: 0.74rem;
  color: var(--muted);
  margin-top: 0.3rem;
  line-height: 1.5;
}

.brand {
  padding: 1rem 1.2rem;
  display: flex;
  flex-direction: column;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.5rem;
}
.brand-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text) !important;
  letter-spacing: -0.5px;
  display: flex;
  align-items: center;
}
.brand-dot {
  display: inline-block;
  width: 8px; height: 8px;
  background: linear-gradient(135deg, #06b6d4, #6366f1);
  border-radius: 50%;
  margin-left: 6px;
  animation: pulse 2.5s infinite;
}
.brand-sub {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 4px;
}
@keyframes pulse {
  0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.5); opacity: 0.7; }
  70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(6, 182, 212, 0); opacity: 1; }
  100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); opacity: 0.7; }
}

.nav-label {
  font-size: 0.68rem;
  letter-spacing: 2px;
  font-weight: 800;
  text-transform: uppercase;
  color: var(--muted) !important;
  opacity: 0.6;
  margin: 1.2rem 0 0.5rem 0.5rem;
}

.tag-pill {
  display: inline-block;
  background: rgba(99,102,241,0.1);
  border: 1px solid rgba(99,102,241,0.22);
  color: var(--accent3);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 20px;
  margin-bottom: 1rem;
}

.placeholder-area {
  height: 100%;
  min-height: 400px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1.5px dashed var(--border);
  border-radius: 16px;
  padding: 3rem;
  background: var(--surface);
}
.placeholder-text {
  font-size: 0.88rem;
  color: var(--muted);
  opacity: 0.8;
  text-align: center;
  line-height: 1.8;
}

.info-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.82rem;
  color: var(--muted);
  padding: 0.25rem 0;
}
.info-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent-cyan);
  flex-shrink: 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
COLORS = {
    "BAIK":        {"hex": "#22c55e", "rgba": "rgba(34,197,94,{})"},
    "SEDANG":      {"hex": "#eab308", "rgba": "rgba(234,179,8,{})"},
    "TIDAK SEHAT": {"hex": "#f97316", "rgba": "rgba(249,115,22,{})"},
    "BERBAHAYA":   {"hex": "#ef4444", "rgba": "rgba(239,68,68,{})"},
}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(128,128,128,0.85)", family="Plus Jakarta Sans", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)
AX = dict(gridcolor="rgba(128,128,128,0.12)", zerolinecolor="rgba(128,128,128,0.2)",
          linecolor="rgba(128,128,128,0.2)", tickfont=dict(color="rgba(128,128,128,0.7)"))

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_resources():
    mp = os.path.join("model", "ann_model.pkl")
    if not os.path.exists(mp):
        from train_model import train
        return train()
    from train_model import load_model, load_metrics
    return load_model(), load_metrics()

try:
    model, metrics = load_resources()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Gagal memuat model: {e}")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-name">SiKuDara<span class="brand-dot"></span></div>
      <div class="brand-sub">Air Quality Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="nav-label">Navigasi</div>', unsafe_allow_html=True)

    page = st.radio("", ["Prediksi", "Visualisasi", "Tentang Model"],
                    label_visibility="collapsed")

    st.divider()

    if model_ready:
        mae_v  = metrics.get("mae",  0)
        rmse_v = metrics.get("rmse", 0)
        r2_v   = metrics.get("r2",   0)
        st.markdown(f"""
        <div style="padding:0.8rem;background:var(--surface);border:1px solid var(--border);border-radius:10px;">
          <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:0.6rem;font-weight:700;">Status Model</div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px;">
            <span style="color:var(--muted);opacity:0.8;">R² Score</span>
            <span style="color:#22c55e;font-weight:700;font-family:'JetBrains Mono',monospace;">{r2_v:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px;">
            <span style="color:var(--muted);opacity:0.8;">MAE</span>
            <span style="color:var(--text);font-family:'JetBrains Mono',monospace;">{mae_v:.2f} µg/m³</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;">
            <span style="color:var(--muted);opacity:0.8;">RMSE</span>
            <span style="color:var(--text);font-family:'JetBrains Mono',monospace;">{rmse_v:.2f} µg/m³</span>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Latih Ulang Model", use_container_width=True):
        for f in ["model/ann_model.pkl", "model/metrics.pkl"]:
            if os.path.exists(f): os.remove(f)
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);">
      <p style="font-size:0.68rem;color:var(--muted);opacity:0.6;line-height:1.6;margin:0;text-align:center;">
         IF25-40404 · Kecerdasan Komputasional<br>
         Institut Teknologi Sumatera<br>
         2025 / 2026
      </p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
if page == "Prediksi":
    st.markdown("""
    <div class="page-header">
      <h1 class="page-title">Prediksi Kualitas Udara</h1>
      <p class="page-sub">Masukkan parameter cuaca dan polutan untuk memperoleh prediksi nilai PM2.5 beserta kategorinya</p>
    </div>""", unsafe_allow_html=True)

    # API functions
    def get_weather_data(lat, lon):
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m"
            f"&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["current"]

    def get_air_quality_data(lat, lon):
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}"
            f"&current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi"
            f"&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["current"]

    col_L, col_R = st.columns([11, 10], gap="large")

    run_pred = False
    feats = {}

    # ── INPUT PANEL ──────────────────────────────────────────────────────────
    with col_L:
        tab_map, tab_manual = st.tabs(["Peta Interaktif (Otomatis)", "Input Manual (Fallback)"])
        
        with tab_map:
            st.markdown('<div class="tag-pill">Pencarian & Pemilihan Lokasi</div>', unsafe_allow_html=True)
            
            # Inisialisasi session state untuk peta
            if "map_center" not in st.session_state:
                st.session_state.map_center = [-5.3971, 105.2668]
            if "map_zoom" not in st.session_state:
                st.session_state.map_zoom = 11
            if "loc_name" not in st.session_state:
                st.session_state.loc_name = None
            if "active_lat" not in st.session_state:
                st.session_state.active_lat = None
            if "active_lon" not in st.session_state:
                st.session_state.active_lon = None
            if "last_click" not in st.session_state:
                st.session_state.last_click = None

            with st.container():
                def search_location_api(searchterm: str) -> list:
                    if not searchterm:
                        return []
                    url = f"https://geocoding-api.open-meteo.com/v1/search?name={searchterm}&count=5&language=id&format=json"
                    try:
                        res = requests.get(url, timeout=5)
                        if res.status_code == 200:
                            data = res.json().get("results", [])
                            results = []
                            for item in data:
                                name_parts = [item.get("name")]
                                if item.get("admin2"): name_parts.append(item.get("admin2"))
                                if item.get("admin1"): name_parts.append(item.get("admin1"))
                                display_name = ", ".join([p for p in name_parts if p])
                                results.append(
                                    (display_name, {"lat": float(item["latitude"]), "lon": float(item["longitude"]), "name": item["name"]})
                                )
                            return results
                    except:
                        pass
                    return []
                    
                selected_loc = st_searchbox(
                    search_location_api,
                    placeholder="🔍 Ketik lokasi (kota, kecamatan, desa) untuk melihat rekomendasi...",
                    key="loc_searchbox"
                )

                if selected_loc and selected_loc != st.session_state.get("last_searched_loc"):
                    st.session_state.last_searched_loc = selected_loc
                    st.session_state.active_lat = selected_loc["lat"]
                    st.session_state.active_lon = selected_loc["lon"]
                    st.session_state.map_center = [selected_loc["lat"], selected_loc["lon"]]
                    st.session_state.map_zoom = 13
                    st.session_state.loc_name = selected_loc["name"]
                    st.rerun()

            m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
            
            if st.session_state.active_lat is not None and st.session_state.active_lon is not None:
                folium.Marker(
                    [st.session_state.active_lat, st.session_state.active_lon],
                    tooltip=st.session_state.loc_name or "Lokasi Terpilih",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)

            map_data = st_folium(m, width="100%", height=380, key="map")
            
            # Deteksi jika user klik langsung di peta
            if map_data and map_data.get("last_clicked"):
                curr_click = map_data["last_clicked"]
                if curr_click != st.session_state.last_click:
                    st.session_state.last_click = curr_click
                    st.session_state.active_lat = curr_click["lat"]
                    st.session_state.active_lon = curr_click["lng"]
                    st.session_state.map_center = [curr_click["lat"], curr_click["lng"]]
                    st.session_state.loc_name = "Koordinat Peta"
                    st.rerun()
            
            # Jika ada koordinat yang aktif, fetch datanya
            if st.session_state.active_lat is not None and st.session_state.active_lon is not None:
                lat = st.session_state.active_lat
                lon = st.session_state.active_lon
                
                # Render notifikasi kecil nama tempat
                st.success(f"**Lokasi:** {st.session_state.loc_name} ({lat:.4f}, {lon:.4f})")
                
                try:
                    with st.spinner("Mengambil data cuaca dan udara..."):
                        weather = get_weather_data(lat, lon)
                        air = get_air_quality_data(lat, lon)
                    
                    feats = {
                        "TEMP": weather.get("temperature_2m", 25.0),
                        "PRES": weather.get("pressure_msl", 1013.0),
                        "DEWP": weather.get("dew_point_2m", 10.0),
                        "WSPM": weather.get("wind_speed_10m", 2.0),
                        "SO2": air.get("sulphur_dioxide", 0.0),
                        "NO2": air.get("nitrogen_dioxide", 0.0),
                        "PM10": air.get("pm10", 0.0)
                    }
                    run_pred = True
                    
                    with st.expander("Detail Data Realtime", expanded=False):
                        w_col, a_col = st.columns(2)
                        w_col.write("**Cuaca (Open-Meteo)**")
                        w_col.json(weather)
                        a_col.write("**Polutan (Open-Meteo)**")
                        a_col.json(air)
                except Exception as e:
                    st.error(f"Gagal mengambil data dari API: {e}")

        with tab_manual:
            with st.container(border=True):
                st.markdown('<div class="tag-pill">Parameter Input</div>', unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.85rem;font-weight:700;color:var(--accent-cyan);margin:0.5rem 0 0.8rem;display:flex;align-items:center;gap:0.4rem;">
                  <span style="font-size:1rem;">🌦️</span> KONDISI ATMOSFER
                </div>""", unsafe_allow_html=True)

                g1, g2 = st.columns(2)
                with g1:
                    temp = st.slider("Suhu (°C)",         -10.0, 40.0,  25.0, 0.5)
                    dewp = st.slider("Titik Embun (°C)",  -30.0, 30.0,  10.0, 0.5)
                with g2:
                    pres = st.slider("Tekanan (hPa)",     990.0,1040.0,1013.0, 0.5)
                    wspm = st.slider("Kec. Angin (m/s)",    0.0, 10.0,   2.0,  0.1)

                st.markdown("""
                <div style="font-size:0.85rem;font-weight:700;color:var(--accent-purple);margin:1.2rem 0 0.8rem;display:flex;align-items:center;gap:0.4rem;">
                  <span style="font-size:1rem;">💨</span> KADAR POLUTAN UDARA
                </div>""", unsafe_allow_html=True)

                g3, g4 = st.columns(2)
                with g3:
                    so2  = st.slider("SO₂ (µg/m³)",        0.0,200.0,  20.0, 1.0)
                    no2  = st.slider("NO₂ (µg/m³)",        0.0,250.0,  40.0,  1.0)
                with g4:
                    pm10 = st.slider("PM10 (µg/m³)",       0.0, 500.0, 60.0, 1.0)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Jalankan Prediksi Manual", use_container_width=True, type="primary"):
                feats = {"TEMP":temp,"PRES":pres,"DEWP":dewp,"WSPM":wspm,"SO2":so2,"NO2":no2,"PM10":pm10}
                run_pred = True

    # ── OUTPUT PANEL ─────────────────────────────────────────────────────────
    with col_R:
        if run_pred and model_ready:
            pm25  = max(0.0, model.predict_single(feats))
            res   = infer_category(pm25)
            cat   = res["category"]
            col_h = COLORS[cat]["hex"]
            col_a = COLORS[cat]["rgba"]

            # Map category to styling glow classes
            glow_class = "glow-baik"
            if cat == "SEDANG":
                glow_class = "glow-sedang"
            elif cat == "TIDAK SEHAT":
                glow_class = "glow-tidak-sehat"
            elif cat == "BERBAHAYA":
                glow_class = "glow-berbahaya"

            # Result card
            st.markdown(f"""
            <div class="result-wrapper {glow_class}">
              <div class="result-badge" style="color:{col_h};">{cat}</div>
              <div class="result-number">{pm25:.1f}</div>
              <div class="result-label">µg/m³  ·  PM2.5</div>
            </div>
            <div class="saran-card">
              <div class="saran-title">Rekomendasi Aktivitas</div>
              {res['saran']}
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pm25,
                title={"text": "Indeks PM2.5", "font": {"size": 12, "color": "rgba(128,128,128,0.7)"}},
                gauge={
                    "axis": {"range": [0, 200],
                             "tickcolor": "rgba(128,128,128,0.3)",
                             "tickfont": {"size": 10},
                             "nticks": 5},
                    "bar":  {"color": col_h, "thickness": 0.2},
                    "bgcolor": "rgba(128,128,128,0.08)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,   50],  "color": "rgba(34,197,94,0.15)"},
                        {"range": [50,  100], "color": "rgba(234,179,8,0.15)"},
                        {"range": [100, 150], "color": "rgba(249,115,22,0.15)"},
                        {"range": [150, 200], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": col_h, "width": 2}, "value": pm25},
                },
                number={"suffix": " µg/m³",
                        "font": {"color": "rgba(128,128,128,0.95)", "size": 15, "family": "JetBrains Mono"}},
            ))
            fig_g.update_layout(
                **{**PLOT_BASE, "margin": dict(l=25, r=25, t=40, b=5)},
                height=240,
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Membership bars
            bars_html = "".join(
                f'<div class="mf-row">'
                f'  <div class="mf-header">'
                f'    <span class="mf-name">{c2}</span>'
                f'    <span class="mf-value">{deg:.3f}</span>'
                f'  </div>'
                f'  <div class="mf-track">'
                f'    <div class="mf-fill" style="width:{deg*100:.1f}%;background:linear-gradient(90deg,{COLORS[c2]["hex"]}88,{COLORS[c2]["hex"]});"></div>'
                f'  </div>'
                f'</div>'
                for c2, deg in res["memberships"].items()
            )
            st.markdown(f"""
            <div class="card-sm">
              <div class="tag-pill">Derajat Keanggotaan Fuzzy</div>
              {bars_html}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="placeholder-area">
              <div class="placeholder-text">
                Atur parameter input pada kolom sebelah kiri<br>
                kemudian klik <strong>Jalankan Prediksi</strong>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── REFERENCE TABLE ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    rows_data = [
        ("BAIK",        "0 – 50",   COLORS["BAIK"]["hex"],        "Aman untuk semua aktivitas luar ruangan"),
        ("SEDANG",      "51 – 100", COLORS["SEDANG"]["hex"],      "Kelompok sensitif sebaiknya membatasi aktivitas berat"),
        ("TIDAK SEHAT", "101 – 150",COLORS["TIDAK SEHAT"]["hex"], "Kurangi aktivitas luar ruangan, gunakan masker"),
        ("BERBAHAYA",   "> 150",    COLORS["BERBAHAYA"]["hex"],   "Hindari semua aktivitas luar ruangan"),
    ]
    tds = "".join(
        f"<tr>"
        f'<td><span style="color:{c};font-weight:700;font-size:0.82rem;">{nm}</span></td>'
        f'<td><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;">{rng}</span></td>'
        f'<td style="max-width:380px;">{desc}</td>'
        f"</tr>"
        for nm, rng, c, desc in rows_data
    )
    st.markdown(f"""
    <div class="card">
      <div class="tag-pill">Standar Kategori ISPU Indonesia</div>
      <table class="ref-tbl">
        <thead><tr><th>Kategori</th><th>PM2.5 (µg/m³)</th><th>Saran Aktivitas</th></tr></thead>
        <tbody>{tds}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualisasi":
    st.markdown("""
    <div class="page-header">
      <h1 class="page-title">Visualisasi & Analisis</h1>
      <p class="page-sub">Eksplorasi performa model, perilaku Fuzzy Logic, dan proses optimasi PSO</p>
    </div>""", unsafe_allow_html=True)

    if not model_ready:
        st.warning("Model belum tersedia.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Prediksi vs Aktual", "Fuzzy Membership", "Konvergensi PSO"])

    with tab1:
        yt = metrics["y_test"];  yp = metrics["y_pred_test"]
        n  = min(300, len(yt))
        ix = np.sort(np.random.choice(len(yt), n, replace=False))

        col_plot, col_hist = st.columns([11, 9], gap="medium")

        with col_plot:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Scatter — Prediksi vs Aktual (300 sampel acak)",
                                "Perbandingan Tren — Nilai Aktual vs Prediksi"),
                vertical_spacing=0.13,
            )
            fig.add_trace(go.Scatter(
                x=yt[ix], y=yp[ix], mode="markers",
                marker=dict(color=yt[ix], colorscale="Plasma", size=5.5, opacity=0.6,
                            colorbar=dict(title="PM2.5", thickness=10, len=0.44, y=0.78, tickfont=dict(size=9))),
                name="Sampel", showlegend=False), row=1, col=1)
            rng = [0, float(max(yt.max(), yp.max()))]
            fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1),
                name="Garis Ideal"), row=1, col=1)

            fig.add_trace(go.Scatter(y=yt[ix], mode="lines", name="Aktual",
                line=dict(color="#6366f1", width=2, shape="spline")), row=2, col=1)
            fig.add_trace(go.Scatter(y=yp[ix], mode="lines", name="Prediksi",
                line=dict(color="#10b981", width=2, shape="spline", dash="dot")), row=2, col=1)

            fig.update_layout(**PLOT_BASE, height=580, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                            bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                hoverlabel=dict(bgcolor="#1f2937", bordercolor="rgba(128,128,128,0.3)", font=dict(family="Plus Jakarta Sans", color="#ffffff")))
            fig.update_xaxes(**AX); fig.update_yaxes(**AX)
            fig.update_annotations(font_color="rgba(128,128,128,0.75)", font_size=11)
            st.plotly_chart(fig, use_container_width=True)

        with col_hist:
            err = yt - yp
            fig_e = px.histogram(x=err, nbins=60, opacity=0.85,
                title="Distribusi Residual  (Aktual − Prediksi)",
                labels={"x": "Residual (µg/m³)", "y": "Frekuensi"},
                color_discrete_sequence=["#06b6d4"])
            fig_e.update_layout(**PLOT_BASE, height=580, hoverlabel=dict(bgcolor="#1f2937", bordercolor="rgba(128,128,128,0.3)", font=dict(family="Plus Jakarta Sans", color="#ffffff")))
            fig_e.update_xaxes(**AX); fig_e.update_yaxes(**AX)
            st.plotly_chart(fig_e, use_container_width=True)

    with tab2:
        curve = get_all_memberships_curve()
        fig_f = go.Figure()
        for cat, cdata in COLORS.items():
            fig_f.add_trace(go.Scatter(
                x=curve["x"], y=curve[cat], name=cat, mode="lines",
                line=dict(color=cdata["hex"], width=2.5, shape="spline"),
                fill="tozeroy",
                fillcolor=cdata["rgba"].format("0.12")))
        fig_f.update_layout(**PLOT_BASE, height=400,
            title="Fungsi Keanggotaan PM2.5  ·  Fuzzy Logic Mamdani",
            xaxis=dict(title="Nilai PM2.5 (µg/m³)", **AX),
            yaxis=dict(title="Derajat Keanggotaan (µ)", range=[0, 1.1], **AX),
            legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#1f2937", bordercolor="rgba(128,128,128,0.3)", font=dict(family="Plus Jakarta Sans", color="#ffffff")))
        st.plotly_chart(fig_f, use_container_width=True)

    with tab3:
        hist = metrics.get("pso_history", [])
        if hist:
            iters = list(range(1, len(hist) + 1))
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(y=hist, x=iters, mode="lines", name="Best MSE",
                line=dict(color="#6366f1", width=2.5, shape="spline"),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            # Best point
            best_i = int(np.argmin(hist))
            fig_p.add_trace(go.Scatter(
                x=[best_i + 1], y=[hist[best_i]],
                mode="markers", name=f"Global Best (iter {best_i+1})",
                marker=dict(color="#10b981", size=10, symbol="diamond")))
            fig_p.update_layout(**PLOT_BASE, height=380,
                title="Konvergensi PSO  ·  Best MSE per Iterasi",
                xaxis=dict(title="Iterasi", **AX),
                yaxis=dict(title="MSE (Val Set)", **AX),
                legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor="#1f2937", bordercolor="rgba(128,128,128,0.3)", font=dict(family="Plus Jakarta Sans", color="#ffffff")))
            st.plotly_chart(fig_p, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MSE Awal",   f"{hist[0]:.2f}")
            c2.metric("MSE Akhir",  f"{hist[-1]:.2f}", delta=f"{hist[-1]-hist[0]:.2f}")
            c3.metric("Reduksi MSE",f"{(1-hist[-1]/hist[0])*100:.1f}%")
            c4.metric("Iterasi Terbaik", f"{best_i+1}")
        else:
            st.info("Data konvergensi tidak tersedia.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TENTANG MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tentang Model":
    st.markdown("""
    <div class="page-header">
      <h1 class="page-title">Tentang Model</h1>
      <p class="page-sub">Arsitektur sistem, metode yang digunakan, dan performa model terlatih</p>
    </div>""", unsafe_allow_html=True)

    # Stat cards
    if model_ready:
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            ("MAE",      f"{metrics['mae']:.2f}",  "Mean Absolute Error · µg/m³"),
            ("RMSE",     f"{metrics['rmse']:.2f}", "Root Mean Square Error · µg/m³"),
            ("R² Score", f"{metrics['r2']:.4f}",  "Koefisien Determinasi"),
            ("Waktu",    f"{metrics.get('elapsed_sec',0):.1f}s", "Durasi Training PSO"),
        ]
        for col_obj, (lbl, val, unit) in zip([c1,c2,c3,c4], cards):
            col_obj.markdown(f"""
            <div class="stat-card">
              <div class="stat-label">{lbl}</div>
              <div class="stat-value">{val}</div>
              <div class="stat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Method cards
    st.markdown('<div class="tag-pill">Metode yang Digunakan</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    methods = [
        ("01", "Artificial Neural Network",
         "Jaringan saraf tiruan dengan arsitektur Input(7) → Hidden1(16) → Hidden2(8) → Output(1). "
         "Fungsi aktivasi ReLU pada hidden layer dan linear pada layer output. "
         "Bertugas memprediksi nilai numerik PM2.5 dari parameter cuaca secara langsung.",
         ["7 fitur input", "2 hidden layer", "Aktivasi ReLU", "Output linear"]),
        ("02", "Particle Swarm Optimization",
         "Algoritma swarm intelligence dengan 30 partikel dan 100 iterasi. "
         "Menggantikan backpropagation untuk mencari bobot ANN secara global, "
         "menghindari local minima, dengan inersia w=0.7 dan koefisien c1=c2=1.5.",
         ["30 partikel", "100 iterasi", "Inertia w = 0.7", "Batas bobot ±5"]),
        ("03", "Fuzzy Logic (Mamdani)",
         "Sistem inferensi fuzzy dengan membership function trapezoidal dan segitiga "
         "untuk memetakan output numerik ANN ke dalam kategori linguistik "
         "BAIK, SEDANG, TIDAK SEHAT, dan BERBAHAYA sesuai standar ISPU.",
         ["MF Trapezoidal", "MF Segitiga", "4 kategori output", "Standar ISPU"]),
    ]
    for col_obj, (num, name, desc, tags) in zip([m1,m2,m3], methods):
        tag_html = "".join(f'<span style="display:inline-block;margin:2px;padding:2px 8px;border-radius:4px;font-size:0.67rem;font-weight:600;background:rgba(99,102,241,0.1);color:#a5b4fc;border:1px solid rgba(99,102,241,0.2);">{t}</span>' for t in tags)
        col_obj.markdown(f"""
        <div class="method-card">
          <div class="method-num">{num}</div>
          <div class="method-name">{name}</div>
          <div class="method-desc">{desc}</div>
          <div style="margin-top:1rem;">{tag_html}</div>
        </div>""", unsafe_allow_html=True)

    # Architecture flow
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="tag-pill">Alur Sistem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-flow">
      <div class="arch-node" style="background:rgba(99,102,241,0.08);border-color:rgba(99,102,241,0.2);">
        <div class="arch-tag">Input</div>
        <div class="arch-label">Parameter Cuaca</div>
        <div class="arch-sub">Suhu · Tekanan · Titik Embun<br>Angin · SO₂ · NO₂ · PM10</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node">
        <div class="arch-tag">Tahap 1</div>
        <div class="arch-label">PSO</div>
        <div class="arch-sub">Optimasi bobot ANN<br>secara global</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node">
        <div class="arch-tag">Tahap 2</div>
        <div class="arch-label">ANN</div>
        <div class="arch-sub">Prediksi nilai<br>PM2.5 (µg/m³)</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node">
        <div class="arch-tag">Tahap 3</div>
        <div class="arch-label">Fuzzy Logic</div>
        <div class="arch-sub">Interpretasi ke<br>kategori linguistik</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node" style="background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.2);">
        <div class="arch-tag">Output</div>
        <div class="arch-label">Kategori Udara</div>
        <div class="arch-sub">BAIK / SEDANG<br>TIDAK SEHAT / BERBAHAYA</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # References
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="tag-pill">Referensi</div>', unsafe_allow_html=True)
    refs = [
        ("Engelbrecht, 2007", "Computational Intelligence: An Introduction. Wiley."),
        ("Kennedy & Eberhart, 1995", "Particle swarm optimization. Proceedings of IEEE ICNN."),
        ("Zadeh, 1965", "Fuzzy sets. Information and Control, 8(3), 338–353."),
        ("UCI ML Repository", "Beijing Multi-Site Air Quality Data Set. archive.ics.uci.edu/dataset/501"),
    ]
    for author, title in refs:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
          <span style="font-size:0.78rem;font-weight:600;color:#a5b4fc;min-width:180px;flex-shrink:0;">{author}</span>
          <span style="font-size:0.78rem;color:rgba(255,255,255,0.4);">{title}</span>
        </div>""", unsafe_allow_html=True)
