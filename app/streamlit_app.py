# ============================================================
# BREAST CANCER DETECTION & DIAGNOSTIC INTELLIGENCE
# Streamlit application — research/educational demonstration
# ============================================================

from pathlib import Path
import json
import html
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ----------------------------- CONFIG -----------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "breast_cancer_logistic_regression.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "processed" / "wdbc_cleaned.csv"
EVAL_DIR = ROOT / "outputs" / "model_evaluation"

st.set_page_config(
    page_title="OncoLens | Breast Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_GROUPS = {
    "Mean morphology": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    ],
    "Standard error": [
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",
    ],
    "Worst morphology": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ],
}
APP_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Independently evaluated holdout metrics from the project's notebook.
# These are fixed project results, not recomputed from uploaded CSV files.
HOLDOUT_METRICS = {
    "Accuracy": 0.9737,
    "Precision": 0.9756,
    "Recall / Sensitivity": 0.9524,
    "Specificity": 0.9861,
    "F1 score": 0.9639,
    "ROC-AUC": 0.9964,
}
HOLDOUT_CM = np.array([[71, 1], [2, 40]])

# ----------------------------- STYLE -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #eaf2ff; --muted: #9fb2ca; --panel: #111f35;
      --panel2: #172943; --line: rgba(157,184,215,.16);
      --teal: #45d6c5; --blue: #6da8ff; --bg: #081321;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 12% 0%, #142b48 0%, #081321 44%, #07111e 100%); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(8,19,33,.82); }
    [data-testid="stSidebar"] { background: #0c1a2c; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: #dce9fa; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1,h2,h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
    .eyebrow { color: var(--teal); font-size: .72rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
    .hero { background: linear-gradient(115deg, rgba(24,53,83,.96), rgba(14,39,60,.92) 55%, rgba(14,73,77,.88));
      border: 1px solid rgba(91,207,199,.25); border-radius: 24px; padding: 30px 32px; margin: 4px 0 24px;
      box-shadow: 0 18px 60px rgba(0,0,0,.22); }
    .hero h1 { color: #f5f9ff; font-size: clamp(2rem, 4vw, 3.15rem); margin: 8px 0 8px; }
    .hero p { color: #c2d3e7; font-size: 1rem; max-width: 780px; margin: 0; line-height: 1.65; }
    .pill { display:inline-block; border:1px solid rgba(69,214,197,.38); color:#8ef0df; background:rgba(69,214,197,.08);
      padding:5px 10px; border-radius:99px; font-size:.72rem; font-weight:700; letter-spacing:.04em; }
    .panel { background: linear-gradient(145deg, rgba(23,41,67,.92), rgba(14,29,48,.96)); border:1px solid var(--line);
      border-radius:18px; padding:19px 20px; height:100%; }
    .panel-label { color:#91a8c3; font-size:.75rem; text-transform:uppercase; letter-spacing:.11em; font-weight:700; }
    .panel-value { color:#f4f8ff; font-family:'Manrope'; font-size:1.8rem; font-weight:800; margin-top:6px; }
    .panel-sub { color:#8fa6c0; font-size:.78rem; margin-top:4px; }
    .section-kicker { color:#6fe0d2; font-size:.72rem; text-transform:uppercase; letter-spacing:.15em; font-weight:800; margin-bottom:5px; }
    .section-title { font-family:'Manrope'; font-size:1.55rem; font-weight:800; color:#f0f6ff; margin:0 0 12px; }
    .muted { color:#9fb2ca; }
    .notice { background:rgba(255,184,77,.09); border:1px solid rgba(255,184,77,.30); border-left:4px solid #ffbd59;
      border-radius:12px; padding:13px 15px; color:#f5d9a8; font-size:.86rem; line-height:1.55; }
    .success-note { background:rgba(69,214,197,.08); border:1px solid rgba(69,214,197,.25); border-radius:12px;
      padding:13px 15px; color:#b7f5e9; font-size:.86rem; }
    .result-malignant { background:linear-gradient(135deg,rgba(120,39,58,.35),rgba(52,25,44,.45)); border:1px solid rgba(255,113,137,.38);
      border-radius:18px; padding:22px; }
    .result-benign { background:linear-gradient(135deg,rgba(24,105,99,.28),rgba(19,54,67,.45)); border:1px solid rgba(69,214,197,.35);
      border-radius:18px; padding:22px; }
    div[data-testid="stMetric"] { background:rgba(23,41,67,.82); border:1px solid var(--line); padding:14px 16px; border-radius:14px; }
    div[data-testid="stMetricLabel"] { color:#9fb2ca; }
    div[data-testid="stMetricValue"] { color:#f3f7ff; }
    /* ---------- CONTROL SYSTEM: persistent colour, hierarchy and states ---------- */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
      min-height: 2.85rem !important;
      border-radius: 12px !important;
      padding: .62rem 1.05rem !important;
      font-family: 'DM Sans', sans-serif !important;
      font-size: .91rem !important;
      font-weight: 750 !important;
      letter-spacing: .01em !important;
      color: #eaf7ff !important;
      background: linear-gradient(135deg, #183c59 0%, #174e62 100%) !important;
      border: 1px solid rgba(104,205,220,.48) !important;
      box-shadow: 0 5px 16px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.07) !important;
      transition: transform .18s ease, box-shadow .18s ease, background .18s ease, border-color .18s ease !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
      color: #ffffff !important;
      background: linear-gradient(135deg, #167f88 0%, #1b9e9a 100%) !important;
      border-color: #65e4d5 !important;
      box-shadow: 0 8px 24px rgba(28,190,177,.22) !important;
      transform: translateY(-1px);
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] > button:focus-visible {
      outline: 3px solid rgba(105,225,215,.55) !important;
      outline-offset: 3px !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active,
    [data-testid="stFormSubmitButton"] > button:active {
      transform: translateY(0) scale(.985) !important;
      box-shadow: 0 2px 8px rgba(0,0,0,.25) !important;
    }
    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
      color: #8498ad !important;
      background: #1a2939 !important;
      border-color: #34475c !important;
      box-shadow: none !important;
      opacity: .58 !important;
      cursor: not-allowed !important;
    }
    /* Primary actions remain visibly primary even before interaction. */
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {
      color: #031d26 !important;
      background: linear-gradient(135deg, #43ddc7 0%, #54b9ed 100%) !important;
      border: 1px solid rgba(154,255,240,.8) !important;
      box-shadow: 0 7px 22px rgba(45,201,190,.20), inset 0 1px 0 rgba(255,255,255,.35) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
      color: #031820 !important;
      background: linear-gradient(135deg, #75f0dc 0%, #7bcaff 100%) !important;
      box-shadow: 0 10px 28px rgba(45,201,190,.32) !important;
    }
    /* Sidebar workspace navigation: selected page is a persistent active pill. */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
      gap: .42rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      background: rgba(21,43,66,.72) !important;
      border: 1px solid rgba(135,170,205,.16) !important;
      border-radius: 11px !important;
      padding: .58rem .72rem !important;
      margin: 0 !important;
      transition: all .18s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
      background: rgba(32,83,105,.75) !important;
      border-color: rgba(91,218,207,.55) !important;
      transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background: linear-gradient(100deg, rgba(34,151,151,.34), rgba(48,111,151,.28)) !important;
      border: 1px solid rgba(88,231,213,.72) !important;
      box-shadow: inset 3px 0 0 #52e0cf, 0 5px 18px rgba(0,0,0,.16) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
      color: #9cf6e8 !important;
      font-weight: 800 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
      display: none !important;
    }
    /* ONCOLENS sidebar: button-based workspace navigation. */
    .nav-heading {
      color: #dceafa;
      font-size: .76rem;
      font-weight: 800;
      letter-spacing: .14em;
      margin: .35rem 0 .25rem;
    }
    .nav-hint {
      color: #829bb5;
      font-size: .72rem;
      margin: 0 0 .8rem;
    }
    [data-testid="stSidebar"] .stButton > button {
      width: 100% !important;
      min-height: 3.35rem !important;
      justify-content: flex-start !important;
      text-align: left !important;
      padding: .72rem .95rem !important;
      border-radius: 13px !important;
      font-size: .91rem !important;
      font-weight: 650 !important;
      letter-spacing: .005em !important;
      margin: .12rem 0 !important;
      position: relative !important;
      overflow: hidden !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
      color: #c7d8eb !important;
      background: linear-gradient(110deg, rgba(22,43,66,.96), rgba(17,36,57,.96)) !important;
      border: 1px solid rgba(113,151,188,.22) !important;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 3px 10px rgba(0,0,0,.10) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
      color: #eaffff !important;
      background: linear-gradient(110deg, rgba(25,86,101,.98), rgba(25,64,91,.98)) !important;
      border-color: rgba(82,220,207,.68) !important;
      box-shadow: 0 7px 20px rgba(30,180,170,.15) !important;
      transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
      color: #06242b !important;
      background: linear-gradient(105deg, #51e1ce 0%, #61c8e9 100%) !important;
      border: 1px solid rgba(168,255,241,.9) !important;
      box-shadow: inset 4px 0 0 #d8fff8, 0 7px 22px rgba(48,211,195,.22) !important;
      font-weight: 850 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
      color: #031c24 !important;
      background: linear-gradient(105deg, #79f1df 0%, #85d9f5 100%) !important;
      box-shadow: inset 4px 0 0 #ffffff, 0 10px 27px rgba(48,211,195,.32) !important;
      transform: translateX(2px) !important;
    }
    [data-testid="stSidebar"] .stButton > button:focus-visible {
      outline: 3px solid rgba(99,231,217,.65) !important;
      outline-offset: 3px !important;
    }

    /* Tabs are styled as compact segmented navigation. */
    [data-testid="stTabs"] [role="tablist"] {
      gap: .45rem !important;
      border-bottom: 1px solid rgba(157,184,215,.16) !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
      color: #a9bfd5 !important;
      background: rgba(19,39,61,.72) !important;
      border: 1px solid rgba(135,170,205,.18) !important;
      border-radius: 10px 10px 0 0 !important;
      padding: .58rem .9rem !important;
      font-weight: 750 !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
      color: #a3f4e8 !important;
      background: linear-gradient(180deg, rgba(39,128,133,.35), rgba(19,49,67,.9)) !important;
      border-color: rgba(89,221,207,.55) !important;
      box-shadow: inset 0 -2px 0 #4ddacb !important;
    }
    /* Toggle switches: high-contrast track and thumb in both states. */
    [data-testid="stToggle"] [role="switch"] {
      background: #31465c !important;
      border: 1px solid #66809a !important;
      box-shadow: inset 0 1px 3px rgba(0,0,0,.28) !important;
    }
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
      background: linear-gradient(90deg, #168f8c, #45d6c5) !important;
      border-color: #75eee0 !important;
    }
    [data-testid="stToggle"] [role="switch"] > div {
      background: #f5fbff !important;
      box-shadow: 0 1px 5px rgba(0,0,0,.38) !important;
    }
    [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
      color: #dceafa !important;
    }
    [data-testid="stProgress"] > div > div > div > div {
      background: linear-gradient(90deg, #28b6ad, #65dcca) !important;
    }
    [data-testid="stFileUploader"] section {
      background: rgba(18,39,62,.72) !important;
      border: 1px dashed rgba(91,205,203,.5) !important;
      border-radius: 14px !important;
    }
    [data-testid="stFileUploader"] button {
      color: #e9f8ff !important;
      background: #1a5264 !important;
      border: 1px solid rgba(98,218,210,.55) !important;
      border-radius: 9px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      border: 1px solid rgba(157,184,215,.20) !important;
      border-radius: 13px !important;
      overflow: hidden !important;
      background: #102239 !important;
    }
    [data-testid="stDataFrame"] div[role="grid"],
    [data-testid="stDataFrame"] canvas { background: #102239 !important; }
    [data-testid="stDataFrame"] [role="columnheader"] {
      background: #19344f !important; color: #dff5ff !important;
      font-weight: 750 !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {
      color: #dceafa !important;
    }
    /* KPI cards: stronger contrast and consistent vertical rhythm. */
    div[data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(25,48,75,.98), rgba(15,32,52,.98)) !important;
      border: 1px solid rgba(104,164,199,.24) !important;
      border-radius: 16px !important;
      padding: 16px 18px !important;
      box-shadow: 0 8px 24px rgba(0,0,0,.14) !important;
      min-height: 112px;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] { color: #b9cce0 !important; font-weight: 650 !important; }
    div[data-testid="stMetricValue"] { color: #f4f9ff !important; font-weight: 800 !important; }
    div[data-testid="stMetricDelta"] { color: #8ce9d9 !important; }
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { min-width: 0; }
    .chart-panel { background: linear-gradient(145deg,#142b45,#0e2035); border:1px solid rgba(157,184,215,.18); border-radius:16px; padding:12px 14px 4px; }
    .chart-caption { color:#91a9c3; font-size:.76rem; margin:.25rem 0 .55rem; }
    .table-caption { color:#9fb2ca; font-size:.78rem; margin:.25rem 0 .5rem; }
    [data-testid="stTabs"] button { font-weight:700; }
    hr { border-color:var(--line); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- LOADERS -----------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    metadata = load_metadata()
    reference_df = load_reference_data()
except Exception as exc:
    st.error(f"Application could not initialize: {exc}")
    st.stop()

# Prefer the feature order captured by the fitted sklearn pipeline.
model_features = list(getattr(model, "feature_names_in_", []))
if not model_features:
    model_features = list(metadata.get("features", []))
if not model_features:
    model_features = APP_FEATURES.copy()

if len(model_features) != 30 or set(model_features) != set(APP_FEATURES):
    st.error("Model feature schema does not match the 30 WDBC application features.")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The loaded model does not support probability predictions.")
    st.stop()

classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
if classifier is None or 1 not in list(classifier.classes_):
    st.error("The saved pipeline does not expose the expected malignant class (1).")
    st.stop()

positive_index = list(classifier.classes_).index(1)
threshold = 0.5  # Matches the notebook's default binary decision threshold.

if reference_df is not None and set(model_features).issubset(reference_df.columns):
    medians = reference_df[model_features].median(numeric_only=True).to_dict()
else:
    medians = {feature: 0.0 for feature in model_features}

# ----------------------------- HELPERS -----------------------------

def predict_frame(frame: pd.DataFrame):
    ordered = frame[model_features].copy()
    probs = model.predict_proba(ordered)[:, positive_index]
    labels = (probs >= threshold).astype(int)
    return labels, probs

def normalize_diagnosis(series):
    mapping = {
        "B": 0, "BENIGN": 0, "0": 0,
        "M": 1, "MALIGNANT": 1, "1": 1,
    }
    normalized = series.astype(str).str.strip().str.upper().map(mapping)
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported diagnosis labels: {bad}. Use B/M or 0/1.")
    return normalized.astype(int)

def validate_upload(uploaded):
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}")

    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")]
    if raw.empty:
        raise ValueError("The uploaded CSV has no records.")

    missing = [f for f in model_features if f not in raw.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    allowed = set(model_features) | {"id", "diagnosis"}
    extra = [str(c) for c in raw.columns if c not in allowed]
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(extra))

    X = raw[model_features].copy()
    for col in model_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError("Missing/non-numeric values in: " + ", ".join(bad_cols))
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("CSV contains infinite or non-finite values.")

    ids = raw["id"].copy() if "id" in raw.columns else None
    y = normalize_diagnosis(raw["diagnosis"]) if "diagnosis" in raw.columns else None
    return raw, X, ids, y

def compact_table_height(row_count, row_height=36, header_height=38, minimum=92, maximum=440):
    """Choose a compact table height so short result sets do not show blank rows."""
    return min(maximum, max(minimum, header_height + row_height * min(int(row_count), 10) + 8))


def dark_table(frame: pd.DataFrame):
    """Apply a restrained dark palette to Streamlit's dataframe renderer."""
    return (
        frame.style
        .set_properties(**{
            "background-color": "#102239",
            "color": "#e3eefb",
            "border-color": "#29435f",
            "font-size": "0.86rem",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a3855"), ("color", "#e8f6ff"),
                ("font-weight", "700"), ("border-color", "#35516d"),
            ]},
            {"selector": "td", "props": [("border-color", "#29435f")]},
        ])
    )


def render_dark_bar_chart(labels, values, title, color="#45d6c5", xlabel="Records", percent=False):
    """Render a compact dark-theme horizontal bar chart with readable labels."""
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=color, edgecolor="#a9f5e8", linewidth=.5, height=.55)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_title(title, color="#edf6ff", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#b9cce0", fontsize=9, labelpad=8)
    ax.tick_params(axis="x", colors="#b9cce0", labelsize=8)
    ax.tick_params(axis="y", colors="#dceafa", labelsize=9, length=0, pad=8)
    ax.grid(axis="x", color="#8ba7c2", alpha=.18, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    max_value = max([float(v) for v in values], default=1.0)
    ax.set_xlim(0, max_value * 1.22 if max_value > 0 else 1)
    for bar, value in zip(bars, values):
        label = f"{value:.1%}" if percent else f"{int(value):,}"
        ax.text(bar.get_width() + max_value * .025, bar.get_y() + bar.get_height()/2,
                label, va="center", ha="left", color="#eff8ff", fontsize=9, fontweight="bold")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_probability_histogram(probabilities):
    """Render probability distribution on a dark canvas."""
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    bins = np.linspace(0, 1, 11)
    ax.hist(probabilities, bins=bins, color="#6da8ff", edgecolor="#d8eaff", alpha=.92)
    ax.axvline(threshold, color="#ffbd59", linestyle="--", linewidth=1.8, label=f"Decision threshold ({threshold:.2f})")
    ax.set_title("Malignant-class probability distribution", color="#edf6ff", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Estimated probability of class M", color="#b9cce0", fontsize=9)
    ax.set_ylabel("Number of records", color="#b9cce0", fontsize=9)
    ax.tick_params(colors="#b9cce0", labelsize=8)
    ax.grid(axis="y", color="#8ba7c2", alpha=.16, linestyle="--")
    ax.legend(facecolor="#19344f", edgecolor="#35516d", labelcolor="#eaf2ff", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_confusion_heatmap(matrix):
    """Display an uploaded-file confusion matrix without default white styling."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    image = ax.imshow(matrix, cmap="GnBu")
    ax.set_xticks([0, 1], ["Predicted B", "Predicted M"])
    ax.set_yticks([0, 1], ["Actual B", "Actual M"])
    ax.set_title("Uploaded dataset confusion matrix", color="#edf6ff", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors="#dceafa", labelsize=8)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="#071a29" if value > matrix.max() * .48 else "#eaf7ff", fontweight="bold", fontsize=12)
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def cell_visual(mode="overview"):
    # Stylized teaching illustration only; not a scan, histology image, or patient result.
    malignant = mode == "malignant"
    nucleus = "#ff7191" if malignant else "#69d9ce"
    accent = "#ff9baf" if malignant else "#9cf0df"
    title = "Illustrative malignant-like nucleus" if malignant else "Illustrative benign-like nucleus"
    return f"""
    <div style="background:linear-gradient(145deg,#122943,#0c1b2e);border:1px solid rgba(145,180,215,.18);
                border-radius:18px;padding:16px;color:#eaf2ff;font-family:Arial,sans-serif;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#8faac7;">Microscopy concept visual</div>
      <div style="font-size:15px;font-weight:700;margin:5px 0 0;">{title}</div>
      <svg viewBox="0 0 520 250" width="100%" role="img" aria-label="Stylized educational cell illustration">
        <defs>
          <radialGradient id="cyto"><stop offset="0%" stop-color="#315a78"/><stop offset="100%" stop-color="#17314c"/></radialGradient>
          <radialGradient id="nuc"><stop offset="0%" stop-color="{accent}"/><stop offset="100%" stop-color="{nucleus}"/></radialGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <g transform="translate(258 124)">
          <circle r="92" fill="url(#cyto)" stroke="#5d91b4" stroke-width="3"/>
          <circle r="80" fill="none" stroke="#7aa9c5" stroke-opacity=".25" stroke-dasharray="4 8"/>
          <g fill="#79a8c6" opacity=".65">
            <circle cx="-53" cy="-37" r="4"/><circle cx="52" cy="-44" r="3"/><circle cx="-60" cy="30" r="3"/>
            <circle cx="57" cy="37" r="4"/><circle cx="0" cy="-67" r="3"/><circle cx="8" cy="66" r="3"/>
          </g>
          <ellipse cx="0" cy="0" rx="{43 if malignant else 34}" ry="{51 if malignant else 36}"
                   fill="url(#nuc)" stroke="{accent}" stroke-width="2.5" filter="url(#glow)"
                   transform="rotate({-18 if malignant else 8})">
            <animate attributeName="rx" values="{43 if malignant else 34};{47 if malignant else 36};{43 if malignant else 34}" dur="3s" repeatCount="indefinite"/>
          </ellipse>
          <g fill="#6b244d" opacity=".8">
            <circle cx="-13" cy="-16" r="5"/><circle cx="14" cy="-4" r="6"/><circle cx="-4" cy="19" r="4"/>
            <circle cx="17" cy="23" r="3"/>
          </g>
          <circle r="104" fill="none" stroke="{accent}" stroke-opacity=".25" stroke-width="1.5">
            <animate attributeName="r" values="98;108;98" dur="4s" repeatCount="indefinite"/>
          </circle>
        </g>
        <g fill="#9fb8d0" font-size="11">
          <text x="20" y="28">Cell boundary</text><path d="M105 25 L180 55" stroke="#9fb8d0" stroke-opacity=".6"/>
          <text x="365" y="215">Nucleus</text><path d="M360 207 L290 145" stroke="#9fb8d0" stroke-opacity=".6"/>
        </g>
      </svg>
      <div style="font-size:11px;color:#91a9c3;line-height:1.5;">
        Conceptual animation only. Real cell appearance varies; this graphic is not generated from an individual's biopsy.
      </div>
    </div>
    """

def body_visual():
    # Non-diagnostic anatomy schematic. Highlight is a generic educational marker only.
    return """
    <div style="background:linear-gradient(145deg,#122943,#0c1b2e);border:1px solid rgba(145,180,215,.18);
                border-radius:18px;padding:16px;color:#eaf2ff;font-family:Arial,sans-serif;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#8faac7;">Anatomy context</div>
      <div style="font-size:15px;font-weight:700;margin:5px 0 0;">Illustrative upper-body schematic</div>
      <svg viewBox="0 0 520 250" width="100%" role="img" aria-label="Non-diagnostic upper-body schematic">
        <g fill="#294967" stroke="#7da4c4" stroke-width="2">
          <circle cx="260" cy="42" r="25" />
          <path d="M238 70 L222 82 Q195 88 184 111 L155 178 Q150 192 164 198 Q177 203 184 188 L211 132 L215 222 Q260 239 305 222 L309 132 L336 188 Q343 203 356 198 Q370 192 365 178 L336 111 Q325 88 298 82 L282 70 Z" />
        </g>
        <path d="M260 83 L260 221" stroke="#9ab7d0" stroke-opacity=".28" stroke-dasharray="4 6" />
        <circle cx="282" cy="125" r="13" fill="#ff7895" fill-opacity=".9">
          <animate attributeName="r" values="10;15;10" dur="2.4s" repeatCount="indefinite" />
          <animate attributeName="fill-opacity" values=".55;1;.55" dur="2.4s" repeatCount="indefinite" />
        </circle>
        <circle cx="282" cy="125" r="24" fill="none" stroke="#ff9db1" stroke-opacity=".55">
          <animate attributeName="r" values="18;31;18" dur="2.4s" repeatCount="indefinite" />
        </circle>
        <path d="M302 124 L380 92" stroke="#ff9db1" stroke-width="1.5" />
        <text x="386" y="88" fill="#f3b8c5" font-size="12">Illustrative marker</text>
        <text x="386" y="104" fill="#91a9c3" font-size="10">not a patient finding</text>
      </svg>
      <div style="font-size:11px;color:#91a9c3;line-height:1.5;">
        This schematic does not show tumour location, size, spread, percentage of body affected or cancer stage.
      </div>
    </div>
    """

def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="panel">
          <div class="panel-label">{html.escape(label)}</div>
          <div class="panel-value">{html.escape(str(value))}</div>
          <div class="panel-sub">{html.escape(note)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def render_prediction_result(label, probability):
    if label == 1:
        st.markdown(
            f"""<div class="result-malignant">
              <div class="eyebrow" style="color:#ff9aad;">Model classification</div>
              <h2 style="margin:6px 0;color:#fff;">Malignant class (M)</h2>
              <p style="color:#f0c3ce;margin:0;">Estimated model probability for class M: <b>{probability:.2%}</b></p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.warning("This is a model classification, not a diagnosis, cancer-volume estimate, or stage. Clinical assessment is required.")
        st.components.v1.html(cell_visual("malignant"), height=330, scrolling=False)
    else:
        st.markdown(
            f"""<div class="result-benign">
              <div class="eyebrow" style="color:#82eadc;">Model classification</div>
              <h2 style="margin:6px 0;color:#fff;">Benign class (B)</h2>
              <p style="color:#b8e9e4;margin:0;">Estimated model probability for class M: <b>{probability:.2%}</b></p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.success("The model classified this sample as benign. This does not rule out disease or replace clinical assessment.")
        st.components.v1.html(cell_visual("benign"), height=330, scrolling=False)

# ----------------------------- SIDEBAR -----------------------------

with st.sidebar:
    st.markdown("## 🧬 ONCOLENS")
    st.caption("BREAST DIAGNOSTIC INTELLIGENCE")
    st.divider()
    st.markdown(
        '<div class="nav-heading">WORKSPACE</div>'
        '<div class="nav-hint">Choose a workspace to explore</div>',
        unsafe_allow_html=True,
    )

    # Button-based navigation gives each destination a real click target
    # and a persistent selected state (rather than radio controls).
    nav_items = [
        ("Overview", "◈"),
        ("Single-sample analysis", "⌕"),
        ("Batch laboratory", "▤"),
        ("Explainability", "✧"),
        ("Model & methodology", "⌘"),
    ]
    if "oncolens_active_page" not in st.session_state:
        st.session_state["oncolens_active_page"] = "Overview"

    for nav_label, nav_icon in nav_items:
        is_active = st.session_state["oncolens_active_page"] == nav_label
        if st.button(
            f"{nav_icon}   {nav_label}",
            key=f"oncolens_nav_{nav_label.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["oncolens_active_page"] = nav_label
            st.rerun()

    page = st.session_state["oncolens_active_page"]
    st.divider()
    st.markdown("**ACTIVE MODEL**")
    st.markdown("Logistic Regression")
    st.caption("StandardScaler + fitted classifier")
    st.caption("30 WDBC nuclear morphology features")
    st.caption("Decision threshold: 0.50")
    st.divider()
    st.caption("Research / educational prototype • Not for clinical use")

# ----------------------------- HERO -----------------------------

st.markdown(
    """
    <div class="hero">
      <span class="pill">MACHINE LEARNING • EXPLAINABLE AI</span>
      <div class="eyebrow" style="margin-top:18px;">Diagnostic intelligence workspace</div>
      <h1>Breast Cancer Detection</h1>
      <p>Explore a reproducible machine-learning workflow for classifying breast-mass samples from
      fine-needle-aspirate nuclear measurements. Review predictions, model behaviour and feature
      importance in one carefully structured workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- OVERVIEW -----------------------------

if page == "Overview":
    st.markdown('<div class="section-kicker">01 / Executive overview</div><div class="section-title">Model at a glance</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("REFERENCE COHORT", "569", "WDBC observations"),
        ("INPUT SPACE", "30", "Continuous nuclear features"),
        ("HOLDOUT ACCURACY", "97.37%", "114 independent test observations"),
        ("HOLDOUT ROC–AUC", "0.9964", "Independent test set"),
    ]
    for col, (label, value, note) in zip(cols, cards):
        with col:
            metric_card(label, value, note)

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-kicker">Performance snapshot</div><div class="section-title">Independent holdout evaluation</div>', unsafe_allow_html=True)
        perf = pd.DataFrame({
            "Metric": list(HOLDOUT_METRICS.keys()),
            "Score": list(HOLDOUT_METRICS.values()),
        }).sort_values("Score", ascending=True)
        fig, ax = plt.subplots(figsize=(8.2, 4.1))
        bars = ax.barh(
            perf["Metric"], perf["Score"],
            color=["#4aa9bd", "#48bcae", "#54c9ba", "#64d7c4", "#6da8ff", "#8ac8f2"],
            edgecolor="#b8f4ec", linewidth=.45, height=.62
        )
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Score", color="#d7e7f5", labelpad=9)
        ax.set_title("Independent holdout metrics", color="#edf6ff", pad=12, weight="bold")
        ax.set_facecolor("#10243a")
        fig.patch.set_facecolor("#10243a")
        ax.tick_params(colors="#c3d5e7", labelsize=9)
        ax.grid(axis="x", color="#8ba7c2", alpha=.18, linestyle="--")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#45617c")
        for bar, score in zip(bars, perf["Score"]):
            ax.text(score + .012, bar.get_y() + bar.get_height()/2, f"{score:.2%}",
                    va="center", ha="left", color="#eaf5ff", fontsize=9, weight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Values are the previously recorded 114-observation independent test results.")
    with right:
        st.markdown('<div class="section-kicker">Classification profile</div><div class="section-title">Holdout confusion matrix</div>', unsafe_allow_html=True)
        cm_df = pd.DataFrame(HOLDOUT_CM, index=["Actual benign", "Actual malignant"], columns=["Predicted benign", "Predicted malignant"])
        st.dataframe(dark_table(cm_df), use_container_width=True, hide_index=False, height=compact_table_height(len(cm_df), row_height=48))
        st.caption("TN = 71 · FP = 1 · FN = 2 · TP = 40")

    st.write("")
    st.markdown('<div class="section-kicker">Visual learning</div><div class="section-title">Cell morphology & anatomy context</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.components.v1.html(cell_visual("benign"), height=330, scrolling=False)
    with b:
        st.components.v1.html(cell_visual("malignant"), height=330, scrolling=False)
    with c:
        st.components.v1.html(body_visual(), height=330, scrolling=False)

    st.markdown(
        '<div class="notice"><b>Interpretation boundary:</b> This model predicts a binary class for a sampled breast mass. '
        'It does not estimate tumour volume, percentage of the body affected, lymph-node involvement, metastasis or clinical stage.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- SINGLE SAMPLE -----------------------------

elif page == "Single-sample analysis":
    st.markdown('<div class="section-kicker">02 / Individual sample</div><div class="section-title">Single-sample analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Enter the 30 measurements extracted from a digitized FNA sample. Median defaults are provided only to demonstrate the interface.</div>', unsafe_allow_html=True)
    with st.form("single_sample_form"):
        entered = {}
        group_cols = st.columns(3, gap="medium")
        for (group_name, features), container in zip(FEATURE_GROUPS.items(), group_cols):
            with container:
                st.markdown(f"### {group_name}")
                for feature in features:
                    default = float(medians.get(feature, 0.0))
                    entered[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=0.0,
                        value=max(0.0, default),
                        format="%.5f",
                        help="Use a value measured from the relevant FNA-derived image analysis. Median default is illustrative.",
                    )
        submitted = st.form_submit_button("Run sample classification", type="primary", use_container_width=True)

    if submitted:
        try:
            sample = pd.DataFrame([[entered[f] for f in model_features]], columns=model_features)
            labels, probabilities = predict_frame(sample)
            render_prediction_result(int(labels[0]), float(probabilities[0]))
            c1, c2 = st.columns(2)
            c1.metric("Predicted class", "Malignant (M)" if labels[0] else "Benign (B)")
            c2.metric("Estimated probability of class M", f"{probabilities[0]:.2%}")
            st.progress(float(probabilities[0]), text="Model-estimated probability for malignant class")
            st.download_button(
                "Download this sample's prediction",
                data=pd.DataFrame([{
                    **entered,
                    "predicted_class": "M" if labels[0] else "B",
                    "malignant_probability": float(probabilities[0]),
                    "decision_threshold": threshold,
                }]).to_csv(index=False).encode("utf-8"),
                file_name="single_sample_prediction.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ----------------------------- BATCH LAB -----------------------------

elif page == "Batch laboratory":
    st.markdown('<div class="section-kicker">03 / Batch workflow</div><div class="section-title">Batch laboratory</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV containing all 30 model features. Optional columns: `id`, `diagnosis`.")
    st.markdown('<div class="notice"><b>Labelled-file caution:</b> If the CSV includes diagnosis labels, the app can compare predictions with those labels. This is an exploratory comparison on the uploaded records, not a new independent test, especially if the file contains records used during model training.</div>', unsafe_allow_html=True)

    template = pd.DataFrame([{f: medians.get(f, 0.0) for f in model_features}])
    st.download_button(
        "Download 30-feature CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="wdbc_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Choose a diagnostic CSV", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            raw, X_upload, ids, y_true = validate_upload(uploaded)
            st.success(f"Validation passed · {len(X_upload):,} records · 30 required features")
            with st.expander("Preview uploaded records", expanded=False):
                st.dataframe(raw.head(10), use_container_width=True)

            if st.button("Run batch predictions", type="primary", use_container_width=True):
                y_pred, prob = predict_frame(X_upload)
                output = pd.DataFrame()
                if ids is not None:
                    output["id"] = ids.to_numpy()
                output["predicted_diagnosis"] = np.where(y_pred == 1, "M", "B")
                output["malignant_probability"] = prob
                output["benign_probability"] = 1 - prob
                output["decision_threshold"] = threshold
                if y_true is not None:
                    output["actual_diagnosis"] = np.where(y_true.to_numpy() == 1, "M", "B")
                    output["prediction_match"] = (y_pred == y_true.to_numpy())

                st.session_state["batch_output"] = output
                st.session_state["batch_labels"] = y_true
                st.session_state["batch_predictions"] = y_pred
                st.session_state["batch_probabilities"] = prob

            if "batch_output" in st.session_state:
                result = st.session_state["batch_output"]
                st.markdown("### Prediction results")
                st.markdown('<div class="table-caption">Predictions generated by the saved Logistic Regression pipeline. A high probability is not a clinical diagnosis.</div>', unsafe_allow_html=True)
                display_result = result.copy()
                if "malignant_probability" in display_result:
                    display_result["malignant_probability"] = display_result["malignant_probability"].map(lambda x: f"{x:.2%}")
                if "benign_probability" in display_result:
                    display_result["benign_probability"] = display_result["benign_probability"].map(lambda x: f"{x:.2%}")
                if "decision_threshold" in display_result:
                    display_result["decision_threshold"] = display_result["decision_threshold"].map(lambda x: f"{x:.2f}")
                if "prediction_match" in display_result:
                    display_result["prediction_match"] = display_result["prediction_match"].map({True: "Match", False: "Mismatch"})
                st.dataframe(
                    dark_table(display_result),
                    use_container_width=True,
                    height=compact_table_height(len(display_result)),
                    hide_index=True,
                )

                pred_counts = result["predicted_diagnosis"].value_counts().reindex(["B", "M"], fill_value=0)
                c1, c2, c3 = st.columns(3, gap="medium")
                c1.metric("Records processed", f"{len(result):,}")
                c2.metric("Predicted benign", f"{pred_counts['B']:,}")
                c3.metric("Predicted malignant", f"{pred_counts['M']:,}")

                chart_left, chart_right = st.columns(2, gap="large")
                with chart_left:
                    st.markdown("### Predicted class distribution")
                    render_dark_bar_chart(["Benign", "Malignant"], [pred_counts["B"], pred_counts["M"]], "Predicted samples", color="#45cdbb", xlabel="Number of samples")
                with chart_right:
                    st.markdown("### Probability distribution")
                    render_probability_histogram(st.session_state["batch_probabilities"])

                labels_now = st.session_state["batch_labels"]
                if labels_now is not None:
                    pred_now = st.session_state["batch_predictions"]
                    st.markdown("### Uploaded dataset comparison")
                    st.warning("These metrics describe only the uploaded labelled file. They are exploratory and are not the original independent holdout evaluation.")
                    metrics = {
                        "Accuracy": accuracy_score(labels_now, pred_now),
                        "Precision (M)": precision_score(labels_now, pred_now, zero_division=0),
                        "Recall (M)": recall_score(labels_now, pred_now, zero_division=0),
                        "F1 (M)": f1_score(labels_now, pred_now, zero_division=0),
                    }
                    mc = st.columns(4, gap="medium")
                    for col, (name, value) in zip(mc, metrics.items()):
                        col.metric(name, f"{value:.2%}")
                    cm = confusion_matrix(labels_now, pred_now, labels=[0, 1])
                    render_confusion_heatmap(cm)

                st.download_button(
                    "Download batch prediction report",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="oncolens_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"CSV validation failed: {exc}")
            st.caption("Check column names, numeric values, missing values and diagnosis labels.")

# ----------------------------- EXPLAINABILITY -----------------------------

elif page == "Explainability":
    st.markdown('<div class="section-kicker">04 / Explainable AI</div><div class="section-title">What influenced the model?</div>', unsafe_allow_html=True)
    st.markdown("This section presents model-level feature analysis from the notebook. Importance is not causation and does not establish biological risk.")
    coef_path = EVAL_DIR / "logistic_regression_coefficients.csv"
    perm_path = EVAL_DIR / "permutation_feature_importance.csv"
    compare_path = EVAL_DIR / "feature_importance_comparison.csv"
    coef_img = EVAL_DIR / "logistic_regression_feature_importance.png"
    perm_img = EVAL_DIR / "permutation_feature_importance.png"
    compare_img = EVAL_DIR / "feature_importance_comparison.png"

    t1, t2, t3 = st.tabs(["Logistic coefficients", "Permutation importance", "Comparison"])
    with t1:
        if coef_path.exists():
            coef = pd.read_csv(coef_path)
            st.dataframe(dark_table(coef.head(15)), use_container_width=True, height=compact_table_height(min(len(coef), 15)), hide_index=True)
        else:
            st.info("Coefficient CSV not found in outputs/model_evaluation.")
        if coef_img.exists():
            st.image(str(coef_img), use_container_width=True)
    with t2:
        if perm_path.exists():
            perm = pd.read_csv(perm_path)
            st.dataframe(dark_table(perm.head(15)), use_container_width=True, height=compact_table_height(min(len(perm), 15)), hide_index=True)
        else:
            st.info("Permutation-importance CSV not found.")
        if perm_img.exists():
            st.image(str(perm_img), use_container_width=True)
    with t3:
        if compare_path.exists():
            comparison = pd.read_csv(compare_path)
            st.dataframe(dark_table(comparison.head(20)), use_container_width=True, height=compact_table_height(min(len(comparison), 20)), hide_index=True)
        else:
            st.info("Comparative feature-importance CSV not found.")
        if compare_img.exists():
            st.image(str(compare_img), use_container_width=True)

    st.markdown('<div class="notice"><b>Interpretation note:</b> Logistic-regression coefficients describe the fitted model on standardized inputs. Permutation importance measures the change in model score when a feature is shuffled. Correlated predictors can share or obscure importance; neither method proves causation.</div>', unsafe_allow_html=True)

# ----------------------------- METHODOLOGY -----------------------------

elif page == "Model & methodology":
    st.markdown('<div class="section-kicker">05 / Technical documentation</div><div class="section-title">Model & methodology</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Dataset")
        st.write("Wisconsin Diagnostic Breast Cancer (WDBC)")
        st.write("569 observations · 30 continuous predictors · binary diagnosis")
        st.write("Features are computed from digitized fine-needle-aspirate images and describe cell-nucleus characteristics.")
        st.markdown("### Pipeline")
        st.code("StandardScaler → LogisticRegression", language="text")
        st.write("The serialized scikit-learn pipeline is loaded directly. The app does not retrain or separately rescale inputs.")
    with right:
        st.markdown("### Decision rule")
        st.code("P(class = malignant) ≥ 0.50 → M\nP(class = malignant) < 0.50 → B", language="text")
        st.markdown("### Holdout results")
        holdout_table = pd.DataFrame({"Metric": list(HOLDOUT_METRICS.keys()), "Score": [f"{v:.4f}" for v in HOLDOUT_METRICS.values()]})
        st.dataframe(dark_table(holdout_table), use_container_width=True, height=compact_table_height(len(holdout_table)), hide_index=True)
    st.markdown("### Feature groups")
    for name, features in FEATURE_GROUPS.items():
        with st.expander(f"{name} · {len(features)} features"):
            st.write(", ".join(features))
    st.markdown(
        '<div class="notice"><b>Clinical boundary:</b> This application does not provide a clinical diagnosis, histological grade, tumour size, TNM stage, lymph-node status, metastatic status, treatment recommendation or whole-body cancer burden. These require clinical examination and additional diagnostic evidence.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Loaded metadata JSON"):
        st.json(metadata)

# ----------------------------- FOOTER -----------------------------

st.divider()
st.markdown(
    '<div style="text-align:center;color:#8299b4;font-size:.78rem;line-height:1.7;">'
    '<b style="color:#b9d1e8;">ONCOLENS</b> · Breast Diagnostic Intelligence<br>'
    'Educational machine-learning prototype · Not clinically validated · Not for medical decision-making'
    '</div>',
    unsafe_allow_html=True,
)
