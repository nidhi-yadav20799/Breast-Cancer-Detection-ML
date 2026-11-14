from pathlib import Path
import hashlib
import json
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ============================================================
# ONCOLENS — BREAST DIAGNOSTIC INTELLIGENCE
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "breast_cancer_logistic_regression.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "processed" / "wdbc_cleaned.csv"
EVAL_DIR = ROOT / "outputs" / "model_evaluation"

st.set_page_config(
    page_title="ONCOLENS | Breast Diagnostic Intelligence",
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

HOLDOUT_METRICS = {
    "Accuracy": 0.9737,
    "Precision": 0.9756,
    "Malignant recall": 0.9524,
    "Specificity": 0.9861,
    "F1 score": 0.9639,
    "ROC-AUC": 0.9964,
}
HOLDOUT_CM = np.array([[71, 1], [2, 40]])
THRESHOLD = 0.50

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
 --ink:#eaf2ff;
 --muted:#9fb2ca;
 --panel:#111f35;
 --panel2:#172943;
 --line:rgba(157,184,215,.18);
 --teal:#45d6c5;
 --blue:#6da8ff;
 --pink:#ff7191;
 --bg:#081321;
}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
 background:radial-gradient(ellipse at 12% 0%,#142b48 0%,#081321 48%,#07111e 100%);
 color:var(--ink);
}
[data-testid="stHeader"] {background:rgba(8,19,33,.85);}
[data-testid="stSidebar"] {
 background:#0c1a2c;
 border-right:1px solid var(--line);
}
[data-testid="stSidebar"] * {color:#dce9fa;}
.block-container {max-width:1500px;padding-top:1.5rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;}
.eyebrow {
 color:var(--teal);
 font-size:.72rem;
 font-weight:800;
 letter-spacing:.18em;
 text-transform:uppercase;
}
.hero {
 background:linear-gradient(115deg,rgba(24,53,83,.96),rgba(14,39,60,.92) 55%,rgba(14,73,77,.88));
 border:1px solid rgba(91,207,199,.25);
 border-radius:24px;
 padding:30px 32px;
 margin:4px 0 24px;
 box-shadow:0 18px 60px rgba(0,0,0,.22);
}
.hero h1 {
 color:#f5f9ff;
 font-size:clamp(2rem,4vw,3.15rem);
 margin:12px 0 8px;
}
.hero p {
 color:#c2d3e7;
 font-size:1rem;
 max-width:780px;
 line-height:1.65;
}
.pill {
 display:inline-block;
 border:1px solid rgba(69,214,197,.38);
 color:#8ef0df;
 background:rgba(69,214,197,.08);
 padding:5px 10px;
 border-radius:99px;
 font-size:.72rem;
 font-weight:700;
}
.panel {
 background:linear-gradient(145deg,rgba(23,41,67,.92),rgba(14,29,48,.96));
 border:1px solid var(--line);
 border-radius:18px;
 padding:19px 20px;
 height:100%;
}
.panel-label {
 color:#b9cce0;
 font-size:.75rem;
 letter-spacing:.11em;
 font-weight:700;
}
.panel-value {
 color:#f4f8ff;
 font-family:'Manrope';
 font-size:1.8rem;
 font-weight:800;
 margin-top:6px;
}
.panel-sub {color:#9fb2ca;font-size:.78rem;margin-top:4px;}
.section-kicker {
 color:#6fe0d2;
 font-size:.72rem;
 letter-spacing:.15em;
 font-weight:800;
 margin-bottom:5px;
}
.section-title {
 font-family:'Manrope';
 font-size:1.55rem;
 font-weight:800;
 color:#f0f6ff;
 margin:0 0 12px;
}
.muted {color:#9fb2ca;}
.notice {
 background:rgba(255,184,77,.09);
 border:1px solid rgba(255,184,77,.30);
 border-left:4px solid #ffbd59;
 border-radius:12px;
 padding:13px 15px;
 color:#f5d9a8;
 font-size:.86rem;
 line-height:1.55;
}
.result-malignant {
 background:linear-gradient(135deg,rgba(120,39,58,.35),rgba(52,25,44,.45));
 border:1px solid rgba(255,113,137,.38);
 border-radius:18px;
 padding:22px;
}
.result-benign {
 background:linear-gradient(135deg,rgba(24,105,99,.28),rgba(19,54,67,.45));
 border:1px solid rgba(69,214,197,.35);
 border-radius:18px;
 padding:22px;
}
/* HIGH-CONTRAST METRIC CARDS */
[data-testid="stMetric"] {
 background: linear-gradient(145deg, #193b58, #10263f) !important;
 border: 1px solid rgba(111, 211, 215, 0.30) !important;
 border-radius: 16px !important;
 padding: 17px !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
[data-testid="stMetric"] [data-testid="stMetricLabel"] div {
 color: #c8dced !important;
 opacity: 1 !important;
 font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] div {
 color: #f4f9ff !important;
 opacity: 1 !important;
 font-weight: 800 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
 color: #83e8d8 !important;
}
.stButton>button,.stDownloadButton>button,
[data-testid="stFormSubmitButton"]>button {
 min-height:2.85rem!important;
 border-radius:12px!important;
 padding:.62rem 1.05rem!important;
 font-weight:750!important;
 color:#eaf7ff!important;
 background:linear-gradient(135deg,#183c59,#174e62)!important;
 border:1px solid rgba(104,205,220,.48)!important;
 box-shadow:0 5px 16px rgba(0,0,0,.18)!important;
 transition:all .18s ease!important;
}
.stButton>button:hover,.stDownloadButton>button:hover {
 color:white!important;
 background:linear-gradient(135deg,#167f88,#1b9e9a)!important;
 border-color:#65e4d5!important;
 transform:translateY(-1px);
}
.stButton>button[kind="primary"],
[data-testid="stFormSubmitButton"]>button[kind="primary"] {
 color:#031d26!important;
 background:linear-gradient(135deg,#43ddc7,#54b9ed)!important;
 border:1px solid rgba(154,255,240,.8)!important;
}
[data-testid="stSidebar"] .stButton>button {
 width:100%;
 min-height:3.2rem!important;
 text-align:left!important;
 justify-content:flex-start!important;
 margin:.12rem 0!important;
}
[data-testid="stSidebar"] .stButton>button[kind="secondary"] {
 color:#c7d8eb!important;
 background:linear-gradient(110deg,#162b42,#112439)!important;
 border:1px solid rgba(113,151,188,.22)!important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"] {
 color:#06242b!important;
 background:linear-gradient(105deg,#51e1ce,#61c8e9)!important;
 border:1px solid rgba(168,255,241,.9)!important;
}
[data-testid="stForm"] {
 background:linear-gradient(145deg,rgba(17,35,56,.78),rgba(12,27,45,.88));
 border:1px solid rgba(112,157,193,.24);
 border-radius:18px;
 padding:1rem;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input {
 color:#eff8ff!important;
 -webkit-text-fill-color:#eff8ff!important;
 background:#10243a!important;
 border:1px solid #35536f!important;
 border-radius:10px!important;
}
[data-testid="stNumberInput"] button {
 color:#dff8f5!important;
 background:#1a3a54!important;
 border:1px solid #35536f!important;
}
[data-testid="stFileUploader"] section {
 background:linear-gradient(145deg,#132b42,#0d1f32)!important;
 border:1px dashed rgba(91,205,203,.62)!important;
 border-radius:15px!important;
}
[data-testid="stFileUploader"] * {color:#dceafa!important;}
[data-testid="stExpander"] {
 background:rgba(16,35,56,.72)!important;
 border:1px solid rgba(135,170,205,.20)!important;
 border-radius:13px!important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {color:#dceafa!important;}
[data-testid="stTabs"] button[role="tab"] {
 color:#a9bfd5!important;
 background:rgba(19,39,61,.72)!important;
 border:1px solid rgba(135,170,205,.18)!important;
 border-radius:10px 10px 0 0!important;
 padding:.58rem .9rem!important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
 color:#a3f4e8!important;
 background:linear-gradient(180deg,rgba(39,128,133,.35),rgba(19,49,67,.9))!important;
 border-color:rgba(89,221,207,.55)!important;
}
[data-testid="stDataFrame"] {
 border:1px solid rgba(157,184,215,.24)!important;
 border-radius:13px!important;
 overflow:hidden!important;
}
hr {border-color:var(--line);}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
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
    st.error(f"Application initialization failed: {exc}")
    st.stop()

model_features = list(getattr(model, "feature_names_in_", []))
if not model_features:
    model_features = list(metadata.get("features", []))
if not model_features:
    model_features = APP_FEATURES.copy()

if len(model_features) != 30 or set(model_features) != set(APP_FEATURES):
    st.error("The saved model feature schema does not match the WDBC features.")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support probability predictions.")
    st.stop()

# Resolve the fitted classifier from the saved pipeline without assuming
# that its pipeline step is literally named "classifier".
if hasattr(model, "named_steps") and model.named_steps:
    classifier = list(model.named_steps.values())[-1]
else:
    classifier = model

if not hasattr(classifier, "classes_"):
    st.error("The saved model does not expose fitted class labels.")
    st.stop()

model_classes = list(classifier.classes_)
if len(model_classes) != 2:
    st.error(
        "ONCOLENS requires a binary classifier with exactly two classes. "
        f"The saved model exposes: {model_classes}"
    )
    st.stop()


def class_label_to_binary(value):
    """Map supported WDBC labels to 0=benign and 1=malignant."""
    label = str(value).strip().upper()
    if label in {"M", "MALIGNANT", "1", "1.0", "TRUE"}:
        return 1
    if label in {"B", "BENIGN", "0", "0.0", "FALSE"}:
        return 0
    raise ValueError(
        f"Unsupported model class label {value!r}. "
        "Expected B/M or 0/1 labels."
    )


try:
    binary_model_classes = [class_label_to_binary(value) for value in model_classes]
except ValueError as exc:
    st.error(
        f"Cannot identify the malignant class in the saved model: {exc} "
        "Check the model target encoding and metadata."
    )
    st.stop()

if set(binary_model_classes) != {0, 1}:
    st.error(
        "The saved model classes could not be mapped unambiguously to "
        "benign (B/0) and malignant (M/1)."
    )
    st.stop()

positive_index = binary_model_classes.index(1)

if reference_df is not None and set(model_features).issubset(reference_df.columns):
    medians = reference_df[model_features].median(numeric_only=True).to_dict()
else:
    medians = {feature: 0.0 for feature in model_features}

# ============================================================
# HELPERS
# ============================================================

def predict_frame(frame):
    """Return 0=benign / 1=malignant predictions and P(malignant)."""
    ordered = frame[model_features].copy()
    probabilities = model.predict_proba(ordered)[:, positive_index]
    predictions = (probabilities >= THRESHOLD).astype(int)
    return predictions, probabilities


def normalize_diagnosis(series):
    mapping = {
        "B": 0, "BENIGN": 0, "0": 0,
        "M": 1, "MALIGNANT": 1, "1": 1,
    }
    normalized = (
        series.astype(str).str.strip().str.upper().map(mapping)
    )
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(
            f"Unsupported diagnosis labels: {bad}. Use B/M or 0/1."
        )
    return normalized.astype(int)


def validate_upload(uploaded):
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}")

    raw.columns = raw.columns.astype(str).str.strip()
    raw = raw.loc[
        :, ~raw.columns.astype(str).str.startswith("Unnamed:")
    ]

    if raw.empty:
        raise ValueError("The uploaded CSV contains no records.")

    missing = [f for f in model_features if f not in raw.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    allowed = set(model_features) | {"id", "diagnosis"}
    extra = [str(c) for c in raw.columns if c not in allowed]
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(extra))

    X = raw[model_features].copy()
    for col in model_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing/non-numeric values in: " + ", ".join(bad)
        )

    numeric_values = X.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("CSV contains infinite or non-finite values.")
    if (numeric_values < 0).any():
        raise ValueError("WDBC feature measurements must be non-negative.")

    ids = raw["id"].copy() if "id" in raw.columns else None
    if ids is not None and ids.isna().any():
        raise ValueError("The optional id column contains missing values.")
    y = (
        normalize_diagnosis(raw["diagnosis"])
        if "diagnosis" in raw.columns else None
    )
    return raw, X, ids, y


def compact_table_height(n, row_height=36):
    return min(440, max(92, 46 + row_height * min(int(n), 10)))


def dark_table(frame):
    return (
        frame.style
        .set_properties(**{
            "background-color": "#102239",
            "color": "#e3eefb",
            "border-color": "#29435f",
            "font-size": "0.84rem",
            "padding": "0.48rem 0.62rem",
            "white-space": "nowrap",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a3855"),
                ("color", "#e8f6ff"),
                ("font-weight", "700"),
                ("border-color", "#35516d"),
                ("padding", "0.58rem 0.65rem"),
            ]},
            {"selector": "td", "props": [("border-color", "#29435f")]},
            {"selector": "tbody tr:nth-child(even) td", "props": [
                ("background-color", "#142b43")
            ]},
        ])
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-label">{html.escape(label)}</div>
          <div class="panel-value">{html.escape(str(value))}</div>
          <div class="panel-sub">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ORIGINAL ANIMATED CELL ILLUSTRATIONS
# ============================================================

def cell_visual(mode="overview"):
    malignant = mode == "malignant"
    nucleus = "#ff7191" if malignant else "#69d9ce"
    accent = "#ff9baf" if malignant else "#9cf0df"
    title = (
        "Illustrative malignant-like nucleus"
        if malignant else "Illustrative benign-like nucleus"
    )

    return f"""
    <div style="background:linear-gradient(145deg,#122943,#0c1b2e);
                border:1px solid rgba(145,180,215,.18);
                border-radius:18px;padding:16px;color:#eaf2ff;
                font-family:Arial,sans-serif;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                  color:#8faac7;">Microscopy concept visual</div>
      <div style="font-size:15px;font-weight:700;margin:5px 0 0;">{title}</div>
      <svg viewBox="0 0 520 250" width="100%" role="img"
           aria-label="Stylized educational cell illustration">
        <defs>
          <radialGradient id="cyto">
            <stop offset="0%" stop-color="#315a78"/>
            <stop offset="100%" stop-color="#17314c"/>
          </radialGradient>
          <radialGradient id="nuc">
            <stop offset="0%" stop-color="{accent}"/>
            <stop offset="100%" stop-color="{nucleus}"/>
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="5" result="b"/>
            <feMerge>
              <feMergeNode in="b"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <g transform="translate(258 124)">
          <circle r="92" fill="url(#cyto)" stroke="#5d91b4" stroke-width="3"/>
          <circle r="80" fill="none" stroke="#7aa9c5"
                  stroke-opacity=".25" stroke-dasharray="4 8"/>
          <g fill="#79a8c6" opacity=".65">
            <circle cx="-53" cy="-37" r="4"/>
            <circle cx="52" cy="-44" r="3"/>
            <circle cx="-60" cy="30" r="3"/>
            <circle cx="57" cy="37" r="4"/>
            <circle cx="0" cy="-67" r="3"/>
            <circle cx="8" cy="66" r="3"/>
          </g>
          <ellipse cx="0" cy="0"
                   rx="{43 if malignant else 34}"
                   ry="{51 if malignant else 36}"
                   fill="url(#nuc)" stroke="{accent}"
                   stroke-width="2.5" filter="url(#glow)"
                   transform="rotate({-18 if malignant else 8})">
            <animate attributeName="rx"
                     values="{43 if malignant else 34};{47 if malignant else 36};{43 if malignant else 34}"
                     dur="3s" repeatCount="indefinite"/>
          </ellipse>
          <g fill="#6b244d" opacity=".8">
            <circle cx="-13" cy="-16" r="5"/>
            <circle cx="14" cy="-4" r="6"/>
            <circle cx="-4" cy="19" r="4"/>
            <circle cx="17" cy="23" r="3"/>
          </g>
          <circle r="104" fill="none" stroke="{accent}"
                  stroke-opacity=".25" stroke-width="1.5">
            <animate attributeName="r" values="98;108;98"
                     dur="4s" repeatCount="indefinite"/>
          </circle>
        </g>
        <g fill="#9fb8d0" font-size="11">
          <text x="20" y="28">Cell boundary</text>
          <path d="M105 25 L180 55" stroke="#9fb8d0" stroke-opacity=".6"/>
          <text x="365" y="215">Nucleus</text>
          <path d="M360 207 L290 145" stroke="#9fb8d0" stroke-opacity=".6"/>
        </g>
      </svg>
      <div style="font-size:11px;color:#91a9c3;line-height:1.5;">
        Conceptual animation only. Not generated from an individual's biopsy.
      </div>
    </div>
    """


def body_visual():
    return """
    <div style="background:linear-gradient(145deg,#122943,#0c1b2e);
                border:1px solid rgba(145,180,215,.18);
                border-radius:18px;padding:16px;color:#eaf2ff;
                font-family:Arial,sans-serif;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                  color:#8faac7;">Anatomy context</div>
      <div style="font-size:15px;font-weight:700;margin:5px 0 0;">
        Illustrative upper-body schematic
      </div>
      <svg viewBox="0 0 520 250" width="100%" role="img"
           aria-label="Non-diagnostic upper-body schematic">
        <g fill="#294967" stroke="#7da4c4" stroke-width="2">
          <circle cx="260" cy="42" r="25"/>
          <path d="M238 70 L222 82 Q195 88 184 111 L155 178
                   Q150 192 164 198 Q177 203 184 188 L211 132
                   L215 222 Q260 239 305 222 L309 132 L336 188
                   Q343 203 356 198 Q370 192 365 178 L336 111
                   Q325 88 298 82 L282 70 Z"/>
        </g>
        <path d="M260 83 L260 221" stroke="#9ab7d0"
              stroke-opacity=".28" stroke-dasharray="4 6"/>
        <circle cx="282" cy="125" r="13"
                fill="#ff7895" fill-opacity=".9">
          <animate attributeName="r" values="10;15;10"
                   dur="2.4s" repeatCount="indefinite"/>
          <animate attributeName="fill-opacity" values=".55;1;.55"
                   dur="2.4s" repeatCount="indefinite"/>
        </circle>
        <circle cx="282" cy="125" r="24" fill="none"
                stroke="#ff9db1" stroke-opacity=".55">
          <animate attributeName="r" values="18;31;18"
                   dur="2.4s" repeatCount="indefinite"/>
        </circle>
        <path d="M302 124 L380 92" stroke="#ff9db1" stroke-width="1.5"/>
        <text x="386" y="88" fill="#f3b8c5" font-size="12">
          Illustrative marker
        </text>
        <text x="386" y="104" fill="#91a9c3" font-size="10">
          not a patient finding
        </text>
      </svg>
      <div style="font-size:11px;color:#91a9c3;line-height:1.5;">
        This schematic does not show tumour location, size, spread,
        percentage of body affected or cancer stage.
      </div>
    </div>
    """


def render_prediction_result(label, probability):
    if label == 1:
        st.markdown(
            f"""
            <div class="result-malignant">
              <div class="eyebrow" style="color:#ff9aad;">Model classification</div>
              <h2 style="margin:6px 0;color:#fff;">Malignant class (M)</h2>
              <p style="color:#f0c3ce;margin:0;">
                Estimated model probability for class M:
                <b>{probability:.2%}</b>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "This is a model classification, not a diagnosis, cancer-volume "
            "estimate or stage. Clinical assessment is required."
        )
        st.components.v1.html(cell_visual("malignant"), height=330, scrolling=False)
    else:
        st.markdown(
            f"""
            <div class="result-benign">
              <div class="eyebrow" style="color:#82eadc;">Model classification</div>
              <h2 style="margin:6px 0;color:#fff;">Benign class (B)</h2>
              <p style="color:#b8e9e4;margin:0;">
                Estimated model probability for class M:
                <b>{probability:.2%}</b>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(
            "The model classified this sample as benign. This does not "
            "rule out disease or replace clinical assessment."
        )
        st.components.v1.html(cell_visual("benign"), height=330, scrolling=False)


# ============================================================
# CHART HELPERS
# ============================================================

def render_dark_bar_chart(labels, values, title, color="#45d6c5", xlabel="Records"):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, values, color=color,
        edgecolor="#a9f5e8", linewidth=.5, height=.55
    )
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_title(title, color="#edf6ff", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color="#b9cce0")
    ax.tick_params(colors="#b9cce0")
    ax.grid(axis="x", color="#8ba7c2", alpha=.18, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    maximum = max([float(v) for v in values], default=1)
    ax.set_xlim(0, maximum * 1.22 if maximum > 0 else 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + maximum * .025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center", color="#eff8ff", fontsize=9, fontweight="bold"
        )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_probability_histogram(probabilities):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    ax.hist(
        probabilities, bins=np.linspace(0, 1, 11),
        color="#6da8ff", edgecolor="#d8eaff", alpha=.92
    )
    ax.axvline(
        THRESHOLD, color="#ffbd59", linestyle="--",
        linewidth=1.8, label=f"Decision threshold ({THRESHOLD:.2f})"
    )
    ax.set_title(
        "Malignant-class probability distribution",
        color="#edf6ff", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Estimated probability of class M", color="#b9cce0")
    ax.set_ylabel("Number of records", color="#b9cce0")
    ax.tick_params(colors="#b9cce0")
    ax.grid(axis="y", color="#8ba7c2", alpha=.16, linestyle="--")
    ax.legend(
        facecolor="#19344f", edgecolor="#35516d",
        labelcolor="#eaf2ff", fontsize=8
    )
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_confusion_heatmap(matrix):
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    fig.patch.set_facecolor("#10243a")
    ax.set_facecolor("#10243a")
    image = ax.imshow(matrix, cmap="GnBu")
    ax.set_xticks([0, 1], ["Predicted B", "Predicted M"])
    ax.set_yticks([0, 1], ["Actual B", "Actual M"])
    ax.set_title("Uploaded dataset confusion matrix", color="#edf6ff")
    ax.tick_params(colors="#dceafa")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j, i, f"{value:,}", ha="center", va="center",
            color="#071a29" if value > matrix.max() * .48 else "#eaf7ff",
            fontweight="bold", fontsize=12
        )
    for spine in ax.spines.values():
        spine.set_color("#35516d")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    ("Overview", "◈"),
    ("Single-sample analysis", "⌕"),
    ("Batch laboratory", "▤"),
    ("Explainability", "✧"),
    ("Model & methodology", "⌘"),
]

if "oncolens_active_page" not in st.session_state:
    st.session_state["oncolens_active_page"] = "Overview"

with st.sidebar:
    st.markdown("## 🧬 ONCOLENS")
    st.caption("BREAST DIAGNOSTIC INTELLIGENCE")
    st.divider()
    st.markdown("**WORKSPACE**")
    st.caption("Choose a workspace to explore")

    for label, icon in PAGES:
        active = st.session_state["oncolens_active_page"] == label
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{label}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state["oncolens_active_page"] = label
            st.rerun()

    page = st.session_state["oncolens_active_page"]

    st.divider()
    st.markdown("**ACTIVE MODEL**")
    st.caption("Logistic Regression")
    st.caption("StandardScaler + fitted classifier")
    st.caption("30 WDBC features")
    st.caption("Decision threshold: 0.50")
    st.divider()
    st.caption("Educational research prototype")
    st.caption("Not for clinical use")

# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <span class="pill">MACHINE LEARNING • EXPLAINABLE AI</span>
    <div class="eyebrow" style="margin-top:18px;">
        Diagnostic intelligence workspace
    </div>
    <h1>Breast Cancer Detection</h1>
    <p>
        Explore a machine-learning workflow for classifying breast-mass
        samples from fine-needle-aspirate nuclear measurements.
        Review predictions, model behaviour and feature importance.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 01 — OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-kicker">01 / Executive overview</div>'
        '<div class="section-title">Model at a glance</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Evaluation note: the holdout split was used to compare tuned candidate "
        "models and select the final model. These scores are exploratory, not "
        "an untouched independent final test estimate."
    )

    cols = st.columns(4)
    cards = [
        ("REFERENCE COHORT", "569", "WDBC observations"),
        ("INPUT SPACE", "30", "Continuous nuclear features"),
        ("HOLDOUT ACCURACY", "97.37%", "114 holdout observations"),
        ("HOLDOUT ROC-AUC", "0.9964", "Exploratory holdout estimate"),
    ]

    for col, (label, value, note) in zip(cols, cards):
        with col:
            metric_card(label, value, note)

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("### Exploratory holdout performance")
        performance = pd.DataFrame({
            "Metric": list(HOLDOUT_METRICS.keys()),
            "Score": list(HOLDOUT_METRICS.values()),
        }).sort_values("Score", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#10243a")
        ax.set_facecolor("#10243a")
        bars = ax.barh(
            performance["Metric"], performance["Score"],
            color=["#4aa9bd", "#48bcae", "#54c9ba",
                   "#64d7c4", "#6da8ff", "#8ac8f2"],
            edgecolor="#b8f4ec", linewidth=.45,
        )
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Score", color="#b9cce0")
        ax.set_title("Holdout performance (exploratory)", color="#edf6ff")
        ax.tick_params(colors="#dceafa")
        for bar, value in zip(bars, performance["Score"]):
            ax.text(
                value + .012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2%}",
                va="center", color="#eff8ff",
                fontsize=9, fontweight="bold",
            )
        for spine in ax.spines.values():
            spine.set_color("#35516d")
        ax.grid(axis="x", color="#8ba7c2", alpha=.16, linestyle="--")
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Holdout confusion matrix")
        fig, ax = plt.subplots(figsize=(5.4, 4))
        fig.patch.set_facecolor("#10243a")
        ax.set_facecolor("#10243a")
        ax.imshow(HOLDOUT_CM, cmap="GnBu")
        ax.set_xticks([0, 1], ["Benign", "Malignant"])
        ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set_xlabel("Predicted class", color="#dceafa")
        ax.set_ylabel("Actual class", color="#dceafa")
        ax.set_title("Holdout confusion matrix", color="#edf6ff")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(HOLDOUT_CM[i, j]),
                    ha="center", va="center",
                    color="#071a29" if HOLDOUT_CM[i, j] > 35 else "#eaf7ff",
                    fontsize=15, fontweight="bold",
                )
        ax.tick_params(colors="#dceafa")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Animated cell illustrations")
    cell_left, cell_right = st.columns(2, gap="large")

    with cell_left:
        st.components.v1.html(
            cell_visual("benign"), height=330, scrolling=False
        )
    with cell_right:
        st.components.v1.html(
            cell_visual("malignant"), height=330, scrolling=False
        )

    st.markdown("### Anatomy context")
    st.components.v1.html(body_visual(), height=700, scrolling=False)

    st.markdown("""
    <div class="notice">
    <b>Important:</b> These cell graphics and anatomy schematic are
    conceptual educational illustrations. They are not patient images,
    biopsy findings, tumour maps, cancer-stage indicators or estimates
    of the percentage of the body affected.
    </div>
    """, unsafe_allow_html=True)

    if reference_df is not None:
        st.markdown("### Dataset snapshot")

        if "diagnosis" in reference_df.columns:
            try:
                reference_labels = normalize_diagnosis(
                    reference_df["diagnosis"]
                )
                benign_count = int((reference_labels == 0).sum())
                malignant_count = int((reference_labels == 1).sum())
            except ValueError:
                benign_count = 0
                malignant_count = 0
                st.warning(
                    "Dataset snapshot counts could not be calculated because "
                    "the diagnosis column contains unsupported labels."
                )
        else:
            benign_count = 0
            malignant_count = 0

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            metric_card("OBSERVATIONS", f"{len(reference_df):,}", "Total WDBC records")
        with c2:
            metric_card("PREDICTORS", "30", "Numeric model features")
        with c3:
            metric_card("BENIGN RECORDS", f"{benign_count:,}", "Reference class B")
        with c4:
            metric_card("MALIGNANT RECORDS", f"{malignant_count:,}", "Reference class M")


# ============================================================
# 02 — SINGLE-SAMPLE ANALYSIS
# ============================================================

elif page == "Single-sample analysis":

    st.markdown(
        '<div class="section-kicker">02 / Individual sample workspace</div>'
        '<div class="section-title">Single-sample analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="notice">
    <b>DEMO DEFAULTS:</b> Initial values are dataset medians, not
    measurements from a real patient. Replace them with valid
    laboratory measurements before interpreting a prediction.
    </div>
    """, unsafe_allow_html=True)

    values = {}

    with st.form("single_sample_form"):

        for group_name, features in FEATURE_GROUPS.items():
            st.markdown(f"### {group_name}")
            columns = st.columns(2)

            for index, feature in enumerate(features):
                with columns[index % 2]:
                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=0.0,
                        value=float(medians.get(feature, 0.0)),
                        step=0.01,
                        format="%.5f",
                        key=f"sample_{feature}",
                    )

        submitted = st.form_submit_button(
            "🔬 ANALYZE SAMPLE",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            sample = pd.DataFrame([values], columns=model_features)
            predictions, probabilities = predict_frame(sample)
            label = int(predictions[0])
            malignant_probability = float(probabilities[0])

            st.markdown("### Analysis result")
            render_prediction_result(label, malignant_probability)

            c1, c2 = st.columns(2)
            c1.metric(
                "Malignant probability",
                f"{malignant_probability:.2%}",
            )
            c2.metric(
                "Benign probability",
                f"{1 - malignant_probability:.2%}",
            )

            fig, ax = plt.subplots(figsize=(8, 1.8))
            fig.patch.set_facecolor("#10243a")
            ax.set_facecolor("#10243a")
            ax.barh(
                ["Class probability"], [malignant_probability],
                color="#ff7191", label="Malignant",
            )
            ax.barh(
                ["Class probability"], [1 - malignant_probability],
                left=[malignant_probability],
                color="#45d6c5", label="Benign",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", color="#b9cce0")
            ax.tick_params(colors="#dceafa")
            ax.legend(
                facecolor="#19344f",
                edgecolor="#35516d",
                labelcolor="#eaf2ff",
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ============================================================
# 03 — BATCH LABORATORY
# ============================================================

elif page == "Batch laboratory":

    st.markdown(
        '<div class="section-kicker">03 / Multi-sample workspace</div>'
        '<div class="section-title">Batch laboratory</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="panel">
      <h3>CSV upload requirements</h3>
      <p class="muted">
      Include all 30 WDBC predictor columns. Optional columns:
      <b>id</b> and <b>diagnosis</b>. Diagnosis labels may be B/M or 0/1.
      </p>
    </div>
    """, unsafe_allow_html=True)

    template = pd.DataFrame([
        {f: medians.get(f, 0.0) for f in model_features}
    ])

    st.download_button(
        "Download 30-feature CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="wdbc_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Choose a diagnostic CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:

        # FIX 1: identify each upload and invalidate stale previous results.
        upload_signature = hashlib.sha256(
            uploaded.getvalue()
        ).hexdigest()

        if st.session_state.get("batch_upload_signature") != upload_signature:
            for key in (
                "batch_output",
                "batch_labels",
                "batch_predictions",
                "batch_probabilities",
                "batch_source_signature",
            ):
                st.session_state.pop(key, None)

            st.session_state["batch_upload_signature"] = upload_signature

        try:
            raw, X_upload, ids, y_true = validate_upload(uploaded)

            st.success(
                f"Validation passed · {len(X_upload):,} records · "
                "30 required features"
            )

            with st.expander("Preview uploaded records", expanded=False):
                st.dataframe(
                    dark_table(raw.head(10)),
                    use_container_width=True,
                    height=compact_table_height(min(len(raw), 10)),
                    hide_index=True,
                )

            if st.button(
                "Run batch predictions",
                type="primary",
                use_container_width=True,
            ):

                y_pred, probabilities = predict_frame(X_upload)

                # FIX 2: ensure predictions retain every input record.
                expected_rows = len(X_upload)

                if (
                    len(y_pred) != expected_rows
                    or len(probabilities) != expected_rows
                ):
                    raise RuntimeError(
                        f"Prediction row-count mismatch. Expected "
                        f"{expected_rows}; got {len(y_pred)} labels and "
                        f"{len(probabilities)} probabilities."
                    )

                output = pd.DataFrame(index=range(expected_rows))

                if ids is not None:
                    output["id"] = ids.to_numpy()

                output["predicted_diagnosis"] = np.where(
                    y_pred == 1, "M", "B"
                )
                output["malignant_probability"] = probabilities
                output["benign_probability"] = 1 - probabilities
                output["decision_threshold"] = THRESHOLD

                if y_true is not None:
                    output["actual_diagnosis"] = np.where(
                        y_true.to_numpy() == 1, "M", "B"
                    )
                    output["prediction_match"] = (
                        y_pred == y_true.to_numpy()
                    )

                if len(output) != expected_rows:
                    raise RuntimeError(
                        "Output report row count does not match input rows."
                    )

                st.session_state["batch_output"] = output
                st.session_state["batch_labels"] = y_true
                st.session_state["batch_predictions"] = y_pred
                st.session_state["batch_probabilities"] = probabilities
                st.session_state["batch_source_signature"] = upload_signature

            # Only show results created for the currently uploaded file.
            has_current_result = (
                "batch_output" in st.session_state
                and st.session_state.get("batch_source_signature")
                == upload_signature
            )

            if has_current_result:

                result = st.session_state["batch_output"]

                # FIX 3: prevent stale or partial output from being displayed.
                if len(result) != len(X_upload):
                    for key in (
                        "batch_output",
                        "batch_labels",
                        "batch_predictions",
                        "batch_probabilities",
                        "batch_source_signature",
                    ):
                        st.session_state.pop(key, None)

                    st.error(
                        "Stored results did not match the uploaded file. "
                        "Click Run batch predictions again."
                    )
                    st.stop()

                st.markdown("### Prediction results")
                st.caption(
                    f"Showing predictions for {len(result):,} uploaded records. "
                    "A high probability is not a clinical diagnosis."
                )

                display_result = result.copy()
                display_result["malignant_probability"] = (
                    display_result["malignant_probability"]
                    .map(lambda x: f"{x:.2%}")
                )
                display_result["benign_probability"] = (
                    display_result["benign_probability"]
                    .map(lambda x: f"{x:.2%}")
                )
                display_result["decision_threshold"] = (
                    display_result["decision_threshold"]
                    .map(lambda x: f"{x:.2f}")
                )

                if "prediction_match" in display_result:
                    display_result["prediction_match"] = (
                        display_result["prediction_match"]
                        .map({True: "Match", False: "Mismatch"})
                    )

                st.dataframe(
                    dark_table(display_result),
                    use_container_width=True,
                    height=compact_table_height(len(display_result)),
                    hide_index=True,
                )

                counts = (
                    result["predicted_diagnosis"]
                    .value_counts()
                    .reindex(["B", "M"], fill_value=0)
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Records processed", f"{len(result):,}")
                c2.metric("Predicted benign", f"{counts['B']:,}")
                c3.metric("Predicted malignant", f"{counts['M']:,}")

                chart_left, chart_right = st.columns(2, gap="large")

                with chart_left:
                    render_dark_bar_chart(
                        ["Benign", "Malignant"],
                        [counts["B"], counts["M"]],
                        "Predicted class distribution",
                        color="#45cdbb",
                        xlabel="Number of samples",
                    )

                with chart_right:
                    render_probability_histogram(
                        st.session_state["batch_probabilities"]
                    )

                labels_now = st.session_state["batch_labels"]

                if labels_now is not None:

                    predictions_now = st.session_state["batch_predictions"]

                    st.markdown("### Uploaded dataset comparison")
                    st.warning(
                        "These metrics describe the uploaded labelled file. "
                        "They are exploratory and are not the original "
                        "holdout evaluation."
                    )

                    metrics = {
                        "Accuracy": accuracy_score(
                            labels_now, predictions_now
                        ),
                        "Precision (M)": precision_score(
                            labels_now, predictions_now, zero_division=0
                        ),
                        "Recall (M)": recall_score(
                            labels_now, predictions_now, zero_division=0
                        ),
                        "F1 (M)": f1_score(
                            labels_now, predictions_now, zero_division=0
                        ),
                    }

                    metric_cols = st.columns(4)
                    for col, (name, value) in zip(metric_cols, metrics.items()):
                        col.metric(name, f"{value:.2%}")

                    matrix = confusion_matrix(
                        labels_now, predictions_now, labels=[0, 1]
                    )
                    render_confusion_heatmap(matrix)

                st.download_button(
                    "⬇ Download batch prediction report",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="oncolens_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"CSV processing failed: {exc}")
            st.caption(
                "Check column names, numeric values, missing values "
                "and diagnosis labels."
            )


# ============================================================
# 04 — EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.markdown(
        '<div class="section-kicker">04 / Explainable AI</div>'
        '<div class="section-title">What influenced the model?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Explore saved feature analysis from the model-development notebook. "
        "Feature importance is not causation."
    )

    files = {
        "Logistic regression coefficients":
            "logistic_regression_coefficients.csv",
        "Permutation feature importance":
            "permutation_feature_importance.csv",
        "Feature importance comparison":
            "feature_importance_comparison.csv",
    }

    images = {
        "Logistic regression feature importance":
            "logistic_regression_feature_importance.png",
        "Permutation feature importance":
            "permutation_feature_importance.png",
        "Feature importance comparison":
            "feature_importance_comparison.png",
    }

    tab1, tab2 = st.tabs(["Feature tables", "Saved visualizations"])

    with tab1:
        for title, filename in files.items():
            path = EVAL_DIR / filename

            with st.expander(title):
                if path.exists():
                    try:
                        frame = pd.read_csv(path)
                        st.dataframe(
                            dark_table(frame),
                            use_container_width=True,
                            height=compact_table_height(len(frame)),
                            hide_index=True,
                        )
                        st.download_button(
                            f"Download {filename}",
                            data=path.read_bytes(),
                            file_name=filename,
                            mime="text/csv",
                            key=f"download_{filename}",
                        )
                    except Exception as exc:
                        st.error(f"Could not read {filename}: {exc}")
                else:
                    st.info(f"File not found: {filename}")

    with tab2:
        for title, filename in images.items():
            path = EVAL_DIR / filename
            if path.exists():
                st.markdown(f"### {title}")
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Visualization not found: {filename}")

    st.markdown("""
    <div class="notice">
    Logistic-regression coefficients describe the fitted model's learned
    relationships. Permutation importance measures the reduction in model
    score after shuffling a feature. Neither method proves causation.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 05 — MODEL & METHODOLOGY
# ============================================================

elif page == "Model & methodology":

    st.markdown(
        '<div class="section-kicker">05 / Technical documentation</div>'
        '<div class="section-title">Model & methodology</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Dataset")
        st.write("Wisconsin Diagnostic Breast Cancer (WDBC)")
        st.write("569 observations · 30 continuous predictors · binary diagnosis")
        st.write(
            "The features describe cell-nucleus characteristics "
            "computed from digitized fine-needle-aspirate images."
        )

        st.markdown("### Model pipeline")
        st.code(
            "StandardScaler → LogisticRegression",
            language="text",
        )
        st.write(
            "The saved scikit-learn pipeline is loaded directly. "
            "The app does not retrain the model or separately scale inputs."
        )

    with right:
        st.markdown("### Decision rule")
        st.code(
            "P(class = malignant) >= 0.50 -> M\n"
            "P(class = malignant) < 0.50 -> B",
            language="text",
        )

        st.markdown("### Exploratory holdout results")
        holdout = pd.DataFrame({
            "Metric": list(HOLDOUT_METRICS.keys()),
            "Score": [f"{v:.4f}" for v in HOLDOUT_METRICS.values()],
        })
        st.dataframe(
            dark_table(holdout),
            use_container_width=True,
            height=compact_table_height(len(holdout)),
            hide_index=True,
        )

    st.markdown("### Feature groups")

    for name, features in FEATURE_GROUPS.items():
        with st.expander(f"{name} · {len(features)} features"):
            st.write(", ".join(features))

    st.markdown("""
    <div class="notice">
    <b>Clinical boundary:</b> This application does not provide a clinical
    diagnosis, histological grade, tumour size, TNM stage, lymph-node status,
    metastatic status, treatment recommendation or whole-body cancer burden.
    These require clinical examination and additional diagnostic evidence.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Loaded model metadata"):
        st.json(metadata)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#8299b4;font-size:.78rem;line-height:1.7;">
    <b style="color:#b9d1e8;">ONCOLENS</b> · Breast Diagnostic Intelligence<br>
    Educational machine-learning prototype · Not clinically validated
    · Not for medical decision-making
</div>
""", unsafe_allow_html=True)