import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Case ML Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — big bold title, gradients, cards
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
    }
    .main {
        background-color: #f4f6fb;
    }

    /* ---------- HERO BANNER ---------- */
    .hero-banner {
        background: linear-gradient(120deg, #4338CA, #7E22CE, #DB2777);
        border-radius: 20px;
        padding: 50px 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.3);
    }
    .hero-title {
        font-size: 72px;
        font-weight: 900;
        color: white;
        letter-spacing: 2px;
        margin: 0;
        text-transform: uppercase;
        text-shadow: 0 4px 14px rgba(0,0,0,0.35);
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 22px;
        color: #EDE9FE;
        margin-top: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    @media (max-width: 900px) {
        .hero-title { font-size: 44px; }
        .hero-subtitle { font-size: 16px; }
    }

    /* ---------- SECTION HEADERS ---------- */
    .section-header {
        font-size: 30px;
        font-weight: 900;
        color: #1E1B4B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-left: 8px solid #9333EA;
        padding: 6px 0 6px 16px;
        margin: 22px 0 18px 0;
        background: linear-gradient(90deg, rgba(147,51,234,0.08), transparent);
    }
    .sub-header {
        font-size: 20px;
        font-weight: 800;
        color: #374151;
        margin: 14px 0 10px 0;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 12px 12px 0 0;
        padding: 14px 28px;
        font-weight: 800;
        font-size: 17px;
        border: 1px solid #E5E7EB;
        color: #4B5563;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4338CA, #9333EA);
        color: white !important;
    }

    /* ---------- METRIC / RESULT CARDS ---------- */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border-top: 6px solid #4F46E5;
        margin-bottom: 14px;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-card h4 {
        margin: 0 0 10px 0;
        color: #6B7280;
        font-size: 15px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-card .value {
        font-size: 30px;
        font-weight: 900;
        color: #111827;
    }

    /* ---------- STAT PILLS (sidebar) ---------- */
    .stat-pill {
        background: linear-gradient(90deg, #4338CA, #9333EA);
        color: white;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-pill .num {
        font-size: 26px;
        font-weight: 900;
    }
    .stat-pill .lbl {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        opacity: 0.9;
    }

    /* ---------- BUTTON ---------- */
    .stButton>button {
        background: linear-gradient(90deg, #4338CA, #9333EA, #DB2777);
        color: white;
        font-weight: 800;
        font-size: 17px;
        border-radius: 12px;
        border: none;
        padding: 14px 0;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(147,51,234,0.35);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #DB2777, #9333EA, #4338CA);
        color: white;
    }

    /* ---------- MISC ---------- */
    .caption-box {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 13px;
        color: #4B5563;
        font-weight: 600;
        text-align: center;
        margin-top: -6px;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "legal_cases_five_tasks.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

CATEGORICAL_COLS = ["Jurisdiction", "Case_Type", "Outcome", "Cross_Jurisdictional",
                     "High_Quality_Law_Firms", "Article_Prediction", "Cause_Prediction"]
DROP_COLS = ["Case_ID", "Case_Analysis"]

TASKS = {
    "case_type": {"type": "classification", "target": "Case_Type", "label": "Case Type", "icon": "📁", "color": "#4338CA"},
    "outcome": {"type": "classification", "target": "Outcome", "label": "Outcome", "icon": "⚖️", "color": "#9333EA"},
    "article_prediction": {"type": "classification", "target": "Article_Prediction", "label": "Article Prediction", "icon": "📜", "color": "#DB2777"},
    "penalty_prediction": {"type": "regression", "target": "Penalty_Prediction", "label": "Penalty Prediction", "icon": "💰", "color": "#F59E0B"},
    "liability_score": {"type": "regression", "target": "Liability_Score", "label": "Liability Score", "icon": "📊", "color": "#10B981"},
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_encoders():
    path = os.path.join(MODELS_DIR, "label_encoders.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_models():
    models = {}
    for task_name, cfg in TASKS.items():
        path = os.path.join(MODELS_DIR, f"{task_name}_model.joblib")
        if os.path.exists(path):
            models[task_name] = joblib.load(path)
    return models


@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "training_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


df = load_data()
encoders = load_encoders()
models = load_models()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Hero banner (BIG BOLD TITLE)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">⚖️ LEGAL CASE ML DASHBOARD</p>
    <p class="hero-subtitle">AI-Powered Prediction of Case Type · Outcome · Article · Penalty · Liability</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚖️ ABOUT THIS PROJECT")
    st.info(
        "This dashboard uses **Random Forest** models trained on "
        "2,000+ legal case records to predict case type, outcome, "
        "applicable article, penalty amount, and liability score."
    )

    st.markdown("## 🧰 TECH STACK")
    st.markdown(
        "- **Python**\n"
        "- **Scikit-learn** (Random Forest)\n"
        "- **Pandas / NumPy**\n"
        "- **Matplotlib / Seaborn**\n"
        "- **Streamlit**"
    )

    st.markdown("## 📌 QUICK STATS")
    if df is not None:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="num">{len(df):,}</div>
            <div class="lbl">Total Cases</div>
        </div>
        """, unsafe_allow_html=True)
    if metrics:
        acc_vals = [m["accuracy"] for m in metrics.values() if "accuracy" in m]
        if acc_vals:
            avg_acc = np.mean(acc_vals)
            st.markdown(f"""
            <div class="stat-pill">
                <div class="num">{avg_acc*100:.1f}%</div>
                <div class="lbl">Avg. Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-pill">
            <div class="num">{len(TASKS)}</div>
            <div class="lbl">ML Models Trained</div>
        </div>
        """, unsafe_allow_html=True)

missing = []
if df is None:
    missing.append(f"Data file not found: `{DATA_PATH}`")
if encoders is None:
    missing.append("`label_encoders.joblib` not found in `models/`")
if not models:
    missing.append("No trained models found in `models/`")

if missing:
    st.error("**SETUP INCOMPLETE — RUN THE TRAINING NOTEBOOK FIRST:**\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_input_widgets(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    values = {}
    cols_per_row = 3
    rows_needed = -(-len(feature_cols) // cols_per_row)
    idx = 0
    for _ in range(rows_needed):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(feature_cols):
                break
            col_name = feature_cols[idx]
            idx += 1
            with c:
                if col_name in ["Cross_Jurisdictional", "High_Quality_Law_Firms"]:
                    values[col_name] = st.selectbox(f"🔘 {col_name}", [True, False], key=col_name)
                elif col_name in CATEGORICAL_COLS:
                    options = list(encoders[col_name].classes_)
                    values[col_name] = st.selectbox(f"🏷️ {col_name}", options, key=col_name)
                else:
                    series = df[col_name]
                    default = float(series.median())
                    if pd.api.types.is_integer_dtype(series):
                        values[col_name] = st.number_input(f"🔢 {col_name}", value=int(default), step=1, key=col_name)
                    else:
                        values[col_name] = st.number_input(f"🔢 {col_name}", value=round(default, 2), step=0.1, key=col_name)
    return values


def predict_task(task_name: str, payload: dict):
    bundle = models[task_name]
    model, feature_cols = bundle["model"], bundle["feature_columns"]
    row = pd.DataFrame([payload])

    for col in CATEGORICAL_COLS:
        if col in row.columns:
            le = encoders[col]
            known = set(le.classes_)
            row[col] = row[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            row[col] = le.transform(row[col])

    missing_cols = [c for c in feature_cols if c not in row.columns]
    if missing_cols:
        return None, f"Missing fields: {missing_cols}"

    X = row[feature_cols]
    pred = model.predict(X)[0]

    target_col = TASKS[task_name]["target"]
    if target_col in encoders:
        pred_label = encoders[target_col].inverse_transform([int(round(pred))])[0]
    else:
        pred_label = float(pred)

    proba = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(X)[0]
        classes = encoders[target_col].inverse_transform(model.classes_)
        proba = pd.Series(raw_proba, index=classes).sort_values(ascending=False)

    return {"prediction": pred_label, "proba": proba}, None


def metric_card(label, value, icon, color):
    st.markdown(f"""
    <div class="metric-card" style="border-top-color:{color};">
        <h4>{icon} {label}</h4>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_eda, tab_metrics = st.tabs(["🔮  PREDICT", "📊  EDA PLOTS", "📈  MODEL METRICS"])

# --- Predict tab ---
with tab_predict:
    st.markdown('<p class="section-header">📝 Enter Case Details</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        input_values = build_input_widgets(df)
        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Select Predictions To Run</p>', unsafe_allow_html=True)
        selected_tasks = st.multiselect(
            "Choose one or more tasks",
            options=list(TASKS.keys()),
            format_func=lambda k: f"{TASKS[k]['icon']} {TASKS[k]['label']}",
            default=list(TASKS.keys()),
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀  RUN PREDICTION")

    if submitted:
        if not selected_tasks:
            st.warning("Select at least one task above.")
        else:
            st.markdown('<p class="section-header">✨ Prediction Results</p>', unsafe_allow_html=True)
            result_cols = st.columns(len(selected_tasks))
            for i, task_name in enumerate(selected_tasks):
                cfg = TASKS[task_name]
                result, error = predict_task(task_name, input_values)
                with result_cols[i]:
                    if error:
                        st.error(error)
                        continue
                    if cfg["type"] == "classification":
                        metric_card(cfg["label"], result["prediction"], cfg["icon"], cfg["color"])
                        if result["proba"] is not None:
                            st.bar_chart(result["proba"], color=cfg["color"])
                    else:
                        metric_card(cfg["label"], f"{result['prediction']:,.2f}", cfg["icon"], cfg["color"])

# --- EDA tab ---
with tab_eda:
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    if not os.path.isdir(PLOTS_DIR):
        st.info("No `plots/` folder found — run the training notebook to generate EDA charts.")
    else:
        eda_files = sorted(f for f in os.listdir(PLOTS_DIR) if f.startswith(("0", "cm_", "fit_", "featimp_")))
        if not eda_files:
            st.info("No plots found in `plots/`.")
        else:
            cols = st.columns(2)
            for i, fname in enumerate(eda_files):
                with cols[i % 2]:
                    st.image(Image.open(os.path.join(PLOTS_DIR, fname)), use_container_width=True)
                    st.markdown(f'<div class="caption-box">{fname}</div>', unsafe_allow_html=True)
                    st.write("")

# --- Metrics tab ---
with tab_metrics:
    st.markdown('<p class="section-header">📈 Training Metrics</p>', unsafe_allow_html=True)
    if not metrics:
        st.info("No `training_metrics.json` found — run the training notebook first.")
    else:
        cols = st.columns(len(metrics))
        for i, (task_name, m) in enumerate(metrics.items()):
            cfg = TASKS[task_name]
            with cols[i]:
                if "accuracy" in m:
                    metric_card(cfg["label"], f"{m['accuracy']*100:.1f}%", cfg["icon"], cfg["color"])
                    st.markdown(f'<div class="caption-box">F1 Score: {m["f1_weighted"]:.3f}</div>', unsafe_allow_html=True)
                else:
                    metric_card(cfg["label"], f"R² {m['r2']:.2f}", cfg["icon"], cfg["color"])
                    st.markdown(f'<div class="caption-box">MAE: {m["mae"]:.2f} | RMSE: {m["rmse"]:.2f}</div>', unsafe_allow_html=True)

        st.markdown('<p class="sub-header">📋 Detailed Comparison</p>', unsafe_allow_html=True)
        rows = []
        for task_name, m in metrics.items():
            label = TASKS[task_name]["label"]
            if "accuracy" in m:
                rows.append({"Task": label, "Type": "Classification",
                             "Accuracy": round(m["accuracy"], 3), "F1 (weighted)": round(m["f1_weighted"], 3)})
            else:
                rows.append({"Task": label, "Type": "Regression",
                             "R²": round(m["r2"], 3), "MAE": round(m["mae"], 2), "RMSE": round(m["rmse"], 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)