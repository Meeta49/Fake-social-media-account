import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Social Media Account Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-fake {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border-left: 5px solid #ff4b4b;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #c0392b;
    }
    .result-real {
        background: linear-gradient(135deg, #00c85122, #00c85111);
        border-left: 5px solid #00c851;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e8449;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Fake Social Media Account Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect fake or bot accounts using ML — analyze profile features instantly</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=80)
    st.title("⚙️ Settings")
    model_choice = st.selectbox(
        "Choose ML Model",
        ["Random Forest", "XGBoost", "Logistic Regression"],
        index=0
    )
    st.markdown("---")
    st.markdown("**About**")
    st.info("This app uses machine learning to detect fake/bot social media accounts based on profile and behavioral features.")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload train.csv (optional)", type=["csv"])
    st.caption("If not uploaded, a synthetic dataset will be used for demo.")

# ─── Data Loading / Generation ───────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    real = pd.DataFrame({
        'profile pic':        np.random.choice([0, 1], n//2, p=[0.05, 0.95]),
        'nums/length username': np.random.uniform(0, 0.2, n//2),
        'fullname words':     np.random.randint(1, 4, n//2),
        'nums/length fullname': np.random.uniform(0, 0.1, n//2),
        'name==username':     np.random.choice([0, 1], n//2, p=[0.9, 0.1]),
        'description length': np.random.randint(20, 150, n//2),
        'external URL':       np.random.choice([0, 1], n//2, p=[0.4, 0.6]),
        'private':            np.random.choice([0, 1], n//2, p=[0.4, 0.6]),
        '#posts':             np.random.randint(10, 500, n//2),
        '#followers':         np.random.randint(50, 10000, n//2),
        '#follows':           np.random.randint(50, 2000, n//2),
        'fake':               np.zeros(n//2, dtype=int)
    })
    fake = pd.DataFrame({
        'profile pic':        np.random.choice([0, 1], n//2, p=[0.6, 0.4]),
        'nums/length username': np.random.uniform(0.3, 1.0, n//2),
        'fullname words':     np.random.randint(0, 2, n//2),
        'nums/length fullname': np.random.uniform(0.3, 1.0, n//2),
        'name==username':     np.random.choice([0, 1], n//2, p=[0.3, 0.7]),
        'description length': np.random.randint(0, 30, n//2),
        'external URL':       np.random.choice([0, 1], n//2, p=[0.8, 0.2]),
        'private':            np.random.choice([0, 1], n//2, p=[0.7, 0.3]),
        '#posts':             np.random.randint(0, 20, n//2),
        '#followers':         np.random.randint(0, 200, n//2),
        '#follows':           np.random.randint(500, 5000, n//2),
        'fake':               np.ones(n//2, dtype=int)
    })
    return pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=42)

@st.cache_data
def load_and_prepare(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = generate_synthetic_data()
    # Standardise column names
    df.columns = [c.strip().lower() for c in df.columns]
    target_col = next((c for c in df.columns if 'fake' in c), None)
    if target_col is None:
        st.error("Could not find a 'fake' target column in the CSV.")
        st.stop()
    feature_cols = [c for c in df.columns if c != target_col]
    df = df.dropna()
    # Keep only numeric columns
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return df, num_cols, target_col

df, feature_cols, target_col = load_and_prepare(uploaded_file)

# ─── Model Training ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(model_name, _df, feature_cols, target_col):
    X = _df[feature_cols]
    y = _df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    return model, scaler, acc, report, cm, feature_cols

model, scaler, acc, report, cm, feat_cols = train_model(model_choice, df, feature_cols, target_col)

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict Account", "📊 Model Performance", "📋 Dataset Explorer"])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Account Features")
    st.caption("Fill in the profile details below and click **Predict** to check if the account is real or fake.")

    col1, col2, col3 = st.columns(3)
    inputs = {}

    # Map known feature names to friendly widgets
    feature_config = {
        'profile pic':           ("Has Profile Picture?", "toggle", 1),
        'nums/length username':  ("Nums/Length of Username", "slider_float", (0.0, 1.0, 0.1)),
        'fullname words':        ("Words in Full Name", "slider_int", (0, 5, 2)),
        'nums/length fullname':  ("Nums/Length of Full Name", "slider_float", (0.0, 1.0, 0.05)),
        'name==username':        ("Name equals Username?", "toggle", 0),
        'description length':    ("Bio / Description Length (chars)", "slider_int", (0, 200, 50)),
        'external url':          ("Has External URL?", "toggle", 0),
        'private':               ("Is Private Account?", "toggle", 0),
        '#posts':                ("Number of Posts", "slider_int", (0, 1000, 30)),
        '#followers':            ("Number of Followers", "slider_int", (0, 50000, 200)),
        '#follows':              ("Number of Follows", "slider_int", (0, 10000, 300)),
    }

    cols = [col1, col2, col3]
    for i, feat in enumerate(feat_cols):
        cfg = feature_config.get(feat)
        with cols[i % 3]:
            if cfg:
                label, wtype, default = cfg
                if wtype == "toggle":
                    inputs[feat] = int(st.checkbox(label, value=bool(default)))
                elif wtype == "slider_float":
                    inputs[feat] = st.slider(label, min_value=default[0], max_value=default[1], value=default[2], step=0.01)
                elif wtype == "slider_int":
                    inputs[feat] = st.slider(label, min_value=default[0], max_value=default[1], value=default[2])
            else:
                inputs[feat] = st.number_input(feat.title(), value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df[feat_cols])
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f'<div class="result-fake">⚠️ FAKE / BOT Account — Confidence: {proba[1]*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-real">✅ REAL Account — Confidence: {proba[0]*100:.1f}%</div>', unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Label": ["Real", "Fake"], "Probability": [proba[0], proba[1]]})
        fig, ax = plt.subplots(figsize=(5, 1.8))
        colors = ["#00c851", "#ff4b4b"]
        ax.barh(prob_df["Label"], prob_df["Probability"], color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Confidence")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig, use_container_width=False)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📈 {model_choice} — Model Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Precision (Fake)", f"{report['1']['precision']*100:.1f}%")
    m3.metric("Recall (Fake)",    f"{report['1']['recall']*100:.1f}%")
    m4.metric("F1-Score (Fake)",  f"{report['1']['f1-score']*100:.1f}%")

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real','Fake'], yticklabels=['Real','Fake'], ax=ax)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        st.pyplot(fig)

    with col_fi:
        st.markdown("**Feature Importances**")
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True)
        else:
            fi = pd.Series(np.abs(model.coef_[0]), index=feat_cols).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        fi.plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title("Feature Importance / Coefficients")
        ax2.spines[['top','right']].set_visible(False)
        st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Real Accounts",  int((df[target_col]==0).sum()))
    c3.metric("Fake Accounts",  int((df[target_col]==1).sum()))

    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("**Class Distribution**")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    counts = df[target_col].value_counts()
    ax3.bar(['Real','Fake'], counts.values, color=['#00c851','#ff4b4b'])
    ax3.set_ylabel("Count"); ax3.spines[['top','right']].set_visible(False)
    st.pyplot(fig3)

    st.markdown("**Correlation Heatmap**")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    corr = df[feat_cols + [target_col]].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, linewidths=0.5)
    st.pyplot(fig4)
