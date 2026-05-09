import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,4)
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Browsing Behavior Analyzer", layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv("merged_browser_ram.csv")
    session = pd.read_csv("session_features.csv")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    session["start_time"] = pd.to_datetime(session["start_time"], errors="coerce")
    session["end_time"] = pd.to_datetime(session["end_time"], errors="coerce")

    return merged, session

merged_df, session_df = load_data()

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    lstm_model = load_model("lstm_model.h5", compile=False)
    le = joblib.load("label_encoder.pkl")
    autoencoder = load_model("autoencoder_model.h5", compile=False)
    ae_scaler = joblib.load("autoencoder_scaler.pkl")

    try:
        ae_threshold = joblib.load("ae_threshold.pkl")
    except Exception:
        ae_threshold = 0.05

    return kmeans, scaler, lstm_model, le, autoencoder, ae_scaler, ae_threshold

kmeans, scaler, lstm_model, le, autoencoder, ae_scaler, ae_threshold = load_models()

# ---------------------------
# TITLE
# ---------------------------
st.title("🧠 Browsing Behavior Analyzer")
st.markdown("Browse patterns, RAM usage, clustering, anomalies, and next-category prediction in one dashboard.")

# ---------------------------
# SIDEBAR FILTER
# ---------------------------
st.sidebar.header("Filter")
days = st.sidebar.selectbox("Select Time Window", ["All", "3 Days", "5 Days"])

filtered_merged = merged_df.copy()
if days != "All":
    n = int(days.split()[0])
    cutoff = filtered_merged["timestamp"].max() - pd.Timedelta(days=n)
    filtered_merged = filtered_merged[filtered_merged["timestamp"] >= cutoff]

# ---------------------------
# OVERVIEW
# ---------------------------
st.header("📊 Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Events", len(filtered_merged))
col2.metric("Total Sessions", int(session_df["session_id"].nunique()) if "session_id" in session_df.columns else len(session_df))
col3.metric(
    "Avg Session Duration (min)",
    round(float(session_df["session_duration"].mean()), 2) if "session_duration" in session_df.columns else 0.0
)

# ---------------------------
# CATEGORY ANALYSIS
# ---------------------------
st.header("🌐 Category Distribution")

cat_counts = filtered_merged["category"].value_counts()

fig, ax = plt.subplots(figsize=(8,4))
cat_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Category")
ax.set_ylabel("Count")
st.pyplot(fig)

st.caption("This shows which browsing categories dominate in the selected time window.")

# ---------------------------
# TIME ANALYSIS
# ---------------------------
st.header("⏰ Time-Based Usage")

filtered_merged["hour"] = filtered_merged["timestamp"].dt.hour
hour_counts = filtered_merged.groupby("hour").size()

fig, ax = plt.subplots()
hour_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Hour")
ax.set_ylabel("Events")
st.pyplot(fig)

st.caption("Peak browsing hours help identify productivity and distraction patterns.")

# ---------------------------
# RAM ANALYSIS
# ---------------------------
st.header("💻 RAM Usage by Category")

ram_usage = filtered_merged.groupby("category")["browser_ram_mb"].mean().sort_values()

fig, ax = plt.subplots()
ram_usage.plot(kind="barh", ax=ax)
ax.set_xlabel("Average Browser RAM (MB)")
ax.set_ylabel("Category")
st.pyplot(fig)

st.caption("Higher RAM categories usually indicate heavier apps or media-heavy activity.")

# ---------------------------
# CLUSTER ANALYSIS
# ---------------------------
st.header("🔍 Session Clustering")

if "cluster" in session_df.columns:
    cluster_summary = session_df.groupby("cluster")[["session_duration", "num_events", "avg_browser_ram", "peak_ram"]].mean()
    st.dataframe(cluster_summary)
else:
    st.warning("Cluster column not found. Run model_building.py first.")

# ---------------------------
# ANOMALY DETECTION
# ---------------------------
st.header("🚨 Anomaly Detection")

if "anomaly" in session_df.columns:
    anomalies = session_df[session_df["anomaly"] == True]
    st.write("Total anomalies:", len(anomalies))
    st.dataframe(anomalies.head(10))
else:
    st.warning("Anomaly column not found. Run model_building.py first.")

# ---------------------------
# PREDICTION SECTION
# ---------------------------
st.header("🔮 Predict User Behavior")

st.caption(f"Allowed categories: {', '.join(le.classes_)}")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Session Duration (min)", value=30.0)
    events = st.number_input("Number of Events", value=10)
    avg_ram = st.number_input("Avg RAM", value=800.0)
    peak_ram = st.number_input("Peak RAM", value=1200.0)
    seq_input = st.text_input(
        "Last 5 categories (comma separated)",
        "social,video,learning,social,video"
    )

analyze = st.button("Analyze")

if analyze:
    session_input = np.array([[duration, events, avg_ram, peak_ram]])

    # ---- Cluster ----
    scaled = scaler.transform(session_input)
    cluster = int(kmeans.predict(scaled)[0])

    # ---- LSTM ----
    seq = [s.strip() for s in seq_input.split(",") if s.strip()]

    if len(seq) != 5:
        st.error("Please enter exactly 5 categories.")
        st.stop()

    try:
        encoded = le.transform(seq).reshape(1, -1)
    except ValueError:
        st.error("Invalid category entered. Use only the categories shown above.")
        st.stop()

    pred = lstm_model.predict(encoded, verbose=0)
    next_cat = le.inverse_transform([int(np.argmax(pred))])[0]

    # ---- Autoencoder ----
    scaled_ae = ae_scaler.transform(session_input)
    recon = autoencoder.predict(scaled_ae, verbose=0)
    error = float(np.mean((scaled_ae - recon) ** 2))
    anomaly = error > ae_threshold

    # ---- Recommendations ----
    recs = []

    if peak_ram > 1200:
        recs.append("⚠️ High RAM usage — close unused tabs.")
    if duration > 60:
        recs.append("⏳ Long session — take a break.")
    if events > 30:
        recs.append("🔁 Too many events — reduce multitasking.")
    if next_cat == "social":
        recs.append("📱 Social usage predicted — avoid distractions.")
    if anomaly:
        recs.append("🚨 Unusual session detected.")

    if not recs:
        recs.append("✅ Usage looks healthy.")

    st.subheader("Results")
    st.write("Cluster:", cluster)
    st.write("Next Category:", next_cat)
    st.write("Anomaly:", anomaly)

    st.subheader("📌 Recommendations")
    for r in recs:
        st.success(r)