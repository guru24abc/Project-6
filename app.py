import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Browsing Behavior Analyzer", layout="wide")


# ---------------------------
# Helpers
# ---------------------------
def safe_lower(series):
    return series.fillna("unknown").astype(str).str.lower().str.strip()

def fmt(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "0.00"


# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv("merged_browser_ram.csv")
    session = pd.read_csv("session_features.csv")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    session["start_time"] = pd.to_datetime(session["start_time"], errors="coerce")
    session["end_time"] = pd.to_datetime(session["end_time"], errors="coerce")

    for col in ["category", "domain", "browser"]:
        if col not in merged.columns:
            merged[col] = "unknown"

    if "browser_ram_mb" not in merged.columns:
        merged["browser_ram_mb"] = np.nan

    merged["category"] = safe_lower(merged["category"])
    merged["domain"] = safe_lower(merged["domain"])
    merged["browser"] = safe_lower(merged["browser"])

    if "cluster" not in session.columns:
        session["cluster"] = -1
    if "anomaly" not in session.columns:
        session["anomaly"] = False

    session["anomaly"] = session["anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])

    return merged, session


merged_df, session_df = load_data()


# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    lstm_model = load_model("lstm_model.h5", compile=False)
    autoencoder = load_model("autoencoder_model.h5", compile=False)
    ae_scaler = joblib.load("autoencoder_scaler.pkl")

    try:
        ae_threshold = joblib.load("ae_threshold.pkl")
    except Exception:
        ae_threshold = 0.05

    return kmeans, scaler, le, lstm_model, autoencoder, ae_scaler, ae_threshold


kmeans, scaler, le, lstm_model, autoencoder, ae_scaler, ae_threshold = load_models()


# ---------------------------
# Title
# ---------------------------
st.title("🧠 Browsing Behavior Analyzer")
st.markdown("Browse patterns, RAM usage, clustering, anomalies, and next-category prediction in one dashboard.")


# ---------------------------
# Sidebar Filter
# ---------------------------
st.sidebar.header("Filter")
window = st.sidebar.selectbox("Select Time Window", ["All", "3 Days", "5 Days"])

filtered = merged_df.copy()
if window != "All":
    days = int(window.split()[0])
    cutoff = filtered["timestamp"].max() - pd.Timedelta(days=days)
    filtered = filtered[filtered["timestamp"] >= cutoff].copy()


# ---------------------------
# Overview
# ---------------------------
st.header("Overview")

total_events = len(filtered)
total_sessions = int(session_df["session_id"].nunique()) if "session_id" in session_df.columns else len(session_df)
avg_session_duration = session_df["session_duration"].mean() if "session_duration" in session_df.columns and len(session_df) else 0
avg_ram = filtered["browser_ram_mb"].mean() if "browser_ram_mb" in filtered.columns and len(filtered) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events", f"{total_events:,}")
c2.metric("Total Sessions", f"{total_sessions:,}")
c3.metric("Avg Session Duration (min)", fmt(avg_session_duration))
c4.metric("Avg Browser RAM (MB)", fmt(avg_ram))


# ---------------------------
# Category Distribution
# ---------------------------
st.header("Category Distribution")
if len(filtered):
    top_categories = filtered["category"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    top_categories.plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_title("Top Categories in Selected Window")
    st.pyplot(fig)
else:
    st.info("No data available.")


# ---------------------------
# Time-Based Usage
# ---------------------------
st.header("Time-Based Usage")
if len(filtered):
    filtered["hour"] = filtered["timestamp"].dt.hour
    hour_counts = filtered.groupby("hour").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    hour_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Events")
    ax.set_title("Browsing Activity by Hour")
    st.pyplot(fig)

    peak_hour = int(hour_counts.idxmax())
    st.write(f"Peak hour: **{peak_hour:02d}:00**")
else:
    st.info("No data available.")


# ---------------------------
# RAM Usage by Category
# ---------------------------
st.header("RAM Usage by Category")
if len(filtered) and "browser_ram_mb" in filtered.columns:
    ram_cat = filtered.groupby("category")["browser_ram_mb"].mean().sort_values().tail(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    ram_cat.plot(kind="barh", ax=ax)
    ax.set_xlabel("Average Browser RAM (MB)")
    ax.set_ylabel("Category")
    ax.set_title("Average Browser RAM by Category")
    st.pyplot(fig)

    st.write(f"Highest RAM category: **{ram_cat.idxmax()}**")
else:
    st.info("No RAM data available.")


# ---------------------------
# Session Clustering
# ---------------------------
st.header("Session Clustering")

if "cluster" in session_df.columns:
    cluster_cols = [c for c in ["session_duration", "num_events", "avg_browser_ram", "peak_ram", "avg_system_ram", "unique_domains"] if c in session_df.columns]
    if cluster_cols:
        cluster_summary = session_df.groupby("cluster")[cluster_cols].mean(numeric_only=True)
        st.dataframe(cluster_summary, use_container_width=True)

    st.write("Cluster sizes:")
    st.write(session_df["cluster"].value_counts().sort_index())
else:
    st.warning("Cluster column not found in session_features.csv. Run model_building.py again.")


# ---------------------------
# Anomaly Detection
# ---------------------------
st.header("Anomaly Detection")

if "anomaly" in session_df.columns:
    anomalies = session_df[session_df["anomaly"] == True].copy()
    st.write(f"Total anomalies: **{len(anomalies)}**")

    if len(anomalies) > 0:
        anomaly_cols = [
            c for c in [
                "session_id", "start_time", "end_time", "session_duration",
                "num_events", "avg_browser_ram", "peak_ram",
                "dominant_category", "anomaly_score"
            ] if c in anomalies.columns
        ]
        st.dataframe(anomalies[anomaly_cols].head(10), use_container_width=True)
    else:
        st.info("No anomalies found in the current trained model output.")
else:
    st.warning("Anomaly column not found in session_features.csv. Run model_building.py again.")


# ---------------------------
# Suggestions / Prediction
# ---------------------------
st.header("Suggestions")

st.caption(f"Allowed categories: {', '.join(le.classes_)}")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Session Duration (min)", min_value=0.0, value=30.0, step=1.0)
    events = st.number_input("Number of Events", min_value=1, value=10, step=1)
    avg_ram_input = st.number_input("Avg Browser RAM (MB)", min_value=0.0, value=800.0, step=10.0)
    peak_ram_input = st.number_input("Peak RAM (MB)", min_value=0.0, value=1200.0, step=10.0)

with col2:
    seq_input = st.text_input(
        "Last 5 categories (comma separated)",
        "social,video,learning,social,video"
    )

analyze = st.button("Analyze")

if analyze:
    session_input = np.array([[duration, events, avg_ram_input, peak_ram_input]], dtype=float)

    # Cluster prediction
    try:
        scaled = scaler.transform(session_input)
        cluster = int(kmeans.predict(scaled)[0])
    except Exception as e:
        st.error(f"Cluster prediction failed: {e}")
        st.stop()

    # LSTM next-category prediction
    seq = [s.strip().lower() for s in seq_input.split(",") if s.strip()]
    if len(seq) != 5:
        st.error("Please enter exactly 5 categories.")
        st.stop()

    try:
        encoded = le.transform(seq).reshape(1, -1)
    except Exception:
        st.error("Invalid category entered. Use only the categories shown above.")
        st.stop()

    try:
        pred = lstm_model.predict(encoded, verbose=0)
        next_cat = le.inverse_transform([int(np.argmax(pred))])[0]
    except Exception as e:
        st.error(f"LSTM prediction failed: {e}")
        st.stop()

    # Autoencoder anomaly check
    try:
        scaled_ae = ae_scaler.transform(session_input)
        recon = autoencoder.predict(scaled_ae, verbose=0)
        error = float(np.mean((scaled_ae - recon) ** 2))
        anomaly = error > float(ae_threshold)
    except Exception as e:
        st.error(f"Autoencoder check failed: {e}")
        st.stop()

    st.subheader("Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Cluster", str(cluster))
    r2.metric("Next Category", str(next_cat))
    r3.metric("Anomaly", "Yes" if anomaly else "No")

    # Single simple suggestion card like your screenshot
    if anomaly:
        suggestion_msg = "Unusual session detected."
    elif peak_ram_input > 1200:
        suggestion_msg = "High RAM usage detected."
    elif duration > 60:
        suggestion_msg = "Long session detected."
    elif events > 30:
        suggestion_msg = "Too many events in one session."
    elif next_cat in ["social", "video", "entertainment"]:
        suggestion_msg = f"Next category may be {next_cat}."
    else:
        suggestion_msg = "Session looks normal."

    st.subheader("Suggestions")
    st.success(suggestion_msg)