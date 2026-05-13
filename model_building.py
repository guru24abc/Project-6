import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

SESSION_FILE = "session_features.csv"
MERGED_FILE = "merged_browser_ram.csv"

FEATURE_COLS = [
    "session_duration",
    "num_events",
    "avg_browser_ram",
    "peak_ram"
]

SEQ_LEN = 5


def load_data():
    session = pd.read_csv(SESSION_FILE)
    merged = pd.read_csv(MERGED_FILE)

    for col in FEATURE_COLS:
        session[col] = pd.to_numeric(session[col], errors="coerce")

    session[FEATURE_COLS] = session[FEATURE_COLS].fillna(session[FEATURE_COLS].median(numeric_only=True))

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "category" not in merged.columns:
        merged["category"] = "unknown"
    else:
        merged["category"] = merged["category"].fillna("unknown").astype(str).str.lower().str.strip()

    return session, merged


def train_clustering(session):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(session[FEATURE_COLS])

    n_samples = len(session)
    n_clusters = min(3, n_samples)

    if n_clusters < 2:
        raise ValueError("Not enough sessions to train clustering model.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    session["cluster"] = kmeans.fit_predict(X_scaled)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")

    return session, scaler, kmeans


def train_autoencoder(session):
    ae_scaler = StandardScaler()
    X_ae = ae_scaler.fit_transform(session[FEATURE_COLS])

    input_dim = X_ae.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)
    encoded = Dense(4, activation="relu")(encoded)
    decoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )

    autoencoder.fit(
        X_ae, X_ae,
        epochs=20,
        batch_size=16,
        shuffle=True,
        verbose=1,
        callbacks=[early_stop]
    )

    recon = autoencoder.predict(X_ae, verbose=0)
    errors = np.mean(np.power(X_ae - recon, 2), axis=1)
    threshold = np.percentile(errors, 95)

    session["anomaly"] = errors > threshold
    session["anomaly_score"] = errors

    autoencoder.save("autoencoder_model.h5")
    joblib.dump(ae_scaler, "autoencoder_scaler.pkl")
    joblib.dump(threshold, "ae_threshold.pkl")

    return session, autoencoder, ae_scaler, threshold


def train_lstm(merged):
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    categories = merged["category"].astype(str).fillna("unknown").str.lower().str.strip()

    le = LabelEncoder()
    category_ids = le.fit_transform(categories)

    if len(le.classes_) < 2:
        raise ValueError("Not enough category variety to train LSTM model.")

    joblib.dump(le, "label_encoder.pkl")

    X_seq = []
    y_seq = []

    for i in range(SEQ_LEN, len(category_ids)):
        X_seq.append(category_ids[i - SEQ_LEN:i])
        y_seq.append(category_ids[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if len(X_seq) < 100:
        raise ValueError("Not enough sequence data to train LSTM model.")

    lstm_model = Sequential([
        Embedding(input_dim=len(le.classes_), output_dim=16, input_length=SEQ_LEN),
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(len(le.classes_), activation="softmax")
    ])

    lstm_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    lstm_model.fit(
        X_seq,
        y_seq,
        epochs=6,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop]
    )

    lstm_model.save("lstm_model.h5")

    return lstm_model, le


def main():
    session, merged = load_data()

    print("Training clustering model...")
    session, scaler, kmeans = train_clustering(session)

    print("Training autoencoder...")
    session, autoencoder, ae_scaler, threshold = train_autoencoder(session)

    print("Training LSTM model...")
    lstm_model, le = train_lstm(merged)

    session.to_csv(SESSION_FILE, index=False)

    print("✅ Models trained and saved successfully!")
    print("✅ Updated session_features.csv with cluster/anomaly columns")


if __name__ == "__main__":
    main()