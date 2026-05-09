import pandas as pd
import numpy as np

SESSION_GAP_MINUTES = 15

def load_data():
    browsing = pd.read_csv("browsing_history.csv")
    ram = pd.read_csv("ram_log.csv")

    browsing["timestamp"] = pd.to_datetime(browsing["timestamp"], errors="coerce")
    ram["timestamp"] = pd.to_datetime(ram["timestamp"], errors="coerce")

    browsing = browsing.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ram = ram.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return browsing, ram

def merge_browser_ram(browsing, ram):
    merged = pd.merge_asof(
        browsing.sort_values("timestamp"),
        ram.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    merged["ram_used_mb"] = merged["ram_used_mb"].ffill().bfill()
    merged["ram_available_mb"] = merged["ram_available_mb"].ffill().bfill()
    merged["browser_ram_mb"] = merged["browser_ram_mb"].ffill().bfill()

    return merged

def add_sessions(df, gap_minutes=SESSION_GAP_MINUTES):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    gaps = df["timestamp"].diff().dt.total_seconds().div(60)
    df["new_session"] = gaps.isna() | (gaps > gap_minutes)
    df["session_id"] = df["new_session"].cumsum().astype(int)
    return df

def build_session_features(df):
    features = df.groupby("session_id").agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        session_duration=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 60.0),
        num_events=("timestamp", "size"),
        avg_browser_ram=("browser_ram_mb", "mean"),
        peak_ram=("browser_ram_mb", "max"),
        avg_system_ram=("ram_used_mb", "mean"),
        unique_domains=("domain", "nunique"),
    ).reset_index()

    dominant_category = (
        df.groupby("session_id")["category"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index(name="dominant_category")
    )

    features = features.merge(dominant_category, on="session_id", how="left")
    return features

def main():
    browsing, ram = load_data()
    merged = merge_browser_ram(browsing, ram)
    merged = add_sessions(merged, gap_minutes=SESSION_GAP_MINUTES)
    features = build_session_features(merged)

    merged.to_csv("merged_browser_ram.csv", index=False)
    features.to_csv("session_features.csv", index=False)

    print("Saved merged_browser_ram.csv")
    print("Saved session_features.csv")
    print("Merged rows:", len(merged))
    print("Sessions:", len(features))

if __name__ == "__main__":
    main()