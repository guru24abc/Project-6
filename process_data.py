import pandas as pd
import numpy as np

SESSION_GAP_MINUTES = 1

BROWSING_FILE = "browsing_history.csv"
RAM_FILE = "ram_log.csv"

MERGED_OUTPUT = "merged_browser_ram.csv"
SESSION_OUTPUT = "session_features.csv"


def load_data():
    browsing = pd.read_csv(BROWSING_FILE)
    ram = pd.read_csv(RAM_FILE)

    browsing["timestamp"] = pd.to_datetime(browsing["timestamp"], errors="coerce")
    ram["timestamp"] = pd.to_datetime(ram["timestamp"], errors="coerce")

    browsing = browsing.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ram = ram.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "category" not in browsing.columns:
        browsing["category"] = "unknown"
    else:
        browsing["category"] = browsing["category"].fillna("unknown").astype(str).str.lower().str.strip()

    if "domain" not in browsing.columns:
        browsing["domain"] = "unknown"
    else:
        browsing["domain"] = browsing["domain"].fillna("unknown").astype(str).str.lower().str.strip()

    if "browser" not in browsing.columns:
        browsing["browser"] = "chrome"
    else:
        browsing["browser"] = browsing["browser"].fillna("chrome").astype(str).str.lower().str.strip()

    return browsing, ram


def merge_browser_ram(browsing, ram):
    browsing_sorted = browsing.sort_values("timestamp").reset_index(drop=True)
    ram_sorted = ram.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        browsing_sorted,
        ram_sorted,
        on="timestamp",
        direction="nearest"
    )

    for col in ["ram_used_mb", "ram_available_mb", "browser_ram_mb"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["ram_used_mb"] = merged["ram_used_mb"].ffill().bfill()
    merged["ram_available_mb"] = merged["ram_available_mb"].ffill().bfill()
    merged["browser_ram_mb"] = merged["browser_ram_mb"].ffill().bfill()

    return merged


def add_sessions(df, gap_minutes=SESSION_GAP_MINUTES):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    time_gaps = df["timestamp"].diff().dt.total_seconds().div(60)
    df["new_session"] = time_gaps.isna() | (time_gaps > gap_minutes)
    df["session_id"] = df["new_session"].cumsum().astype(int)

    return df


def _mode_or_first(series):
    series = series.dropna().astype(str)
    if series.empty:
        return "unknown"
    mode_values = series.mode()
    if not mode_values.empty:
        return mode_values.iloc[0]
    return series.iloc[0]


def build_session_features(df):
    df = df.copy()
    df["category"] = df["category"].fillna("unknown").astype(str).str.lower().str.strip()
    df["domain"] = df["domain"].fillna("unknown").astype(str).str.lower().str.strip()

    session_features = df.groupby("session_id").agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        session_duration=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 60.0 if len(s) > 1 else 0.0),
        num_events=("timestamp", "size"),
        avg_browser_ram=("browser_ram_mb", "mean"),
        peak_ram=("browser_ram_mb", "max"),
        avg_system_ram=("ram_used_mb", "mean"),
        unique_domains=("domain", "nunique"),
        unique_categories=("category", "nunique"),
    ).reset_index()

    category_switches = (
        df.groupby("session_id")["category"]
        .apply(lambda s: (s.astype(str).fillna("unknown").shift() != s.astype(str).fillna("unknown")).sum() - 1 if len(s) > 1 else 0)
        .reset_index(name="category_switches")
    )

    domain_switches = (
        df.groupby("session_id")["domain"]
        .apply(lambda s: (s.astype(str).fillna("unknown").shift() != s.astype(str).fillna("unknown")).sum() - 1 if len(s) > 1 else 0)
        .reset_index(name="domain_switches")
    )

    dominant_category = (
        df.groupby("session_id")["category"]
        .apply(_mode_or_first)
        .reset_index(name="dominant_category")
    )

    dominant_domain = (
        df.groupby("session_id")["domain"]
        .apply(_mode_or_first)
        .reset_index(name="dominant_domain")
    )

    session_features = session_features.merge(category_switches, on="session_id", how="left")
    session_features = session_features.merge(domain_switches, on="session_id", how="left")
    session_features = session_features.merge(dominant_category, on="session_id", how="left")
    session_features = session_features.merge(dominant_domain, on="session_id", how="left")

    session_features["category_switches"] = session_features["category_switches"].fillna(0).astype(int)
    session_features["domain_switches"] = session_features["domain_switches"].fillna(0).astype(int)

    # A few helpful derived ratios
    session_features["events_per_minute"] = session_features.apply(
        lambda row: row["num_events"] / row["session_duration"] if row["session_duration"] > 0 else float(row["num_events"]),
        axis=1
    )

    return session_features


def main():
    browsing, ram = load_data()
    merged = merge_browser_ram(browsing, ram)
    merged = add_sessions(merged, gap_minutes=SESSION_GAP_MINUTES)
    session_features = build_session_features(merged)

    merged.to_csv(MERGED_OUTPUT, index=False)
    session_features.to_csv(SESSION_OUTPUT, index=False)

    print(f"Saved {MERGED_OUTPUT} with {len(merged)} rows.")
    print(f"Saved {SESSION_OUTPUT} with {len(session_features)} sessions.")
    print("Done.")


if __name__ == "__main__":
    main()