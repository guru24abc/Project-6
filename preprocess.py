import pandas as pd
import numpy as np
from urllib.parse import urlparse, urlunparse

HISTORY_INPUT = "history_last_5d_100k.csv"
RAM_INPUT = "ram_log_5d_5s.csv"
DOMAIN_MAP_INPUT = "domain_category_map.csv"

HISTORY_OUTPUT = "browsing_history.csv"
RAM_OUTPUT = "ram_log.csv"

def extract_domain_from_url(url):
    if pd.isna(url) or str(url).strip() == "":
        return np.nan
    url = str(url).strip()
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "").strip()
    return domain if domain else np.nan

def clean_url(url, domain=None):
    if pd.notna(url) and str(url).strip() != "":
        url = str(url).strip()
        if not url.startswith("http"):
            url = "https://" + url
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    if pd.notna(domain) and str(domain).strip() != "":
        return f"https://{str(domain).strip()}"
    return np.nan

def preprocess_history():
    df = pd.read_csv(HISTORY_INPUT)

    # timestamp
    if "timestamp_local" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("No timestamp column found in history file.")

    # domain
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str).str.lower().str.replace("www.", "", regex=False).str.strip()
        df.loc[df["domain"].isin(["nan", "none", ""]), "domain"] = np.nan
    else:
        df["domain"] = np.nan

    # If domain missing, derive from URL
    if "url" in df.columns:
        missing_domain = df["domain"].isna()
        df.loc[missing_domain, "domain"] = df.loc[missing_domain, "url"].apply(extract_domain_from_url)
    else:
        df["url"] = np.nan

    # If url missing, rebuild from domain
    if "url" in df.columns:
        df["url"] = df.apply(lambda r: clean_url(r["url"], r["domain"]), axis=1)
    else:
        df["url"] = df["domain"].apply(lambda d: f"https://{d}" if pd.notna(d) else np.nan)

    # category
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.lower().str.strip()
        df.loc[df["category"].isin(["nan", "none", ""]), "category"] = np.nan
    else:
        df["category"] = np.nan

    # If category missing, map by domain
    if df["category"].isna().any():
        domain_map = pd.read_csv(DOMAIN_MAP_INPUT)
        domain_map["domain"] = domain_map["domain"].astype(str).str.lower().str.replace("www.", "", regex=False).str.strip()
        domain_map["category"] = domain_map["category"].astype(str).str.lower().str.strip()
        df = df.drop(columns=["category"], errors="ignore").merge(domain_map, on="domain", how="left")

    # browser
    if "browser" not in df.columns:
        df["browser"] = "chrome"
    else:
        df["browser"] = df["browser"].astype(str).str.lower().str.strip()

    # title
    if "title" not in df.columns:
        df["title"] = ""
    else:
        df["title"] = df["title"].fillna("")

    # Clean bad values
    df = df.dropna(subset=["timestamp", "domain"])
    df = df.drop_duplicates(subset=["timestamp", "domain", "browser"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    keep_cols = ["timestamp", "url", "title", "domain", "category", "browser", "hour", "date"]
    df = df[keep_cols]

    df.to_csv(HISTORY_OUTPUT, index=False)
    print(f"Saved {HISTORY_OUTPUT} with {len(df)} rows.")

def preprocess_ram():
    df = pd.read_csv(RAM_INPUT)

    if "timestamp_local" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("No timestamp column found in RAM file.")

    for col in ["ram_used_mb", "ram_available_mb", "browser_ram_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "ram_used_mb", "ram_available_mb", "browser_ram_mb"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df[["timestamp", "ram_used_mb", "ram_available_mb", "browser_ram_mb"]]
    df.to_csv(RAM_OUTPUT, index=False)
    print(f"Saved {RAM_OUTPUT} with {len(df)} rows.")

if __name__ == "__main__":
    preprocess_history()
    preprocess_ram()