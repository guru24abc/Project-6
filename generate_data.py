import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------
# CONFIG
# ---------------------------
NUM_ROWS = 100000
START_DATE = datetime(2026, 2, 1)

random.seed(42)
np.random.seed(42)

# Domain → Category mapping
domain_map = {
    "instagram.com": "social",
    "facebook.com": "social",
    "x.com": "social",
    "reddit.com": "social",
    "youtube.com": "video",
    "netflix.com": "video",
    "hotstar.com": "video",
    "primevideo.com": "video",
    "github.com": "learning",
    "kaggle.com": "learning",
    "stackoverflow.com": "learning",
    "coursera.org": "learning",
    "medium.com": "learning",
    "amazon.com": "shopping",
    "flipkart.com": "shopping",
    "wikipedia.org": "other"
}

domains = list(domain_map.keys())

# RAM usage base per category
ram_profile = {
    "social": (500, 900),
    "video": (1200, 2000),
    "learning": (700, 1200),
    "shopping": (600, 1000),
    "other": (400, 800)
}

# ---------------------------
# GENERATE BROWSING DATA
# ---------------------------
rows = []
current_time = START_DATE

for _ in range(NUM_ROWS):
    if random.random() < 0.05:
        current_time += timedelta(minutes=random.randint(20, 60))
    else:
        current_time += timedelta(seconds=random.randint(10, 120))

    domain = random.choice(domains)
    category = domain_map[domain]

    rows.append({
        "timestamp": current_time,
        "url": f"https://{domain}/page{random.randint(1,100)}",
        "domain": domain,
        "category": category,
        "browser": random.choice(["chrome", "edge"])
    })

browser_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

# ---------------------------
# GENERATE RAM DATA
# ---------------------------
ram_times = pd.DataFrame({
    "timestamp": pd.date_range(
        start=START_DATE,
        end=browser_df["timestamp"].max(),
        freq="5s"
    )
})

ram_times = pd.merge_asof(
    ram_times.sort_values("timestamp"),
    browser_df[["timestamp", "category"]].sort_values("timestamp"),
    on="timestamp",
    direction="backward"
)

ram_times["category"] = ram_times["category"].fillna("other")

ram_times["ram_used_mb"] = np.random.uniform(5000, 8000, size=len(ram_times))
ram_times["browser_ram_mb"] = ram_times["category"].apply(
    lambda c: np.random.uniform(*ram_profile[c])
)
ram_times["ram_available_mb"] = 16000 - ram_times["ram_used_mb"]

ram_df = ram_times[[
    "timestamp",
    "ram_used_mb",
    "ram_available_mb",
    "browser_ram_mb"
]].copy()

# ---------------------------
# SAVE FILES
# ---------------------------
browser_df.to_csv("browsing_history.csv", index=False)
ram_df.to_csv("ram_log.csv", index=False)

domain_df = pd.DataFrame(list(domain_map.items()), columns=["domain", "category"])
domain_df.to_csv("domain_category_map.csv", index=False)

print("✅ Synthetic data generated successfully!")
print("Browsing rows:", len(browser_df))
print("RAM rows:", len(ram_df))