"""Exploratory summary tables and figures for the engineered dataset.

Writes feature summary statistics, per-store weekly statistics and the
sales distribution / boxplot / representative-store figures to
results/figures/.

Usage:
    python src/eda_summary.py
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "features.csv"
OUT_DIR = ROOT / "results" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"], dayfirst=True)
print("Loaded", len(df), "rows from", INPUT_PATH)

# --- Basic cleaning (if needed) ---
# Ensure required columns exist
required_cols = ["Store","Date","Weekly_Sales","sales_log1p","lag_1","roll_mean_4","Temperature","CPI"]
present = [c for c in required_cols if c in df.columns]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("Warning: missing columns (these will be omitted from summary):", missing)

# --- Aggregate to store-week level if multiple dept rows exist (safe if already aggregated) ---
# We assume 'Weekly_Sales' is already store-aggregated. If duplicates exist per (Store,Date), sum them.
dup_check = df.groupby(["Store","Date"]).size().max()
if dup_check > 1:
    print("Detected multiple rows per (Store,Date). Aggregating Weekly_Sales by sum.")
    df = df.groupby(["Store","Date"], as_index=False).agg({"Weekly_Sales":"sum", 
                                                           "sales_log1p":"sum" if "sales_log1p" in df.columns else "first",
                                                           "Temperature":"first",
                                                           "CPI":"first"})
    # Recompute sales_log1p if not present
    if "sales_log1p" not in df.columns:
        df["sales_log1p"] = np.log1p(df["Weekly_Sales"].fillna(0))

# --- Summary statistics for placeholders ---
selected = [c for c in ["Weekly_Sales","sales_log1p","lag_1","roll_mean_4","Temperature","CPI"] if c in df.columns]
summary = df[selected].describe().transpose()[["mean","std","min","25%","50%","75%","max"]]
summary = summary.rename(columns={"std":"std","50%":"median"})
summary_out = os.path.join(OUT_DIR, "feature_summary.csv")
summary.to_csv(summary_out)
print("Saved feature summary table to:", summary_out)

# Optionally show the table (if running in notebook)
print("\nFeature summary (first rows):")
print(summary.head())

# --- Figure 1: Distribution of Weekly Sales (histogram) ---
fig1_path = os.path.join(OUT_DIR, "weekly_sales_distribution.png")
plt.figure(figsize=(8,4))
plt.hist(df["Weekly_Sales"].dropna(), bins=100)
plt.title("Distribution of Weekly Sales (store-aggregated)")
plt.xlabel("Weekly Sales (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Saved figure:", fig1_path)

# --- Figure 2: Time series for representative store (Store 37 if present) ---
import matplotlib.dates as mdates

store_list = sorted(df["Store"].unique())
rep_store = 37 if 37 in store_list else store_list[len(store_list)//2]
df_ts = df[df["Store"] == rep_store].sort_values("Date")
fig2_path = os.path.join(OUT_DIR, f"store_{rep_store}_timeseries.png")
plt.figure(figsize=(10,3))
ax = plt.gca()
ax.plot(df_ts["Date"], df_ts["Weekly_Sales"], linestyle="-")
ax.set_title(f"Weekly Sales Time Series for Store {rep_store}")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales (USD)")

# Show at most N x-ticks (evenly spaced) and format dates
max_ticks = 10
if len(df_ts) > 0:
    tick_count = min(max_ticks, len(df_ts))
    idx = np.linspace(0, len(df_ts) - 1, tick_count, dtype=int)
    ticks = df_ts["Date"].iloc[idx]
    ax.set_xticks(ticks)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Saved figure:", fig2_path)

# --- Figure 3: Boxplot of weekly sales by store (ordered by median) ---
fig3_path = os.path.join(OUT_DIR, "weekly_sales_boxplot_by_store.png")
# compute medians to order stores
medians = df.groupby("Store")["Weekly_Sales"].median().sort_values()
ordered_stores = medians.index.tolist()
# Prepare data for boxplot; to avoid plotting thousands of points, we will not show fliers
data_for_box = [df.loc[df["Store"]==s, "Weekly_Sales"].dropna().values for s in ordered_stores]
plt.figure(figsize=(12,5))
plt.boxplot(data_for_box, showfliers=False)
plt.title("Boxplot of Weekly Sales by Store (ordered by median)")
plt.xlabel("Store (ordered by median)")
plt.ylabel("Weekly Sales (USD)")
# Replace x-ticks with store ids but limit tick labels to avoid clutter
tick_pos = list(range(1, len(ordered_stores)+1))
plt.xticks(tick_pos, ordered_stores, rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(fig3_path, dpi=300)
plt.close()
print("Saved figure:", fig3_path)

# --- Optional: Save a small CSV with per-store medians for LaTeX table or figure annotation ---
store_stats = df.groupby("Store")["Weekly_Sales"].agg(["count","mean","median","std"]).reset_index()
store_stats_out = os.path.join(OUT_DIR, "store_weekly_stats.csv")
store_stats.to_csv(store_stats_out, index=False)
print("Saved per-store stats to:", store_stats_out)

print("\nAll artifacts saved in:", OUT_DIR)
print("Files created:")
for f in os.listdir(OUT_DIR):
    print(" -", os.path.join(OUT_DIR, f))
