"""Compare per-store MAPE across ARIMA, XGBoost, LSTM and STGNN.

Reads the per-store result files in results/metrics/ and writes a MAPE
distribution boxplot to results/figures/mape_distribution.png, plus an
aggregate summary table to results/metrics/aggregate_metrics.csv.

Usage:
    python src/plot_model_comparison.py
"""
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

# (file, MAPE column, forecast column, actual column)
MODELS = {
    "ARIMA": ("arima_results_filtered.csv", "MAPE (%)", "forecast", "real"),
    "XGBoost": ("xgboost_results.csv", "MAPE (%)", "pred", "real"),
    "LSTM": ("lstm_store_results.csv", "mape", "forecast", "real"),
    "STGNN": ("gnn_results.csv", "MAPE (%)", None, None),
}


def ntae(df, fc, rc):
    """Normalised total absolute error (%) pooled over all stores and weeks."""
    err = tot = 0.0
    for f, r in zip(df[fc], df[rc]):
        f, r = np.asarray(ast.literal_eval(f)), np.asarray(ast.literal_eval(r))
        n = min(len(f), len(r))
        err += np.abs(f[-n:] - r[-n:]).sum()
        tot += r[-n:].sum()
    return err / tot * 100


def main():
    mape, rows = {}, []
    for name, (fname, mcol, fc, rc) in MODELS.items():
        df = pd.read_csv(METRICS / fname)
        m = df[mcol].to_numpy()
        mape[name] = m
        rows.append({
            "model": name,
            "stores": len(m),
            "mean_mape": m.mean(),
            "median_mape": np.median(m),
            "p90_mape": np.percentile(m, 90),
            "var_mape": m.var(),
            "ntae": ntae(df, fc, rc) if fc else np.nan,
        })

    summary = pd.DataFrame(rows).round(2)
    summary.to_csv(METRICS / "aggregate_metrics.csv", index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(list(mape.values()), showmeans=True,
               meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    ax.set_xticks(range(1, len(mape) + 1), list(mape.keys()))
    ax.set_ylabel("Per-store MAPE (%)")
    ax.set_title("Distribution of per-store MAPE across 45 Walmart stores")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "mape_distribution.png", dpi=200)
    print(f"Saved {FIGURES / 'mape_distribution.png'}")


if __name__ == "__main__":
    main()
