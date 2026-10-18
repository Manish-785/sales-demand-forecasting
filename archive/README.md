# Archive

Superseded experiments kept for provenance. They are **not** part of the
reported pipeline and may reference old file locations.

| File | What it was | Superseded by |
|---|---|---|
| `legacy_gconvgru_stgnn.ipynb` | Early EDA plus a first STGNN prototype using `GConvGRU` (torch-geometric-temporal) on a fixed correlation-threshold store graph (\|ρ\| > 0.7). | `notebooks/04_stgnn.ipynb` (learned adjacency + dilated TCN) |
| `walmart_arima_new.ipynb` | Early SARIMAX experiment with inline feature engineering. | `notebooks/01_arima.ipynb` |
| `csv_to_parquet.py` | First feature pipeline writing Parquet; its rolling/EWMA features were **not** shifted and leak the current week. | `src/features.py` |
