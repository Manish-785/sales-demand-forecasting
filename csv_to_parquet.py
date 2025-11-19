import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Configuration ---
INPUT_FILE = "Walmart.csv"
PARQUET_FILE = "features.parquet"

def thanksgiving_date(year):
    d = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = 1 + ((3 - d.weekday()) % 7)
    return pd.Timestamp(year=year, month=11, day=first_thursday + 21).date()

def add_features(df):
    print("Feature Engineering: Starting...")
    df = df.copy()
    
    # 1. Date Parsing and Sorting
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # 2. Holiday Features
    years = df['Date'].dt.year.dropna().unique()
    tg_dates = {int(y): thanksgiving_date(int(y)) for y in years}

    def is_major(d):
        if pd.isna(d): return 0
        if d.month == 12 and d.day == 25: return 1
        if tg_dates.get(int(d.year)) == d.date(): return 1
        return 0

    df['is_major_holiday'] = df['Date'].apply(is_major).astype(int)
    
    if 'Holiday_Flag' in df.columns:
        df['is_minor_holiday'] = ((df['Holiday_Flag'] == 1) & (df['is_major_holiday'] == 0)).astype(int)
    else:
        df['is_minor_holiday'] = 0

    # 3. Time Features
    df['year'] = df['Date'].dt.year.astype('Int64')
    df['month'] = df['Date'].dt.month.astype('Int64')
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype('Int64')
    df['day_of_week'] = df['Date'].dt.weekday.astype('Int64')

    # Cyclical Encoding
    week_float = df['weekofyear'].astype(float)
    month_float = df['month'].astype(float)
    df['weekofyear_sin'] = np.sin(2 * np.pi * (week_float - 1) / 52)
    df['weekofyear_cos'] = np.cos(2 * np.pi * (week_float - 1) / 52)
    df['month_sin'] = np.sin(2 * np.pi * (month_float - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month_float - 1) / 12)

    # Ensure Target is Numeric
    df['Weekly_Sales'] = pd.to_numeric(df.get('Weekly_Sales'), errors='coerce')

    # 4. Lag Features
    lag_periods = [1, 2, 3, 7, 14, 28, 52]
    for lag in lag_periods:
        df[f'lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)

    # 5. Rolling Window Statistics
    roll_means = [3, 4, 8, 12, 52]
    roll_stds = [3, 4, 8, 52]
    for w in roll_means:
        df[f'roll_mean_{w}'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    for w in roll_stds:
        df[f'roll_std_{w}'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        )

    # 6. EWMA Features
    ewm_spans = [3, 7, 14]
    for s in ewm_spans:
        df[f'ewm_span_{s}'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.ewm(span=s, adjust=False).mean()
        )

    # --- REMOVED: Store-Level Static Statistics ---
    # (No store_mean, store_std, store_autocorr, etc.)

    # 7. Target Transformations
    # KEEP: Log1p (Safe, helps convergence)
    df['sales_log1p'] = np.log1p(df['Weekly_Sales'].fillna(0))

    # REMOVED: Z-score and MinMax (Depended on static stats / caused leakage)
    
    print("Feature Engineering: Complete.")
    return df

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    df_processed = add_features(df)
    
    # Clean NaNs
    exclude_cols = ['Date']
    numeric_cols = [c for c in df_processed.columns if c not in exclude_cols]
    df_processed[numeric_cols] = df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    df_final = df_processed.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    print(f"Saving to {PARQUET_FILE}...")
    try:
        df_final.to_parquet(PARQUET_FILE, index=False)
        print("Success! Parquet file created.")
    except ImportError:
        print("Error: 'pyarrow' or 'fastparquet' missing. Install with: pip install pyarrow")
    except Exception as e:
        print(f"Error saving: {e}")

if __name__ == "__main__":
    main()