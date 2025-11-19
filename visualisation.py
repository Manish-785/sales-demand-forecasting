import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --- Configuration ---
INPUT_FILE = 'features.csv'

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Prepare Data
    print("Aggregating features...")
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop non-useful columns for clustering
    cols_to_drop = ['Store', 'year']
    cols_to_use = [c for c in numeric_df.columns if c not in cols_to_drop]
    
    # Average features per store
    store_features = numeric_df.groupby('Store')[cols_to_use].mean()

    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(store_features)

    # 3. Run DBSCAN (The Logic)
    print("Running DBSCAN...")
    # eps=2.5 is robust for high-dimensional data
    dbscan = DBSCAN(eps=2.5, min_samples=3)
    clusters = dbscan.fit_predict(X_scaled)
    store_features['Cluster'] = clusters

    # 4. Get Recommendations
    print("\n--- GNN Experiment Candidates ---")
    cluster_counts = store_features['Cluster'].value_counts()
    print("Cluster Counts:\n", cluster_counts)
    
    valid_clusters = cluster_counts[cluster_counts.index != -1]
    for cluster_id in valid_clusters.index:
        stores_in_cluster = store_features[store_features['Cluster'] == cluster_id].index.tolist()
        print(f"\nCluster {cluster_id} (Stores: {len(stores_in_cluster)}) -> Store IDs: {stores_in_cluster}")

if __name__ == "__main__":
    main()