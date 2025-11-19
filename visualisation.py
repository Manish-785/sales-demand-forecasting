import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def generate_visualizations(file_path):
    print(f"Loading dataset from {file_path}...")
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please ensure 'features.csv' is in the same directory.")
        return

    # Convert Date column to datetime objects if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # 5. DBSCAN Clustering with PCA and t-SNE Visualization
    # ---------------------------------------------------------
    print("Generating DBSCAN Clustering (PCA vs t-SNE)...")
    
    # 1. Aggregate data by Store to get a 'Store Profile'
    # We need to ensure these columns exist in the dataframe
    required_columns = ['Store', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Dataset is missing one of the required columns: {required_columns}")
        return

    store_features = df.groupby('Store')[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].mean()
    
    # 2. Standardize the data
    scaler = StandardScaler()
    store_features_scaled = scaler.fit_transform(store_features)
    
    # 3. Apply DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    clusters = dbscan.fit_predict(store_features_scaled)
    
    # 4. Dimensionality Reduction
    
    # A. PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(store_features_scaled)
    
    # B. t-SNE
    # Perplexity must be less than n_samples. We use min(30, n-1) to be safe.
    n_samples = store_features_scaled.shape[0]
    perplex = 15
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
    tsne_coords = tsne.fit_transform(store_features_scaled)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Store': store_features.index,
        'Cluster': clusters,
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'TSNE1': tsne_coords[:, 0],
        'TSNE2': tsne_coords[:, 1]
    })
    
    # 5. Plotting Side-by-Side
    plt.figure(figsize=(20, 8))
    
    # PCA Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=plot_df, x='PC1', y='PC2', 
        hue='Cluster', palette='tab10', s=100, style='Cluster'
    )
    # Annotate PCA
    for i in range(plot_df.shape[0]):
        plt.text(plot_df.PC1[i]+0.1, plot_df.PC2[i], str(plot_df.Store[i]), fontsize=9, alpha=0.7)
    plt.title('PCA Projection of Store Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # t-SNE Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=plot_df, x='TSNE1', y='TSNE2', 
        hue='Cluster', palette='tab10', s=100, style='Cluster'
    )
    # Annotate t-SNE
    for i in range(plot_df.shape[0]):
        plt.text(plot_df.TSNE1[i]+0.1, plot_df.TSNE2[i], str(plot_df.Store[i]), fontsize=9, alpha=0.7)
    plt.title(f't-SNE Projection of Store Clusters (Perplexity={perplex})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.show()

    # Print cluster summary
    print("\n--- DBSCAN Cluster Summary ---")
    print(plot_df['Cluster'].value_counts().sort_index())
    print("\nNote: Cluster -1 represents 'Noise' or Outliers.")

if __name__ == "__main__":
    # Run the function
    file_path = 'features.csv'
    generate_visualizations(file_path)