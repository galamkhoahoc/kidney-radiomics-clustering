import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

def main():
    print("Loading data...")
    df_feat = pd.read_csv('/content/kits23_210626_2100/data/processed/features_clean.csv')
    df_lab = pd.read_csv('/content/kits23_210626_2100/outputs/cluster_labels.csv')

    # Drop Lesion_Type from df_lab to avoid duplicate columns after merge
    if 'Lesion_Type' in df_lab.columns:
        df_lab = df_lab.drop(columns=['Lesion_Type'])

    # Merge to ensure correct ordering
    df = pd.merge(df_feat, df_lab, on=['PatientID', 'Instance'])

    # Extract features and scale
    feature_cols = [c for c in df.columns if c.startswith('original_')]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # Compute UMAP
    print("Computing UMAP embeddings (this might take a few seconds)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]

    # Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Ground Truth
    sns.scatterplot(
        data=df, x='UMAP1', y='UMAP2', 
        hue='Lesion_Type', palette={'Tumor': '#d62728', 'Cyst': '#1f77b4'}, 
        ax=axes[0], s=20, alpha=0.8
    )
    axes[0].set_title("Ground Truth (Tumor vs Cyst)")

    # Helper function to plot clustering results (handles noise as grey)
    def plot_clusters(ax, label_col, title):
        unique_labels = sorted(df[label_col].unique())
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        color_map = {lbl: ('#d3d3d3' if lbl == -1 else palette[i % len(palette)]) for i, lbl in enumerate(unique_labels)}
        
        sns.scatterplot(
            data=df, x='UMAP1', y='UMAP2', 
            hue=label_col, palette=color_map, 
            ax=ax, s=20, alpha=0.8, legend='full'
        )
        ax.set_title(title)

    # 2. OPTICS
    plot_clusters(axes[1], 'OPTICS_label', 'OPTICS Clusters')

    # 3. HDBSCAN
    plot_clusters(axes[2], 'HDBSCAN_label', 'HDBSCAN Clusters')

    plt.tight_layout()
    out_file = '/content/kits23_210626_2100/outputs/umap_clustering_comparison.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
