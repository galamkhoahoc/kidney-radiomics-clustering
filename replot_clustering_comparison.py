import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from pathlib import Path

def main():
    print("Loading data...")
    # Adjust paths if needed
    data_dir = Path('data/processed')
    outputs_dir = Path('outputs')
    
    df_feat = pd.read_csv(data_dir / 'features_clean.csv')
    df_lab = pd.read_csv(outputs_dir / 'cluster_labels.csv')

    # Drop Lesion_Type from df_lab to avoid duplicate columns after merge
    if 'Lesion_Type' in df_lab.columns:
        df_lab = df_lab.drop(columns=['Lesion_Type'])

    # Merge to ensure correct ordering
    df = pd.merge(df_feat, df_lab, on=['PatientID', 'Instance'])

    # Extract features and scale
    feature_cols = [c for c in df.columns if c.startswith('original_')]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # Compute UMAP in the same way as plot_umap.py
    print("Computing UMAP embeddings (this might take a few seconds)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]

    # The clustering methods to plot
    methods = [
        ('HDBSCAN_label', 'HDBSCAN'),
        ('OPTICS_label', 'OPTICS'),
        ('K-Means(k=2)_label', 'K-Means(k=2)'),
        ('GMM(k=2)_label', 'GMM(k=2)')
    ]

    # Filter methods that are actually present in the data
    methods = [(col, title) for col, title in methods if col in df.columns]

    # Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5.5))
    if len(methods) == 1:
        axes = [axes]

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
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    for i, (col, title) in enumerate(methods):
        plot_clusters(axes[i], col, title)

    plt.suptitle("Clustering Comparison (2D embedding)", fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = outputs_dir / 'clustering_comparison.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
