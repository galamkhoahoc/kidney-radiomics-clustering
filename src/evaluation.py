"""Evaluation metrics, visualization, and stability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from loguru import logger
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import resample

from src.clustering import ClusteringResult, STATUS_OK, STATUS_FAILED
from src.config import (
    BOOTSTRAP_ARI_THRESHOLD,
    BOOTSTRAP_N_ITERATIONS,
    BOOTSTRAP_SAMPLE_RATIO,
    FIGURE_DPI,
    FONT_SIZE,
    KITS23_METADATA_PATH,
    LESION_COLORS,
    OUTPUTS_DIR,
    PLOT_PALETTE,
    UMAP_2D_PARAMS,
)
import src.config as _cfg  # for dynamic access to RANDOM_STATE (updated by --seed)
from src.data_ingestion import load_kits23_metadata


def setup_plot_style() -> None:
    """Configure matplotlib/seaborn defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = FIGURE_DPI
    plt.rcParams["font.size"] = FONT_SIZE


def compute_external_metrics(
    y_true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    method_name: str,
) -> Optional[dict]:
    """
    Compute external clustering metrics against ground-truth lesion types.

    Returns both noise-excluding and noise-including variants so that
    density methods (HDBSCAN/OPTICS) can be fairly compared to centroid
    methods (KMeans/GMM) which label 100% of samples.
    """
    y_true = np.array([1 if y == "Tumor" else 0 for y in y_true_labels])
    non_noise = cluster_labels != -1

    if non_noise.sum() < 2:
        logger.warning("{}: insufficient non-noise data ({}).", method_name, non_noise.sum())
        return None

    # --- Non-noise metrics (density-method diagnostic) ---
    y_nn = y_true[non_noise]
    l_nn = cluster_labels[non_noise]

    ari_nn = float(adjusted_rand_score(y_nn, l_nn))
    nmi_nn = float(normalized_mutual_info_score(y_nn, l_nn))

    ct = pd.crosstab(y_nn, l_nn)
    purity = float(ct.max(axis=0).sum() / non_noise.sum())
    coverage = float(non_noise.sum() / len(cluster_labels))

    # --- Noise-inclusive metrics (fair cross-method comparison) ---
    # Treat noise label -1 as its own cluster
    ari_all = float(adjusted_rand_score(y_true, cluster_labels))
    nmi_all = float(normalized_mutual_info_score(y_true, cluster_labels))

    # Composite score
    coverage_x_ari = coverage * ari_nn

    # Comparison category
    is_density = method_name in ("HDBSCAN", "OPTICS")
    category = "unsupervised_density" if is_density else "supervised_k"

    logger.info(
        "{}: ARI_nn={:.4f}, ARI_all={:.4f}, NMI_nn={:.4f}, NMI_all={:.4f}, "
        "Purity={:.4f}, Coverage={:.1%}, Cov×ARI={:.4f}",
        method_name, ari_nn, ari_all, nmi_nn, nmi_all, purity, coverage,
        coverage_x_ari,
    )

    return {
        "Method": method_name,
        "Comparison_Category": category,
        "ARI_non_noise": ari_nn,
        "NMI_non_noise": nmi_nn,
        "ARI_with_noise": ari_all,
        "NMI_with_noise": nmi_all,
        "Purity": purity,
        "Coverage": coverage,
        "Coverage_x_ARI": coverage_x_ari,
    }


def evaluate_all_methods(
    y_true: np.ndarray,
    results: dict[str, ClusteringResult],
) -> pd.DataFrame:
    """Evaluate all clustering methods and return a comparison DataFrame.

    Methods with status=FAILED_NO_VALID_CONFIG get a row with NaN metrics
    so the failure is visible in the output CSV rather than silently omitted.

    After computing per-method metrics, logs a "best per category" summary:
    - Density methods (HDBSCAN/OPTICS): ranked by Coverage×ARI
    - Centroid methods (K-Means/GMM): ranked by ARI
    """
    metrics: list[dict] = []

    for name, result in results.items():
        if result.status == STATUS_FAILED:
            logger.warning(
                "{}: {} — writing NaN metrics row.", name, result.status,
            )
            is_density = name in ("HDBSCAN", "OPTICS")
            metrics.append({
                "Method": name,
                "Status": STATUS_FAILED,
                "Comparison_Category": "unsupervised_density" if is_density else "supervised_k",
                "ARI_non_noise": float("nan"),
                "NMI_non_noise": float("nan"),
                "ARI_with_noise": float("nan"),
                "NMI_with_noise": float("nan"),
                "Purity": float("nan"),
                "Coverage": 0.0,
                "Coverage_x_ARI": float("nan"),
                "DBCV": float("nan"),
            })
            continue

        m = compute_external_metrics(y_true, result.labels, name)
        if m:
            m["Status"] = STATUS_OK
            # Add DBCV (internal quality) for density methods
            m["DBCV"] = float(result.score) if result.score is not None else float("nan")
            metrics.append(m)

    df_metrics = pd.DataFrame(metrics)
    if not df_metrics.empty:
        df_valid = df_metrics[df_metrics["Status"] == STATUS_OK]

        if not df_valid.empty:
            # ── Headline: best by Coverage×ARI (fair cross-method comparison) ──
            best_cov_ari = df_valid.loc[df_valid["Coverage_x_ARI"].idxmax()]
            logger.info(
                "═══ HEADLINE: Best method by Coverage×ARI: {} ({:.4f}) ═══",
                best_cov_ari["Method"],
                best_cov_ari["Coverage_x_ARI"],
            )

            # ── Best per category summary ──
            density_methods = df_valid[df_valid["Comparison_Category"] == "unsupervised_density"]
            centroid_methods = df_valid[df_valid["Comparison_Category"] == "supervised_k"]

            logger.info("─── Best Per Category ───")

            if not density_methods.empty:
                best_density = density_methods.loc[density_methods["Coverage_x_ARI"].idxmax()]
                logger.info(
                    "  Density winner:  {} — Coverage×ARI={:.4f}  (DBCV={:.4f}, Coverage={:.1%}, ARI_nn={:.4f})",
                    best_density["Method"],
                    best_density["Coverage_x_ARI"],
                    best_density["DBCV"],
                    best_density["Coverage"],
                    best_density["ARI_non_noise"],
                )
            else:
                logger.warning("  Density winner:  NONE — all density methods failed")

            if not centroid_methods.empty:
                best_centroid = centroid_methods.loc[centroid_methods["ARI_with_noise"].idxmax()]
                logger.info(
                    "  Centroid winner: {} — ARI={:.4f}  (Coverage=100%, NMI={:.4f})",
                    best_centroid["Method"],
                    best_centroid["ARI_with_noise"],
                    best_centroid["NMI_with_noise"],
                )
            else:
                logger.warning("  Centroid winner: NONE — no centroid methods ran")

            # ── Full comparison table ──
            display_cols = [
                "Method", "Comparison_Category", "Coverage_x_ARI",
                "DBCV", "ARI_with_noise", "Coverage", "Purity",
            ]
            available_cols = [c for c in display_cols if c in df_valid.columns]
            logger.info(
                "─── Full Metrics Table (sorted by Coverage×ARI) ───\n{}",
                df_valid[available_cols]
                .sort_values("Coverage_x_ARI", ascending=False)
                .to_string(index=False),
            )

        # Report failures
        n_failed = (df_metrics["Status"] == STATUS_FAILED).sum()
        if n_failed > 0:
            logger.warning(
                "{} method(s) FAILED — no valid density structure found. "
                "This is a valid scientific result: the data may not have "
                "strong density-separated groups in PCA space.", n_failed,
            )

        # Expected-outcome framing
        logger.info(
            "NOTE: Use DBCV for internal (unsupervised) quality claims; "
            "ARI for external validation against ground truth; "
            "Coverage×ARI for fair cross-category comparison "
            "(density vs centroid). This is the methodologically correct split."
        )

    return df_metrics


def compute_umap_2d(X_pca: np.ndarray) -> np.ndarray:
    """Reduce PCA features to 2D with UMAP for visualization."""
    logger.info("Computing UMAP 2D embedding.")
    reducer = umap.UMAP(**UMAP_2D_PARAMS)
    return reducer.fit_transform(X_pca)


def compute_tsne_2d(
    X_pca: np.ndarray,
    perplexity: Optional[float] = None,
) -> np.ndarray:
    """Reduce PCA features to 2D with t-SNE for visualization."""
    n_samples = X_pca.shape[0]
    if perplexity is None:
        perplexity = min(30, max(5, n_samples // 10))

    logger.info("Computing t-SNE 2D (perplexity={}).", perplexity)
    tsne = TSNE(
        n_components=2,
        random_state=_cfg.RANDOM_STATE,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X_pca)


def plot_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    ax: plt.Axes,
    palette: tuple[str, ...] = PLOT_PALETTE,
    max_legend_entries: int = 10,
    subtitle: str = "",
) -> None:
    """Draw a 2D scatter plot colored by cluster labels (legend capped)."""
    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    cluster_labels_only = [l for l in unique_labels if l != -1]

    # If too many clusters, show top by size + lump the rest
    if len(cluster_labels_only) > max_legend_entries:
        sizes = {l: int((labels == l).sum()) for l in cluster_labels_only}
        top_labels = sorted(sizes, key=sizes.get, reverse=True)[:max_legend_entries]
        other_labels = set(cluster_labels_only) - set(top_labels)
        n_other = len(other_labels)
    else:
        top_labels = cluster_labels_only
        other_labels = set()
        n_other = 0

    # Plot noise
    if has_noise:
        mask = labels == -1
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c="#bdc3c7", marker="x", s=20, alpha=0.4, label="Noise",
        )

    # Plot top clusters
    for label in top_labels:
        mask = labels == label
        color = palette[label % len(palette)]
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, s=40, alpha=0.7, edgecolors="white",
            linewidth=0.5, label=f"Cluster {label}",
        )

    # Plot remaining clusters lumped
    if other_labels:
        mask = np.isin(labels, list(other_labels))
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c="#95a5a6", s=25, alpha=0.5, edgecolors="white",
            linewidth=0.3, label=f"Other ({n_other} clusters)",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if subtitle:
        ax.text(
            0.5, -0.02, subtitle, transform=ax.transAxes,
            fontsize=8, ha="center", va="top", color="#7f8c8d", style="italic",
        )
    ax.legend(fontsize=8, loc="best")


def plot_ground_truth_umap(
    X_2d: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot 2D embedding colored by Tumor vs Cyst ground truth."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    for lt in ["Tumor", "Cyst"]:
        mask = y_true == lt
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=LESION_COLORS[lt], s=40, alpha=0.7,
            edgecolors="white", linewidth=0.5, label=lt,
        )

    ax.set_title("Ground Truth (Tumor vs Cyst)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Lesion Type")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ground truth plot to {}", output_path)


def plot_clustering_comparison(
    X_2d_default: np.ndarray,
    results: dict[str, ClusteringResult],
    output_path: Path,
    umap_cache: dict = None,
) -> None:
    """Plot clustering results for all methods side by side.

    Uses per-method UMAP embeddings when available (from umap_cache),
    falling back to X_2d_default for centroid methods.
    """
    if umap_cache is None:
        umap_cache = {}
    setup_plot_style()
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5.5))

    if n_methods == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):
        # Determine source space label for caption
        space_label = ""
        if result.clustering_space is not None:
            pca_comp = result.params.get("pca_components", "")
            space_label = f"Clustered in {pca_comp}-dim space"
        elif result.params.get("n_clusters"):
            space_label = f"PCA-{X_2d_default.shape[1] if X_2d_default is not None else '?'}D space"

        # Use method-specific UMAP if available, else default
        X_2d = umap_cache.get(id(result.clustering_space), X_2d_default) if result.clustering_space is not None else X_2d_default

        plot_clusters(
            X_2d,
            result.labels,
            f"{name} ({result.n_clusters} clusters)",
            ax,
            subtitle=space_label,
        )

    plt.suptitle("Clustering Comparison (2D embedding)", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved clustering comparison plot to {}", output_path)


def plot_clustering_with_ground_truth(
    X_2d_gt: np.ndarray,
    X_2d_hdb: np.ndarray,
    X_2d_opt: np.ndarray,
    y_true: np.ndarray,
    hdbscan_result: ClusteringResult,
    optics_result: ClusteringResult,
    output_path: Path,
) -> None:
    """Three-panel plot: ground truth, HDBSCAN, OPTICS.

    Each panel uses the UMAP embedding from the space where its labels
    were discovered, so visualization matches the clustering space.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for lt in ["Tumor", "Cyst"]:
        mask = y_true == lt
        axes[0].scatter(
            X_2d_gt[mask, 0], X_2d_gt[mask, 1],
            c=LESION_COLORS[lt], s=40, alpha=0.7,
            edgecolors="white", linewidth=0.5, label=lt,
        )
    axes[0].set_title("Ground Truth (Tumor vs Cyst)", fontweight="bold")
    axes[0].legend()

    hdb_space = hdbscan_result.params.get("pca_components", "default")
    opt_space = optics_result.params.get("pca_components", "default")

    plot_clusters(
        X_2d_hdb, hdbscan_result.labels,
        f"HDBSCAN ({hdbscan_result.n_clusters} clusters)", axes[1],
        subtitle=f"Clustered in {hdb_space}-dim space",
    )
    plot_clusters(
        X_2d_opt, optics_result.labels,
        f"OPTICS ({optics_result.n_clusters} clusters)", axes[2],
        subtitle=f"Clustered in {opt_space}-dim space",
    )

    plt.suptitle("Clustering Results (2D embedding)", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved clustering visualization to {}", output_path)


def plot_optics_reachability(
    optics_result: ClusteringResult,
    output_path: Path,
) -> None:
    """Plot OPTICS reachability diagram."""
    model = optics_result.model
    if model is None:
        logger.warning("OPTICS model not available — skipping reachability plot.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    reachability = model.reachability_[model.ordering_]
    labels_ordered = optics_result.labels[model.ordering_]

    reach_finite = reachability[np.isfinite(reachability)]
    y_max = np.percentile(reach_finite, 99) * 1.1 if len(reach_finite) > 0 else 1.0

    for label in sorted(set(labels_ordered)):
        mask = labels_ordered == label
        if label == -1:
            color, name = "#bdc3c7", "Noise"
        else:
            color = PLOT_PALETTE[label % len(PLOT_PALETTE)]
            name = f"Cluster {label}"

        ax.bar(
            np.where(mask)[0],
            np.clip(reachability[mask], 0, y_max),
            width=1, color=color, alpha=0.7, label=name,
        )

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Points (OPTICS ordering)")
    ax.set_ylabel("Reachability Distance")
    ax.set_title("OPTICS Reachability Plot", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved OPTICS reachability plot to {}", output_path)


def plot_cluster_feature_heatmap(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Z-score heatmap of top discriminating features per cluster."""
    df_analysis = df_clean.copy()
    df_analysis["Cluster"] = labels
    df_no_noise = df_analysis[df_analysis["Cluster"] != -1]

    if len(df_no_noise) == 0 or df_no_noise["Cluster"].nunique() < 2:
        logger.warning("Insufficient clusters for feature heatmap.")
        return

    cluster_means = df_no_noise.groupby("Cluster")[feature_cols].mean()
    cluster_zscore = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-8)

    feat_var = cluster_zscore.var(axis=0)
    top_features = feat_var.nlargest(min(top_n, len(feature_cols))).index.tolist()
    short_names = [
        f.split("_", 2)[-1][:25] if len(f) > 25 else f for f in top_features
    ]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, max(6, len(top_features) * 0.4)))
    sns.heatmap(
        cluster_zscore[top_features].T.rename(index=dict(zip(top_features, short_names))),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        f"Feature Z-score Heatmap by Cluster (Top {len(top_features)})",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature heatmap to {}", output_path)


def plot_contingency_tables(
    df_clean: pd.DataFrame,
    results: dict[str, ClusteringResult],
    output_path: Path,
) -> None:
    """Bar charts of Tumor/Cyst proportions per cluster."""
    setup_plot_style()
    methods = ["HDBSCAN", "OPTICS"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, method in zip(axes, methods):
        if method not in results:
            continue
        df_tmp = df_clean.copy()
        df_tmp["Cluster"] = results[method].labels
        ct_norm = pd.crosstab(df_tmp["Lesion_Type"], df_tmp["Cluster"], normalize="columns")
        if ct_norm.shape[1] > 0:
            ct_norm.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_title(f"{method}: Tumor/Cyst ratio per cluster", fontweight="bold")
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Cluster")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved contingency plots to {}", output_path)


def analyze_tumor_subclusters(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> Optional[pd.DataFrame]:
    """Run ANOVA on Tumor samples across HDBSCAN clusters."""
    df_analysis = df_clean.copy()
    df_analysis["Cluster"] = labels

    df_tumor = df_analysis[
        (df_analysis["Lesion_Type"] == "Tumor") & (df_analysis["Cluster"] != -1)
    ]
    tumor_clusters = df_tumor["Cluster"].unique()

    logger.info(
        "Tumor samples in HDBSCAN clusters: {} (clusters: {})",
        len(df_tumor), sorted(tumor_clusters),
    )

    if len(tumor_clusters) < 2:
        logger.info("All Tumors in one cluster — ANOVA skipped.")
        return None

    anova_results: list[dict] = []
    for feat in feature_cols:
        groups = [
            df_tumor[df_tumor["Cluster"] == c][feat].values
            for c in tumor_clusters
            if len(df_tumor[df_tumor["Cluster"] == c]) > 1
        ]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results.append({"feature": feat, "F_statistic": f_stat, "p_value": p_value})

    if not anova_results:
        logger.warning("Insufficient data for ANOVA.")
        return None

    df_anova = pd.DataFrame(anova_results).sort_values("p_value")
    df_anova["significant"] = df_anova["p_value"] < 0.05
    n_sig = int(df_anova["significant"].sum())
    logger.info("Significant features (p < 0.05): {}/{}", n_sig, len(df_anova))
    logger.info("Top 10 discriminating features:\n{}", df_anova.head(10).to_string(index=False))
    return df_anova


def analyze_histologic_subtypes(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    metadata_path: Path = KITS23_METADATA_PATH,
    output_path: Optional[Path] = None,
) -> None:
    """Cross-tabulate histologic subtypes with HDBSCAN clusters."""
    subtype_map = load_kits23_metadata(metadata_path)
    if not subtype_map:
        return

    df_analysis = df_clean.copy()
    df_analysis["Cluster"] = labels
    df_analysis["Histologic_Subtype"] = df_analysis["PatientID"].map(subtype_map).fillna("Unknown")

    logger.info(
        "Histologic subtype distribution:\n{}",
        df_analysis["Histologic_Subtype"].value_counts().to_string(),
    )

    known = df_analysis[df_analysis["Histologic_Subtype"] != "Unknown"]
    if len(known) == 0 or known["Histologic_Subtype"].nunique() <= 1:
        logger.warning("Insufficient subtype data for analysis.")
        return

    ct_sub = pd.crosstab(known["Histologic_Subtype"], known["Cluster"], margins=True)
    logger.info("Subtype × Cluster contingency:\n{}", ct_sub.to_string())

    df_tumor_sub = known[
        (known["Lesion_Type"] == "Tumor") & (known["Cluster"] != -1)
    ]
    if len(df_tumor_sub) == 0 or df_tumor_sub["Histologic_Subtype"].nunique() < 2:
        return

    ct_norm = pd.crosstab(
        df_tumor_sub["Histologic_Subtype"],
        df_tumor_sub["Cluster"],
        normalize="columns",
    )
    logger.info("Tumor subtype proportions per cluster:\n{}", ct_norm.round(3).to_string())

    if output_path and ct_norm.shape[0] >= 2 and ct_norm.shape[1] >= 1:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, max(4, ct_norm.shape[0] * 0.6)))
        sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax)
        ax.set_title("Tumor Subtypes / HDBSCAN Cluster", fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved subtype heatmap to {}", output_path)


def bootstrap_stability(
    X: np.ndarray,
    labels_original: np.ndarray,
    params: dict,
    n_bootstrap: int = BOOTSTRAP_N_ITERATIONS,
    ratio: float = BOOTSTRAP_SAMPLE_RATIO,
    output_path: Optional[Path] = None,
) -> list[float]:
    """
    Bootstrap resampling to assess HDBSCAN cluster stability via ARI.

    Returns:
        List of ARI scores from valid bootstrap iterations.
    """
    n = X.shape[0]
    n_sub = int(n * ratio)
    ari_scores: list[float] = []

    logger.info("Bootstrap stability: {} iterations × {:.0%} subsample.", n_bootstrap, ratio)

    for i in range(n_bootstrap):
        idx = resample(range(n), n_samples=n_sub, random_state=i)
        X_boot = X[idx]

        hdb_boot = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_method=params.get("method", "eom"),
        )
        labels_boot = hdb_boot.fit_predict(X_boot)

        n_clusters_boot = len(set(labels_boot)) - (1 if -1 in labels_boot else 0)
        if n_clusters_boot >= 2:
            ari = float(adjusted_rand_score(np.array(labels_original)[idx], labels_boot))
            ari_scores.append(ari)

    if not ari_scores:
        logger.warning("No valid bootstrap iterations.")
        return ari_scores

    mean_ari = float(np.mean(ari_scores))
    std_ari = float(np.std(ari_scores))
    median_ari = float(np.median(ari_scores))

    logger.info(
        "Bootstrap ARI: {:.4f} ± {:.4f} (median={:.4f}, n={}/{})",
        mean_ari, std_ari, median_ari, len(ari_scores), n_bootstrap,
    )

    if mean_ari < BOOTSTRAP_ARI_THRESHOLD:
        logger.warning("Mean ARI < {:.1f} — clusters may be unstable.", BOOTSTRAP_ARI_THRESHOLD)
    else:
        logger.info("Mean ARI ≥ {:.1f} — clusters appear stable.", BOOTSTRAP_ARI_THRESHOLD)

    if output_path:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ari_scores, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(mean_ari, color="#e74c3c", linestyle="--", lw=2, label=f"Mean={mean_ari:.3f}")
        ax.axvline(BOOTSTRAP_ARI_THRESHOLD, color="orange", linestyle=":", lw=2, label="Threshold=0.5")
        ax.set_xlabel("ARI Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Bootstrap Stability Analysis", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved bootstrap histogram to {}", output_path)

    return ari_scores


def _delete_stale_file(path: Path) -> None:
    """Remove a file if it exists (prevents stale outputs from prior runs)."""
    if path.exists():
        path.unlink()
        logger.info("Deleted stale output: {}", path.name)


def run_evaluation(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    X_pca: np.ndarray,
    results: dict[str, ClusteringResult],
    hdbscan_params: dict,
    grid_dfs: Optional[dict[str, pd.DataFrame]] = None,
    output_dir: Path = OUTPUTS_DIR,
    use_tsne: bool = False,
    run_bootstrap: bool = True,
) -> pd.DataFrame:
    """
    Full evaluation pipeline: metrics, visualizations, ANOVA, bootstrap.

    Also saves grid-search CSVs and sample-level cluster labels.
    Methods with status=FAILED_NO_VALID_CONFIG get NaN metrics rows and
    cluster-dependent plots are skipped (stale files are cleaned).

    Returns:
        Metrics comparison DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    y_true = df_clean["Lesion_Type"].values
    df_metrics = evaluate_all_methods(y_true, results)
    df_metrics.to_csv(output_dir / "clustering_metrics.csv", index=False)

    # --- Save grid search CSVs ---
    if grid_dfs:
        for name, df_grid in grid_dfs.items():
            if not df_grid.empty:
                path = output_dir / f"{name}.csv"
                df_grid.to_csv(path, index=False)
                logger.info("Saved {} ({} rows) to {}", name, len(df_grid), path)

    # --- Save cluster labels CSV ---
    df_labels = df_clean[["PatientID", "Instance", "Lesion_Type"]].copy()
    for name, result in results.items():
        col_name = f"{name}_label"
        if len(result.labels) == len(df_labels):
            df_labels[col_name] = result.labels
        else:
            # Density methods may have run on different PCA dims;
            # labels still align with df_clean row order.
            df_labels[col_name] = -999  # sentinel for length mismatch
            logger.warning(
                "{} labels length ({}) != df_clean ({}); writing sentinel.",
                name, len(result.labels), len(df_labels),
            )
    df_labels.to_csv(output_dir / "cluster_labels.csv", index=False)
    logger.info("Saved cluster labels to {}", output_dir / "cluster_labels.csv")

    # --- Save contingency tables as CSV (only for OK methods) ---
    for name, result in results.items():
        ct_path = output_dir / f"contingency_{name}.csv"
        if result.status == STATUS_FAILED:
            _delete_stale_file(ct_path)
            continue
        df_tmp = df_clean.copy()
        df_tmp["Cluster"] = result.labels
        ct = pd.crosstab(df_tmp["Lesion_Type"], df_tmp["Cluster"], margins=True)
        ct.to_csv(ct_path)

    # --- 2D embedding ---
    # Compute per-method UMAP from each method's clustering space
    # Cache by array identity to avoid redundant computation
    umap_cache: dict[int, np.ndarray] = {}  # id(clustering_space) -> UMAP 2D

    for name, result in results.items():
        if result.clustering_space is not None and id(result.clustering_space) not in umap_cache:
            pca_label = result.params.get("pca_components", "?")
            logger.info(
                "Computing UMAP 2D for {} from {}-dim clustering space (matches where labels were discovered).",
                name, result.clustering_space.shape[1],
            )
            umap_cache[id(result.clustering_space)] = compute_umap_2d(result.clustering_space)

    # Default UMAP from X_pca for centroid methods
    X_2d_default = compute_tsne_2d(X_pca) if use_tsne else compute_umap_2d(X_pca)

    plot_ground_truth_umap(X_2d_default, y_true, output_dir / "ground_truth_2d.png")

    # --- Clustering comparison (include all methods, even failed ones) ---
    plot_clustering_comparison(
        X_2d_default, results, output_dir / "clustering_comparison.png",
        umap_cache=None,  # Use unified X_2d_default space for all subplots
    )

    hdb = results["HDBSCAN"]
    opt = results["OPTICS"]
    hdb_ok = hdb.status == STATUS_OK
    opt_ok = opt.status == STATUS_OK

    # For the three-panel plot, use the same default UMAP space for consistency
    X_2d_hdb = X_2d_default
    X_2d_opt = X_2d_default

    # --- Three-panel ground truth comparison ---
    plot_clustering_with_ground_truth(
        X_2d_default, X_2d_hdb, X_2d_opt, y_true, hdb, opt,
        output_dir / "hdbscan_optics_ground_truth.png",
    )

    # --- OPTICS reachability: only if OPTICS has a fitted model ---
    reachability_path = output_dir / "optics_reachability.png"
    if opt_ok and opt.model is not None:
        plot_optics_reachability(opt, reachability_path)
    else:
        _delete_stale_file(reachability_path)
        logger.info("Skipping OPTICS reachability plot (method {}).", opt.status)

    # --- Feature heatmap: only if HDBSCAN found valid clusters ---
    heatmap_path = output_dir / "cluster_feature_heatmap.png"
    if hdb_ok and hdb.n_clusters >= 2:
        plot_cluster_feature_heatmap(
            df_clean, hdb.labels, feature_cols, heatmap_path,
        )
    else:
        _delete_stale_file(heatmap_path)
        logger.info("Skipping cluster feature heatmap (HDBSCAN {}).", hdb.status)

    # --- Contingency bar charts ---
    plot_contingency_tables(df_clean, results, output_dir / "contingency_tables.png")

    # --- Tumor subcluster ANOVA: only if HDBSCAN found clusters ---
    if hdb_ok and hdb.n_clusters >= 2:
        analyze_tumor_subclusters(df_clean, hdb.labels, feature_cols)
    else:
        logger.info("Skipping tumor subcluster ANOVA (HDBSCAN {}).", hdb.status)

    # --- Histologic subtype analysis: only if HDBSCAN found clusters ---
    subtype_path = output_dir / "subtype_heatmap.png"
    if hdb_ok and hdb.n_clusters >= 2:
        analyze_histologic_subtypes(
            df_clean, hdb.labels, output_path=subtype_path,
        )
    else:
        _delete_stale_file(subtype_path)
        logger.info("Skipping subtype heatmap (HDBSCAN {}).", hdb.status)

    # --- Bootstrap stability: only if HDBSCAN found clusters ---
    bootstrap_path = output_dir / "bootstrap_stability.png"
    if run_bootstrap and hdb_ok and hdb.n_clusters >= 2:
        bootstrap_stability(
            X_pca, hdb.labels, hdbscan_params,
            output_path=bootstrap_path,
        )
    else:
        _delete_stale_file(bootstrap_path)
        if run_bootstrap:
            logger.info("Skipping bootstrap stability (HDBSCAN {}).", hdb.status)

    return df_metrics
