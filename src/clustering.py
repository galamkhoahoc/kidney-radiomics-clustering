"""Unsupervised clustering: HDBSCAN, OPTICS, GMM, and K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hdbscan
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.config import (
    BASELINE_N_CLUSTERS,
    CLUSTER_COUNT_PENALTY,
    GMM_COVARIANCE_TYPE,
    HDBSCAN_CLUSTER_METHODS,
    HDBSCAN_EPSILON_VALUES,
    HDBSCAN_MAX_CLUSTERS,
    HDBSCAN_MIN_CLUSTER_SIZES,
    HDBSCAN_MIN_COVERAGE,
    HDBSCAN_MIN_SAMPLES_LIST,
    KMEANS_N_INIT,
    OPTICS_MAX_CLUSTERS,
    OPTICS_MIN_CLUSTER_SIZES,
    OPTICS_MIN_COVERAGE,
    OPTICS_MIN_SAMPLES_LIST,
    OPTICS_XI_VALUES,
    OPTICS_EPS_VALUES,
    OPTICS_CLUSTER_METHODS,
    PCA_N_COMPONENTS_LIST,
    TARGET_N_CLUSTERS,
    UMAP_CLUSTERING_N_COMPONENTS,
    UMAP_CLUSTERING_N_COMPONENTS_LIST,
    UMAP_CLUSTERING_PARAMS,
)
import src.config as _cfg  # for dynamic access to RANDOM_STATE + USE_UMAP_FOR_DENSITY_CLUSTERING

# Status constants
STATUS_OK = "OK"
STATUS_FAILED = "FAILED_NO_VALID_CONFIG"


@dataclass
class ClusteringResult:
    """Container for a single clustering method output."""

    method: str
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    params: dict
    status: str = STATUS_OK
    model: object = None
    score: Optional[float] = None
    clustering_space: Optional[np.ndarray] = None  # the actual space labels were discovered in


def _count_clusters(labels: np.ndarray) -> int:
    """Count clusters excluding noise label (-1)."""
    return len(set(labels)) - (1 if -1 in labels else 0)


def _coverage(labels: np.ndarray) -> float:
    """Fraction of samples assigned to a cluster (not noise)."""
    return float((labels != -1).sum() / len(labels))


def _cluster_count_penalty(n_clusters: int) -> float:
    """Soft penalty nudging toward TARGET_N_CLUSTERS.

    A 2-cluster solution keeps its full score, 3 clusters get ~0.87,
    9 clusters get ~0.49.  Keeps it discoverable but discourages fragmentation.
    """
    distance = abs(n_clusters - TARGET_N_CLUSTERS)
    return 1.0 / (1.0 + CLUSTER_COUNT_PENALTY * distance)


def compute_umap_embedding(
    X_scaled: np.ndarray,
    n_components: int = UMAP_CLUSTERING_N_COMPONENTS,
) -> np.ndarray:
    """Compute a UMAP embedding for clustering (NOT visualization).

    SCIENTIFIC CAVEAT: Clustering on UMAP embeddings is common but contested.
    UMAP can manufacture apparent clusters and distorts global density.
    Results should be labeled as 'UMAP-space clustering' and reported
    alongside PCA-space results, never as a replacement.
    """
    import umap

    params = {**UMAP_CLUSTERING_PARAMS, "n_components": n_components}
    reducer = umap.UMAP(**params)
    X_umap = reducer.fit_transform(X_scaled)
    logger.info(
        "UMAP clustering embedding: {} → {}D (min_dist={}, n_neighbors={}).",
        X_scaled.shape[1], n_components,
        params["min_dist"], params["n_neighbors"],
    )
    return X_umap


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def grid_search_hdbscan(
    X_scaled: np.ndarray,
    X_pca_default: np.ndarray,
) -> tuple[Optional[dict], pd.DataFrame, np.ndarray]:
    """
    Grid search HDBSCAN hyperparameters across PCA dimensions.

    Scores by ``dbcv * coverage``, rejecting configs with coverage < threshold
    or cluster count outside [2, max_clusters].

    Args:
        X_scaled: Standardised feature matrix (pre-PCA) for multi-PCA search.
        X_pca_default: Default PCA-95% space (used as one candidate).

    Returns:
        Tuple of (best params dict or None, full grid DataFrame, best PCA-projected X).
        Returns None as first element when no valid configuration meets constraints.
    """
    n_samples = X_scaled.shape[0]
    best_score = -1.0
    best_params: Optional[dict] = None
    best_X: np.ndarray = X_pca_default
    results: list[dict] = []

    # Build PCA candidates: explicit n_components list + default PCA-95%
    pca_candidates: list[tuple[str, np.ndarray]] = []
    for n_pc in PCA_N_COMPONENTS_LIST:
        if n_pc >= X_scaled.shape[1]:
            continue
        pca = PCA(n_components=n_pc, random_state=_cfg.RANDOM_STATE)
        X_proj = pca.fit_transform(X_scaled)
        pca_candidates.append((str(n_pc), X_proj))
    # Also include the default PCA-95% result
    n_default = X_pca_default.shape[1]
    pca_candidates.append((f"{n_default}(auto)", X_pca_default))

    # Optionally add UMAP embeddings as candidates (multiple dimensionalities)
    if _cfg.USE_UMAP_FOR_DENSITY_CLUSTERING:
        for umap_dim in UMAP_CLUSTERING_N_COMPONENTS_LIST:
            X_umap = compute_umap_embedding(X_scaled, umap_dim)
            pca_candidates.append((f"UMAP-{umap_dim}D", X_umap))

    logger.info(
        "HDBSCAN grid search: {} space variants × {} configs (n_samples={}).",
        len(pca_candidates),
        len(HDBSCAN_MIN_CLUSTER_SIZES) * len(HDBSCAN_MIN_SAMPLES_LIST)
        * len(HDBSCAN_CLUSTER_METHODS) * len(HDBSCAN_EPSILON_VALUES),
        n_samples,
    )

    for pca_label, X in pca_candidates:
        for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
            for ms in HDBSCAN_MIN_SAMPLES_LIST:
                for method in HDBSCAN_CLUSTER_METHODS:
                    for eps in HDBSCAN_EPSILON_VALUES:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            cluster_selection_method=method,
                            cluster_selection_epsilon=eps,
                            gen_min_span_tree=True,
                        )
                        labels = clusterer.fit_predict(X)

                        n_clusters = _count_clusters(labels)
                        n_noise = int((labels == -1).sum())
                        cov = _coverage(labels)
                        dbcv = float(clusterer.relative_validity_)

                        # Scoring: base × soft cluster-count penalty
                        base_score = dbcv * cov
                        penalty = _cluster_count_penalty(n_clusters)
                        final_score = base_score * penalty

                        row = {
                            "pca_components": pca_label,
                            "min_cluster_size": mcs,
                            "min_samples": ms,
                            "method": method,
                            "epsilon": eps,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "coverage": round(cov, 4),
                            "dbcv": round(dbcv, 4),
                            "base_score": round(base_score, 4),
                            "count_penalty": round(penalty, 4),
                            "final_score": round(final_score, 4),
                        }
                        results.append(row)

                        # Accept only if constraints are met
                        meets_constraints = (
                            cov >= HDBSCAN_MIN_COVERAGE
                            and 2 <= n_clusters <= HDBSCAN_MAX_CLUSTERS
                        )

                        if meets_constraints and final_score > best_score:
                            best_score = final_score
                            best_params = {
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "method": method,
                                "epsilon": eps,
                                "pca_components": pca_label,
                            }
                            best_X = X

    df_grid = pd.DataFrame(results)
    top = df_grid.sort_values("final_score", ascending=False).head(10)
    logger.info("Top HDBSCAN grid results:\n{}", top.to_string(index=False))

    if best_params is None:
        logger.warning(
            "HDBSCAN grid search: NO configuration met constraints "
            "(min_coverage={:.0%}, 2 ≤ k ≤ {}). "
            "Best candidate coverage: {:.1%}. Method will be marked FAILED.",
            HDBSCAN_MIN_COVERAGE,
            HDBSCAN_MAX_CLUSTERS,
            df_grid["coverage"].max() if not df_grid.empty else 0,
        )
    else:
        logger.info(
            "Best HDBSCAN params: {} (score={:.4f})", best_params, best_score,
        )
        logger.info(
            ">>> HDBSCAN grid winner used PCA space: {} dimensions",
            best_params.get("pca_components", "unknown"),
        )

    return best_params, df_grid, best_X


def fit_hdbscan(
    X_scaled: np.ndarray,
    X_pca_default: np.ndarray,
    params: Optional[dict] = None,
) -> tuple[ClusteringResult, pd.DataFrame]:
    """Fit HDBSCAN with given or grid-searched parameters.

    If grid search finds no valid configuration, returns a result with
    status=FAILED_NO_VALID_CONFIG and all-noise labels.

    Returns:
        Tuple of (ClusteringResult, grid search DataFrame).
    """
    if params is None:
        params, df_grid, X_best = grid_search_hdbscan(X_scaled, X_pca_default)
    else:
        df_grid = pd.DataFrame()
        X_best = X_pca_default

    # --- No valid config found: report failure ---
    if params is None:
        n_samples = X_pca_default.shape[0]
        logger.warning(
            "HDBSCAN: FAILED — no valid density configuration found. "
            "Reporting all {} samples as noise.", n_samples,
        )
        return ClusteringResult(
            method="HDBSCAN",
            labels=np.full(n_samples, -1, dtype=int),
            n_clusters=0,
            n_noise=n_samples,
            params={"reason": "no grid config met coverage/cluster constraints"},
            status=STATUS_FAILED,
            model=None,
            score=None,
            clustering_space=None,
        ), df_grid

    # --- Valid config: fit the model ---
    model = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        cluster_selection_method=params.get("method", "eom"),
        cluster_selection_epsilon=params.get("epsilon", 0.0),  # default 0.0 = safe no-op
        gen_min_span_tree=True,
    )
    labels = model.fit_predict(X_best)

    result = ClusteringResult(
        method="HDBSCAN",
        labels=labels,
        n_clusters=_count_clusters(labels),
        n_noise=int((labels == -1).sum()),
        params=params,
        status=STATUS_OK,
        model=model,
        score=float(model.relative_validity_),
        clustering_space=X_best,
    )

    cov = _coverage(labels)
    logger.info(
        "HDBSCAN: {} clusters, {} noise ({:.1f}%), coverage={:.1%}, DBCV={:.4f}",
        result.n_clusters,
        result.n_noise,
        result.n_noise / len(labels) * 100,
        cov,
        result.score,
    )

    if cov < 0.5:
        logger.warning(
            "HDBSCAN coverage is low ({:.1%}). Density clusters only cover "
            "part of the dataset — centroid methods may be more appropriate "
            "for full-coverage Tumor/Cyst separation.", cov,
        )

    return result, df_grid


# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------

def grid_search_optics(
    X_scaled: np.ndarray,
    X_pca_default: np.ndarray,
) -> tuple[Optional[dict], pd.DataFrame, np.ndarray]:
    """
    Independent grid search for OPTICS over (min_samples, min_cluster_size)
    with both xi and DBSCAN-style cluster extraction methods.

    Scores by ``silhouette * coverage**2 * penalty``, rejecting configs with
    coverage < OPTICS_MIN_COVERAGE or cluster count outside [2, max_clusters].
    Coverage is squared to punish low-coverage solutions harder (4c).

    Returns:
        Tuple of (best params dict or None, full grid DataFrame, best PCA-projected X).
        Returns None as first element when no valid configuration meets constraints.
    """
    n_samples = X_scaled.shape[0]
    best_score = -1.0
    best_params: Optional[dict] = None
    best_X: np.ndarray = X_pca_default
    results: list[dict] = []

    # Build PCA candidates
    pca_candidates: list[tuple[str, np.ndarray]] = []
    for n_pc in PCA_N_COMPONENTS_LIST:
        if n_pc >= X_scaled.shape[1]:
            continue
        pca = PCA(n_components=n_pc, random_state=_cfg.RANDOM_STATE)
        X_proj = pca.fit_transform(X_scaled)
        pca_candidates.append((str(n_pc), X_proj))
    n_default = X_pca_default.shape[1]
    pca_candidates.append((f"{n_default}(auto)", X_pca_default))

    # Optionally add UMAP embeddings as candidates (multiple dimensionalities)
    if _cfg.USE_UMAP_FOR_DENSITY_CLUSTERING:
        for umap_dim in UMAP_CLUSTERING_N_COMPONENTS_LIST:
            X_umap = compute_umap_embedding(X_scaled, umap_dim)
            pca_candidates.append((f"UMAP-{umap_dim}D", X_umap))

    # Count total configs for logging
    n_xi_configs = (
        len(pca_candidates)
        * len(OPTICS_MIN_SAMPLES_LIST)
        * len(OPTICS_MIN_CLUSTER_SIZES)
        * len(OPTICS_XI_VALUES)
    )
    n_dbscan_configs = (
        len(pca_candidates)
        * len(OPTICS_MIN_SAMPLES_LIST)
        * len(OPTICS_MIN_CLUSTER_SIZES)
        * len(OPTICS_EPS_VALUES)
    )
    total_configs = 0
    if "xi" in OPTICS_CLUSTER_METHODS:
        total_configs += n_xi_configs
    if "dbscan" in OPTICS_CLUSTER_METHODS:
        total_configs += n_dbscan_configs
    logger.info(
        "OPTICS grid search: {} total configs across {} methods (n_samples={}).",
        total_configs,
        list(OPTICS_CLUSTER_METHODS),
        n_samples,
    )

    for pca_label, X in pca_candidates:
        for ms in OPTICS_MIN_SAMPLES_LIST:
            for mcs in OPTICS_MIN_CLUSTER_SIZES:
                for cluster_method in OPTICS_CLUSTER_METHODS:
                    if cluster_method == "xi":
                        # Xi extraction: sweep xi values
                        sweep_values = OPTICS_XI_VALUES
                    else:
                        # DBSCAN extraction: sweep eps values
                        sweep_values = OPTICS_EPS_VALUES

                    for sweep_val in sweep_values:
                        if cluster_method == "xi":
                            opt = OPTICS(
                                min_samples=ms,
                                xi=sweep_val,
                                min_cluster_size=mcs,
                                cluster_method="xi",
                            )
                        else:  # dbscan
                            opt = OPTICS(
                                min_samples=ms,
                                min_cluster_size=mcs,
                                cluster_method="dbscan",
                                eps=sweep_val,
                            )

                        labels = opt.fit_predict(X)
                        n_clusters = _count_clusters(labels)
                        n_noise = int((labels == -1).sum())
                        cov = _coverage(labels)

                        non_noise = labels != -1
                        if n_clusters >= 2 and non_noise.sum() > n_clusters:
                            sil = float(silhouette_score(X[non_noise], labels[non_noise]))
                        else:
                            sil = -1.0

                        # Scoring: sil * cov**2 * penalty (4c: coverage squared)
                        base_score = sil * (cov ** 2) if sil > 0 else -1.0
                        penalty = _cluster_count_penalty(n_clusters) if base_score > 0 else 1.0
                        final_score = base_score * penalty if base_score > 0 else -1.0

                        row = {
                            "pca_components": pca_label,
                            "cluster_method": cluster_method,
                            "min_samples": ms,
                            "min_cluster_size": mcs,
                            "xi": sweep_val if cluster_method == "xi" else None,
                            "eps": sweep_val if cluster_method == "dbscan" else None,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "coverage": round(cov, 4),
                            "silhouette": round(sil, 4),
                            "base_score": round(base_score, 4),
                            "count_penalty": round(penalty, 4),
                            "final_score": round(final_score, 4),
                        }
                        results.append(row)

                        meets_constraints = (
                            cov >= OPTICS_MIN_COVERAGE
                            and 2 <= n_clusters <= OPTICS_MAX_CLUSTERS
                        )

                        if meets_constraints and final_score > best_score:
                            best_score = final_score
                            best_params = {
                                "cluster_method": cluster_method,
                                "min_samples": ms,
                                "min_cluster_size": mcs,
                                "xi": sweep_val if cluster_method == "xi" else None,
                                "eps": sweep_val if cluster_method == "dbscan" else None,
                                "pca_components": pca_label,
                            }
                            best_X = X

    df_grid = pd.DataFrame(results)
    top = df_grid.sort_values("final_score", ascending=False).head(10)
    logger.info("Top OPTICS grid results:\n{}", top.to_string(index=False))

    if best_params is None:
        logger.warning(
            "OPTICS grid search: NO configuration met constraints "
            "(min_coverage={:.0%}, 2 ≤ k ≤ {}). "
            "Best candidate coverage: {:.1%}. Method will be marked FAILED.",
            OPTICS_MIN_COVERAGE,
            OPTICS_MAX_CLUSTERS,
            df_grid["coverage"].max() if not df_grid.empty else 0,
        )
    else:
        logger.info(
            "Best OPTICS params: {} (score={:.4f})", best_params, best_score,
        )
        logger.info(
            ">>> OPTICS grid winner: method={}, space={}",
            best_params.get("cluster_method", "unknown"),
            best_params.get("pca_components", "unknown"),
        )

    return best_params, df_grid, best_X


def fit_optics(
    X_scaled: np.ndarray,
    X_pca_default: np.ndarray,
) -> tuple[ClusteringResult, pd.DataFrame]:
    """Fit OPTICS with independently grid-searched parameters.

    If grid search finds no valid configuration, returns a result with
    status=FAILED_NO_VALID_CONFIG and all-noise labels.

    Returns:
        Tuple of (ClusteringResult, grid search DataFrame).
    """
    params, df_grid, X_best = grid_search_optics(X_scaled, X_pca_default)

    # --- No valid config found: report failure ---
    if params is None:
        n_samples = X_pca_default.shape[0]
        logger.warning(
            "OPTICS: FAILED — no valid density configuration found. "
            "Reporting all {} samples as noise.", n_samples,
        )
        return ClusteringResult(
            method="OPTICS",
            labels=np.full(n_samples, -1, dtype=int),
            n_clusters=0,
            n_noise=n_samples,
            params={"reason": "no grid config met coverage/cluster constraints"},
            status=STATUS_FAILED,
            model=None,
            score=None,
            clustering_space=None,
        ), df_grid

    # --- Valid config: fit the model ---
    cluster_method = params.get("cluster_method", "xi")
    if cluster_method == "xi":
        model = OPTICS(
            min_samples=params["min_samples"],
            xi=params["xi"],
            min_cluster_size=params["min_cluster_size"],
            cluster_method="xi",
        )
    else:  # dbscan
        model = OPTICS(
            min_samples=params["min_samples"],
            min_cluster_size=params["min_cluster_size"],
            cluster_method="dbscan",
            eps=params["eps"],
        )
    labels = model.fit_predict(X_best)

    result = ClusteringResult(
        method="OPTICS",
        labels=labels,
        n_clusters=_count_clusters(labels),
        n_noise=int((labels == -1).sum()),
        params=params,
        status=STATUS_OK,
        model=model,
        score=None,
        clustering_space=X_best,
    )

    cov = _coverage(labels)
    non_noise = labels != -1
    if result.n_clusters >= 2 and non_noise.sum() > result.n_clusters:
        result.score = float(silhouette_score(X_best[non_noise], labels[non_noise]))

    logger.info(
        "OPTICS ({}): {} clusters, {} noise ({:.1f}%), coverage={:.1%}",
        cluster_method,
        result.n_clusters,
        result.n_noise,
        result.n_noise / len(labels) * 100,
        cov,
    )

    if cov < 0.5:
        logger.warning(
            "OPTICS coverage is low ({:.1%}). Density clusters only cover "
            "part of the dataset.", cov,
        )

    return result, df_grid


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def fit_kmeans(X: np.ndarray, n_clusters: int = BASELINE_N_CLUSTERS) -> ClusteringResult:
    """Fit K-Means baseline."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=_cfg.RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        method="K-Means(k=2)",
        labels=labels,
        n_clusters=n_clusters,
        n_noise=0,
        params={"n_clusters": n_clusters},
        status=STATUS_OK,
        model=model,
    )


def fit_gmm(X: np.ndarray, n_components: int = BASELINE_N_CLUSTERS) -> ClusteringResult:
    """Fit Gaussian Mixture Model baseline."""
    model = GaussianMixture(
        n_components=n_components,
        random_state=_cfg.RANDOM_STATE,
        covariance_type=GMM_COVARIANCE_TYPE,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        method="GMM(k=2)",
        labels=labels,
        n_clusters=n_components,
        n_noise=0,
        params={"n_components": n_components, "covariance_type": GMM_COVARIANCE_TYPE},
        status=STATUS_OK,
        model=model,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_all_clustering(
    X_pca: np.ndarray,
    X_scaled: np.ndarray,
) -> tuple[dict[str, ClusteringResult], dict[str, pd.DataFrame]]:
    """
    Run HDBSCAN, OPTICS, K-Means, and GMM.

    HDBSCAN and OPTICS do multi-PCA grid search using X_scaled.
    KMeans and GMM use the default PCA-95% space (X_pca).

    Density methods that fail to find valid configurations are returned
    with status=FAILED_NO_VALID_CONFIG. Downstream code should check
    result.status before using labels for analysis.

    Returns:
        Tuple of (results dict, grid DataFrames dict).
    """
    logger.info(
        "Running clustering pipeline: X_pca={}, X_scaled={}.",
        X_pca.shape,
        X_scaled.shape,
    )

    hdbscan_result, hdbscan_grid = fit_hdbscan(X_scaled, X_pca)
    optics_result, optics_grid = fit_optics(X_scaled, X_pca)
    kmeans_result = fit_kmeans(X_pca)
    gmm_result = fit_gmm(X_pca)

    # Summary
    for name, r in [("HDBSCAN", hdbscan_result), ("OPTICS", optics_result),
                     ("K-Means", kmeans_result), ("GMM", gmm_result)]:
        if r.status == STATUS_FAILED:
            logger.warning("{}: {} — no valid density structure found.", name, r.status)
        else:
            logger.info("{}: {} — {} clusters.", name, r.status, r.n_clusters)

    results = {
        "HDBSCAN": hdbscan_result,
        "OPTICS": optics_result,
        "K-Means(k=2)": kmeans_result,
        "GMM(k=2)": gmm_result,
    }
    grids = {
        "hdbscan_grid": hdbscan_grid,
        "optics_grid": optics_grid,
    }

    return results, grids
