"""
KiTS23 Radiomics Clustering Pipeline — entry point.

Run the full pipeline:
    python main.py

Run individual stages:
    python main.py --stage ingest
    python main.py --stage features
    python main.py --stage cluster
    python main.py --stage evaluate

Filter tiny lesions:
    python main.py --min-voxels 500

Enable supervised feature selection (ANOVA F-test, top-K):
    python main.py --feature-selection
    python main.py --feature-selection 30

Keep PCA space comparable across different voxel filters (fit once on the
full data, then subset rows instead of refitting PCA on the filtered data):
    python main.py --min-voxels 500 --consistent-pca

Save to a custom output directory:
    python main.py --stage evaluate --min-voxels 500 --output-dir outputs_v500

Run voxel-filter sweep (thresholds configurable via --voxel-thresholds):
    python main.py --stage voxel-sweep
    python main.py --stage voxel-sweep --voxel-thresholds 0,100,300,800 --consistent-pca

Set a random seed (numpy/python stdlib; see --seed help for caveats):
    python main.py --seed 123
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.clustering import run_all_clustering
from src.config import (
    CLEAN_FEATURES_CSV_PATH,
    FEATURES_CSV_PATH,
    FEATURE_SELECTION_TOP_K,
    KITS23_DATASET_DIR,
    LESION_MIN_VOXELS_FILTER,
    LESIONS_CSV_PATH,
    LOG_FORMAT,
    META_COLUMNS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
)
from src.data_ingestion import ingest_dataset, prepare_lesions
from src.evaluation import run_evaluation
from src.feature_extraction import (
    apply_pca,
    run_feature_pipeline,
    scale_features,
    select_discriminative_features,
)
from src.supervised_check import run_supervised_sanity_check


# ---------------------------------------------------------------------------
# Stale output handling (quarantine instead of hard delete)
# ---------------------------------------------------------------------------
STALE_EXTENSIONS = {".png", ".csv"}
# Files that should never be auto-removed (keep data caches and summary files)
PROTECTED_FILES = {
    "features.csv",
    "features_clean.csv",
    "lesions.csv",
    "clustering_metrics.csv",
    "voxel_sweep_summary.csv",
}


def _quarantine_stale_outputs(output_dir: Path) -> Path | None:
    """Move stale plot/csv files from a prior run into a temporary folder.

    This replaces a hard delete: if the upcoming evaluation run fails partway
    through, the previous outputs can be restored instead of being lost.
    Returns the quarantine directory, or None if there was nothing to move.
    """
    if not output_dir.exists():
        return None

    to_move = [
        f for f in output_dir.iterdir()
        if f.is_file() and f.suffix in STALE_EXTENSIONS and f.name not in PROTECTED_FILES
    ]
    if not to_move:
        return None

    quarantine_dir = Path(tempfile.mkdtemp(prefix="stale_outputs_"))
    for f in to_move:
        shutil.move(str(f), str(quarantine_dir / f.name))

    logger.info(
        "Quarantined {} stale output file(s) from {} (deleted only after a successful run).",
        len(to_move),
        output_dir,
    )
    return quarantine_dir


def _commit_quarantine(quarantine_dir: Path | None) -> None:
    """Permanently discard quarantined files after a successful run."""
    if quarantine_dir is not None and quarantine_dir.exists():
        shutil.rmtree(quarantine_dir, ignore_errors=True)


def _restore_quarantine(quarantine_dir: Path | None, output_dir: Path) -> None:
    """Move quarantined files back after a failed run, so old results survive."""
    if quarantine_dir is None or not quarantine_dir.exists():
        return
    restored = 0
    for f in quarantine_dir.iterdir():
        shutil.move(str(f), str(output_dir / f.name))
        restored += 1
    shutil.rmtree(quarantine_dir, ignore_errors=True)
    logger.warning(
        "Evaluation failed — restored {} previous output file(s) in {}.",
        restored,
        output_dir,
    )


def configure_logging(log_level: str = "INFO") -> None:
    """Configure loguru logger for console output."""
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=log_level)


def _parse_int_list(s: str) -> list[int]:
    """Parse a comma-separated string of ints, e.g. '0,250,500,1000'."""
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {s!r}") from e


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KiTS23 radiomics clustering pipeline (HDBSCAN / OPTICS / GMM)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "features", "cluster", "evaluate", "supervised-check", "voxel-sweep"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--no-resume-extraction",
        action="store_true",
        help="Re-run radiomics extraction even if features.csv exists",
    )
    parser.add_argument(
        "--use-tsne",
        action="store_true",
        help="Use t-SNE instead of UMAP for 2D visualization",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap stability analysis",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=LESION_MIN_VOXELS_FILTER,
        help=(
            "Minimum n_voxels to keep a lesion. Filters out tiny masks that "
            "produce noisy radiomics features. 0 = no filtering. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--consistent-pca",
        action="store_true",
        help=(
            "Fit the scaler/PCA once on the unfiltered data and subset rows when a "
            "voxel filter is applied, instead of refitting PCA on the filtered subset. "
            "Recommended when comparing clustering results across different "
            "--min-voxels thresholds, since it keeps everyone in the same PCA space. "
            "Default (off) preserves the original behavior of refitting per filter."
        ),
    )
    parser.add_argument(
        "--voxel-thresholds",
        type=_parse_int_list,
        default=[0, 100, 250, 500, 1000],
        help="Comma-separated min-voxel thresholds for --stage voxel-sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for numpy and the python stdlib random module (default: %(default)s). "
            "Note: this does not guarantee full reproducibility on its own — UMAP/t-SNE and "
            "other steps inside src/ must also accept and use a random_state for that."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Custom output directory (relative to project root or absolute). "
            "Default: outputs/. Also used as the base directory for --stage "
            "voxel-sweep (each threshold gets its own subfolder under it)."
        ),
    )
    parser.add_argument(
        "--umap-cluster",
        action="store_true",
        help=(
            "Include a UMAP embedding as a candidate space for HDBSCAN/OPTICS "
            "grid search (in addition to PCA). UMAP sharpens local density gaps "
            "that PCA smears out. The UMAP dimensionality is configurable via "
            "UMAP_CLUSTERING_N_COMPONENTS in config.py (default: 10D)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--feature-selection",
        nargs="?",
        type=int,
        const=FEATURE_SELECTION_TOP_K,
        default=None,
        metavar="TOP_K",
        help=(
            "Enable supervised feature selection (ANOVA F-test) before "
            "scaling/PCA. Keeps the top-K most discriminative features by "
            "F-statistic against the Tumor/Cyst label. This preserves "
            "low-variance but highly discriminative HU/firstorder features "
            "that PCA would otherwise bury. Pass without a value to use the "
            f"default K={FEATURE_SELECTION_TOP_K}, or specify an integer. "
            "Present this in your thesis as a semi-supervised feature "
            'discovery step ("phân cụm có hướng dẫn đặc trưng").'
        ),
    )

    args = parser.parse_args()

    if args.min_voxels < 0:
        parser.error(f"--min-voxels must be >= 0 (got {args.min_voxels})")
    if any(v < 0 for v in args.voxel_thresholds):
        parser.error(f"--voxel-thresholds must all be >= 0 (got {args.voxel_thresholds})")

    return args


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    """Resolve the output directory from CLI argument."""
    if output_dir_arg is None:
        return OUTPUTS_DIR

    p = Path(output_dir_arg)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _apply_voxel_filter(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    min_voxels: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Filter lesions by voxel count using the cached lesions.csv.

    Merges df_clean with lesions.csv to get n_voxels, then drops rows
    below the threshold. Returns the filtered df_clean, unchanged
    feature_cols, and a boolean keep-mask aligned with the ORIGINAL
    (pre-filter) row order of df_clean — useful for subsetting any other
    array (e.g. X_scaled, X_pca) that shares that row order.
    """
    n = len(df_clean)

    if min_voxels <= 0:
        return df_clean, feature_cols, np.ones(n, dtype=bool)

    if not LESIONS_CSV_PATH.exists():
        logger.warning(
            "Cannot apply voxel filter: lesions.csv not found at {}.",
            LESIONS_CSV_PATH,
        )
        return df_clean, feature_cols, np.ones(n, dtype=bool)

    df_lesions = pd.read_csv(LESIONS_CSV_PATH)

    # Merge on PatientID + Instance to get n_voxels. reset_index first so the
    # row order (and therefore the resulting keep-mask) is guaranteed to line
    # up positionally with df_clean as passed in.
    merge_cols = ["case_id", "instance_name", "n_voxels"]
    df_merge = df_lesions[merge_cols].rename(
        columns={"case_id": "PatientID", "instance_name": "Instance"}
    )

    df_clean_reset = df_clean.reset_index(drop=True)
    merged = df_clean_reset.merge(df_merge, on=["PatientID", "Instance"], how="left")

    unmatched_mask = merged["n_voxels"].isna()
    n_unmatched = int(unmatched_mask.sum())
    if n_unmatched:
        logger.warning(
            "{} lesion(s) had no matching n_voxels entry in lesions.csv "
            "(PatientID/Instance mismatch) and will be dropped by the voxel "
            "filter as if n_voxels=0 — this may not reflect their true size.",
            n_unmatched,
        )

    keep_mask = (merged["n_voxels"].fillna(0) >= min_voxels).to_numpy()
    n_before = n
    n_after = int(keep_mask.sum())

    df_filtered = (
        merged.loc[keep_mask]
        .drop(columns=["n_voxels"], errors="ignore")
        .reset_index(drop=True)
    )

    logger.info(
        "Voxel filter (>= {}): {} → {} lesions ({} removed, {} of which had no n_voxels match).",
        min_voxels,
        n_before,
        n_after,
        n_before - n_after,
        n_unmatched,
    )

    return df_filtered, feature_cols, keep_mask


def _filter_and_project(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    min_voxels: int,
    *,
    consistent_pca: bool = False,
    existing_scaler=None,
    existing_pca_model=None,
    existing_X_scaled: np.ndarray | None = None,
    existing_X_pca: np.ndarray | None = None,
    feature_selection_top_k: int | None = None,
) -> dict:
    """
    Apply the voxel filter (if any) and produce a dict with
    {df_clean, feature_cols, X_scaled, scaler, X_pca, pca_model, n_pc}.

    Shared by stage_features and stage_cluster so the filter -> scale -> PCA
    logic only lives in one place.

    If consistent_pca=True, the scaler/PCA are fit ONCE on the full
    (unfiltered) data — reusing existing_* if already fit, otherwise fitting
    them here — and rows are simply subset by the voxel-filter mask
    afterwards. This keeps the PCA space identical across different
    --min-voxels runs, which matters if you want to compare cluster
    structure across thresholds.

    If consistent_pca=False (default), the scaler/PCA are refit on the
    filtered subset only, matching the original behavior.
    """
    have_existing_fit = existing_X_scaled is not None and existing_X_pca is not None

    if consistent_pca and not have_existing_fit:
        # Apply feature selection if requested before fitting scaler/PCA
        cols_for_fit = feature_cols
        if feature_selection_top_k is not None and "Lesion_Type" in df_clean.columns:
            y = df_clean["Lesion_Type"].values
            cols_for_fit = select_discriminative_features(
                df_clean, feature_cols, y, top_k=feature_selection_top_k,
            )
            feature_cols = cols_for_fit
        existing_X_scaled, existing_scaler = scale_features(df_clean, cols_for_fit)
        existing_X_pca, existing_pca_model, _ = apply_pca(existing_X_scaled, feature_names=cols_for_fit)
        have_existing_fit = True

    if min_voxels <= 0:
        if have_existing_fit:
            return {
                "df_clean": df_clean,
                "feature_cols": feature_cols,
                "X_scaled": existing_X_scaled,
                "scaler": existing_scaler,
                "X_pca": existing_X_pca,
                "pca_model": existing_pca_model,
                "n_pc": existing_X_pca.shape[1],
            }
        # Apply feature selection if requested before fitting scaler/PCA
        cols_for_fit = feature_cols
        if feature_selection_top_k is not None and "Lesion_Type" in df_clean.columns:
            y = df_clean["Lesion_Type"].values
            cols_for_fit = select_discriminative_features(
                df_clean, feature_cols, y, top_k=feature_selection_top_k,
            )
            feature_cols = cols_for_fit
        X_scaled, scaler = scale_features(df_clean, cols_for_fit)
        X_pca, pca_model, n_pc = apply_pca(X_scaled, feature_names=cols_for_fit)
        return {
            "df_clean": df_clean,
            "feature_cols": feature_cols,
            "X_scaled": X_scaled,
            "scaler": scaler,
            "X_pca": X_pca,
            "pca_model": pca_model,
            "n_pc": n_pc,
        }

    df_filtered, feature_cols, keep_mask = _apply_voxel_filter(
        df_clean, feature_cols, min_voxels,
    )

    if consistent_pca and have_existing_fit:
        X_scaled = existing_X_scaled[keep_mask]
        X_pca = existing_X_pca[keep_mask]
        return {
            "df_clean": df_filtered,
            "feature_cols": feature_cols,
            "X_scaled": X_scaled,
            "scaler": existing_scaler,
            "X_pca": X_pca,
            "pca_model": existing_pca_model,
            "n_pc": X_pca.shape[1],
        }

    # Legacy behavior: refit scaler + PCA on the filtered subset only.
    # Apply feature selection if requested
    cols_for_fit = feature_cols
    if feature_selection_top_k is not None and "Lesion_Type" in df_filtered.columns:
        y = df_filtered["Lesion_Type"].values
        cols_for_fit = select_discriminative_features(
            df_filtered, feature_cols, y, top_k=feature_selection_top_k,
        )
        feature_cols = cols_for_fit
    X_scaled, scaler = scale_features(df_filtered, cols_for_fit)
    X_pca, pca_model, n_pc = apply_pca(X_scaled, feature_names=cols_for_fit)
    return {
        "df_clean": df_filtered,
        "feature_cols": feature_cols,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "X_pca": X_pca,
        "pca_model": pca_model,
        "n_pc": n_pc,
    }


def stage_ingest() -> None:
    """Stage 1: Verify and prepare KiTS23 dataset."""
    logger.info("=== Stage 1: Data Ingestion ===")
    dataset_dir = ingest_dataset()
    prepare_lesions(dataset_dir=dataset_dir, use_cache=False)
    logger.success("Data ingestion complete.")


def stage_features(
    resume_extraction: bool = True,
    min_voxels: int = 0,
    consistent_pca: bool = False,
    feature_selection_top_k: int | None = None,
) -> dict:
    """Stage 2: Extract radiomics features, clean, scale, and apply PCA."""
    logger.info("=== Stage 2: Feature Extraction ===")

    if not LESIONS_CSV_PATH.exists():
        logger.info("Lesions cache not found — running ingestion scan first.")
        prepare_lesions(dataset_dir=KITS23_DATASET_DIR)

    if resume_extraction and FEATURES_CSV_PATH.exists():
        mtime = FEATURES_CSV_PATH.stat().st_mtime
        logger.info(
            "Resuming from cached {} (last modified {}). If the extraction code or "
            "input data has changed since then, use --no-resume-extraction to force "
            "a fresh run.",
            FEATURES_CSV_PATH,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        )

    df_lesions = prepare_lesions(use_cache=True)
    artifacts = run_feature_pipeline(
        df_lesions,
        features_path=FEATURES_CSV_PATH,
        clean_path=CLEAN_FEATURES_CSV_PATH,
        resume_extraction=resume_extraction,
        feature_selection_top_k=feature_selection_top_k,
    )

    # Apply voxel filter if requested
    if min_voxels > 0:
        filtered = _filter_and_project(
            artifacts["df_clean"],
            artifacts["feature_cols"],
            min_voxels,
            consistent_pca=consistent_pca,
            existing_scaler=artifacts.get("scaler"),
            existing_pca_model=artifacts.get("pca_model"),
            existing_X_scaled=artifacts.get("X_scaled"),
            existing_X_pca=artifacts.get("X_pca"),
            feature_selection_top_k=feature_selection_top_k,
        )
        artifacts.update(filtered)

    logger.success("Feature extraction complete.")
    return artifacts


def stage_cluster(
    artifacts: dict | None = None,
    min_voxels: int = 0,
    consistent_pca: bool = False,
    feature_selection_top_k: int | None = None,
) -> tuple[dict, dict]:
    """Stage 3: Run clustering algorithms on PCA-reduced data."""
    logger.info("=== Stage 3: Clustering ===")

    if artifacts is None:
        if not CLEAN_FEATURES_CSV_PATH.exists():
            raise FileNotFoundError(
                f"Cleaned features not found at {CLEAN_FEATURES_CSV_PATH}. "
                "Run --stage features first."
            )

        from src.config import META_COLUMNS

        df_clean = pd.read_csv(CLEAN_FEATURES_CSV_PATH)
        feature_cols = [c for c in df_clean.columns if c not in META_COLUMNS]

        artifacts = _filter_and_project(
            df_clean, feature_cols, min_voxels, consistent_pca=consistent_pca,
            feature_selection_top_k=feature_selection_top_k,
        )

    # run_all_clustering now needs X_scaled for multi-PCA grid search
    results, grids = run_all_clustering(artifacts["X_pca"], artifacts["X_scaled"])

    # Extract hdbscan_params (may be a failure sentinel)
    hdbscan_result = results["HDBSCAN"]
    hdbscan_params = hdbscan_result.params

    logger.success("Clustering complete.")
    return artifacts, {
        "results": results,
        "hdbscan_params": hdbscan_params,
        "grids": grids,
    }


def stage_evaluate(
    artifacts: dict,
    cluster_output: dict,
    output_dir: Path,
    use_tsne: bool = False,
    run_bootstrap: bool = True,
) -> None:
    """Stage 4: Evaluate clustering and generate visualizations.

    Stale plot/csv files from a prior run are quarantined (not deleted)
    before evaluation starts. If evaluation succeeds, the quarantine is
    discarded; if it fails, the previous files are restored so a crash
    never leaves the output directory in a worse state than before.
    """
    logger.info("=== Stage 4: Evaluation ===")

    # Clean up stale metrics CSV in data/processed/ (old schema location)
    stale_metrics = PROCESSED_DATA_DIR / "clustering_metrics.csv"
    if stale_metrics.exists():
        stale_metrics.unlink()
        logger.info("Deleted stale metrics CSV at {} (old schema; canonical location is now outputs/).", stale_metrics)

    quarantine_dir = _quarantine_stale_outputs(output_dir)

    try:
        df_metrics = run_evaluation(
            df_clean=artifacts["df_clean"],
            feature_cols=artifacts["feature_cols"],
            X_pca=artifacts["X_pca"],
            results=cluster_output["results"],
            hdbscan_params=cluster_output["hdbscan_params"],
            grid_dfs=cluster_output.get("grids"),
            output_dir=output_dir,
            use_tsne=use_tsne,
            run_bootstrap=run_bootstrap,
        )

        # Also save to the standard metrics path if output_dir != OUTPUTS_DIR
        metrics_path = output_dir / "clustering_metrics.csv"
        df_metrics.to_csv(metrics_path, index=False)
        logger.info("Metrics saved to {}", metrics_path)
    except Exception:
        _restore_quarantine(quarantine_dir, output_dir)
        raise
    else:
        _commit_quarantine(quarantine_dir)
        logger.success("Evaluation complete. Plots saved to {}", output_dir)


def stage_voxel_sweep(
    args: argparse.Namespace,
) -> None:
    """Run clustering + evaluation across multiple voxel thresholds.

    Each threshold gets its own output directory so results don't overwrite.
    A summary CSV comparing all thresholds is saved alongside them.
    """
    voxel_thresholds = args.voxel_thresholds
    base_dir = _resolve_output_dir(args.output_dir) if args.output_dir else PROJECT_ROOT
    summary_rows: list[dict] = []

    logger.info(
        "=== Voxel-Filter Sweep: thresholds = {} (base dir: {}) ===",
        voxel_thresholds,
        base_dir,
    )

    for min_v in voxel_thresholds:
        tag = f"v{min_v}" if min_v > 0 else "v0_no_filter"
        out_dir = base_dir / f"outputs_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Sweep: min_voxels={}, output={} ---", min_v, out_dir)

        try:
            # Feature stage (with voxel filter)
            artifacts = stage_features(
                resume_extraction=True,
                min_voxels=min_v,
                consistent_pca=args.consistent_pca,
                feature_selection_top_k=args.feature_selection,
            )

            n_samples = len(artifacts["df_clean"])
            n_pca = artifacts["X_pca"].shape[1]

            # Cluster stage
            artifacts, cluster_output = stage_cluster(
                artifacts, min_voxels=min_v, consistent_pca=args.consistent_pca,
                feature_selection_top_k=args.feature_selection,
            )

            # Evaluate
            stage_evaluate(
                artifacts,
                cluster_output,
                output_dir=out_dir,
                use_tsne=args.use_tsne,
                run_bootstrap=not args.no_bootstrap,
            )

            # Collect summary (consistent schema across success/error rows)
            for name, result in cluster_output["results"].items():
                summary_rows.append({
                    "min_voxels": min_v,
                    "n_samples": n_samples,
                    "n_pca_components": n_pca,
                    "method": name,
                    "status": result.status,
                    "n_clusters": result.n_clusters,
                    "n_noise": result.n_noise,
                    "output_dir": str(out_dir),
                    "error": None,
                })

        except Exception as e:
            logger.opt(exception=True).error(
                "Sweep failed for min_voxels={}: {}", min_v, e,
            )
            summary_rows.append({
                "min_voxels": min_v,
                "n_samples": None,
                "n_pca_components": None,
                "method": "ALL",
                "status": "ERROR",
                "n_clusters": None,
                "n_noise": None,
                "output_dir": str(out_dir),
                "error": f"{e}\n{traceback.format_exc()}",
            })

    # Save sweep summary
    df_summary = pd.DataFrame(summary_rows)
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / "voxel_sweep_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(
        "Voxel sweep summary:\n{}",
        df_summary.drop(columns=["error"], errors="ignore").to_string(index=False),
    )
    logger.success("Voxel sweep complete. Summary saved to {}", summary_path)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the requested pipeline stage(s)."""
    warnings.filterwarnings("ignore")

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Propagate seed to all config-level random_state values so every
    # estimator (PCA, K-Means, GMM, UMAP, t-SNE, bootstrap) uses the
    # same seed.  This is the single source of truth for reproducibility.
    import src.config as cfg
    cfg.RANDOM_STATE = args.seed
    cfg.UMAP_2D_PARAMS["random_state"] = args.seed
    cfg.UMAP_3D_PARAMS["random_state"] = args.seed
    cfg.UMAP_CLUSTERING_PARAMS["random_state"] = args.seed

    logger.info(
        "Random seed set to {} (numpy, stdlib, and all estimator random_state values). "
        "Residual non-determinism: parallel UMAP may produce slightly different "
        "embeddings across runs due to floating-point ordering in multi-threaded "
        "nearest-neighbor computation — this is inherent to the algorithm and "
        "does not affect scientific conclusions.",
        args.seed,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable UMAP-based clustering if requested via CLI
    if args.umap_cluster:
        import src.config as cfg
        cfg.USE_UMAP_FOR_DENSITY_CLUSTERING = True
        logger.info(
            "UMAP clustering ENABLED: density methods will include a UMAP-{}D "
            "candidate space alongside PCA variants.",
            cfg.UMAP_CLUSTERING_N_COMPONENTS,
        )

    stage = args.stage
    min_voxels = args.min_voxels
    artifacts: dict | None = None
    cluster_output: dict | None = None

    # Special stage: voxel sweep
    if stage == "voxel-sweep":
        stage_voxel_sweep(args)
        return

    if stage in ("all", "ingest"):
        stage_ingest()
        if stage == "ingest":
            return

    if stage in ("all", "features"):
        artifacts = stage_features(
            resume_extraction=not args.no_resume_extraction,
            min_voxels=min_voxels,
            consistent_pca=args.consistent_pca,
            feature_selection_top_k=args.feature_selection,
        )
        if stage == "features":
            return

    if stage in ("all", "cluster"):
        artifacts, cluster_output = stage_cluster(
            artifacts, min_voxels=min_voxels, consistent_pca=args.consistent_pca,
            feature_selection_top_k=args.feature_selection,
        )
        if stage == "cluster":
            return

    if stage in ("all", "evaluate"):
        if artifacts is None or cluster_output is None:
            artifacts, cluster_output = stage_cluster(
                artifacts, min_voxels=min_voxels, consistent_pca=args.consistent_pca,
                feature_selection_top_k=args.feature_selection,
            )
        stage_evaluate(
            artifacts,
            cluster_output,
            output_dir=output_dir,
            use_tsne=args.use_tsne,
            run_bootstrap=not args.no_bootstrap,
        )

    if stage in ("all", "supervised-check"):
        if artifacts is None:
            # Load features for supervised check
            if not CLEAN_FEATURES_CSV_PATH.exists():
                raise FileNotFoundError(
                    f"Cleaned features not found at {CLEAN_FEATURES_CSV_PATH}. "
                    "Run --stage features first."
                )
            df_clean = pd.read_csv(CLEAN_FEATURES_CSV_PATH)
            feature_cols = [c for c in df_clean.columns if c not in META_COLUMNS]
            X_scaled, _ = scale_features(df_clean, feature_cols)
            y_true = df_clean["Lesion_Type"].values
        else:
            X_scaled = artifacts["X_scaled"]
            y_true = artifacts["df_clean"]["Lesion_Type"].values
            feature_cols = artifacts["feature_cols"]

        logger.info("=== Supervised Sanity Check ===")
        run_supervised_sanity_check(
            X_scaled, y_true,
            feature_names=feature_cols,
            output_dir=output_dir,
        )
        logger.success("Supervised sanity check complete.")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging(args.log_level)
    logger.info("KiTS23 Clustering Pipeline — project root: {}", PROJECT_ROOT)
    run_pipeline(args)


if __name__ == "__main__":
    main()