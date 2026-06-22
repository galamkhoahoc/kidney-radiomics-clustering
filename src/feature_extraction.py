"""Radiomics feature extraction, cleaning, scaling, and dimensionality reduction."""

from __future__ import annotations

import gc
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import (
    CLEAN_FEATURES_CSV_PATH,
    CT_WINDOW,
    EXTRACTION_BATCH_SIZE,
    FEATURES_CSV_PATH,
    FEATURE_SELECTION_TOP_K,
    META_COLUMNS,
    NAN_RATIO_THRESHOLD,
    OUTPUTS_DIR,
    PCA_VARIANCE_THRESHOLD,
    RADIOMICS_FEATURE_CLASSES,
    RADIOMICS_SETTINGS,
    SPEARMAN_CORR_THRESHOLD,
    VARIANCE_THRESHOLD,
)


def build_radiomics_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Create and configure a Pyradiomics feature extractor."""
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    for feature_class in RADIOMICS_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    logger.info("Pyradiomics extractor initialized with classes: {}", RADIOMICS_FEATURE_CLASSES)
    return extractor


def load_medical_image_sitk(
    nifti_path: str | Path,
    window: tuple[float, float] = CT_WINDOW,
    hu_baseline: Optional[float] = None,
) -> sitk.Image:
    """Load NIfTI image, apply HU baseline correction and CT windowing."""
    sitk_img = sitk.ReadImage(str(nifti_path))
    img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

    if hu_baseline is not None and np.isfinite(hu_baseline):
        img_array -= np.float32(hu_baseline)

    if window:
        np.clip(img_array, window[0], window[1], out=img_array)

    new_img = sitk.GetImageFromArray(img_array)
    new_img.CopyInformation(sitk_img)
    return new_img


def load_mask_sitk(nifti_path: str | Path) -> sitk.Image:
    """Load a binary NIfTI mask for Pyradiomics."""
    return sitk.ReadImage(str(nifti_path))


def extract_radiomics(
    df_lesions: pd.DataFrame,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    out_csv: Path = FEATURES_CSV_PATH,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Extract radiomics features for all lesions and write incrementally to CSV.

    Args:
        df_lesions: Lesion metadata DataFrame.
        extractor: Configured Pyradiomics extractor.
        out_csv: Output CSV path.
        batch_size: Number of rows per CSV append.
        resume: Skip extraction if output file already exists.

    Returns:
        DataFrame of extracted features.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if resume and out_csv.exists() and out_csv.stat().st_size > 0:
        logger.info("Features file already exists at {} — loading cached results.", out_csv)
        return pd.read_csv(out_csv)

    buffer: list[dict] = []
    errors: list[dict] = []
    cached_case: Optional[str] = None
    cached_image: Optional[sitk.Image] = None
    n_written = 0

    for _, row in tqdm(df_lesions.iterrows(), total=len(df_lesions), desc="Extracting radiomics"):
        try:
            hu_bl = row.get("kidney_hu_baseline", None)
            case_id = row["case_id"]

            if case_id != cached_case:
                if cached_image is not None:
                    del cached_image
                    gc.collect()
                cached_image = load_medical_image_sitk(row["image_path"], hu_baseline=hu_bl)
                cached_case = case_id

            mask = load_mask_sitk(row["mask_path"])
            result = extractor.execute(cached_image, mask)

            feature_dict: dict = {
                "PatientID": row["case_id"],
                "Lesion_Type": row["lesion_type"],
                "Instance": row["instance_name"],
            }

            for key, value in result.items():
                if not key.startswith("diagnostics_"):
                    feature_dict[key] = float(value)

            buffer.append(feature_dict)

            if len(buffer) >= batch_size:
                pd.DataFrame(buffer).to_csv(
                    out_csv,
                    mode="a",
                    header=not out_csv.exists(),
                    index=False,
                )
                n_written += len(buffer)
                buffer.clear()

            del mask, result, feature_dict
            gc.collect()

        except Exception as exc:
            if not errors:
                logger.error(
                    "First extraction error ({}/{}):",
                    row["case_id"],
                    row["instance_name"],
                )
                traceback.print_exc()
            errors.append(
                {
                    "case": row["case_id"],
                    "instance": row["instance_name"],
                    "error": str(exc),
                }
            )
            gc.collect()

    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_csv,
            mode="a",
            header=not out_csv.exists(),
            index=False,
        )
        n_written += len(buffer)
        buffer.clear()

    logger.info("Extraction complete: {}/{} lesions written to {}", n_written, len(df_lesions), out_csv)
    if errors:
        logger.warning("{} lesions failed extraction.", len(errors))
        for err in errors[:5]:
            logger.warning("  - {}/{}: {}", err["case"], err["instance"], err["error"][:80])

    if out_csv.exists():
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def clean_features(df_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Clean and filter radiomics features.

    Returns:
        Tuple of (cleaned DataFrame, metadata columns, feature column names).
    """
    df = df_features.copy()
    meta_cols = list(META_COLUMNS)

    diag_cols = [c for c in df.columns if c.startswith("diagnostics_")]
    df = df.drop(columns=diag_cols, errors="ignore")
    logger.info("Removed {} diagnostics columns. Shape features retained.", len(diag_cols))

    feature_cols = [c for c in df.columns if c not in meta_cols]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    feature_cols = [c for c in feature_cols if df[c].dtype in ("float64", "int64", "float32")]
    logger.info("Numeric features remaining: {}", len(feature_cols))

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    nan_ratio = df[feature_cols].isna().mean()
    high_nan_cols = nan_ratio[nan_ratio > NAN_RATIO_THRESHOLD].index.tolist()
    df = df.drop(columns=high_nan_cols)
    feature_cols = [c for c in feature_cols if c not in high_nan_cols]
    logger.info("Removed {} columns with NaN > {:.0%}", len(high_nan_cols), NAN_RATIO_THRESHOLD)

    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].values
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(X)
    low_var_mask = ~selector.get_support()
    low_var_cols = [feature_cols[i] for i in range(len(feature_cols)) if low_var_mask[i]]
    df = df.drop(columns=low_var_cols)
    feature_cols = [c for c in feature_cols if c not in low_var_cols]
    logger.info("Removed {} near-zero variance columns.", len(low_var_cols))

    corr_matrix = df[feature_cols].corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > SPEARMAN_CORR_THRESHOLD)]
    df = df.drop(columns=high_corr_cols)
    feature_cols = [c for c in feature_cols if c not in high_corr_cols]
    logger.info("Removed {} highly correlated columns (Spearman > {}).", len(high_corr_cols), SPEARMAN_CORR_THRESHOLD)

    # --- HU feature survival diagnostic ---
    hu_key_features = [
        "firstorder_Mean", "firstorder_Median",
        "firstorder_10Percentile", "firstorder_90Percentile",
    ]
    all_firstorder = [c for c in df.columns if "firstorder" in c.lower() and c not in meta_cols]
    survived_firstorder = [c for c in all_firstorder if c in feature_cols]
    removed_firstorder = [c for c in all_firstorder if c not in feature_cols]

    logger.info(
        "HU DIAGNOSTIC: {}/{} firstorder features survived cleaning.",
        len(survived_firstorder), len(survived_firstorder) + len(removed_firstorder),
    )
    for key_feat in hu_key_features:
        # Check partial match since pyradiomics prefixes with 'original_'
        matches = [c for c in feature_cols if key_feat.lower() in c.lower()]
        if matches:
            logger.info("  ✓ {} SURVIVED → {}", key_feat, matches)
        else:
            logger.warning("  ✗ {} was REMOVED by cleaning pipeline", key_feat)

    if survived_firstorder:
        logger.info("  Surviving firstorder features: {}", survived_firstorder[:10])
    if removed_firstorder:
        logger.info("  Removed firstorder features: {}", removed_firstorder[:10])

    logger.info("Final feature set: {} features, {} samples.", len(feature_cols), len(df))
    return df, meta_cols, feature_cols


def scale_features(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """Standard-scale feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    logger.info(
        "Scaled features: shape={}, mean≈{:.6f}, std≈{:.6f}",
        X_scaled.shape,
        X_scaled.mean(axis=0).mean(),
        X_scaled.std(axis=0).mean(),
    )
    return X_scaled, scaler


def apply_pca(
    X_scaled: np.ndarray,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, PCA, int]:
    """
    Fit PCA and retain components explaining ``variance_threshold`` cumulative variance.

    Returns:
        Tuple of (transformed matrix, fitted PCA model, number of components).
    """
    n_components_max = min(X_scaled.shape[0], X_scaled.shape[1])
    pca_full = PCA(n_components=n_components_max)
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X_scaled)

    logger.info(
        "PCA: {} → {} components ({:.2%} variance retained).",
        X_scaled.shape[1],
        n_pc,
        pca.explained_variance_ratio_.sum(),
    )

    # --- Save PCA loadings diagnostic ---
    if feature_names is not None and len(feature_names) == X_scaled.shape[1]:
        loadings = pca.components_  # shape: (n_pc, n_features)
        top_n = min(5, len(feature_names))
        rows = []
        for pc_idx in range(min(n_pc, 5)):  # top 5 PCs
            abs_loadings = np.abs(loadings[pc_idx])
            top_indices = np.argsort(abs_loadings)[::-1][:top_n]
            for rank, feat_idx in enumerate(top_indices):
                feat_name = feature_names[feat_idx]
                rows.append({
                    "PC": f"PC{pc_idx + 1}",
                    "Rank": rank + 1,
                    "Feature": feat_name,
                    "Loading": float(loadings[pc_idx, feat_idx]),
                    "Abs_Loading": float(abs_loadings[feat_idx]),
                    "Is_Firstorder": "firstorder" in feat_name.lower(),
                })

        df_loadings = pd.DataFrame(rows)
        loadings_path = OUTPUTS_DIR / "pca_loadings.csv"
        loadings_path.parent.mkdir(parents=True, exist_ok=True)
        df_loadings.to_csv(loadings_path, index=False)
        logger.info("Saved PCA loadings to {}", loadings_path)

        # Log PC1/PC2 top features with firstorder flag
        for pc_label in ["PC1", "PC2"]:
            pc_rows = df_loadings[df_loadings["PC"] == pc_label]
            has_hu = pc_rows["Is_Firstorder"].any()
            top_feats = pc_rows["Feature"].tolist()
            logger.info(
                "  {} top features: {} {}",
                pc_label, top_feats,
                "(includes firstorder/HU ✓)" if has_hu else "(NO firstorder/HU features ✗)",
            )

    return X_pca, pca, n_pc


def select_discriminative_features(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    top_k: int = FEATURE_SELECTION_TOP_K,
) -> list[str]:
    """Rank features by ANOVA F-statistic and keep the top-K most discriminative.

    This is a supervised feature-selection step (Option A) that ensures
    low-variance but highly discriminative features (e.g. HU/firstorder)
    survive into the PCA/clustering space.  Frame this in your thesis as
    a semi-supervised feature-discovery step ("phân cụm có hướng dẫn đặc trưng").

    Args:
        df_clean: Cleaned feature DataFrame.
        feature_cols: List of feature column names.
        y: Label array (e.g. 'Tumor' / 'Cyst') aligned with df_clean rows.
        top_k: Number of features to retain.

    Returns:
        List of the top-K feature column names, ordered by descending F-score.
    """
    X = df_clean[feature_cols].values
    F_scores, p_values = f_classif(X, y)

    ranked = sorted(
        zip(feature_cols, F_scores, p_values),
        key=lambda t: t[1],
        reverse=True,
    )

    # Clamp top_k to the number of available features
    top_k = min(top_k, len(ranked))
    selected = [name for name, _, _ in ranked[:top_k]]

    # --- Diagnostic logging ---
    logger.info(
        "Feature selection (ANOVA F-test): {} → {} features retained.",
        len(feature_cols), len(selected),
    )

    # Save full ranking to CSV for thesis reference
    ranking_rows = [
        {
            "Rank": i + 1,
            "Feature": name,
            "F_score": float(f),
            "p_value": float(p),
            "Selected": name in selected,
            "Is_Firstorder": "firstorder" in name.lower(),
        }
        for i, (name, f, p) in enumerate(ranked)
    ]
    df_ranking = pd.DataFrame(ranking_rows)
    ranking_path = OUTPUTS_DIR / "feature_selection_ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    df_ranking.to_csv(ranking_path, index=False)
    logger.info("Saved feature selection ranking to {}", ranking_path)

    # Log how many firstorder/HU features survived
    n_firstorder_selected = sum(1 for f in selected if "firstorder" in f.lower())
    n_firstorder_total = sum(1 for f in feature_cols if "firstorder" in f.lower())
    logger.info(
        "  Firstorder/HU features selected: {}/{} (total available: {})",
        n_firstorder_selected,
        top_k,
        n_firstorder_total,
    )
    for name, f, p in ranked[:10]:
        marker = "✓" if name in selected else "✗"
        logger.info("  {} {:50s}  F={:10.2f}  p={:.2e}", marker, name, f, p)

    return selected


def run_feature_pipeline(
    df_lesions: pd.DataFrame,
    features_path: Path = FEATURES_CSV_PATH,
    clean_path: Path = CLEAN_FEATURES_CSV_PATH,
    resume_extraction: bool = True,
    feature_selection_top_k: int | None = None,
) -> dict:
    """End-to-end feature pipeline: extract → clean → [select] → scale → PCA.

    When ``feature_selection_top_k`` is not None, an ANOVA F-test step
    selects the top-K most discriminative features (using Lesion_Type as
    the label) *before* scaling and PCA.  This keeps HU/firstorder
    features alive in the projection space.

    Returns:
        Dictionary with intermediate artifacts.
    """
    extractor = build_radiomics_extractor()
    df_features = extract_radiomics(df_lesions, extractor, out_csv=features_path, resume=resume_extraction)

    if df_features.empty:
        logger.error("Feature extraction produced an empty DataFrame. Please ensure the dataset contains valid masks.")
        import sys
        sys.exit(1)

    df_clean, meta_cols, feature_cols = clean_features(df_features)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    logger.info("Saved cleaned features to {}", clean_path)

    # --- Optional: supervised feature selection (Option A) ---
    if feature_selection_top_k is not None:
        if "Lesion_Type" not in df_clean.columns:
            logger.warning(
                "--feature-selection requested but 'Lesion_Type' column not found. "
                "Skipping feature selection."
            )
        else:
            y = df_clean["Lesion_Type"].values
            feature_cols = select_discriminative_features(
                df_clean, feature_cols, y, top_k=feature_selection_top_k,
            )
            logger.info(
                "Feature set narrowed to {} discriminative features before scaling/PCA.",
                len(feature_cols),
            )

    X_scaled, scaler = scale_features(df_clean, feature_cols)
    X_pca, pca_model, n_pc = apply_pca(X_scaled, feature_names=feature_cols)

    return {
        "df_features": df_features,
        "df_clean": df_clean,
        "meta_cols": meta_cols,
        "feature_cols": feature_cols,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "X_pca": X_pca,
        "pca_model": pca_model,
        "n_pc": n_pc,
    }
