"""Download KiTS23 dataset and scan lesion instances."""

from __future__ import annotations

import gc
import glob
import json
import subprocess
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    ANNOTATION_SUFFIX,
    KIDNEY_MASK_FILENAME,
    KITS23_DATASET_DIR,
    KITS23_DOWNLOAD_CMD,
    KITS23_METADATA_PATH,
    KITS23_REPO_DIR,
    KITS23_REPO_URL,
    LESIONS_CSV_PATH,
    MIN_KIDNEY_VOXELS,
    MIN_VOXELS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def ensure_directories() -> None:
    """Create required data and output directories if missing."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def ingest_dataset() -> Path:
    """
    Verify that the KiTS23 dataset directory exists.

    Returns:
        Path to the KiTS23 dataset directory.
    """
    ensure_directories()
    logger.info("Verifying KiTS23 dataset directory.")
    
    if not KITS23_DATASET_DIR.exists():
        logger.error(f"Dataset directory not found: {KITS23_DATASET_DIR}")
        logger.error("Please run the following commands manually to download the dataset:")
        logger.error(f"  git clone {KITS23_REPO_URL} {KITS23_REPO_DIR}")
        logger.error(f"  cd {KITS23_REPO_DIR}")
        logger.error("  pip install -e .")
        logger.error("  (Restart your runtime if on Colab)")
        logger.error("  kits23_download_data")
        raise FileNotFoundError(f"Dataset directory not found: {KITS23_DATASET_DIR}")

    return KITS23_DATASET_DIR


def scan_instances(
    dataset_dir: Path,
    min_voxels: int = MIN_VOXELS,
) -> pd.DataFrame:
    """
    Scan all cases and collect valid tumor/cyst lesion masks (annotation-1).

    Args:
        dataset_dir: Root path to KiTS23 dataset folder.
        min_voxels: Minimum voxel count to keep a mask.

    Returns:
        DataFrame with lesion metadata and file paths.
    """
    records: list[dict] = []
    skipped_empty = 0
    skipped_small = 0

    case_dirs = sorted(glob.glob(str(dataset_dir / "case_*")))
    logger.info("Scanning {} case directories in {}", len(case_dirs), dataset_dir)

    for case_dir_str in case_dirs:
        case_dir = Path(case_dir_str)
        case_id = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        instances_dir = case_dir / "instances"

        if not instances_dir.exists() or not img_path.exists():
            continue

        for mask_file in sorted(instances_dir.iterdir()):
            if not mask_file.name.endswith(".nii.gz"):
                continue
            if "kidney_instance" in mask_file.name:
                continue
            if ANNOTATION_SUFFIX not in mask_file.name:
                continue

            if "tumor_instance" in mask_file.name:
                lesion_type = "Tumor"
            elif "cyst_instance" in mask_file.name:
                lesion_type = "Cyst"
            else:
                continue

            mask_path = mask_file
            instance_name = mask_file.name.replace("_annotation-1.nii.gz", "")

            try:
                mask_data = nib.load(str(mask_path)).get_fdata()
                n_voxels = int(np.sum(mask_data > 0))

                if n_voxels == 0:
                    skipped_empty += 1
                    continue
                if n_voxels < min_voxels:
                    skipped_small += 1
                    continue

                records.append(
                    {
                        "case_id": case_id,
                        "lesion_type": lesion_type,
                        "instance_name": instance_name,
                        "mask_path": str(mask_path),
                        "image_path": str(img_path),
                        "n_voxels": n_voxels,
                    }
                )
            except Exception as exc:
                logger.warning("Failed to read {}: {}", mask_path, exc)

    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=[
            "case_id", "lesion_type", "instance_name", "mask_path", 
            "image_path", "n_voxels"
        ])
    logger.info("Valid lesions: {} (Tumor: {}, Cyst: {})", len(df),
                (df["lesion_type"] == "Tumor").sum() if len(df) else 0,
                (df["lesion_type"] == "Cyst").sum() if len(df) else 0)
    logger.info("Skipped empty: {}, skipped < {} voxels: {}", skipped_empty, min_voxels, skipped_small)
    return df


def compute_kidney_baselines(df_lesions: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """
    Compute mean HU of kidney cortex per case for contrast-phase normalization.

    Args:
        df_lesions: Lesion DataFrame from scan_instances.
        dataset_dir: KiTS23 dataset root.

    Returns:
        Lesion DataFrame with added ``kidney_hu_baseline`` column.
    """
    baselines: dict[str, float] = {}
    unique_cases = df_lesions["case_id"].unique()
    n_total = len(unique_cases)

    logger.info("Computing kidney HU baseline for {} cases.", n_total)

    for i, case_id in enumerate(unique_cases):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("Baseline progress: {}/{}", i + 1, n_total)

        case_dir = dataset_dir / case_id
        img_path = case_dir / "imaging.nii.gz"
        kidney_mask_path = case_dir / "instances" / KIDNEY_MASK_FILENAME

        if not kidney_mask_path.exists() or not img_path.exists():
            continue

        try:
            img_data = nib.load(str(img_path)).get_fdata()
            mask_data = nib.load(str(kidney_mask_path)).get_fdata() > 0

            if mask_data.sum() > MIN_KIDNEY_VOXELS:
                baselines[case_id] = float(np.mean(img_data[mask_data]))

            del img_data, mask_data
            gc.collect()
        except Exception as exc:
            logger.warning("Baseline error for {}: {}", case_id, exc)

    df = df_lesions.copy()
    df["kidney_hu_baseline"] = df["case_id"].map(baselines)

    n_found = int(df["kidney_hu_baseline"].notna().sum())
    mean_bl = df["kidney_hu_baseline"].mean()
    std_bl = df["kidney_hu_baseline"].std()
    logger.info(
        "Baseline computed: {}/{} lesions ({} cases). Mean HU: {:.1f} ± {:.1f}",
        n_found, len(df), len(baselines), mean_bl, std_bl,
    )
    return df


def load_kits23_metadata(metadata_path: Path = KITS23_METADATA_PATH) -> dict[str, str]:
    """
    Load histologic subtype metadata from kits23.json.

    Returns:
        Mapping of case_id → histologic subtype string.
    """
    if not metadata_path.exists():
        logger.warning("Metadata file not found: {}", metadata_path)
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        kits_meta = json.load(f)

    subtype_map: dict[str, str] = {}

    if isinstance(kits_meta, list):
        for i, case_data in enumerate(kits_meta):
            case_id = case_data.get("case_id", f"case_{i:05d}")
            subtype = (
                case_data.get("tumor_histologic_subtype")
                or case_data.get("histologic_subtype")
                or case_data.get("pathology_t_stage")
                or "Unknown"
            )
            subtype_map[case_id] = subtype
    elif isinstance(kits_meta, dict):
        for key, value in kits_meta.items():
            if isinstance(value, dict):
                subtype = (
                    value.get("tumor_histologic_subtype")
                    or value.get("histologic_subtype")
                    or "Unknown"
                )
                subtype_map[key] = subtype

    if subtype_map and all(
        isinstance(k, int) or (isinstance(k, str) and k.isdigit())
        for k in list(subtype_map.keys())[:10]
    ):
        subtype_map = {f"case_{int(k):05d}": v for k, v in subtype_map.items()}

    logger.info("Loaded metadata for {} cases.", len(subtype_map))
    return subtype_map


def prepare_lesions(
    dataset_dir: Optional[Path] = None,
    save_path: Path = LESIONS_CSV_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Scan lesions, compute HU baselines, and optionally cache to CSV.

    Returns:
        Prepared lesion DataFrame.
    """
    dataset_dir = dataset_dir or KITS23_DATASET_DIR

    if use_cache and save_path.exists():
        logger.info("Loading cached lesions from {}", save_path)
        return pd.read_csv(save_path)

    df_lesions = scan_instances(dataset_dir)
    df_lesions = compute_kidney_baselines(df_lesions, dataset_dir)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_lesions.to_csv(save_path, index=False)
    logger.info("Saved lesions to {}", save_path)
    return df_lesions
