"""Central configuration for paths, constants, and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

KITS23_REPO_DIR: Path = RAW_DATA_DIR / "kits23"
KITS23_DATASET_DIR: Path = KITS23_REPO_DIR / "dataset"
KITS23_METADATA_PATH: Path = KITS23_DATASET_DIR / "kits23.json"

FEATURES_CSV_PATH: Path = PROCESSED_DATA_DIR / "features.csv"
LESIONS_CSV_PATH: Path = PROCESSED_DATA_DIR / "lesions.csv"
CLEAN_FEATURES_CSV_PATH: Path = PROCESSED_DATA_DIR / "features_clean.csv"
METRICS_CSV_PATH: Path = OUTPUTS_DIR / "clustering_metrics.csv"

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
KITS23_REPO_URL: str = "https://github.com/neheller/kits23"
KITS23_DOWNLOAD_CMD: str = "kits23_download_data"

# ---------------------------------------------------------------------------
# Lesion scanning
# ---------------------------------------------------------------------------
MIN_VOXELS: int = 10
ANNOTATION_SUFFIX: str = "annotation-1"
KIDNEY_MASK_FILENAME: str = "kidney_instance-1_annotation-1.nii.gz"
MIN_KIDNEY_VOXELS: int = 100

# ---------------------------------------------------------------------------
# Radiomics (Pyradiomics)
# ---------------------------------------------------------------------------
RADIOMICS_SETTINGS: dict = {
    "binWidth": 25.0,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "verbose": False,
}
RADIOMICS_FEATURE_CLASSES: tuple[str, ...] = (
    "firstorder",
    "shape",
    "glcm",
    "gldm",
    "glrlm",
    "glszm",
    "ngtdm",
)
CT_WINDOW: tuple[float, float] = (-200.0, 500.0)
EXTRACTION_BATCH_SIZE: int = 10

# ---------------------------------------------------------------------------
# Feature cleaning
# ---------------------------------------------------------------------------
META_COLUMNS: tuple[str, ...] = ("PatientID", "Lesion_Type", "Instance")
NAN_RATIO_THRESHOLD: float = 0.5
VARIANCE_THRESHOLD: float = 1e-5
SPEARMAN_CORR_THRESHOLD: float = 0.95

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
PCA_VARIANCE_THRESHOLD: float = 0.95
PCA_N_COMPONENTS_LIST: tuple[int, ...] = (2, 5, 10, 16)

# ---------------------------------------------------------------------------
# Supervised feature selection (ANOVA F-test, Option A)
# ---------------------------------------------------------------------------
FEATURE_SELECTION_TOP_K: int = 25  # number of features to keep when --feature-selection is active

# ---------------------------------------------------------------------------
# UMAP (visualization)
# ---------------------------------------------------------------------------
UMAP_2D_PARAMS: dict = {
    "n_components": 2,
    "random_state": 42,
    "n_neighbors": 30,
    "min_dist": 0.1,
    "init": "spectral",
}
UMAP_3D_PARAMS: dict = {
    "n_components": 3,
    "random_state": 42,
    "n_neighbors": 30,
    "min_dist": 0.0,
}

# ---------------------------------------------------------------------------
# UMAP for clustering (not visualization — min_dist=0.0 maximizes separation)
# ---------------------------------------------------------------------------
USE_UMAP_FOR_DENSITY_CLUSTERING: bool = False  # enabled via --umap-cluster CLI
UMAP_CLUSTERING_N_COMPONENTS: int = 10  # default when a single value is needed
UMAP_CLUSTERING_N_COMPONENTS_LIST: tuple[int, ...] = (5, 10, 15)  # grid search candidates
UMAP_CLUSTERING_PARAMS: dict = {
    "n_neighbors": 15,   # lower = emphasize local structure (was 30)
    "min_dist": 0.0,     # maximize cluster separation for density methods
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# HDBSCAN grid search
# ---------------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZES: tuple[int, ...] = (10, 15, 25, 40, 60, 80)  # added 10 for discriminative/UMAP space
HDBSCAN_MIN_SAMPLES_LIST: tuple[int, ...] = (1, 3, 5, 10)
HDBSCAN_CLUSTER_METHODS: tuple[str, ...] = ("eom", "leaf")  # leaf finds finer sub-structure
HDBSCAN_MIN_COVERAGE: float = 0.4
HDBSCAN_MAX_CLUSTERS: int = 6  # raised from 4 to let leaf reveal natural sub-groups
HDBSCAN_EPSILON_VALUES: tuple[float, ...] = (0.0, 0.3, 0.5, 1.0)

# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------
OPTICS_MIN_SAMPLES_LIST: tuple[int, ...] = (3, 5, 10, 20)
OPTICS_MIN_CLUSTER_SIZES: tuple[int, ...] = (10, 20, 30, 50)
OPTICS_XI_VALUES: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1, 0.2)
OPTICS_EPS_VALUES: tuple[float, ...] = (0.2, 0.3, 0.5, 1.0)  # for DBSCAN-style extraction
OPTICS_CLUSTER_METHODS: tuple[str, ...] = ("xi", "dbscan")  # search both extraction methods
OPTICS_MIN_COVERAGE: float = 0.5   # raised from 0.1 — reject cherry-picked dense pockets
OPTICS_MAX_CLUSTERS: int = 15

# ---------------------------------------------------------------------------
# Grid search scoring — soft cluster-count penalty
# ---------------------------------------------------------------------------
TARGET_N_CLUSTERS: int = 2           # conceptual target for Tumor/Cyst
CLUSTER_COUNT_PENALTY: float = 0.15  # mild penalty (0.1-0.2 range); don't force 2

# ---------------------------------------------------------------------------
# Baseline clustering (GMM / K-Means)
# ---------------------------------------------------------------------------
BASELINE_N_CLUSTERS: int = 2
RANDOM_STATE: int = 42
GMM_COVARIANCE_TYPE: str = "full"
KMEANS_N_INIT: int = 10

# ---------------------------------------------------------------------------
# Lesion size filtering
# ---------------------------------------------------------------------------
LESION_MIN_VOXELS_FILTER: int = 250  # filter out tiny masks with noisy radiomics; use --min-voxels 0 to disable

# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------
BOOTSTRAP_N_ITERATIONS: int = 100
BOOTSTRAP_SAMPLE_RATIO: float = 0.8
BOOTSTRAP_ARI_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 120
FONT_SIZE: int = 11
PLOT_PALETTE: tuple[str, ...] = (
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
)
LESION_COLORS: dict[str, str] = {"Tumor": "#e74c3c", "Cyst": "#3498db"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)
