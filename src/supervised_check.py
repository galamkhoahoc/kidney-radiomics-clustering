"""Supervised sanity check: LDA projection + stratified-CV classifiers.

Answers the question: does a supervised model find a Tumor/Cyst direction
that unsupervised methods cannot recover?  If yes, the signal exists but
lives on a low-variance axis that PCA discards — a publishable finding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import FIGURE_DPI, FONT_SIZE, OUTPUTS_DIR
import src.config as _cfg


def _setup_plot_style() -> None:
    """Configure matplotlib/seaborn defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = FIGURE_DPI
    plt.rcParams["font.size"] = FONT_SIZE


def run_lda_projection(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    output_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit LDA and project to 1D.  Compute Fisher's discriminant ratio.

    Returns:
        Tuple of (1D projections, binary labels, Fisher's ratio).
    """
    y = np.array([1 if v == "Tumor" else 0 for v in y_true])

    lda = LinearDiscriminantAnalysis(n_components=1)
    proj = lda.fit_transform(X_scaled, y).ravel()

    t = proj[y == 1]  # tumor projections
    c = proj[y == 0]  # cyst projections

    # Fisher's discriminant ratio: between-class gap vs within-class spread
    fisher = float((t.mean() - c.mean()) ** 2 / (t.var() + c.var() + 1e-9))

    logger.info(
        "LDA projection: Fisher's discriminant ratio = {:.4f}", fisher,
    )
    logger.info(
        "  Tumor: mean={:.3f}, std={:.3f}, n={}",
        t.mean(), t.std(), len(t),
    )
    logger.info(
        "  Cyst:  mean={:.3f}, std={:.3f}, n={}",
        c.mean(), c.std(), len(c),
    )

    if fisher > 2.0:
        logger.info(
            "Fisher ratio > 2.0 — strong supervised separation exists. "
            "This direction is likely low-variance (PCA misses it)."
        )
    elif fisher > 0.5:
        logger.info(
            "Fisher ratio 0.5–2.0 — moderate separation. Classes overlap "
            "but a supervised model can partially distinguish them."
        )
    else:
        logger.info(
            "Fisher ratio < 0.5 — weak separation. Classes genuinely "
            "overlap in feature space."
        )

    # --- Plot ---
    if output_path:
        _setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            c, bins=30, alpha=0.6, color="#3498db", label="Cyst",
            edgecolor="white", density=True,
        )
        ax.hist(
            t, bins=30, alpha=0.6, color="#e74c3c", label="Tumor",
            edgecolor="white", density=True,
        )

        ax.axvline(c.mean(), color="#2980b9", linestyle="--", lw=2, label=f"Cyst mean ({c.mean():.2f})")
        ax.axvline(t.mean(), color="#c0392b", linestyle="--", lw=2, label=f"Tumor mean ({t.mean():.2f})")

        ax.set_xlabel("LDA Projection (1D)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"LDA Projection: Tumor vs Cyst (Fisher ratio = {fisher:.3f})",
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved LDA projection plot to {}", output_path)

    return proj, y, fisher


def run_quick_classifier(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Stratified 5-fold CV with class-balanced classifiers.

    Reports ROC-AUC, balanced accuracy, and macro-F1 — not raw accuracy,
    which would be misleading given the class imbalance (Tumors >> Cysts).

    Returns:
        DataFrame with one row per model and per-fold stats.
    """
    y = np.array([1 if v == "Tumor" else 0 for v in y_true])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_cfg.RANDOM_STATE)
    scoring = ["roc_auc", "balanced_accuracy", "f1_macro"]

    models = {
        # class_weight="balanced" so the minority Cyst class isn't ignored
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=_cfg.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=_cfg.RANDOM_STATE,
        ),
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
        row = {
            "model": name,
            "roc_auc": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "balanced_acc": float(scores["test_balanced_accuracy"].mean()),
            "balanced_acc_std": float(scores["test_balanced_accuracy"].std()),
            "f1_macro": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
        }
        rows.append(row)

        logger.info(
            "{}: ROC-AUC={:.4f}±{:.4f}, Balanced-Acc={:.4f}±{:.4f}, F1-Macro={:.4f}±{:.4f}",
            name,
            row["roc_auc"], row["roc_auc_std"],
            row["balanced_acc"], row["balanced_acc_std"],
            row["f1_macro"], row["f1_macro_std"],
        )

    return pd.DataFrame(rows)


def run_supervised_sanity_check(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    feature_names: list[str] | None = None,
    output_dir: Path = OUTPUTS_DIR,
) -> dict:
    """Full supervised sanity check: LDA + classifiers.

    Saves:
        - outputs/lda_projection.png
        - outputs/supervised_sanity_check.csv

    Returns:
        Dict with fisher_ratio, classifier_df, and interpretation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SUPERVISED SANITY CHECK")
    logger.info("=" * 60)

    n_tumor = sum(1 for v in y_true if v == "Tumor")
    n_cyst = sum(1 for v in y_true if v == "Cyst")
    logger.info(
        "Class distribution: {} Tumor, {} Cyst (ratio {:.2f}:1)",
        n_tumor, n_cyst, n_tumor / max(n_cyst, 1),
    )

    # LDA
    proj, y, fisher = run_lda_projection(
        X_scaled, y_true, output_path=output_dir / "lda_projection.png",
    )

    # Classifiers
    df_clf = run_quick_classifier(X_scaled, y_true)

    # Save results
    df_clf["fisher_ratio"] = fisher
    csv_path = output_dir / "supervised_sanity_check.csv"
    df_clf.to_csv(csv_path, index=False)
    logger.info("Saved supervised results to {}", csv_path)

    # --- Interpretation ---
    best_auc = df_clf["roc_auc"].max()

    if best_auc > 0.85 and fisher > 1.0:
        interpretation = (
            "STRONG supervised separation (AUC={:.3f}, Fisher={:.3f}). "
            "Tumor/Cyst signal EXISTS but lives on a low-variance axis "
            "that PCA discards. Unsupervised clustering in PCA space "
            "cannot recover it — this is the key finding."
        ).format(best_auc, fisher)
    elif best_auc > 0.70:
        interpretation = (
            "MODERATE supervised separation (AUC={:.3f}, Fisher={:.3f}). "
            "Some discriminative signal exists but classes partially overlap. "
            "The boundary may be roughly linear if LogReg ≈ RF."
        ).format(best_auc, fisher)
    else:
        interpretation = (
            "WEAK supervised separation (AUC={:.3f}, Fisher={:.3f}). "
            "Classes genuinely overlap in feature space. Radiomics alone "
            "may not cleanly separate Tumor/Cyst — a valid finding."
        ).format(best_auc, fisher)

    logger.info("INTERPRETATION: {}", interpretation)

    return {
        "fisher_ratio": fisher,
        "classifier_df": df_clf,
        "interpretation": interpretation,
    }
