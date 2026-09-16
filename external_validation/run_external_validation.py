#!/usr/bin/env python3
"""Run DeepH-Immune external validation directly from released models/features."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeph_immune import (  # noqa: E402
    CUSTOM_OBJECTS,
    MHC_ALIGNED_LEN,
    PEP_MAX_LEN,
    tokenize_mhc_aligned,
    tokenize_peptide,
)

PREPARED_CSV = ROOT / "data/external/MHC2_External_dataset_V2_prepared.csv"
FEATURE_DIR = ROOT / "data/external/features"
MODEL_DIR = ROOT / "models"
OUTDIR = ROOT / "results/external_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)


def fold_number(path: Path) -> int:
    m = re.search(r"fold(\d+)", path.stem)
    return int(m.group(1)) if m else 999


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing:\n  {path}\n\n"
            "See README.md and data/external/features/README.md."
        )


def main():
    required_files = [
        PREPARED_CSV,
        FEATURE_DIR / "ESM2_peptide_external_v2.npy",
        FEATURE_DIR / "ESM2_mhc_external_v2.npy",
        FEATURE_DIR / "Labels_external_v2.npy",
        FEATURE_DIR / "Sample_ids_external_v2.npy",
    ]

    for path in required_files:
        require(path)

    model_paths = sorted(
        MODEL_DIR.glob("DeepH_ImmuneV1_fold*.keras"),
        key=fold_number,
    )

    if len(model_paths) != 5:
        raise FileNotFoundError(
            f"Expected 5 fold models in {MODEL_DIR}; found {len(model_paths)}."
        )

    ext = pd.read_csv(PREPARED_CSV)

    pep_tokens = np.asarray(
        [tokenize_peptide(s) for s in ext["Peptide"]],
        dtype=np.int32,
    )

    mhc_tokens = np.asarray(
        [tokenize_mhc_aligned(s) for s in ext["MHC_aligned_281"]],
        dtype=np.int32,
    )

    esm_pep = np.load(
        FEATURE_DIR / "ESM2_peptide_external_v2.npy"
    ).astype(np.float32)

    esm_mhc = np.load(
        FEATURE_DIR / "ESM2_mhc_external_v2.npy"
    ).astype(np.float32)

    y = np.load(
        FEATURE_DIR / "Labels_external_v2.npy"
    ).astype(np.int32)

    ids = np.load(
        FEATURE_DIR / "Sample_ids_external_v2.npy",
        allow_pickle=True,
    ).astype(str)

    n = len(ext)

    assert pep_tokens.shape == (n, PEP_MAX_LEN)
    assert mhc_tokens.shape == (n, MHC_ALIGNED_LEN)
    assert esm_pep.shape == (n, 1280)
    assert esm_mhc.shape == (n, 1280)
    assert y.shape == (n,)
    assert ids.shape == (n,)
    assert np.array_equal(
        y,
        ext["Immunogenicity"].to_numpy(dtype=np.int32),
    )
    assert np.array_equal(
        ids,
        ext["sample_id"].astype(str).to_numpy(),
    )

    print("External rows:", n)
    print("True positives:", int(y.sum()))
    print("True negatives:", int((y == 0).sum()))
    print("All external inputs are aligned.")

    X = [pep_tokens, mhc_tokens, esm_pep, esm_mhc]

    fold_predictions = []
    fold_metrics = []

    for fold, model_path in enumerate(model_paths, start=1):
        print(f"\nLoading fold {fold}: {model_path.name}")

        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        p = model.predict(
            X,
            batch_size=64,
            verbose=0,
        ).reshape(-1)

        fold_predictions.append(p)

        fold_metrics.append({
            "fold": fold,
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
        })

        del model

    fold_predictions = np.stack(fold_predictions, axis=1)
    ensemble_prob = fold_predictions.mean(axis=1)
    ensemble_sd = fold_predictions.std(axis=1)

    threshold = 0.50
    pred = (ensemble_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    summary = {
        "n": int(n),
        "true_positive_prevalence": float(y.mean()),
        "ensemble_auroc": float(roc_auc_score(y, ensemble_prob)),
        "ensemble_auprc": float(
            average_precision_score(y, ensemble_prob)
        ),
        "brier_score": float(
            brier_score_loss(y, ensemble_prob)
        ),
        "threshold_for_diagnostic_metrics": threshold,
        "predicted_positive_rate": float(pred.mean()),
        "mean_probability_all": float(ensemble_prob.mean()),
        "mean_probability_true_positive": float(
            ensemble_prob[y == 1].mean()
        ),
        "mean_probability_true_negative": float(
            ensemble_prob[y == 0].mean()
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y, pred, zero_division=0)
        ),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print("\nFold metrics")
    print(pd.DataFrame(fold_metrics).to_string(index=False))

    print("\nEnsemble summary")
    print(json.dumps(summary, indent=2))

    result = ext.copy()

    for i in range(5):
        result[f"fold{i+1}_probability"] = fold_predictions[:, i]

    result["ensemble_probability"] = ensemble_prob
    result["ensemble_sd"] = ensemble_sd
    result["prediction_at_0.5"] = pred

    result.to_csv(
        OUTDIR / "external_predictions.csv",
        index=False,
    )

    pd.DataFrame(fold_metrics).to_csv(
        OUTDIR / "external_fold_metrics.csv",
        index=False,
    )

    with (OUTDIR / "external_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    fpr, tpr, _ = roc_curve(y, ensemble_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {summary['ensemble_auroc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("DeepH-Immune external ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "external_ROC.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(
        y, ensemble_prob
    )
    plt.figure(figsize=(6, 5))
    plt.plot(
        recall,
        precision,
        label=f"AUPRC = {summary['ensemble_auprc']:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("DeepH-Immune external precision-recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "external_PR.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(
        ensemble_prob[y == 0],
        bins=25,
        alpha=0.55,
        density=True,
        label="True negative",
    )
    plt.hist(
        ensemble_prob[y == 1],
        bins=25,
        alpha=0.55,
        density=True,
        label="True positive",
    )
    plt.axvline(0.5, linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted immunogenicity probability")
    plt.ylabel("Density")
    plt.title("External prediction probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTDIR / "external_probability_distribution.png",
        dpi=300,
    )
    plt.close()

    print("\nSaved results to:", OUTDIR)


if __name__ == "__main__":
    main()
