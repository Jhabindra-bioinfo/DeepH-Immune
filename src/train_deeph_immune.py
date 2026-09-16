#!/usr/bin/env python3
"""Five-fold DeepH-Immune training.

The released repository intentionally excludes precomputed training ESM-2
features. Generate them first with src/extract_esm2.py.

Every fold creates a fresh model and optimizer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from deeph_immune import (
    PEP_MAX_LEN,
    MHC_ALIGNED_LEN,
    build_deeph_immune_v2,
    tokenize_mhc_aligned,
    tokenize_peptide,
)


def length_bin(lengths):
    lengths = np.asarray(lengths, dtype=int)
    out = np.empty(lengths.shape, dtype=object)
    out[lengths <= 11] = "9-11"
    out[(lengths >= 12) & (lengths <= 15)] = "12-15"
    out[(lengths >= 16) & (lengths <= 20)] = "16-20"
    out[lengths >= 21] = "21-25"
    return out


def make_strata(y, lengths):
    bins = length_bin(lengths)
    return np.asarray([f"y{int(label)}_{b}" for label, b in zip(y, bins)])


def length_label_weights(y_train, lengths_train, min_w=0.5, max_w=3.0):
    strata = make_strata(y_train, lengths_train)
    counts = Counter(strata)
    median_count = float(np.median(list(counts.values())))
    weights = np.asarray([
        np.sqrt(median_count / counts[s]) for s in strata
    ], dtype=np.float32)
    weights = np.clip(weights, min_w, max_w)
    weights /= weights.mean()
    return weights


def load_inputs(csv_path: Path, esm_dir: Path):
    df = pd.read_csv(csv_path)

    required = {
        "sample_id",
        "MHC_allele",
        "Peptide",
        "Immunogenicity",
        "Peptide_length",
        "MHC_aligned_281",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    pep_tokens = np.asarray(
        [tokenize_peptide(s) for s in df["Peptide"]],
        dtype=np.int32,
    )

    mhc_tokens = np.asarray(
        [tokenize_mhc_aligned(s) for s in df["MHC_aligned_281"]],
        dtype=np.int32,
    )

    y = df["Immunogenicity"].to_numpy(dtype=np.int32)

    esm_pep = np.load(esm_dir / "ESM2_peptide.npy").astype(np.float32)
    esm_mhc = np.load(esm_dir / "ESM2_mhc.npy").astype(np.float32)
    esm_labels = np.load(esm_dir / "Labels.npy").astype(np.int32)
    esm_ids = np.load(
        esm_dir / "Sample_ids.npy",
        allow_pickle=True,
    ).astype(str)

    n = len(df)

    assert pep_tokens.shape == (n, PEP_MAX_LEN)
    assert mhc_tokens.shape == (n, MHC_ALIGNED_LEN)
    assert esm_pep.shape == (n, 1280)
    assert esm_mhc.shape == (n, 1280)

    if not np.array_equal(y, esm_labels):
        raise ValueError(
            "Labels in CSV and ESM feature directory are not in the same order"
        )

    if not np.array_equal(
        df["sample_id"].astype(str).to_numpy(),
        esm_ids,
    ):
        raise ValueError(
            "sample_id order mismatch between CSV and ESM features"
        )

    return df, pep_tokens, mhc_tokens, esm_pep, esm_mhc, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/training/Training_MHCII_V1_to_V2_12_15_clean.csv",
    )
    parser.add_argument(
        "--esm-dir",
        default="data/training/esm2_features",
    )
    parser.add_argument(
        "--outdir",
        default="results/cross_validation",
    )
    parser.add_argument(
        "--split-mode",
        choices=["stratified_length", "peptide_group"],
        default="stratified_length",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-length-weights", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    esm_dir = Path(args.esm_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)

    df, pep_tokens, mhc_tokens, esm_pep, esm_mhc, y = load_inputs(
        csv_path, esm_dir
    )

    lengths = df["Peptide_length"].to_numpy(dtype=int)
    strata = make_strata(y, lengths)

    print("Samples:", len(df))
    print("Labels:", Counter(y))
    print("Length-bin x label strata:", Counter(strata))
    print("Split mode:", args.split_mode)

    if args.split_mode == "stratified_length":
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        split_iter = splitter.split(np.zeros(len(df)), strata)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        split_iter = splitter.split(
            np.zeros(len(df)),
            strata,
            groups=df["Peptide"].astype(str).to_numpy(),
        )

    fold_rows = []
    pred_rows = []

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):
        print(f"\n========== Fold {fold}/{args.folds} ==========")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold)

        model, _attention_model = build_deeph_immune_v2()

        X_train = [
            pep_tokens[train_idx],
            mhc_tokens[train_idx],
            esm_pep[train_idx],
            esm_mhc[train_idx],
        ]

        X_val = [
            pep_tokens[val_idx],
            mhc_tokens[val_idx],
            esm_pep[val_idx],
            esm_mhc[val_idx],
        ]

        y_train = y[train_idx]
        y_val = y[val_idx]

        sample_weight = (
            None
            if args.no_length_weights
            else length_label_weights(y_train, lengths[train_idx])
        )

        model_path = outdir / f"DeepH_ImmuneV1_fold{fold}.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auprc",
                mode="max",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auprc",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_auprc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            sample_weight=sample_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )

        val_pred = model.predict(
            X_val,
            batch_size=args.batch_size,
            verbose=0,
        ).reshape(-1)

        auroc = roc_auc_score(y_val, val_pred)
        auprc = average_precision_score(y_val, val_pred)

        print(
            f"Fold {fold}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
        )

        fold_rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "auroc": auroc,
            "auprc": auprc,
            "best_epoch": int(
                np.argmax(history.history["val_auprc"]) + 1
            ),
            "model_path": str(model_path),
        })

        for local_i, global_i in enumerate(val_idx):
            pred_rows.append({
                "sample_id": df.iloc[global_i]["sample_id"],
                "fold": fold,
                "MHC_allele": df.iloc[global_i]["MHC_allele"],
                "Peptide": df.iloc[global_i]["Peptide"],
                "Peptide_length": int(
                    df.iloc[global_i]["Peptide_length"]
                ),
                "y_true": int(y_val[local_i]),
                "y_score": float(val_pred[local_i]),
            })

        pd.DataFrame(history.history).to_csv(
            outdir / f"history_fold{fold}.csv",
            index=False,
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows).sort_values("sample_id")

    fold_df.to_csv(
        outdir / "cv_fold_metrics.csv",
        index=False,
    )

    pred_df.to_csv(
        outdir / "cv_out_of_fold_predictions.csv",
        index=False,
    )

    summary = {
        "n": len(df),
        "split_mode": args.split_mode,
        "folds": args.folds,
        "length_aware_weights": not args.no_length_weights,
        "mean_auroc": float(fold_df["auroc"].mean()),
        "std_auroc": float(fold_df["auroc"].std(ddof=0)),
        "mean_auprc": float(fold_df["auprc"].mean()),
        "std_auprc": float(fold_df["auprc"].std(ddof=0)),
    }

    with (outdir / "cv_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nFinal CV summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
