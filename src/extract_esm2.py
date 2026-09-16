#!/usr/bin/env python3
"""Frozen ESM-2 feature extraction for DeepH-Immune.

This script generates the ESM-2 features required for model training.

Important:
- Training ESM features are intentionally NOT distributed in this repository.
- Peptides are embedded as their biological sequence (no '*' padding).
- MHC sequences are embedded from MHC_ungapped (no alignment gaps).
- Mean pooling excludes BOS/EOS and batch-padding tokens.

Model:
    facebook/esm2_t33_650M_UR50D
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ALLOWED = set("ACDEFGHIKLMNPQRSTVWYX")


def residue_mean_pool(last_hidden_state, attention_mask, special_tokens_mask):
    residue_mask = attention_mask.bool() & (~special_tokens_mask.bool())
    weights = residue_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def embed_unique_sequences(
    sequences, tokenizer, model, device, batch_size=32, desc="ESM-2"
):
    unique = list(dict.fromkeys(sequences))
    embedding_map = {}

    for start in tqdm(range(0, len(unique), batch_size), desc=desc):
        batch = unique[start:start + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )

        special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = residue_mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                special_tokens_mask,
            )

        pooled = pooled.detach().float().cpu().numpy()

        for seq, emb in zip(batch, pooled):
            embedding_map[seq] = emb

    return embedding_map


def validate_sequence_column(series, column_name):
    bad = []
    for i, seq in enumerate(series.astype(str)):
        invalid = set(seq.upper()) - ALLOWED
        if invalid:
            bad.append((i, seq, sorted(invalid)))
    if bad:
        preview = "\n".join(
            f"row={i}, invalid={inv}, seq={seq}"
            for i, seq, inv in bad[:10]
        )
        raise ValueError(
            f"{column_name} contains unsupported residues.\n{preview}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/training/Training_MHCII_V1_to_V2_12_15_clean.csv",
    )
    parser.add_argument(
        "--outdir",
        default="data/training/esm2_features",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--peptide-batch-size", type=int, default=128)
    parser.add_argument("--mhc-batch-size", type=int, default=16)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {
        "sample_id",
        "Peptide",
        "MHC_ungapped",
        "Immunogenicity",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id is not unique")

    validate_sequence_column(df["Peptide"], "Peptide")
    validate_sequence_column(df["MHC_ungapped"], "MHC_ungapped")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Rows:", len(df))
    print("Unique peptides:", df["Peptide"].nunique())
    print("Unique MHC sequences:", df["MHC_ungapped"].nunique())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False)
    model.eval().to(device)

    pep_map = embed_unique_sequences(
        df["Peptide"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        batch_size=args.peptide_batch_size,
        desc="Unique peptide embeddings",
    )

    mhc_map = embed_unique_sequences(
        df["MHC_ungapped"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        batch_size=args.mhc_batch_size,
        desc="Unique MHC embeddings",
    )

    esm_peptide = np.stack(
        [pep_map[s] for s in df["Peptide"].astype(str)]
    ).astype(np.float32)

    esm_mhc = np.stack(
        [mhc_map[s] for s in df["MHC_ungapped"].astype(str)]
    ).astype(np.float32)

    labels = df["Immunogenicity"].to_numpy(dtype=np.int32)
    sample_ids = df["sample_id"].astype(str).to_numpy(dtype=str)

    assert esm_peptide.shape == (len(df), 1280)
    assert esm_mhc.shape == (len(df), 1280)

    np.save(outdir / "ESM2_peptide.npy", esm_peptide)
    np.save(outdir / "ESM2_mhc.npy", esm_mhc)
    np.save(outdir / "Labels.npy", labels)
    np.save(outdir / "Sample_ids.npy", sample_ids)

    pd.DataFrame({
        "sample_id": sample_ids,
        "row_index": np.arange(len(df), dtype=int),
    }).to_csv(outdir / "ESM2_row_manifest.csv", index=False)

    print("Saved training ESM-2 features to:", outdir.resolve())


if __name__ == "__main__":
    main()
