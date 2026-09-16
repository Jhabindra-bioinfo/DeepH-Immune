# DeepH-Immune

Reproducible code, model weights, training data, and external-validation resources for **DeepH-Immune**, a deep-learning framework for MHC class II peptide immunogenicity prediction.

## What is included

This repository is designed for two reproducibility workflows:

1. **Retrain DeepH-Immune from the released model-ready training table**
   - Training data are provided.
   - Precomputed training ESM-2 embeddings are intentionally **not** provided.
   - Users generate training ESM-2 features from the beginning with the supplied extraction script, then run five-fold training.

2. **Reproduce the external validation directly**
   - The five released fold models are provided.
   - The processed external-validation table is provided.
   - Precomputed external ESM-2 features are provided so users do **not** need to run ESM-2 again for external validation.

## Repository layout

```text
DeepH-Immune/
├── README.md
├── requirements_esm2.txt
├── requirements_tensorflow.txt
├── data/
│   ├── training/
│   │   └── Training_MHCII_V1_to_V2_12_15_clean.csv
│   └── external/
│       ├── MHC2_External_dataset_V2_prepared.csv
│       └── features/
│           ├── ESM2_peptide_external_v2.npy
│           ├── ESM2_mhc_external_v2.npy
│           ├── Labels_external_v2.npy
│           ├── Sample_ids_external_v2.npy
│           └── ESM2_external_row_manifest_v2.csv
├── src/
│   ├── deeph_immune.py
│   ├── extract_esm2.py
│   └── train_deeph_immune.py
├── models/
│   ├── DeepH_ImmuneV1_fold1.keras
│   ├── DeepH_ImmuneV1_fold2.keras
│   ├── DeepH_ImmuneV1_fold3.keras
│   ├── DeepH_ImmuneV1_fold4.keras
│   └── DeepH_ImmuneV1_fold5.keras
├── notebooks/
│   ├── 01_Generate_Training_ESM2_Features.ipynb
│   ├── 02_Train_DeepH_Immune.ipynb
│   └── 03_External_Validation.ipynb
├── external_validation/
│   └── run_external_validation.py
└── results/
    ├── cross_validation/
    └── external_validation/
```

## Model inputs

DeepH-Immune combines:

- masked peptide sequence encoding,
- masked MHC-II sequence encoding,
- peptide-to-MHC cross-attention,
- frozen ESM-2 peptide embeddings,
- frozen ESM-2 MHC embeddings,
- feature fusion followed by immunogenicity classification.

The released sequence encoder uses:

- maximum peptide tensor length: **25**
- MHC aligned length: **281**
- `PAD=0`
- `X` as an unknown biological residue rather than a padding token.

The released training table contains the study training cohort. The architecture can represent peptide inputs up to 25 residues through masking; predictive interpretation outside the training-length distribution should be made cautiously unless separately validated.

## Quick start

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DeepH-Immune
```

### A. Reproduce external validation directly

Install the TensorFlow-side dependencies:

```bash
pip install -r requirements_tensorflow.txt
```

Then run:

```bash
python external_validation/run_external_validation.py
```

The script:

- checks alignment of the external CSV, labels, sample IDs, peptide ESM features, and MHC ESM features;
- loads all five released fold models;
- predicts every external sample with each fold model;
- averages the five probabilities;
- reports per-fold and ensemble AUROC/AUPRC;
- saves sample-level predictions;
- saves ROC, precision-recall, and probability-distribution plots.

Results are written to:

```text
results/external_validation/
```

### B. Retrain DeepH-Immune

Training ESM-2 embeddings are **not distributed**. Generate them first.

Create/use a PyTorch + Transformers environment:

```bash
pip install -r requirements_esm2.txt
```

Generate the training embeddings:

```bash
python src/extract_esm2.py   --csv data/training/Training_MHCII_V1_to_V2_12_15_clean.csv   --outdir data/training/esm2_features
```

This downloads and uses:

```text
facebook/esm2_t33_650M_UR50D
```

and creates:

```text
data/training/esm2_features/
├── ESM2_peptide.npy
├── ESM2_mhc.npy
├── Labels.npy
├── Sample_ids.npy
└── ESM2_row_manifest.csv
```

Then use the TensorFlow environment:

```bash
pip install -r requirements_tensorflow.txt
```

Run five-fold training:

```bash
python src/train_deeph_immune.py   --csv data/training/Training_MHCII_V1_to_V2_12_15_clean.csv   --esm-dir data/training/esm2_features   --outdir results/cross_validation   --split-mode stratified_length   --folds 5   --epochs 80   --batch-size 64   --seed 42
```

The training script creates a **fresh model for every fold**.

## Jupyter workflow

Equivalent notebook entry points are provided:

1. `notebooks/01_Generate_Training_ESM2_Features.ipynb`
2. `notebooks/02_Train_DeepH_Immune.ipynb`
3. `notebooks/03_External_Validation.ipynb`

## Data-order integrity

The `.npy` arrays must remain in the exact same row order as the corresponding CSV.

External-validation alignment is checked automatically against:

- `sample_id`
- `Immunogenicity`
- feature-array dimensions.

Do not independently shuffle the CSV and `.npy` files.

## Deliberately omitted

This public reproducibility package intentionally does **not** include:

- upstream preprocessing notebooks/scripts;
- precomputed training ESM-2 features.

The released training CSV is the final model-ready input table.

## External-validation threshold note

AUROC and AUPRC are threshold-independent. The external-validation script uses a threshold of `0.5` only for descriptive classification/bias diagnostics; it does not optimize a threshold on the external cohort.

## Environment reproducibility

The uploaded bundle did not contain an exact `pip freeze` or Conda environment export. The included requirement files list the required packages but are not version-pinned. For archival reproducibility, adding the exact package versions from the environment used for the final reported run is recommended.

## License

No software license was included in the supplied study bundle. Add the license approved by your institution before public release.
