# External ESM-2 features

These arrays are mapped to the row order of:

`../MHC2_External_dataset_V2_prepared.csv`

Expected files:

- `ESM2_peptide_external_v2.npy`
- `ESM2_mhc_external_v2.npy`
- `Labels_external_v2.npy`
- `Sample_ids_external_v2.npy`
- `ESM2_external_row_manifest_v2.csv`

The external validation script checks alignment before prediction.

## Current packaging note

`ESM2_mhc_external_v2.npy` was **not present in the uploaded ZIP used to build this organized repository**. Add that file here before publishing the repository if you want external validation to run immediately after cloning.
