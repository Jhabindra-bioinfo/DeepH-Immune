# Training data

`Training_MHCII_V1_to_V2_12_15_clean.csv` is the final model-ready training table supplied for this release.


Precomputed training ESM-2 features can be Generate  using:

```bash
python src/extract_esm2.py   --csv data/training/Training_MHCII_V1_to_V2_12_15_clean.csv   --outdir data/training/esm2_features
```
