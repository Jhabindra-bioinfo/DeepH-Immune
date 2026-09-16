# Before publishing to GitHub

The repository has been organized to match the requested reproducibility design.

## Required before the external validation is fully runnable

- [ ] Add `data/external/features/ESM2_mhc_external_v2.npy`

This file was referenced by the external-validation workflow and visible in your server directory screenshots, but it was not included in the uploaded `Github_deepH-Immune.zip`.

## Strongly recommended before public release

- [ ] Add the exact package versions used in the final run (`pip freeze` or Conda YAML).
- [ ] Add the institution-approved software/data license.
- [ ] Replace `<YOUR_GITHUB_REPOSITORY_URL>` in `README.md`.
- [ ] Run `python external_validation/run_external_validation.py` once from a clean clone.
- [ ] Run the two training notebooks/scripts once from a clean clone if feasible.
