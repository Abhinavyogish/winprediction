# Models

This directory is reserved for trained model artifacts (for example, scikit-learn pipelines).

The current repository focuses on:
- parsing Cricsheet IPL CSVs into normalized Parquet tables under `data/processed/`
- building validated feature tables under `data/processed/` via `python -m src.features.build_features`

Recommended conventions:
- store versioned artifacts under subfolders (e.g. `models/v1/`)
- store any training metadata (feature list, metrics, data snapshot hashes) alongside the artifact

