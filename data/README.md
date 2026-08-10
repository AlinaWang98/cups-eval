# Data Metadata

This directory is metadata-only. It documents the datasets used by the project
without redistributing dataset payloads.

Included files:

- `datasets_manifest.json`: consolidated dataset inventory with Hugging Face IDs,
  configs, split sizes, feature names, and conventional local paths.
- `datasets/**/dataset_dict.json`: Hugging Face `DatasetDict` metadata.
- `datasets/**/dataset_info.json`: split-level schema and dataset metadata.
- `datasets/**/state.json`: Hugging Face saved-dataset state metadata.
- `datasets/download_manifest.reasoning_nonmedical.json`: non-medical pilot
  download manifest.

Excluded files:

- `*.arrow`
- `*.parquet`
- `*.jsonl`
- `hf_cache/`

The experiment loader first checks whether `local_path` contains a real dataset
payload. If only metadata is present, it falls back to Hugging Face
`load_dataset(...)`. To run fully offline, download the datasets yourself and
place them at the paths declared in the configs, for example
`data/datasets/openlifescienceai__medmcqa/default`.

Dataset access remains governed by the upstream dataset licenses and terms.
