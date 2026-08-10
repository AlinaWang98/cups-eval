# Data Availability Notes

This GitHub release is code-first. It includes dataset metadata and compact derived tables, but not full dataset payloads or per-item model outputs.

## Public Benchmark Inputs

The benchmark inputs are loaded from their upstream sources, including Hugging Face datasets listed in `data/datasets_manifest.json` and `configs/datasets/*.yaml`. Access and reuse are governed by the upstream licenses and terms.

## Derived Data For A Data Repository

The following artifacts should be deposited in Zenodo, Mendeley Data, Figshare, OSF, or an institutional repository:

- Per-item predictions by dataset, model, and strategy.
- Raw or normalized model outputs needed to audit answer extraction.
- `router_features.csv` files for all runs used in CUPS/router/mechanism analyses.
- Run manifests that map compact paper tables to exact run directories.
- Any large generated artifacts not suitable for GitHub.

After deposit, cite the repository DOI in the manuscript Data Availability statement and add the DOI/URL to this file.

## Licensing Note

The authors should select and add a formal software license before public release. This repository does not choose a license on their behalf.

## Why Not Put Everything In GitHub?

GitHub is best for code, configuration, compact tables, and reproduction instructions. Full per-item outputs and model response logs are research data: they can be large, may require clearer licensing/provenance metadata, and should receive a stable DOI.
