# One Earth Manuscript Review Workspace

This workspace contains review materials for manuscript `ONE-EARTH-D-26-00944_R1`, titled **"Groundwater Recharge Is More Climate Sensitive than Previously Indicated"**.

## Directory Layout

```text
.
├── ONE-EARTH-D-26-00944_R1.pdf          # Original manuscript PDF
├── code/
│   └── dPL_Hydro_SNTEMP/                # Downloaded differentiable hydrology + SNTEMP code
├── data/
│   ├── inputs/                          # Key model input arrays from Google Drive
│   ├── results/                         # Key author model outputs/checkpoint from Google Drive
│   ├── data/                            # Partial auxiliary/GHM/baseflow data from Google Drive
│   ├── archive/                         # Original downloaded archives
│   └── manifests/                       # Download logs, file lists, missing-file inventories
└── outputs/
    ├── review/                          # Review report, extracted manuscript text, figures, infographic
    └── acquisition/                     # Code/data acquisition notes and helper script
```

## Code

The author-provided code was downloaded from the anonymous 4open repository.

- Local path: `code/dPL_Hydro_SNTEMP/`
- Archive: `data/archive/dPL_Hydro_SNTEMP_DR-93F0.zip`
- Size: about 1.8 MB, 143 files

Important files:

- `code/dPL_Hydro_SNTEMP/main_hydro_temp.py`
- `code/dPL_Hydro_SNTEMP/config/config_hydro_temp.yaml`
- `code/dPL_Hydro_SNTEMP/MODELS/Differentiable_models.py`
- `code/dPL_Hydro_SNTEMP/MODELS/train_test.py`
- `code/dPL_Hydro_SNTEMP/MODELS/hydro_models/`
- `code/dPL_Hydro_SNTEMP/MODELS/temp_models/SNTEMP/`

Reproducibility note: the example config contains machine-specific Windows paths such as `D:\data_attr_inputs2003\...`; these must be changed to the local `data/inputs/` paths before running.

## Data Status

The Google Drive folder listing contains **333 files**. The full data package has not been fully downloaded.

Downloaded locally:

| Category | Local path | Status |
|---|---|---|
| Model inputs | `data/inputs/` | Complete, 4/4 files |
| Key author results | `data/results/` | Complete, 18/18 files |
| Baseflow data | `data/data/bf/` | Complete, 30/30 files |
| Partial GHM data | `data/data/GHMs_out_qr_2005soc/` | Partial |
| Full Drive manifest | `data/manifests/gdrive_filelist.json` | Complete |
| Current found-file list | `data/manifests/found_after_organization.json` | 59/333 files |
| Current missing-file list | `data/manifests/missing_after_organization.json` | 274/333 files |

Approximate local data sizes:

- `data/inputs/`: 2.29 GB
- `data/results/`: 2.49 GB
- `data/data/`: 0.42 GB
- `data/archive/`: 2.85 GB

The key input/result bundle was downloaded on `yuyun-hk`, compressed, copied back, and unpacked locally. The remote temporary files were deleted after transfer.

## Review Outputs

Main review artifacts:

- `outputs/review/ONE-EARTH-D-26-00944_R1_review.md`
- `outputs/review/ONE-EARTH-D-26-00944_R1_infographic.html`
- `outputs/review/ONE-EARTH-D-26-00944_R1.extracted.md`
- `outputs/review/ONE-EARTH-D-26-00944_R1.main_figures/`

Acquisition and reproducibility logs:

- `outputs/acquisition/acquisition_log.md`
- `outputs/acquisition/download_oneearth_remote.py`
- `data/manifests/found_after_remote.json`
- `data/manifests/missing_after_remote.json`
- `data/manifests/found_after_organization.json`
- `data/manifests/missing_after_organization.json`

## Current Review-Relevant Findings

- Code is available and contains a plausible implementation of the differentiable hydrology + stream-temperature framework.
- Key model inputs and author result files were obtainable only after using `yuyun-hk`; direct local Google Drive downloads were unreliable.
- The full Google Drive data package remains incomplete locally.
- The manuscript/code package lacks a turnkey reproducibility workflow because paths are machine-specific and the data archive is not packaged with checksums or a robust downloader.

## Suggested Next Steps

1. If full GHM comparison reproduction is needed, download the missing files listed in `data/manifests/missing_after_remote.json`.
2. Patch `code/dPL_Hydro_SNTEMP/config/config_hydro_temp.yaml` to point at `data/inputs/`.
3. Create a small reproducibility smoke test before attempting full model runs.
4. Keep `data/archive/oneearth_critical.tar.gz` until the unpacked `inputs/` and `results/` are confirmed sufficient.
