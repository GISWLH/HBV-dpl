# AGENTS.md

## Project Purpose

This workspace supports peer review of One Earth manuscript `ONE-EARTH-D-26-00944_R1`.

Main goals:

- Inspect the manuscript and extracted figures/text.
- Review author-provided differentiable hydrology code.
- Track Google Drive data availability and reproducibility gaps.
- Preserve review reports and acquisition logs.

## Directory Conventions

- `code/` contains downloaded source code only.
  - Main code path: `code/dPL_Hydro_SNTEMP/`
- `data/` contains downloaded manuscript data, archives, and manifests.
  - `data/inputs/`: key dPL model input arrays.
  - `data/results/`: key author model outputs and checkpoint.
  - `data/data/`: auxiliary downloaded data, currently partial.
  - `data/archive/`: original downloaded archives; do not delete unless explicitly asked.
  - `data/manifests/`: file inventories, found/missing lists, and download logs.
- `outputs/review/` contains review-facing artifacts.
  - Review report: `outputs/review/ONE-EARTH-D-26-00944_R1_review.md`
  - Infographic: `outputs/review/ONE-EARTH-D-26-00944_R1_infographic.html`
  - Extracted manuscript text: `outputs/review/ONE-EARTH-D-26-00944_R1.extracted.md`
  - Extracted figures: `outputs/review/ONE-EARTH-D-26-00944_R1.main_figures/`
- `outputs/acquisition/` contains acquisition notes and helper scripts.

Keep the root directory clean. New analysis outputs should normally go under `outputs/`; new downloaded data should go under `data/`.

## Data Status

The Google Drive listing has 333 files. Local data is incomplete.

Known complete local pieces:

- `data/inputs/` is complete.
- `data/results/` is complete for the key author dPL result bundle.
- `data/data/bf/` is complete.

Known incomplete local pieces:

- `data/data/GHMs_out_qr_2005soc/`
- `data/data/out_2003_np_qr_ISIMIP2b_GHMs/`
- `data/data/out_qr_isimip2a_gages_2003/`
- `data/data/qtot/`

Use `data/manifests/missing_after_remote.json` before claiming full data availability.

## Remote Host Notes

The user asked to use `yuyun-hk` for Google Drive downloads. Use the SSH alias `yuyun-hk`.

Remote cleanup already performed:

- `/root/oneearth_critical.tar.gz` deleted.
- `/root/oneearth_gd` deleted.

Do not assume remote temporary data still exists. If more data must be downloaded, recreate a short remote path such as `/root/oneearth_gd` and watch free disk space first with `df -h /root`.

## Reproducibility Notes

The code config contains hard-coded original paths, for example:

- `D:\data_attr_inputs2003\f1223_1023_daymet_20240826.npy`
- `D:\data_attr_inputs2003\attr1223_1023_daymet_20240826.npy`
- `D:\tmp_dirs`

Before running code, patch a copy of the config to use this workspace:

- `data/inputs/f1223_1023_daymet_20240826.npy`
- `data/inputs/attr1223_1023_daymet_20240826.npy`

Prefer a small smoke test before attempting full model reproduction.

## Safety Rules

- Do not delete `data/archive/oneearth_critical.tar.gz` or `data/archive/dPL_Hydro_SNTEMP_DR-93F0.zip` unless explicitly requested.
- Do not delete the original PDF.
- Do not overwrite review reports without preserving the previous version or confirming intent.
- Treat incomplete data as incomplete; do not report that the full Google Drive data package has been downloaded.
- Avoid moving files back into temporary names such as `review_materials/` or `gd/`.

## Useful Files

- `README.md`: human-readable project summary.
- `data/manifests/gdrive_filelist.json`: full Google Drive file listing.
- `data/manifests/found_after_remote.json`: files found locally after remote transfer.
- `data/manifests/missing_after_remote.json`: files still missing locally.
- `outputs/acquisition/acquisition_log.md`: acquisition history and reproducibility notes.

