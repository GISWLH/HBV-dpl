# Version 2 Acceptance Notes

Source copied from `D:\OneDrive\Acdemic\review\OneEarth`.

Copied content:

- `code/dPL_Hydro_SNTEMP/`
- `data/inputs/`
- `data/results/`
- `data/data/`
- `data/archive/`
- `data/manifests/`
- root `AGENTS.md`, `README.md`, and `package.json`

Large review-only artifacts under `outputs/` and manuscript PDFs/docx review files were not copied into this code/data version folder.

## Smoke Test

Run from:

```powershell
cd D:\OneDrive\GitHub\HBV-dpl\version2\code\dPL_Hydro_SNTEMP
python smoke_acceptance.py
```

What it verifies:

- local copied input arrays can be read;
- the author data loader can load a small local subset;
- normalization statistics can be initialized;
- PyTorch/CUDA and the author LSTM wrapper can run;
- the differentiable `marrmot_PRMS` hydrology model completes one small forward pass.

Latest result:

- status: `pass`
- report: `code/dPL_Hydro_SNTEMP/smoke_acceptance_report.json`
- tested model path: hydrology-only smoke path using `marrmot_PRMS`

The full coupled SNTEMP temperature workflow remains non-turnkey: the upstream config has hard-coded machine paths and the temperature model expects longer historical windows than the small smoke subset. The original code/data are preserved for deeper reproduction work.
