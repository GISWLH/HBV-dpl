<p align="center">
  <img src="figures/hbv_dpl_logo_header.png" alt="HBV-dPL project header" width="720">
</p>

# HBV-dPL

Differentiable hydrology model notes and reproducibility workspace for two HBV/dPL-style implementations:

- `version1/`: a compact differentiable HBV rainfall-runoff model.
- `version2/`: a differentiable parameter learning hydrology and stream-temperature model copied from the OneEarth review workspace and smoke-tested locally.

The repository is organized for code review, model comparison, and lightweight acceptance testing before deeper reproduction work.

## Repository Layout

```text
HBV-dpl/
├── figures/                         # README images, image prompts, and API response logs
├── version1/                        # Differentiable HBV implementation
│   ├── dHBV.py
│   └── CAMELS_GB_hydromet_timeseries_67025_19701001-20150930.csv
└── version2/                        # Copied OneEarth dPL code/data workspace
    ├── ACCEPTANCE.md
    ├── code/dPL_Hydro_SNTEMP/
    └── data/
```

## Model Architectures

### Version 1: Differentiable HBV

![Version 1 differentiable HBV architecture](figures/version1_dhbv_architecture.png)

Version 1 wraps a classical HBV rainfall-runoff structure as a differentiable PyTorch recurrent layer.

Forward path:

1. Daily forcing inputs enter the model as temperature, precipitation, and PET.
2. `HBVCell` maps normalized learnable parameters into physical ranges.
3. The cell updates snow, meltwater, soil moisture, upper groundwater, and lower groundwater states through time.
4. The runoff components are summed into simulated discharge, `Qsim`.
5. `Qsim` is compared with observed discharge through an NSE-based loss.

Backward path:

The NSE loss is backpropagated through the full time-unrolled HBV recurrence. Gradients update the learnable HBV parameters directly, so the physical model itself is the trainable layer.

Key file:

- `version1/dHBV.py`

### Version 2: dPL Hydro + SNTEMP

![Version 2 dPL Hydro SNTEMP architecture](figures/version2_dpl_architecture.png)

Version 2 uses differentiable parameter learning: a neural network generates physical model parameters, and the process model remains inside the computational graph.

Forward path:

1. Dynamic forcings and basin attributes are loaded from NumPy input arrays.
2. An LSTM or MLP generates hydrology and temperature parameter tensors.
3. Sigmoid transforms and physical bounds convert neural outputs into process-model parameters.
4. The hydrology module, typically `marrmot_PRMS`, simulates flow components, storage states, PET/AET terms, BFI, and routed streamflow.
5. When enabled, SNTEMP uses hydrologic source flows plus radiation, shade, vapor pressure, channel geometry, and groundwater effects to simulate stream temperature.
6. Multi-target losses can compare simulated flow, stream temperature, BFI, SWE, or AET against observations.

Backward path:

The loss backpropagates through the differentiable process modules and into the LSTM/MLP parameter generator. This trains the neural network to emit physically bounded parameters rather than directly predicting discharge only.

Key files:

- `version2/code/dPL_Hydro_SNTEMP/main_hydro_temp.py`
- `version2/code/dPL_Hydro_SNTEMP/MODELS/Differentiable_models.py`
- `version2/code/dPL_Hydro_SNTEMP/MODELS/hydro_models/marrmot_PRMS/prms_marrmot.py`
- `version2/code/dPL_Hydro_SNTEMP/MODELS/temp_models/SNTEMP/SNTEMP.py`

## Acceptance Test

Version 2 includes a local smoke acceptance script:

```powershell
cd D:\OneDrive\GitHub\HBV-dpl\version2\code\dPL_Hydro_SNTEMP
python smoke_acceptance.py
```

The latest smoke test passed for:

- copied input-array loading;
- author data-loader execution;
- CUDA/PyTorch runtime;
- LSTM parameter generation;
- a small `marrmot_PRMS` hydrology forward pass.

Acceptance artifacts:

- `version2/ACCEPTANCE.md`
- `version2/code/dPL_Hydro_SNTEMP/smoke_acceptance.py`
- `version2/code/dPL_Hydro_SNTEMP/smoke_acceptance_report.json`

## Figures And Tools

The README graphics were generated from editable prompts:

- header logo: Poe `GPT-Image-2`;
- model architecture figures: GetGoAPI `gpt-image-2`.

Generated assets:

- `figures/hbv_dpl_logo_header.png`
- `figures/version1_dhbv_architecture.png`
- `figures/version2_dpl_architecture.png`

Editable/reproducible prompt files:

- `figures/hbv_dpl_logo_header_prompt.txt`
- `figures/version1_dhbv_prompt.txt`
- `figures/version2_dpl_prompt.txt`

Generation response logs are stored next to the images as `*.response.json`.

## Notes Before Publishing

- `version2/data/` contains large copied data artifacts from the OneEarth review workspace.
- Files larger than GitHub LFS per-file limits are stored as split parts with reconstruction notes in `version2/data/archive/README.md` and `version2/data/inputs/README.md`.
- The full coupled SNTEMP workflow is not yet a turnkey reproduction path because the upstream config contains machine-specific paths and long-window assumptions.
- The included smoke test is an acceptance check for copied code/data and the hydrology forward path, not a full manuscript reproduction.
