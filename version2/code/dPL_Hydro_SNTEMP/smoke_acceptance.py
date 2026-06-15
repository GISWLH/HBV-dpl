"""Local smoke acceptance test for the copied OneEarth dPL code/data.

This does not attempt a full 100-epoch reproduction. It verifies that the
copied input arrays, config assumptions, data loader, model construction, and a
small forward pass are usable from this workspace.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

from core.load_data.dataFrame_loading import loadData
from core.load_data.data_prep import sub_Nans_for_mean, take_sample_test
from core.load_data.normalizing import init_norm_stats, transNorm
from core.utils.small_codes import create_output_dirs
from MODELS.Differentiable_models import diff_hydro_temp_model


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "inputs"
SMOKE_DIR = ROOT / "data" / "smoke_inputs"
REPORT_PATH = PROJECT_ROOT / "smoke_acceptance_report.json"


def make_smoke_inputs(n_basins: int = 2, n_days: int = 760) -> None:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    forcing_src = INPUT_DIR / "f1223_1023_daymet_20240826.npy"
    attr_src = INPUT_DIR / "attr1223_1023_daymet_20240826.npy"
    forcing_dst = SMOKE_DIR / forcing_src.name
    attr_dst = SMOKE_DIR / attr_src.name

    if not forcing_dst.exists():
        forcing = np.load(forcing_src, mmap_mode="r")[:n_basins, :n_days, :]
        np.save(forcing_dst, forcing)
    if not attr_dst.exists():
        attr = np.load(attr_src, mmap_mode="r")[:n_basins, :]
        np.save(attr_dst, attr)

    for src in INPUT_DIR.glob("*_name.json"):
        dst = SMOKE_DIR / src.name
        if not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def load_local_args() -> dict:
    cfg = YAML(typ="safe").load((PROJECT_ROOT / "config" / "config_hydro_temp.yaml").read_text())
    args = copy.deepcopy(cfg)
    args.update(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "forcing_path": str(SMOKE_DIR / "f1223_1023_daymet_20240826.npy"),
            "attr_path": str(SMOKE_DIR / "attr1223_1023_daymet_20240826.npy"),
            "output_model": str(ROOT / "outputs" / "smoke_acceptance"),
            "Action": [1],
            "tRange": [19800101, 19820130],
            "t_train": [19800101, 19820130],
            "t_test": [19800101, 19820130],
            "warm_up": 30,
            "rho": 7,
            "batch_size": 2,
            "no_basins": 2,
            "EPOCHS": 1,
            "EPOCH_testing": 1,
            "hidden_size": 16,
            "dropout": 0.0,
            "nmul": 2,
            "saveEpoch": 1,
            "loss_function": "RmseLoss_flow_comb",
            "target": ["00060_Mean"],
            "temp_model_name": "None",
            "routing_temp_model": False,
            "lat_temp_adj": False,
            "varT_temp_model": [],
            "varC_temp_model": [],
            "dyn_params_list_hydro": [],
            "dyn_params_list_temp": [],
            "static_params_list_prms": [],
            "semi_static_params_list_prms": [],
            "static_params_list": [],
            "semi_static_params_list": [],
            "static_params_list_SNTEMP": [],
            "semi_static_params_list_SNTEMP": [],
        }
    )
    return args


def prepare_sample(args: dict) -> dict:
    dataset = loadData(args, trange=args["t_test"])
    dataset = sub_Nans_for_mean(dataset)
    init_norm_stats(args, dataset["x_NN"], dataset["c_NN"], dataset["obs"])

    x_nn_scaled = transNorm(args, dataset["x_NN"], varLst=args["varT_NN"], toNorm=True)
    c_nn_scaled = transNorm(args, dataset["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
    dataset["inputs_NN_scaled"] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)

    for key, value in list(dataset.items()):
        if isinstance(value, np.ndarray):
            dataset[key] = torch.from_numpy(value).float()

    return take_sample_test(args, dataset, 0, args["no_basins"])


def main() -> None:
    make_smoke_inputs()
    args = load_local_args()
    args = create_output_dirs(args)
    report = {
        "workspace": str(ROOT),
        "code_path": str(PROJECT_ROOT),
        "forcing_path": args["forcing_path"],
        "attr_path": args["attr_path"],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args["device"],
    }

    forcing = np.load(args["forcing_path"], mmap_mode="r")
    attr = np.load(args["attr_path"], mmap_mode="r")
    report["smoke_forcing_shape"] = list(forcing.shape)
    report["smoke_attr_shape"] = list(attr.shape)

    sample = prepare_sample(args)
    report["sample_shapes"] = {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")}

    model = diff_hydro_temp_model(args).to(args["device"])
    model.eval()
    with torch.no_grad():
        out = model(sample)

    report["forward_output_shapes"] = {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")}
    report["status"] = "pass"
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
