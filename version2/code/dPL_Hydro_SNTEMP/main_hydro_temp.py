import sys

sys.path.append("../")
import torch
import os
from config.read_configurations import config_hydro_temp as config
from core.utils.randomseed_config import randomseed_config
from core.utils.small_codes import create_output_dirs
from MODELS.Differentiable_models import diff_hydro_temp_model
from MODELS import train_test


def main_hydro_temp(args):
    randomseed_config(seed=args["randomseed"][0])
    # Creating output directories and adding it to args
    args = create_output_dirs(args)

    if 0 in args["Action"]:  # training mode
        diff_model = diff_hydro_temp_model(args)
        optim = torch.optim.Adadelta(diff_model.parameters())
        train_test.train_differentiable_model(
            args=args,
            diff_model=diff_model,
            optim=optim
        )
    if 1 in args["Action"]:  # testing mode
        modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCH_testing"]) + ".pt")
        diff_model = torch.load(modelFile)
        train_test.test_differentiable_model(
            args=args,
            diff_model=diff_model
        )


if __name__ == "__main__":
    args = config
    main_hydro_temp(args)
    print("END")
