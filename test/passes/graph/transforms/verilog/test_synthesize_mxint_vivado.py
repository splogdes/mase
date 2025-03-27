import os, re, random
import optuna
from optuna import study
from optuna.samplers import TPESampler, GridSampler, RandomSampler
import json
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger
from test_emit_verilog_mxint import MLP, shared_emit_verilog_mxint
import os, sys, logging, traceback, pdb
import pytest
import toml
import torch
import torch.nn as nn
import chop as chop
import chop.passes as passes
from pathlib import Path
from chop.actions import simulate
from chop.passes.graph.analysis.report.report_node import report_node_type_analysis_pass
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger

config_file = "config.tcl"

set_logging_verbosity("debug")

logger = get_logger(__name__)


def dump_param(trial_number, quan_args, filename="output.json"):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data[str(trial_number)] = quan_args

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def write_value(trial_number, name, value, filename="output.json"):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    if str(trial_number) in data.keys():
        data[str(trial_number)][name] = value
    else:
        data[str(trial_number)] = {name: value}

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

ss = {
    "block_size": [2 ** i for i in range(1, 5)],
    "batch_parallelism": [2 ** i for i in range(1, 5)],
    "m_width": [i for i in range(4, 11)],
    "e_width": [i for i in range(3, 11)],
    "batches": [32],
    "num_batches": [1],
    "seed": [i for i in range(10)]
}

def get_params(trial):

    block_size = 2 ** trial.suggest_int("block_size", 1, 4)
    batch_parallelism = 2 ** trial.suggest_int("batch_parallelism", 1, 4)
    mlp_depth = 3
    mlp_features = [32 for i in range(mlp_depth + 1)]

    params = {
        "seed": trial.number,
        "block_size": block_size,
        "batch_parallelism": batch_parallelism,
        "m_width": (m_width := trial.suggest_int("m_width", 4, 10)),
        "e_width": trial.suggest_int("e_width", 3, min(m_width - 1, 10)),
        "batches": 32,
        "num_batches": 1,
    }

    mlp = MLP(mlp_features)
    input_shape = (mlp_features[0],)

    logger.info(
        f"{block_size=}, {batch_parallelism=}, {params['e_width']=}, {params['m_width']=}, {params['batches']=}"
    )

    mg, mlp = shared_emit_verilog_mxint(mlp, input_shape, params)

    return params, mg, mlp


def writeTrialNumber(trial_number):
    with open(config_file, "w") as f:
        f.write(f"set trial_number {trial_number}\n")
        f.write(f"set top_dir {Path.home()}/.mase/top/\n")
        f.write(f"set mase_dir {Path.cwd()}/")


def extract_site_type_used_util(filename):
    site_data = {}
    with open(filename, "r") as file:
        lines = file.readlines()

    pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|.*?\|\s*(\d+\.\d+)\s*\|")

    for line in lines:
        match = pattern.match(line)
        if match:
            site_type = match.group(1).strip()
            used = int(match.group(2).strip())
            util = float(match.group(3).strip())
            site_data[site_type] = {"Used": used, "Util%": util}

    return site_data


def get_bram_uram_util(filename):
    site_data = extract_site_type_used_util(filename)
    bram_util = site_data.get("Block RAM Tile", {}).get("Util%", 0.0)
    uram_util = site_data.get("URAM", {}).get("Util%", 0.0)
    return {"bram": bram_util, "uram": uram_util}


def getResources(trial):
    params, mg, mlp = get_params(trial)
    dump_param(trial.number, params)
    writeTrialNumber(trial.number)
    # os.system(
        # f"vivado -mode batch -nolog -nojou -source {Path.cwd()}/test/passes/graph/transforms/verilog/generate.tcl"
    # )
    bram_utils = get_bram_uram_util(f"{Path.cwd()}/resources/util_{trial.number}.txt")
    clb_luts = extract_site_type_used_util(
        f"{Path.cwd()}/resources/util_{trial.number}.txt"
    )
    out = (
        clb_luts["CLB LUTs*"]["Util%"]
        + clb_luts["CLB Registers"]["Util%"]
        + clb_luts["CARRY8"]["Util%"]
        + bram_utils["bram"]
        + bram_utils["uram"]
    )
    write_value(trial.number, "resource_score", out)
    return out


def getAccuracy(trial):
    params, mg, mlp = get_params(trial)
    quantized = mg.model

    criterion = nn.MSELoss()
    total_mse = 0.0

    for _ in range(100):
        x = torch.randn(params["batches"], mg.model[0].in_features)
        y1 = quantized(x)
        y2 = mlp(x)
        mse = criterion(y1, y2)
        total_mse += mse.item()

    avg_mse = total_mse / 100

    write_value(trial.number, "avg_mse", avg_mse)
    return avg_mse


def main():
    sampler = RandomSampler()

    study = optuna.create_study(
        direction="minimize",
        study_name="resource_accuracy_optimiser",
        sampler=sampler,
    )

    study.optimize(
        getAccuracy,
        n_trials=100,
        timeout=60 * 60 * 24,
        n_jobs=1,
    )

    print("Best trials:")

    for trial in study.best_trials:
        print(f"Trial {trial.number}: {trial.values}")

if __name__ == "__main__":

    try:
        os.mkdir(f"{Path.cwd()}/resources/")
    except:
        pass

    with open("thor.csv", "w") as f:
        f.write("seed,block_size,batch_parallelism,m_width,e_width,acc,time\n")

    main()
