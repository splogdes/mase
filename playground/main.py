from chop.models.nerf import get_nerf_model,get_nerf_model_info
from chop.dataset.nerf import get_nerf_dataset
from chop.dataset import get_dataset_info
from pathlib import Path
import torch

from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.logger import set_logging_verbosity

import toml
import torch
import torch.nn as nn
import os

torch.manual_seed(0)
set_logging_verbosity("debug")

datapath = f'{Path.cwd()}/../data/data'

print(f"DataPath: {datapath}")

dataset = get_nerf_dataset(name="nerf-lego", split="train", path=Path(datapath))
dataset_info = get_dataset_info("nerf-lego")
model = get_nerf_model("nerf", dataset_info=dataset_info)

mgn = MaseGraph(model=model)

print("Done !")