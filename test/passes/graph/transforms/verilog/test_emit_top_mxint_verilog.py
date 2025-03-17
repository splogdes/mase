#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import random
import os, sys, logging, traceback, pdb
from chop.passes.graph.analysis.report.report_node import report_node_type_analysis_pass
import pytest
import toml

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes

from pathlib import Path

from chop.actions import simulate
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger

set_logging_verbosity("debug")


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook

IN_FEATURES = 16
OUT_FEATURES = 8


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        x = self.fc1(x)
        return x


@pytest.mark.dev
def test_emit_verilog_linear(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)

    mlp = MLP()
    mg = chop.MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 6
    x = torch.randn((batch_size, IN_FEATURES))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    block_size = 4
    m_width = 8
    e_width = 6
    # Quantize to int
    quan_args = {
        "by": "type",
        "default": {
            "config": {
                "name": "mxint",
                # data
                "data_in_width": m_width,
                "data_in_exponent_width": e_width,
                "data_in_block_size": [1, block_size],
                # weight
                "weight_width": m_width,
                "weight_exponent_width": e_width,
                "weight_block_size": [1, block_size],
                # bias
                "bias_width": m_width,
                "bias_exponent_width": e_width,
                "bias_block_size": [1, block_size],
            }
        },
    }

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    _ = report_node_type_analysis_pass(mg)

    # hack to pass the correct parallelism parameters around
    for node in mg.fx_graph.nodes:
        node_meta = node.meta["mase"].parameters["common"]
        match node_meta["mase_op"]:
            case "linear":
                args = node_meta["args"]
                args["data_in_0"]["parallelism_0"] = block_size
                args["data_in_0"]["parallelism_1"] = batch_size
                args["weight"]["parallelism_0"] = block_size
                args["weight"]["parallelism_1"] = block_size
                args["bias"]["parallelism_0"] = block_size
                args["bias"]["parallelism_1"] = 1

                results = node_meta["results"]
                results["data_out_0"]["parallelism_0"] = block_size
                results["data_out_0"]["parallelism_1"] = batch_size

    # mg.model.fc1.weight.data = torch.eye(IN_FEATURES) * -2
    # mg.model.fc1.bias.data = torch.zeros(mg.model.fc1.bias.data.shape)

    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "num_batches": random.randint(1,100)}
    )
    # mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(skip_build=False, skip_test=False, simulator="verilator", waves=True)


if __name__ == "__main__":
    seed = os.getenv("COCOTB_SEED")
    if seed is None:
        seed = random.randrange(sys.maxsize)
        logger.info(f"Generated {seed=}")
    else:
        seed = int(seed)
        logger.info(f"Using provided {seed=}")
    test_emit_verilog_linear(seed)
    logger.info(f"{seed=}")
