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

logger = get_logger(__name__)


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)
        self.fc4 = nn.Linear(out_features, out_features)

    def forward(self, x):
        l1 = (
            torch.relu(
                self.fc1(x)
            )
        )
        l2 = torch.relu(self.fc2(l1))
        l3 = torch.relu(self.fc2(l2))
        l4 = torch.relu(self.fc2(l3))
        return l4


def test_emit_verilog_mxint(seed: int = 10):
    torch.manual_seed(seed)
    random.seed(seed)

    # size of the overall block is given by block_size * batch_parallelism
    block_size = random.randint(1,10)          # block dim 0
    batch_parallelism = random.randint(1,10)   # block dim 1

    IN_FEATURES = block_size * random.randint(1, 10)
    OUT_FEATURES = block_size * random.randint(1, 10)
    m_width = random.randint(4, 10)
    e_width = random.randint(4, min(m_width, 10))

    # number of batches to be processed by the hw block at one time
    batches = batch_parallelism * random.randint(1, 20)
    # number of times to send a set of batches to the hardware module
    num_batches = random.randint(1,20) 
    logger.info(
        f"{block_size=}, {batch_parallelism=}, {IN_FEATURES=}, {OUT_FEATURES=}, {m_width=}, {e_width=}, {batches=}"
    )

    mlp = MLP(IN_FEATURES, OUT_FEATURES)
    mg = chop.MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batches, IN_FEATURES))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    # Quantize to mxint
    quan_args = {
        "by": "type",
        "default": {
            "config": {
                "name": "mxint",
                # data
                "data_in_width": m_width,
                "data_in_exponent_width": e_width,
                "data_in_block_size": [batch_parallelism, block_size],
                # weight
                "weight_width": m_width,
                "weight_exponent_width": e_width,
                "weight_block_size": [block_size, block_size],
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
        args = node_meta["args"]
        results = node_meta["results"]
        match node_meta["mase_op"]:
            case "linear":
                args["data_in_0"]["parallelism_0"] = block_size
                args["data_in_0"]["parallelism_1"] = batch_parallelism
                args["weight"]["parallelism_0"] = block_size
                args["weight"]["parallelism_1"] = block_size
                args["bias"]["parallelism_0"] = block_size
                args["bias"]["parallelism_1"] = 1

                results["data_out_0"]["parallelism_0"] = block_size
                results["data_out_0"]["parallelism_1"] = batch_parallelism
            case 'relu':
                args["data_in_0"]["parallelism_0"] = block_size
                args["data_in_0"]["parallelism_1"] = batch_parallelism
                results["data_out_0"]["parallelism_0"] = block_size
                results["data_out_0"]["parallelism_1"] = batch_parallelism

    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)
    mg, _ = passes.report_node_meta_param_analysis_pass(mg)

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg,
        pass_args={
            "wait_time": 10 * num_batches,
            "wait_unit": "us",
            "num_batches": num_batches,
        },
    )

    simulate(
        skip_build=False,
        skip_test=False,
        simulator="verilator",
        waves=True,
    )

    logger.info(
        f"{block_size=}, {batch_parallelism=}, {IN_FEATURES=}, {OUT_FEATURES=}, {m_width=}, {e_width=}, {batches=}"
    )


if __name__ == "__main__":
    seed = os.getenv("COCOTB_SEED")
    if seed is None:
        seed = random.randrange(sys.maxsize)
        logger.info(f"Generated {seed=}")
    else:
        seed = int(seed)
        logger.info(f"Using provided {seed=}")
    test_emit_verilog_mxint(seed)
    logger.info(f"{seed=}")
