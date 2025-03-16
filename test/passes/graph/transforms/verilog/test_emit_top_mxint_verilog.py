#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
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

IN_FEATURES = 8
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
def test_emit_verilog_linear():
    mlp = MLP()
    mg = chop.MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 2
    x = torch.randn((batch_size, IN_FEATURES))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    block_size = 2
    # Quantize to int
    quan_args = {
            "by": "type",
            "default": {
                "config": {
                    "name": "mxint",
                    # data
                    "data_in_width": 12,
                    "data_in_exponent_width": 4,
                    "weight_block_size": [1, block_size],
                    # weight
                    "weight_width": 12,
                    "weight_exponent_width": 4,
                    "bias_block_size": [1, block_size],
                    # bias
                    "bias_width": 12,
                    "bias_exponent_width": 4,
                    "data_in_block_size": [1, block_size],
                }
            },
        }

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    _ = report_node_type_analysis_pass(mg)


    # Increase weight range
    mg.model.fc1.weight = torch.nn.Parameter(
        10 * torch.randn(mg.model.fc1.weight.shape)
    )

    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg
    )
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    )
    # mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(skip_build=False, skip_test=False, simulator="verilator")


if __name__ == "__main__":
    test_emit_verilog_linear()
