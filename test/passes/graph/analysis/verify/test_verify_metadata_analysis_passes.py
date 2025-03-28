"""
test all verify metadata passes

    verify_common_metadata_analysis_pass,
    verify_hardware_metadata_analysis_pass,
    verify_metadata_analysis_pass,
    verify_software_metadata_analysis_pass,
"""

import logging
import os
import sys

import toml

from chop.passes.graph.transforms.quantize.quantize import quantize_transform_pass
import torch


from chop.tools.logger import set_logging_verbosity
from chop.ir.graph import MaseGraph
from chop.models.toy.toy import ToyNet
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    init_metadata_analysis_pass,
    verify_metadata_analysis_pass,
)

logger = logging.getLogger("chop.test")
set_logging_verbosity("debug")


def test_verify_metadata():
    mlp = ToyNet(image_size=(1, 28, 28), num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    # Quantize to fixed-point
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
        "fixed.toml",
    )

    # load toml config file
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = quantize_transform_pass(mg, quan_args)

    mg, _ = add_hardware_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    # all three verify passes are bundled in one
    # mg, _ = verify_metadata_analysis_pass(mg, dummy_in)


test_verify_metadata()
