from mase_components.linear_layers.mxint_operators.test.utils import (
    block_mxint_quant,
    pack_tensor_to_mx_listed_chunk,
)
import logging, torch
from pathlib import Path
from textwrap import indent

from chop.passes.graph.utils import vf, v2p, init_project
from chop.nn.quantizers import (
    integer_quantizer_for_hw,
    integer_quantizer,
)

logger = logging.getLogger(__name__)

from pathlib import Path

torch.manual_seed(0)

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
    StreamDriver,
    StreamMonitor,
)


import dill
import inspect


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):

    wait_time = pass_args.get("wait_time", 100)
    wait_unit = pass_args.get("wait_units", "ms")
    batch_size = pass_args.get("batch_size", 1)

    test_template = f"""
import cocotb

@cocotb.test()
async def test(dut):
    from pathlib import Path
    import dill
    from cocotb.triggers import Timer

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut, fail_on_checks=True)

    await tb.initialize()

    in_tensors = tb.generate_inputs(batches={batch_size})
    exp_out = tb.model(*list(in_tensors.values()))

    tb.load_drivers(in_tensors)
    tb.load_monitors(exp_out)

    # await tb.wait_end(timeout={wait_time}, timeout_unit="{wait_unit}")
    await cocotb.triggers.Timer(100, 'us')
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)


def _emit_cocotb_tb(graph):
    class MaseGraphTB(Testbench):
        def __init__(self, dut, fail_on_checks=True):
            super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)

            # Instantiate as many drivers as required inputs to the model
            self.input_drivers = {}
            self.output_monitors = {}

            for node in graph.nodes_in:
                for arg in node.meta["mase"]["common"]["args"].keys():
                    if "data_in" not in arg:
                        continue
                    self.input_drivers[arg] = MultiSignalStreamDriver(
                        dut.clk,
                        (getattr(dut, f"m_{arg}"), getattr(dut, f"e_{arg}")),
                        getattr(dut, f"{arg}_valid"),
                        getattr(dut, f"{arg}_ready"),
                    )
                    self.input_drivers[arg].log.setLevel(logging.DEBUG)

            # Instantiate as many monitors as required outputs
            for node in graph.nodes_out:
                for result in node.meta["mase"]["common"]["results"].keys():
                    if "data_out" not in result:
                        continue
                    self.output_monitors[result] = MultiSignalStreamMonitor(
                        dut.clk,
                        (getattr(dut, f"m_{result}"), getattr(dut, f"e_{result}")),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                        check=True,
                        signed=False,
                        off_by_one=True,
                    )
                    self.output_monitors[result].log.setLevel(logging.DEBUG)

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

        def generate_inputs(self, batches):
            """
            Generate inputs for the model by sampling a random tensor
            for each input argument, according to its shape

            :param batches: number of batches to generate for each argument
            :type batches: int
            :return: a dictionary of input arguments and their corresponding tensors
            :rtype: Dict
            """
            # ! TO DO: iterate through graph.args instead to generalize
            inputs = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    # Batch dimension always set to 1 in metadata
                    if "data_in" not in arg:
                        continue
                    # print(f"Generating data for node {node}, arg {arg}: {arg_info}")
                    inputs[f"{arg}"] = torch.rand(([batches] + arg_info["shape"][1:]))
            return inputs

        def load_drivers(self, in_tensors):
            for arg, arg_batches in in_tensors.items():
                # Quantize input tensor according to precision
                if len(self.input_precision) > 1:
                    config = {
                        "width": self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                        "exponent_width": self.get_parameter(
                            f"{_cap(arg)}_PRECISION_1"
                        ),
                    }
                    parallelism = [
                        self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                        self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                    ]
                    print(config, parallelism, arg_batches.shape)
                    (qtensor, mtensor, etensor) = block_mxint_quant(
                        arg_batches, config, parallelism
                    )
                    tensor_inputs = pack_tensor_to_mx_listed_chunk(
                        mtensor, etensor, parallelism
                    )

                else:
                    # TO DO: convert to integer equivalent of floating point representation
                    pass

                self.input_drivers[arg].load_driver(tensor_inputs)

        def load_monitors(self, expectation):
            # Process the expectation tensor
            config = {
                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            }
            parallelism = [
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ]

            print(config, parallelism)

            (qtensor, mtensor, etensor) = block_mxint_quant(
                expectation, config, parallelism
            )
            tensor_output = pack_tensor_to_mx_listed_chunk(
                mtensor, etensor, parallelism
            )

            # # convert the exponents from the biased form to signed
            # bias = 2 ** (config["exponent_width"] - 1) - 1
            # for i, (tensor, exp) in enumerate(tensor_output):
            #     new_exp = exp - bias
            #     tensor_output[i] = (tensor, new_exp)

            self.output_monitors["data_out_0"].load_monitor(tensor_output)

            # Drive the in-flight flag for each monitor
            # self.output_monitors["data_out_0"].in_flight = True

    # Serialize testbench object to be instantiated within test by cocotb runner
    cls_obj = MaseGraphTB
    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "tb_obj.dill", "wb") as file:
        dill.dump(cls_obj, file)
    with open(tb_path / "__init__.py", "w") as file:
        file.write("from .test import test")


def emit_cocotb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project
        - trace -> bool : trace waves in the simulation
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    init_project(project_dir)

    _emit_cocotb_test(graph, pass_args=pass_args)
    _emit_cocotb_tb(graph)

    return graph, None
