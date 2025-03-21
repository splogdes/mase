from typing import Literal
from mase_components.linear_layers.mxint_operators.test.utils import (
    block_mxint_quant,
    pack_tensor_to_mx_listed_chunk,
)
import numpy as np
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
from cocotb.result import TestFailure


import dill
import inspect


class MxIntStreamMonitor(MultiSignalStreamMonitor):
    def __init__(self, clk, e_data, m_data, valid, ready, off_by_value=0):
        self.off_by = off_by_value
        super().__init__(
            clk,
            (m_data, e_data),
            valid,
            ready,
            check=True,
            signed=True,
            off_by_one=False,
        )

    def _check(self, got, exp):
        got_m, got_e = got
        exp_m, exp_e = exp

        def check_equality(got, exp):
            if not np.equal(got, exp).all():
                diff = np.subtract(got, exp)
                if np.isclose(got, exp, atol=self.off_by).all():
                    self.log.warning(
                        f"Off-by-{max(abs(diff))} error: {diff=}\nGot {got}\nExp {exp}"
                    )
                else:
                    raise TestFailure(
                        "\nGot \n%s, \nExp \n%s,\nDiff \n%s" % (got, exp, diff)
                    )

        # breakpoint()
        if exp_e == got_e:
            check_equality(got_m, exp_m)
        elif abs(diff := (exp_e - got_e)) == 1:
            # normalisation related error
            # in the case where a single off by 1 error causes the dut to normalise
            # and get a different output exponent
            adj_m = np.array(got_m) * 2 ** (-diff)
            self.log.warning(f"Normalisation Error {exp_e=} {got_e=}")
            check_equality(adj_m, exp_m)


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):

    wait_time = pass_args.get("wait_time", 100)
    wait_unit = pass_args.get("wait_units", "ms")
    num_batches = pass_args.get("num_batches", 1)
    print(pass_args)
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

    in_tensors = tb.generate_inputs(num_batches={num_batches})
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
                    self.output_monitors[result] = MxIntStreamMonitor(
                        dut.clk,
                        getattr(dut, f"e_{result}"),
                        getattr(dut, f"m_{result}"),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                        off_by_value=1,
                    )
                    self.output_monitors[result].log.setLevel(logging.DEBUG)

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

        def generate_inputs(self, num_batches):
            """
            Generate inputs for the model by sampling a random tensor
            for each input argument, according to its shape

            :param num_batches: number of batches to generate for each argument
            :type num_batches: int
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
                    print(
                        f"Generating data for node {node}, arg {arg}: {arg_info} {arg_info['shape']}"
                    )
                    inputs[f"{arg}"] = torch.randn(([num_batches] + arg_info["shape"]))
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

            # convert the exponents from the biased form to signed
            exp_max_val = 2 ** config["exponent_width"]
            for i, (tensor, exp) in enumerate(tensor_output):
                # sign extend by doing (2e) mod 2^b - (e mod 2^b)
                exp_signed = (2 * exp) % exp_max_val - (exp % exp_max_val)
                tensor_output[i] = (tensor, exp_signed)

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
