#!/usr/bin/env python3
"""
Testbench for the fixed_threshold module.
This module implements:
    if (data_in <= THRESHOLD_VALUE)
         data_out = VALUE_VALUE;
    else data_out = data_in;
The inputs and outputs are fixed-point numbers.
"""

import logging
import torch
import pytest
import cocotb
from cocotb.triggers import Timer
from chop.nn.quantizers import integer_quantizer
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# Configure logging
logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def get_in_and_out_threshold(x, threshold_value, value_value, width, frac_width):
    """
    Quantizes the floating-point tensor `x` to fixed-point representation
    and computes the expected output for the threshold module.
    
    For each element a in x (after quantization), the expected output is:
        if (a <= THRESHOLD_VALUE) then VALUE_VALUE else a.
        
    The quantization is done using the provided bit width and fractional width.
    """
    # Quantize the input and convert to fixed point (i.e. multiply by 2^frac_width)
    ins = integer_quantizer(x, width=width, frac_width=frac_width)
    ins_fixed = ins * (2**frac_width)
    
    # Compute expected outputs:
    # If the fixed-point input is less than or equal to threshold_value, then output value_value,
    # otherwise output the input value.
    outs_fixed = torch.where(ins_fixed <= threshold_value,
                             torch.tensor(value_value, dtype=torch.int),
                             ins_fixed)
    return ins_fixed.int(), outs_fixed.int()


class ThresholdTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        # Assign DUT parameters so they can be accessed as attributes
        self.assign_self_params([
            "DATA_IN_0_PRECISION_0",
            "DATA_IN_0_PRECISION_1",
            "DATA_IN_0_TENSOR_SIZE_DIM_0",
            "DATA_IN_0_TENSOR_SIZE_DIM_1",
            "DATA_IN_0_PARALLELISM_DIM_0",
            "DATA_IN_0_PARALLELISM_DIM_1",
            "DATA_OUT_0_PRECISION_0",
            "DATA_OUT_0_PRECISION_1",
            "DATA_OUT_0_TENSOR_SIZE_DIM_0",
            "DATA_OUT_0_TENSOR_SIZE_DIM_1",
            "DATA_OUT_0_PARALLELISM_DIM_0",
            "DATA_OUT_0_PARALLELISM_DIM_1",
        ])
        # Create stream interfaces for the input and output ports
        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

    def generate_inputs_outputs(self, width, frac_width, threshold_value, value_value):
        """
        Generate a test vector of inputs (as a torch tensor) and compute the expected outputs.
        In this example, we use a set of six test values.
        """
        # Example test inputs (in floating point)
        test_values = [3.0, 5.0, 6.0, 7.5, -1.0, 4.5]
        inputs = torch.tensor(test_values).float()
        ins, outs = get_in_and_out_threshold(inputs, threshold_value, value_value, width, frac_width)
        logger.info(f"Test Inputs (fixed point): {ins.tolist()}")
        logger.info(f"Expected Outputs (fixed point): {outs.tolist()}")
        return ins, outs


@cocotb.test()
async def cocotb_test(dut):
    """
    Cocotb test for the fixed_threshold module.
    """
    tb = ThresholdTB(dut)
    await tb.reset()
    logger.info("Reset finished.")
    # Make sure the DUT is ready to send/receive data
    tb.data_out_0_monitor.ready.value = 1

    # Retrieve fixed-point settings from the DUT parameters
    width = tb.DATA_IN_0_PRECISION_0    # e.g. 8 bits
    frac_width = tb.DATA_IN_0_PRECISION_1  # e.g. 3 fractional bits

    # Define the threshold and replacement values in floating point,
    # then convert to fixed point. For example:
    #
    #   threshold_float = 5.0   -> threshold_value = int(5.0 * 2^frac_width) = 40
    #   value_float     = 2.0   -> value_value     = int(2.0 * 2^frac_width) = 16
    threshold_float = 5.0
    value_float = 2.0
    threshold_value = int(threshold_float * (2**frac_width))
    value_value = int(value_float * (2**frac_width))

    # Generate the test inputs and expected outputs
    inputs, exp_outs = tb.generate_inputs_outputs(width, frac_width, threshold_value, value_value)

    # Drive the inputs into the DUT
    tb.data_in_0_driver.append(inputs.tolist())
    # Set up the expected output sequence
    tb.data_out_0_monitor.expect(exp_outs.tolist())

    # Wait long enough for the transaction to complete
    await Timer(1000, units="us")
    # Check that all expected transactions have been observed
    assert tb.data_out_0_monitor.exp_queue.empty(), "Not all expected outputs were received."


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_threshold():
    """
    Pytest wrapper that runs the simulation using the MASE runner.
    The module parameters below are used to configure the DUT.
    """
    mase_runner(
        module_param_list=[
            {
                # Input port configuration
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 6,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 6,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 3,
                # Output port configuration
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 6,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 6,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_PRECISION_0": 8,
                "DATA_OUT_0_PRECISION_1": 3,
                # Threshold-specific parameters:
                # Note that THRESHOLD_VALUE and VALUE_VALUE should be given in fixed point.
                # For 3 fractional bits, 5.0 -> 5*8 = 40 and 2.0 -> 2*8 = 16.
                "THRESHOLD_VALUE": 40,
                "VALUE_VALUE": 16,
            }
        ]
    )


if __name__ == "__main__":
    test_fixed_threshold()
