#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)

from mase_cocotb.runner import mase_runner
from utils import mxint_quantize

import torch
from math import ceil, log2
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


class MXIntDotProductTB(Testbench):
    def __init__(self, dut, num) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready,
        )
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mweight, dut.eweight),
            dut.weight_valid,
            dut.weight_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
            signed=False,
        )

    def generate_inputs(self):
        inputs = []
        weights = []
        exp_outputs = []
        for _ in range(self.num):
            data = torch.randn(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = mxint_quantize(
                data,
                int(self.dut.DATA_IN_0_PRECISION_0),
                int(self.dut.DATA_IN_0_PRECISION_1),
            )
            w = torch.randn(int(self.dut.BLOCK_SIZE))
            (weight, mweight, eweight) = mxint_quantize(
                w,
                int(self.dut.WEIGHT_PRECISION_0),
                int(self.dut.WEIGHT_PRECISION_1),
            )

            # print(self.dut.DATA_OUT_0_PRECISION_0, self.dut.DATA_OUT_0_PRECISION_1)
            # out_man_w = self.dut.DATA_OUT_0_PRECISION_0.value
            # print(ebias_out, ebias_data, ebias_weight)

            # logger.info(f"expected mantissa out {mantout}, expout = {(edata_in - ebias_data) + (eweight - ebias_weight) + ebias_out}")
            # logger.info(f"expected value = {(mdp_out*2**(-(self.dut.DATA_OUT_0_PRECISION_0.value-2))) * (2**(edp_out - ebias_out))}")
            # data_mant_w = self.dut.DATA_IN_0_PRECISION_0.value
            # data_exp_w = self.dut.DATA_IN_0_PRECISION_1.value

            # print(2**(edata_in - ebias_data) * mdata_in.int() / (2 **(data_mant_w-2)))
            # breakpoint()

            ebias_data = (2 ** (self.dut.DATA_IN_0_PRECISION_1.value - 1)) - 1
            ebias_weight = (2 ** (self.dut.WEIGHT_PRECISION_1.value - 1)) - 1
            ebias_out = (2 ** (self.dut.DATA_OUT_0_PRECISION_1.value - 1)) - 1

            w_man_w = self.dut.WEIGHT_PRECISION_0.value
            in_man_w = self.dut.DATA_IN_0_PRECISION_0.value
            out_man_w = self.dut.DATA_OUT_0_PRECISION_0.value
            expout = (edata_in - ebias_data) + (eweight - ebias_weight) + ebias_out
            mantout = mdata_in.int() @ mweight.int()
            logger.debug(
                f"expected value = {(mantout * 2 ** (-(w_man_w + in_man_w - 4))) * (2 ** (expout - ebias_out))}"
            )

            # compute the mantissa and take the mod since the comparison is unsigned
            mdp_out = (mdata_in @ mweight).int()%(2**out_man_w)
            # adjust the exponent by the biases of the different widths
            edp_out = (edata_in - ebias_data) + (eweight - ebias_weight) + ebias_out

            # logger.info(f"{data} @ {w} = {out}")
            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            weights.append((mweight.int().tolist(), eweight.int().tolist()))
            exp_outputs.append((mdp_out.tolist(), edp_out.int().tolist()))
        print(inputs)
        print(weights)
        print(exp_outputs)
        return inputs, weights, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, weights, exp_outputs = self.generate_inputs()

        # self.log.info(f"inputs: {inputs}\n{}")

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        self.weight_driver.load_driver(weights)
        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)
        # breakpoint()

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = MXIntDotProductTB(dut, num=40)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": random.randint(2,16),
                "DATA_IN_0_PRECISION_1": random.randint(2,16),
                "WEIGHT_PRECISION_0": random.randint(2,16),
                "WEIGHT_PRECISION_1": random.randint(2,16),
                "BLOCK_SIZE": random.randint(2,16),
            },
        ],
    )
