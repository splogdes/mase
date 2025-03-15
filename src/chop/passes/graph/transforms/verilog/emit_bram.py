import logging
import math
import os
import struct
import time

from chop.passes.graph.transforms.verilog.mxint_bram_template import mxint_template
import torch

from chop.passes.graph.utils import vf, v2p, get_module_by_name, init_project
from chop.nn.quantizers import (
    integer_quantizer_for_hw, 
    integer_floor_quantizer_for_hw,
    mxint_quantizer_for_hw
)

logger = logging.getLogger(__name__)
from pathlib import Path


def iceil(x):
    return int(math.ceil(x))


def clog2(x):
    return iceil(math.log2(x))


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def emit_parameters_in_mem_internal(node, param_name, file_name, data_name):
    verilog_param_name = param_name.replace(".", "_")
    match node.meta["mase"].parameters["common"]["args"][verilog_param_name]["type"]:
        case "fixed":
            emit_parameters_in_mem_internal_fixed_point(
                node, verilog_param_name, file_name, data_name
            )
        case "mxint":
            emit_parameters_in_mem_internal_mxint(
                node, verilog_param_name, file_name, data_name
            )
        case unsupported_type:
            raise NotImplementedError(f"Unsupported BRAM data-type {unsupported_type}")


def emit_parameters_in_mem_internal_mxint(
    node, verilog_param_name, file_name, data_name
):
    return
    node_type_info = node.meta["mase"].parameters["common"]["args"][verilog_param_name]
    node_verilog_info = node.meta["mase"].parameters["hardware"]["verilog_param"]

    total_size = math.prod(node_type_info["shape"])

    out_size = int(
        node_verilog_info[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_0"]
        * node_verilog_info[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_1"]
    )

    out_depth = int((total_size + out_size - 1) / out_size)

    mantissa_width = int(node_type_info["precision"][0])
    exponent_width = int(node_type_info["precision"][1])

    node_param_name = f"{vf(node.name)}_{verilog_param_name}"

    rom_str = mxint_template.format(
        node_param_name=node_param_name,
        date_time=time.strftime("%d/%m/%Y %H:%M:%S"),
        edwith=exponent_width,
        emem_size=out_size,
        mwidth=mantissa_width,
    )


def emit_parameters_in_mem_internal_fixed_point(
    node, verilog_param_name, file_name, data_name
):
    """
    Emit single-port ROM hardware components for each parameter
    (Mostly because Vivado does not support string type parameters...)
    """
    # ! TO DO: currently emitting too many parameters

    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][verilog_param_name]["shape"]
    )
    # TO DO: change setting parallelism for weight in metadata
    # node.meta["mase"].parameters["hardware"]["verilog_param"][f"{_cap(param_name)}_PARALLELISM_DIM_1"]
    out_size = int(
        node.meta["mase"].parameters["hardware"]["verilog_param"][
            f"{_cap(verilog_param_name)}_PARALLELISM_DIM_0"
        ]
        * node.meta["mase"].parameters["hardware"]["verilog_param"][
            f"{_cap(verilog_param_name)}_PARALLELISM_DIM_1"
        ]
    )
    out_depth = int((total_size + out_size - 1) / out_size)
    out_width = int(
        node.meta["mase"].parameters["common"]["args"][verilog_param_name]["precision"][
            0
        ]
    )

    addr_width = clog2(out_depth) + 1

    node_param_name = f"{vf(node.name)}_{verilog_param_name}"

    rom_verilog = f"""
// =====================================
//     Mase Hardware
//     Parameter: {node_param_name}
//     {time.strftime("%d/%m/%Y %H:%M:%S")}
// =====================================

`timescale 1 ns / 1 ps
module {node_param_name}_rom #(
  parameter DWIDTH = {out_size * out_width},
  parameter MEM_SIZE = {out_depth},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input clk,
    input logic [AWIDTH-1:0] addr0,
    input ce0,
    output logic [DWIDTH-1:0] q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("{data_name}", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module {node_param_name} #(
  parameter DATA_WIDTH = 32'd{out_width * out_size},
  parameter ADDR_RANGE = 32'd{out_depth},
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  {node_param_name}_rom {node_param_name}_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module {node_param_name}_source #(
    parameter {_cap(verilog_param_name)}_TENSOR_SIZE_DIM_0  = 32,
    parameter {_cap(verilog_param_name)}_TENSOR_SIZE_DIM_1  = 1,
    parameter {_cap(verilog_param_name)}_PRECISION_0 = 16,
    parameter {_cap(verilog_param_name)}_PRECISION_1 = 3,

    parameter {_cap(verilog_param_name)}_PARALLELISM_DIM_0 = 1,
    parameter {_cap(verilog_param_name)}_PARALLELISM_DIM_1 = 1,
    parameter OUT_DEPTH = (({_cap(verilog_param_name)}_TENSOR_SIZE_DIM_0 + {_cap(verilog_param_name)}_PARALLELISM_DIM_0 - 1) / {_cap(verilog_param_name)}_PARALLELISM_DIM_0) * (({_cap(verilog_param_name)}_TENSOR_SIZE_DIM_1 + {_cap(verilog_param_name)}_PARALLELISM_DIM_1 - 1) / {_cap(verilog_param_name)}_PARALLELISM_DIM_1)
) (
    input clk,
    input rst,

    output logic [{_cap(verilog_param_name)}_PRECISION_0-1:0] data_out      [{_cap(verilog_param_name)}_PARALLELISM_DIM_0 * {_cap(verilog_param_name)}_PARALLELISM_DIM_1-1:0],
    output                       data_out_valid,
    input                        data_out_ready
);
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(OUT_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_ready) begin
        if (counter == OUT_DEPTH - 1) counter <= 0;
        else counter <= counter + 1;
      end
    end

  logic [1:0] clear;
  always_ff @(posedge clk)
    if (rst) clear <= 0;
    else if ((data_out_ready == 1) && (clear != 2)) clear <= clear + 1;
  logic ce0;
  assign ce0 = data_out_ready;

  logic [{_cap(verilog_param_name)}_PRECISION_0*{_cap(verilog_param_name)}_PARALLELISM_DIM_0*{_cap(verilog_param_name)}_PARALLELISM_DIM_1-1:0] data_vector;
  {node_param_name} #(
      .DATA_WIDTH({_cap(verilog_param_name)}_PRECISION_0 * {_cap(verilog_param_name)}_PARALLELISM_DIM_0 * {_cap(verilog_param_name)}_PARALLELISM_DIM_1),
      .ADDR_RANGE(OUT_DEPTH)
  ) {node_param_name}_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < {_cap(verilog_param_name)}_PARALLELISM_DIM_0 * {_cap(verilog_param_name)}_PARALLELISM_DIM_1; j++)
    assign data_out[j] = data_vector[{_cap(verilog_param_name)}_PRECISION_0*j+{_cap(verilog_param_name)}_PRECISION_0-1:{_cap(verilog_param_name)}_PRECISION_0*j];

  assign data_out_valid = clear == 2;

endmodule
"""

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(rom_verilog)
    logger.debug(
        f"ROM module {verilog_param_name} successfully written into {file_name}"
    )
    assert os.path.isfile(file_name), "ROM Verilog generation failed."
    # os.system(f"verible-verilog-format --inplace {file_name}")


def emit_parameters_in_dat_internal(node, param_name, file_name):
    """
    Emit initialized data for the ROM block.
    Each element is represented in fixed-width hexadecimal format.
    """
    verilog_param_name = param_name.replace(".", "_")
    
    mase = node.meta["mase"]
    common_args = mase.parameters["common"]["args"][verilog_param_name]
    hw_verilog = mase.parameters["hardware"]["verilog_param"]
    hw_interface = mase.parameters["hardware"]["interface"][verilog_param_name]

    total_size = math.prod(common_args["shape"])
    out_size = int(
        hw_verilog[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_0"] *
        hw_verilog[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_1"]
    )
    out_depth = (total_size + out_size - 1) // out_size

    param_data = mase.module.get_parameter(param_name).data
    if hw_interface["transpose"]:
        param_data = torch.reshape(
            param_data,
            (
                hw_verilog["DATA_OUT_0_SIZE"],
                hw_verilog["DATA_IN_0_DEPTH"],
                hw_verilog["DATA_IN_0_SIZE"],
            ),
        )
        param_data = torch.transpose(param_data, 0, 1)


    match common_args["type"]:
        
        case "fixed":
            param_data = torch.flatten(param_data).tolist()
            width, frac_width = common_args["precision"]

            if mase.module.config.get("floor", False):
                base_quantizer = integer_floor_quantizer_for_hw
            else:
                base_quantizer = integer_quantizer_for_hw

            data_buff = ""
            for i in range(out_depth):
                line_values = []
                start_idx = i * out_size
                end_idx = start_idx + out_size
                
                for idx in range(end_idx - 1, start_idx - 1, -1):
                    if idx >= len(param_data):
                        value = 0
                    else:
                        value = param_data[idx]

                    quantized = base_quantizer(torch.tensor(value), width, frac_width).item()
                    hex_str = format(quantized, '0{}X'.format(width // 4))
                    line_values.append(hex_str)
                    
                data_buff += "".join(line_values) + "\n"

            dat_file = file_name + ".dat"
            with open(dat_file, "w", encoding="utf-8") as outf:
                outf.write(data_buff)

            logger.debug(f"Init data {param_name} successfully written into {dat_file}")
            assert os.path.isfile(dat_file), "ROM data generation failed."

        case "mxint":
            data_width, exponent_width = common_args["precision"]
            floor_values = mase.module.config.get("floor", False)
            block_size = [
                hw_verilog[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_1"],
                hw_verilog[f"{_cap(verilog_param_name)}_PARALLELISM_DIM_0"],
            ]

            mxint_blocks, mxint_exp = mxint_quantizer_for_hw(
                param_data,
                data_width,
                exponent_width,
                block_size,
                floor=floor_values,
            )

            block_buff = ""
            for i in range(mxint_blocks.shape[1]):
                line_values = []
                for j in range(mxint_blocks.shape[0]):
                    value = int(mxint_blocks[j, i].item())
                    mask = 2**(data_width - 1) - 1
                    value = (value & mask) - (value & ~mask)
                    hex_str = format(value, '0{}X'.format(data_width // 4))
                    line_values.append(hex_str)
                block_buff += " ".join(line_values[::-1]) + "\n"

            exp_buff = ""
            for exp in mxint_exp.flatten().tolist():
                hex_str = format(int(exp), '0{}X'.format(exponent_width // 4))
                exp_buff += hex_str + "\n"

            block_file = file_name + "_block.dat"
            exp_file = file_name + "_exp.dat"
            
            with open(block_file, "w", encoding="utf-8") as outf:
                outf.write(block_buff)
            with open(exp_file, "w", encoding="utf-8") as outf:
                outf.write(exp_buff)

            assert os.path.isfile(block_file), "ROM data generation failed."
            logger.debug(f"Init data {param_name} successfully written into {block_file}")
            
            assert os.path.isfile(exp_file), "ROM data generation failed."
            logger.debug(f"Init data {param_name} successfully written into {exp_file}")

        case _:
            raise ValueError("Emitting non-fixed parameters is not supported.")


def emit_parameters_in_dat_hls(node, param_name, file_name):
    """
    Emit initialised data for the ROM block. Each element must be in 8 HEX digits.
    """
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_depth = total_size
    out_size = 1
    out_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
        "{}_WIDTH".format(param_name.upper())
    ]

    data_buff = ""
    param_data = node.meta["mase"].module.get_parameter(param_name).data
    param_data = torch.flatten(param_data).tolist()

    if node.meta["mase"].parameters["common"]["args"][param_name]["type"] == "fixed":
        width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
            0
        ]
        frac_width = node.meta["mase"].parameters["common"]["args"][param_name][
            "precision"
        ][1]

        scale = 2**frac_width
        thresh = 2**width
        for i in range(0, out_depth):
            line_buff = ""
            for j in range(0, out_size):
                value = param_data[i * out_size + out_size - 1 - j]
                value = integer_quantizer_for_hw(
                    torch.tensor(value), width, frac_width
                ).item()
                value = str(bin(int(value * scale) % thresh))
                value_bits = value[value.find("0b") + 2 :]
                value_bits = "0" * (width - len(value_bits)) + value_bits
                assert len(value_bits) == width
                line_buff += value_bits

            hex_buff = hex(int(line_buff, 2))
            data_buff += hex_buff[hex_buff.find("0x") + 2 :] + "\n"
    elif node.meta["mase"].parameters["common"]["args"][param_name]["type"] == "float":
        width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
            0
        ]
        assert width == 32, "Only float32 is supported for now."

        for i in range(0, out_depth):
            line_buff = ""
            value = param_data[i]
            hex_buff = hex(struct.unpack("<I", struct.pack("<f", value))[0])
            # Double will then be:
            # hex(struct.unpack('<Q', struct.pack('<d', value))[0])
            data_buff += hex_buff[hex_buff.find("0x") + 2 :] + "\n"
    else:
        assert False, "Emitting unknown type of parameters is not supported."

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(data_buff)
    logger.debug(f"Init data {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM data generation failed."


def emit_bram_handshake(node, rtl_dir):
    """
    Enumerate input parameters of the internal node and emit a ROM block
    with handshake interface for each parameter
    """
    node_name = vf(node.name)
    for param_name, parameter in node.meta["mase"].module.named_parameters():
        param_verilog_name = param_name.replace(".", "_")
        if (
            node.meta["mase"].parameters["hardware"]["interface"][param_verilog_name][
                "storage"
            ]
            == "BRAM"
        ):
            logger.debug(
                f"Emitting DAT file for node: {node_name}, parameter: {param_verilog_name}"
            )
            verilog_name = os.path.join(
                rtl_dir, f"{node_name}_{param_verilog_name}_source.sv"
            )
            data_name = os.path.join(
                rtl_dir, f"{node_name}_{param_verilog_name}_rom"
            )
            emit_parameters_in_mem_internal(node, param_name, verilog_name, data_name)
            emit_parameters_in_dat_internal(node, param_name, data_name)
        else:
            assert False, "Emtting parameters in non-BRAM hardware is not supported."


def emit_parameters_in_mem_hls(node, param_name, file_name, data_name):
    """
    Emit single-port ROM hardware components for each parameter
    (Mostly because Vivado does not support string type parameters...)
    """

    # The depth of parameters matches with the input depth
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_depth = total_size
    addr_width = clog2(out_depth) + 1
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_size = iceil(total_size / out_depth)
    assert total_size % out_depth == 0, (
        f"Cannot partition imperfect size for now = {total_size} / {out_depth}."
    )
    # Assume the first index is the total width
    out_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
        "{}_WIDTH".format(param_name.upper())
    ]

    node_name = vf(node.name)
    node_param_name = f"{node_name}_{param_name}"
    time_to_emit = time.strftime("%d/%m/%Y %H:%M:%S")

    rom_verilog = f"""
// =====================================
//     Mase Hardware
//     Parameter: {node_param_name}
//     {time_to_emit}
// =====================================

`timescale 1 ns / 1 ps
module {node_param_name}_rom #(
  parameter DWIDTH = {out_size * out_width},
  parameter MEM_SIZE = {out_depth},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input clk,
    input logic [AWIDTH-1:0] addr0,
    input ce0,
    output logic [DWIDTH-1:0] q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("{data_name}", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module {node_param_name}_source #(
  parameter DATA_WIDTH = 32'd{out_width * out_size},
  parameter ADDR_RANGE = 32'd{out_depth},
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  {node_param_name}_rom {node_param_name}_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule
"""

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(rom_verilog)
    logger.debug(f"ROM module {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM Verilog generation failed."
    # os.system(f"verible-verilog-format --inplace {file_name}")


def emit_bram_hls(node, rtl_dir):
    """
    Enumerate input parameters of the hls node and emit a ROM block
    with handshake interface for each parameter
    """
    node_name = vf(node.name)
    for param_name, parameter in node.meta["mase"].module.named_parameters():
        if (
            node.meta["mase"].parameters["hardware"]["interface"][param_name]["storage"]
            == "BRAM"
        ):
            # Verilog code of the ROM has been emitted using mlir passes
            verilog_name = os.path.join(rtl_dir, f"{node_name}_{param_name}.sv")
            data_name = os.path.join(rtl_dir, f"{node_name}_{param_name}_rom.dat")
            emit_parameters_in_mem_hls(node, param_name, verilog_name, data_name)
            emit_parameters_in_dat_hls(node, param_name, data_name)
        else:
            assert False, "Emtting parameters in non-BRAM hardware is not supported."


def emit_bram_transform_pass(graph, pass_args={}):
    """
    Enumerate input parameters of the node and emit a ROM block with
    handshake interface for each parameter


    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : name of the top module
    """

    logger.info("Emitting BRAM...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"

    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        # Only modules have internal parameters
        if node.meta["mase"].module is None:
            continue
        if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            emit_bram_handshake(node, rtl_dir)
        elif "MLIR_HLS" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            emit_bram_hls(node, rtl_dir)

    return graph, None
