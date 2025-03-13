`timescale 1ns / 1ps
/*
Module      : mxint_accumulator
Description : 
  - This module implements an accumulator for MxInt.
  - When receiving WIDTH_different exponent values, it adjusts the mantissa to a common bit-width before accumulating.
  - The highest exponent is used as the reference, and smaller exponents shift their mantissas accordingly.
  - The accumulated value is stored until the required depth is reached, at which point it is output.
  - The exponent may increase based on the accumulation result.

Features:
  - Dynamic shifting of mantissas for exponent alignment.
  - Controlled accumulation with a depth accum_count.
  - Adaptive exponent update when needed.
*/

module mxint_accumulator #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter BLOCK_SIZE = 4,
    parameter IN_DEPTH = 2,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + 2 ** DATA_IN_0_PRECISION_1 + $clog2(
        IN_DEPTH
    ),
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1
) (
    input logic clk,
    input logic rst,

    // Input Data
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] mdata_in_0     [BLOCK_SIZE - 1:0],
    input  logic        [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input  logic                                    data_in_0_valid,
    output logic                                    data_in_0_ready,

    // Output Data
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0     [BLOCK_SIZE - 1:0],
    output logic        [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic                                     data_out_0_valid,
    input  logic                                     data_out_0_ready
);

  localparam WIDTH_DIFF = DATA_OUT_0_PRECISION_0 - DATA_IN_0_PRECISION_0;
  localparam COUNTER_WIDTH = $clog2(IN_DEPTH);  // 1-bit wider so IN_DEPTH also fits.


  /* verilator lint_off WIDTH */
  assign data_in_0_ready  = (accum_count != IN_DEPTH) || data_out_0_ready;
  assign data_out_0_valid = (accum_count == IN_DEPTH);
  /* verilator lint_on WIDTH */

  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_in_0 [BLOCK_SIZE - 1:0];
  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_out_0[BLOCK_SIZE - 1:0];
  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] tmp_accumulator    [BLOCK_SIZE - 1:0];

  logic                                       exponent_increment [BLOCK_SIZE - 1:0];
  logic        [ DATA_IN_0_PRECISION_1 - 1:0] max_exponent;

  logic        [             COUNTER_WIDTH:0] accum_count;
  logic signed [ DATA_IN_0_PRECISION_1 - 1:0] shift;
  logic                                       no_reg_value;


  // =============================
  // Exponent Calculation
  // =============================
  assign no_reg_value =(accum_count == 0 || (data_out_0_valid && data_out_0_ready && data_in_0_valid));
  assign max_exponent = (edata_out_0 < edata_in_0) ? edata_in_0 : edata_out_0;
  assign shift = edata_out_0 - edata_in_0;

  // count
  always_ff @(posedge clk)
    if (rst) accum_count <= 0;
    else begin

      if (data_out_0_valid) begin

        if (data_out_0_ready) begin
          if (data_in_0_valid) accum_count <= 1;
          else accum_count <= 0;
        end

      end else if (data_in_0_valid && data_in_0_ready) accum_count <= accum_count + 1;
    end

  // =============================
  // Mantissa Shift and Accumulation
  // =============================
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin

    always_comb begin

      if (shift > 0) begin
        shifted_mdata_in_0[i] = no_reg_value ? mdata_in_0[i] <<< (WIDTH_DIFF - 1) : mdata_in_0[i] <<< (shift + WIDTH_DIFF - 1);
        shifted_mdata_out_0[i] = mdata_out_0[i] >>> 1;
      end else begin
        shifted_mdata_in_0[i]  = mdata_in_0[i] <<< (WIDTH_DIFF - 1);
        shifted_mdata_out_0[i] = mdata_out_0[i] >>> (-shift + 1);
      end

      tmp_accumulator[i] = shifted_mdata_out_0[i] + shifted_mdata_in_0[i];

      if ((tmp_accumulator[i] < 2 ** (DATA_OUT_0_PRECISION_0 - 2) - 1) && (tmp_accumulator[i] > -2 ** (DATA_OUT_0_PRECISION_0 - 2) + 1))
        exponent_increment[i] = 1;
      else exponent_increment[i] = 0;

    end

  end


  // =============================
  // Determine If Exponent Needs Increment
  // =============================

  logic increase_exponent;

  always_comb begin
    increase_exponent = 0;
    
    for (int i = 0; i < BLOCK_SIZE; i++) begin

      if (exponent_increment[i]) begin
        increase_exponent = 1;
        break;
      end

    end
  end

  // =============================
  // Mantissa Output Update Logic
  // =============================

  genvar i;
  for (i = 0; i < BLOCK_SIZE; i++) begin
    always_ff @(posedge clk) begin
      if (rst) mdata_out_0[i] <= '0;
      else begin
        if (data_out_0_valid) begin
          if (data_out_0_ready) begin

            if (data_in_0_valid) mdata_out_0[i] <= shifted_mdata_in_0[i] <<< 1;
            else mdata_out_0[i] <= '0;

          end
        end else if (data_in_0_valid && data_in_0_ready)
          mdata_out_0[i] <= tmp_accumulator[i] <<< (increase_exponent ? 0 : 1);
      end
    end
  end

  // =============================
  // Exponent Output Update Logic
  // =============================
  always_ff @(posedge clk)
    if (rst) edata_out_0 <= '0;
    else if (data_out_0_valid) begin
      if (data_out_0_ready) begin
        if (data_in_0_valid) edata_out_0 <= edata_in_0;
        else edata_out_0 <= '0;
      end
    end else if (data_in_0_valid && data_in_0_ready)
      edata_out_0 <= max_exponent + increase_exponent;

endmodule
