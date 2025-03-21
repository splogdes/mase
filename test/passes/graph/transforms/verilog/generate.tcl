
create_project -in_memory -part xcku5p-ffvb676-2-e

set_property board_part xilinx.com:kcu116:part0:1.5 [current_project]

add_files {/home/omar/.mase/top/hardware/rtl/input_buffer.sv /home/omar/.mase/top/hardware/rtl/split2.sv /home/omar/.mase/top/hardware/rtl/or_tree_layer.sv /home/omar/.mase/top/hardware/rtl/top.sv /home/omar/.mase/top/hardware/rtl/unpacked_register_slice.sv /home/omar/.mase/top/hardware/rtl/join2.sv /home/omar/.mase/top/hardware/rtl/log2_max_abs.sv /home/omar/.mase/top/hardware/rtl/fc1_weight_source.sv /home/omar/.mase/top/hardware/rtl/ultraram.v /home/omar/.mase/top/hardware/rtl/skid_buffer.sv /home/omar/.mase/top/hardware/rtl/unpacked_mx_fifo.sv /home/omar/.mase/top/hardware/rtl/ultraram_fifo.sv /home/omar/.mase/top/hardware/rtl/mxint_linear.sv /home/omar/.mase/top/hardware/rtl/fixed_adder_tree_layer.sv /home/omar/.mase/top/hardware/rtl/fixed_dot_product.sv /home/omar/.mase/top/hardware/rtl/mxint_accumulator.sv /home/omar/.mase/top/hardware/rtl/mxint_dot_product.sv /home/omar/.mase/top/hardware/rtl/fixed_mult.sv /home/omar/.mase/top/hardware/rtl/fixed_vector_mult.sv /home/omar/.mase/top/hardware/rtl/fc1_bias_source.sv /home/omar/.mase/top/hardware/rtl/or_tree.sv /home/omar/.mase/top/hardware/rtl/mxint_cast.sv /home/omar/.mase/top/hardware/rtl/fixed_adder_tree.sv /home/omar/.mase/top/hardware/rtl/mxint_circular.sv /home/omar/.mase/top/hardware/rtl/mxint_register_slice.sv /home/omar/.mase/top/hardware/rtl/unpacked_skid_buffer.sv /home/omar/.mase/top/hardware/rtl/fifo.sv /home/omar/.mase/top/hardware/rtl/simple_dual_port_ram.sv /home/omar/.mase/top/hardware/rtl/join_n.sv /home/omar/.mase/top/hardware/rtl/blk_mem_gen_0.sv /home/omar/.mase/top/hardware/rtl/register_slice.sv}

set_property top top [current_fileset]

# Load parameters from the file
source config.tcl

# Apply parameters dynamically
set generic_params ""
foreach key [array names PARAMS] {
    append generic_params " -generic $key=$PARAMS($key)"
}

# Run synthesis with dynamic parameters
eval "synth_design -mode out_of_context -top top -part xcku5p-ffvb676-2-e $generic_params"

#launch_runs synth_1 -jobs 8
#wait_on_run synth_1

#open_run synth_1 -name synth_1
#report_utilization -name utilization_1
