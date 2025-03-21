import os

config_file = "config.tcl"

template = {
    "fc1_DATA_IN_0_PRECISION_0": 10,
    "fc1_DATA_IN_0_PRECISION_1": 10,
    "fc1_DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
    "fc1_DATA_IN_0_PARALLELISM_DIM_0": 4,
    "fc1_DATA_IN_0_TENSOR_SIZE_DIM_1": 6,
    "fc1_DATA_IN_0_PARALLELISM_DIM_1": 6,
    "fc1_WEIGHT_PRECISION_0": 10,
    "fc1_WEIGHT_PRECISION_1": 10,
    "fc1_WEIGHT_TENSOR_SIZE_DIM_0": 20,
    "fc1_WEIGHT_PARALLELISM_DIM_0": 4,
    "fc1_WEIGHT_TENSOR_SIZE_DIM_1": 40,
    "fc1_WEIGHT_PARALLELISM_DIM_1": 4,
    "fc1_BIAS_PRECISION_0": 10,
    "fc1_BIAS_PRECISION_1": 10,
    "fc1_BIAS_TENSOR_SIZE_DIM_0": 40,
    "fc1_BIAS_PARALLELISM_DIM_0": 4,
    "fc1_BIAS_TENSOR_SIZE_DIM_1": 1,
    "fc1_BIAS_PARALLELISM_DIM_1": 1,
    "fc1_DATA_OUT_0_PRECISION_0": 10,
    "fc1_DATA_OUT_0_TENSOR_SIZE_DIM_0": 40,
    "fc1_DATA_OUT_0_PARALLELISM_DIM_0": 4,
    "fc1_DATA_OUT_0_TENSOR_SIZE_DIM_1": 6,
    "fc1_DATA_OUT_0_PARALLELISM_DIM_1": 6,
    "fc1_DATA_OUT_0_PRECISION_1": 10,
    "DATA_IN_0_PRECISION_0": 10,
    "DATA_IN_0_PRECISION_1": 10,
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
    "DATA_IN_0_PARALLELISM_DIM_0": 4,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 6,
    "DATA_IN_0_PARALLELISM_DIM_1": 6,
    "DATA_OUT_0_PRECISION_0": 10,
    "DATA_OUT_0_TENSOR_SIZE_DIM_0": 40,
    "DATA_OUT_0_PARALLELISM_DIM_0": 4,
    "DATA_OUT_0_TENSOR_SIZE_DIM_1": 6,
    "DATA_OUT_0_PARALLELISM_DIM_1": 6,
    "DATA_OUT_0_PRECISION_1": 10,
}


def writeConfig(parameters):
    with open(config_file, "w") as f:
        for key, value in parameters.items():
            f.write(f"set PARAMS({key}) {value}\n")


def main():
    writeConfig(template)
    print(f"Configuration written to {config_file}")
    os.system('vivado -mode batch -nolog -nojou -source generate.tcl')


if __name__ == '__main__':
    main()
