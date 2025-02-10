from functools import partial

import torch
import torch.nn.functional as F

from chop.nn.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
)


def leaky_relu_integer(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=True
        )
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_binary(x, negative_slope=0.1, inplace=False, config=None):
    # Note: the binary quantizer is provided for path-completion.
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_stochastic = config["data_in_stochastic"]
        x_bipolar = config["data_in_bipolar"]
        x_quantizer = partial(binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar)
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_ternary(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_scaling_factor = config["data_in_scaling_factor"]
        x_quantizer = partial(ternary_quantizer, scaling_factor=x_scaling_factor)
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_minifloat_denorm(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_width = config["data_in_exponent_width"]
        x_exponent_bias = config["data_in_exponent_bias"]

        x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_minifloat_ieee(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_width = config["data_in_exponent_width"]
        x_exponent_bias = config["data_in_exponent_bias"]

        x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_log(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_bias = config["data_in_exponent_bias"]

        x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope, inplace=inplace)


def leaky_relu_block_fp(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_width = config["data_in_exponent_width"]
        x_exponent_bias = config["data_in_exponent_bias"]
        x_block_size = config["data_in_block_size"]

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )

        x_shape = list(x.shape)
        if x_more_than_2_dims:
            x = torch.flatten(x, start_dim=0, end_dim=-3)
        x = x_quantizer(x)
        x = torch.reshape(x, x_shape)
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)


def leaky_relu_block_minifloat(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_width = config["data_in_exponent_width"]
        x_exponent_bias_width = config["data_in_exponent_bias_width"]
        x_block_size = config["data_in_block_size"]

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )

        x_shape = list(x.shape)
        if x_more_than_2_dims:
            x = torch.flatten(x, start_dim=0, end_dim=-3)
        x = x_quantizer(x)
        x = torch.reshape(x, x_shape)
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)


def leaky_relu_block_log(x, negative_slope=0.1, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    else:
        x_width = config["data_in_width"]
        x_exponent_bias_width = config["data_in_exponent_bias_width"]
        x_block_size = config["data_in_block_size"]

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_log_quantizer,
            width=x_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )

        x_shape = list(x.shape)
        if x_more_than_2_dims:
            x = torch.flatten(x, start_dim=0, end_dim=-3)
        x = x_quantizer(x)
        x = torch.reshape(x, x_shape)
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
