import torch
from torch import Tensor

from .utils import block, my_clamp, my_round, unblock


def _mxint_quantize(
    x: Tensor,
    width: int = 8,
    exponent_width: int = 4,
    block_size: list[int] = [16],
    skip_first_dim: bool = True,
):
    """
    - Convert IEEE FP32/64 to Microscaling Interger (MXINT), where an exponent is shared over all elements in a block.
    - https://arxiv.org/pdf/2310.10537.pdf
    - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    ---
    - forward: convert IEEE FP32/64 to MXINT
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.
    """
    
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    exponent_bias = 2 ** (exponent_width - 1)

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    # exponent
    exponent = torch.ceil(torch.log2(torch.tensor([15]))) - exponent_bias
    exponent = torch.clamp(exponent, exponent_min, exponent_max)
    # mantissa
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = blocked_x / 2**exponent
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)
    q_x = (2**exponent) * mantissa
    
    mxint_x = unblock(
        q_x,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )
    
    return mxint_x


class MXINTQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        width: int = 8,
        exponent_width: int = 4,
        block_size: list[int] = [16],
        skip_first_dim: bool = True,
    ):
        return _mxint_quantize(
            x,
            width=width,
            exponent_width=exponent_width,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


def mxint_old_quantizer(
    x: Tensor,
    width: int = 8,
    exponent_width: int = 4,
    block_size: list[int] = [16],
    skip_first_dim: bool = True,
):
    return MXINTQuantize.apply(
        x,
        width,
        exponent_width,
        block_size,
        skip_first_dim,
    )
