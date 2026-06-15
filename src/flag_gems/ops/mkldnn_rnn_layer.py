import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

exp = tl_extra_shim.exp


@libentry()
@triton.jit
def lstm_gates_kernel(
    gates_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel that applies sigmoid and tanh activations to LSTM gates.

    This kernel takes pre-computed gates (from matmul) and applies the
    LSTM activation functions element-wise.
    """
    # Each program processes one batch element
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Offsets for this batch element
    base_off = pid * 4 * hidden_size

    # Process each gate group (i, f, g, o)
    for gate_idx in range(4):
        gate_base = base_off + gate_idx * hidden_size

        # Load gate values
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        gate_vals = tl.load(gates_ptr + gate_base + offsets, mask=mask, other=0.0)

        # Apply activation based on gate type
        if gate_idx == 2:  # g gate - tanh
            # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            activated = (exp(2.0 * gate_vals) - 1.0) / (exp(2.0 * gate_vals) + 1.0)
        else:  # i, f, o gates - sigmoid
            # sigmoid(x) = 1 / (1 + exp(-x))
            activated = 1.0 / (1.0 + exp(-gate_vals))

        # Store activated gates
        tl.store(output_ptr + gate_base + offsets, activated, mask=mask)


def mkldnn_rnn_layer(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_,
    reverse=False,
    batch_sizes=[],
    mode=2,  # LSTM
    hidden_size=5,
    num_layers=1,
    has_biases=False,
    bidirectional=False,
    batch_first=False,
    train=False,
):
    """Triton implementation of mkldnn_rnn_layer for LSTM.

    This implementation:
    1. Uses PyTorch's matmul (cuBLAS) for matrix multiplications
    2. Uses a custom Triton kernel for activation functions

    Falls back to PyTorch's mkldnn_rnn_layer for unsupported configurations.
    """
    logger.debug("GEMS MKLDNN_RNN_LAYER")

    # Validate parameters - only support limited configuration
    supported_config = (
        mode == 2 and  # LSTM
        num_layers == 1 and
        not bidirectional and
        not has_biases and
        not reverse and
        len(batch_sizes) == 0 and
        not batch_first
    )

    if not supported_config:
        logger.debug("GEMS MKLDNN_RNN_LAYER - falling back to PyTorch")
        result = torch.mkldnn_rnn_layer(
            input, weight0, weight1, weight2, weight3,
            hx_, cx_,
            reverse=reverse,
            batch_sizes=batch_sizes,
            mode=mode,
            hidden_size=hidden_size,
            num_layers=num_layers,
            has_biases=has_biases,
            bidirectional=bidirectional,
            batch_first=batch_first,
            train=train
        )
        if len(result) == 4:
            output, hy, cy, reserved = result
        else:
            output, hy, cy = result
            reserved = torch.empty(0, device=input.device)
        return (output, hy, cy, reserved)

    # Get dimensions: input is (seq_len, batch, input_size)
    seq_len, batch_size, input_size = input.shape

    # Prepare weights
    weight0 = weight0.contiguous()
    weight1 = weight1.contiguous()

    # Allocate output tensors
    output = torch.empty((seq_len, batch_size, hidden_size), dtype=input.dtype, device=input.device)
    hy = torch.empty((num_layers, batch_size, hidden_size), dtype=input.dtype, device=input.device)
    cy = torch.empty((num_layers, batch_size, hidden_size), dtype=input.dtype, device=input.device)
    reserved = torch.empty(36864, dtype=torch.uint8, device=input.device)

    # Initialize states
    h = hx_[0].clone()
    c = cx_[0].clone()

    # Block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = max(32, min(BLOCK_SIZE, 128))

    # Process each timestep
    with torch_device_fn.device(input.device):
        for t in range(seq_len):
            # Get input at timestep t
            x_t = input[t]

            # Compute gates using matrix multiplication (cuBLAS)
            gates_ih = torch.matmul(x_t, weight0.T)
            gates_hh = torch.matmul(h, weight1.T)
            gates = gates_ih + gates_hh

            # Apply activations using Triton kernel
            activated_gates = torch.empty_like(gates)
            lstm_gates_kernel[(batch_size,)](
                gates,
                activated_gates,
                hidden_size,
                batch_size,
                BLOCK_SIZE,
            )

            # Extract gates
            i = activated_gates[:, 0:hidden_size]
            f = activated_gates[:, hidden_size:2*hidden_size]
            g = activated_gates[:, 2*hidden_size:3*hidden_size]
            o = activated_gates[:, 3*hidden_size:4*hidden_size]

            # Update states
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)

            # Store output
            output[t] = h_new

            c = c_new
            h = h_new

    hy[0] = h
    cy[0] = c

    return (output, hy, cy, reserved)