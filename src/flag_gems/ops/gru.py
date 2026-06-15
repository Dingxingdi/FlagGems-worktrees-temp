import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2
tanh = tl_extra_shim.tanh


@triton.jit
def sigmoid(x):
    """Sigmoid function implemented using exp2."""
    log2e: tl.constexpr = 1.4426950408889634
    return 1.0 / (1.0 + exp2(-x * log2e))


@libentry()
@triton.jit
def gru_cell_kernel(
    # Input
    input_ptr,
    hx_ptr,
    # Weights
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    # Output
    output_ptr,
    # Dimensions
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_size: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """GRU cell kernel - computes one time step of GRU."""
    # Each program handles one batch element
    pid = tl.program_id(0)

    # Offsets
    input_offset = pid * input_size
    hx_offset = pid * hidden_size
    output_offset = pid * hidden_size

    # Load bias (broadcast to all batches)
    # For bias_ih: shape (3*hidden_size,)
    # For bias_hh: shape (3*hidden_size,)

    # GRU gates: reset (r), update (z), new (n)
    # Each gate has hidden_size elements

    # Compute gates for r, z, n
    # gates_ih = input @ weight_ih.T + bias_ih
    # gates_hh = hx @ weight_hh.T + bias_hh
    # gates = gates_ih + gates_hh
    # r = sigmoid(gates[0:hidden_size])
    # z = sigmoid(gates[hidden_size:2*hidden_size])
    # n = tanh(gates[2*hidden_size:3*hidden_size])
    # h_new = (1 - z) * n + z * hx

    # Compute input contribution
    gates_ih = tl.zeros((hidden_size,), tl.float32)

    # Matrix multiplication: input (1, input_size) @ weight_ih (3*hidden_size, input_size).T
    # Actually we need: input @ weight_ih.T
    # weight_ih.T has shape (input_size, 3*hidden_size)
    # Result has shape (1, 3*hidden_size), but we want (3*hidden_size,)

    # Loop over hidden_size * 3 gates
    for i in range(3 * hidden_size):
        # Compute dot product of input with weight_ih column i
        val = 0.0
        # Unroll the inner loop for efficiency
        for j in range(input_size):
            input_val = tl.load(input_ptr + input_offset + j).to(tl.float32)
            weight_val = tl.load(weight_ih_ptr + i * input_size + j).to(tl.float32)
            val = val + input_val * weight_val
        gates_ih = tl.where(i == tl.arange(0, 3 * hidden_size), val, gates_ih)

    # Load bias_ih
    bias_ih = tl.load(bias_ih_ptr + tl.arange(0, 3 * hidden_size)).to(tl.float32)
    gates_ih = gates_ih + bias_ih

    # Compute hidden contribution
    gates_hh = tl.zeros((hidden_size,), tl.float32)
    for i in range(3 * hidden_size):
        val = 0.0
        for j in range(hidden_size):
            hx_val = tl.load(hx_ptr + hx_offset + j).to(tl.float32)
            weight_val = tl.load(weight_hh_ptr + i * hidden_size + j).to(tl.float32)
            val = val + hx_val * weight_val
        gates_hh = tl.where(i == tl.arange(0, 3 * hidden_size), val, gates_hh)

    # Load bias_hh
    bias_hh = tl.load(bias_hh_ptr + tl.arange(0, 3 * hidden_size)).to(tl.float32)
    gates_hh = gates_hh + bias_hh

    # Total gates
    gates = gates_ih + gates_hh

    # Split into r, z, n gates
    r = sigmoid(gates[:hidden_size])
    z = sigmoid(gates[hidden_size:2 * hidden_size])
    n = tanh(gates[2 * hidden_size:3 * hidden_size])

    # Load hx
    hx = tl.load(hx_ptr + hx_offset + tl.arange(0, hidden_size)).to(tl.float32)

    # Compute new hidden state: h_new = (1 - z) * n + z * hx
    h_new = (1.0 - z) * n + z * hx

    # Store result
    tl.store(output_ptr + output_offset + tl.arange(0, hidden_size), h_new)


def gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    """
    GRU cell operation - single time step.

    Args:
        input: Input tensor of shape (batch, input_size)
        hx: Hidden state of shape (batch, hidden_size)
        weight_ih: Input-to-hidden weights of shape (3*hidden_size, input_size)
        weight_hh: Hidden-to-hidden weights of shape (3*hidden_size, hidden_size)
        bias_ih: Input-to-hidden bias of shape (3*hidden_size,)
        bias_hh: Hidden-to-hidden bias of shape (3*hidden_size,)

    Returns:
        New hidden state of shape (batch, hidden_size)
    """
    batch = input.shape[0]
    hidden_size = hx.shape[1]
    input_size = input.shape[1]

    output = torch.empty_like(hx)

    # For small hidden sizes, use the Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (batch,)

    gru_cell_kernel[grid](
        input,
        hx,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        output,
        input_size,
        hidden_size,
        batch,
        BLOCK_SIZE,
    )

    return output


def gru(
    input: torch.Tensor,
    hx: torch.Tensor,
    params: list,
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple:
    """
    GRU forward pass.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
        hx: Initial hidden state of shape (num_layers * num_directions, batch, hidden_size)
        params: List of weight tensors
        has_biases: Whether to use bias
        num_layers: Number of layers
        dropout: Dropout probability (only used if num_layers > 1)
        train: Whether in training mode
        bidirectional: Whether to use bidirectional GRU
        batch_first: Whether input is (batch, seq, feature) format

    Returns:
        output: Output tensor of shape (seq_len, batch, hidden_size * num_directions)
        hidden: Final hidden state of shape (num_layers * num_directions, batch, hidden_size)
    """
    logger.debug("GEMS GRU")

    # Handle batch_first
    if batch_first:
        input = input.transpose(0, 1)  # (seq_len, batch, input_size)

    seq_len, batch, input_size = input.shape
    num_directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]

    # For now, only support single layer, non-bidirectional
    if num_layers != 1:
        raise NotImplementedError("Only single layer GRU is supported")
    if bidirectional:
        raise NotImplementedError("Bidirectional GRU is not supported")
    if dropout > 0 and num_layers > 1:
        raise NotImplementedError("Dropout is not supported for multi-layer GRU")

    # Extract weights
    # For single layer with bias: [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]
    # For single layer without bias: [weight_ih_l0, weight_hh_l0]
    if has_biases:
        weight_ih = params[0]
        weight_hh = params[1]
        bias_ih = params[2]
        bias_hh = params[3]
    else:
        weight_ih = params[0]
        weight_hh = params[1]
        bias_ih = torch.zeros(3 * hidden_size, device=input.device, dtype=input.dtype)
        bias_hh = torch.zeros(3 * hidden_size, device=input.device, dtype=input.dtype)

    # Process each time step
    output_seq = []
    h = hx[0]  # Shape: (batch, hidden_size)

    for t in range(seq_len):
        x_t = input[t]  # Shape: (batch, input_size)
        h = gru_cell(x_t, h, weight_ih, weight_hh, bias_ih, bias_hh)
        output_seq.append(h)

    # Stack outputs
    output = torch.stack(output_seq, dim=0)  # (seq_len, batch, hidden_size)

    # Final hidden state
    hidden = h.unsqueeze(0)  # (1, batch, hidden_size)

    return output, hidden