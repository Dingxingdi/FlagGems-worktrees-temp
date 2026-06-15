import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["seq_len"],
)
@triton.jit
def rnn_tanh_kernel(
    input_ptr,
    hx_ptr,
    w_ih_ptr,
    w_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hn_ptr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    seq_len: tl.constexpr,
    num_layers: tl.constexpr,
    has_biases: tl.constexpr,
    batch_first: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RNN tanh forward kernel for single layer, unidirectional."""
    # This kernel handles a single layer unidirectional RNN
    pid = tle.program_id(0)

    # Calculate which batch and layer this program handles
    batch_idx = pid
    layer_idx = 0  # Only single layer supported in this kernel

    # Load input for first timestep
    if batch_first:
        # input shape: (batch, seq_len, input_size)
        input_offset = batch_idx * seq_len * input_size
    else:
        # input shape: (seq_len, batch, input_size)
        input_offset = batch_idx * input_size

    # Load initial hidden state
    # hx shape: (num_layers, batch, hidden_size)
    hx_offset = layer_idx * batch_size * hidden_size
    h_ptr = hx_ptr + hx_offset + batch_idx * hidden_size

    # Load weights
    # w_ih: (hidden_size, input_size)
    # w_hh: (hidden_size, hidden_size)
    w_ih_offset = layer_idx * hidden_size * input_size
    w_hh_offset = layer_idx * hidden_size * hidden_size

    # Load hidden state h_0
    h_current = tl.load(h_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size)
    h_current = h_current.to(tl.float32)

    # Process each timestep
    for t in range(seq_len):
        # Load input x_t
        if batch_first:
            x_offset = batch_idx * seq_len * input_size + t * input_size
        else:
            x_offset = t * batch_size * input_size + batch_idx * input_size

        x_t = tl.load(input_ptr + x_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < input_size)
        x_t = x_t.to(tl.float32)

        # Compute h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)
        # wx = W_ih @ x_t
        wx = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(hidden_size):
            for j in range(input_size):
                w_ih_val = tl.load(w_ih_ptr + w_ih_offset + i * input_size + j)
                wx = wx + w_ih_val * x_t[j]

        # wh = W_hh @ h_{t-1}
        wh = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(hidden_size):
            for j in range(hidden_size):
                w_hh_val = tl.load(w_hh_ptr + w_hh_offset + i * hidden_size + j)
                wh = wh + w_hh_val * h_current[j]

        # Add biases
        if has_biases:
            b_ih = tl.load(bias_ih_ptr + layer_idx * hidden_size + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size)
            b_hh = tl.load(bias_hh_ptr + layer_idx * hidden_size + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size)
            h_t = wx + wh + b_ih.to(tl.float32) + b_hh.to(tl.float32)
        else:
            h_t = wx + wh

        # Apply tanh activation
        h_t = tl.math.tanh(h_t)

        # Store output
        if batch_first:
            out_offset = batch_idx * seq_len * hidden_size + t * hidden_size
        else:
            out_offset = t * batch_size * hidden_size + batch_idx * hidden_size

        tl.store(output_ptr + out_offset + tl.arange(0, BLOCK_SIZE), h_t, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)

        # Update hidden state
        h_current = h_t

    # Store final hidden state
    tl.store(hn_ptr + hx_offset + batch_idx * hidden_size + tl.arange(0, BLOCK_SIZE), h_current, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)


def rnn_tanh(
    input,
    hx,
    params,
    has_biases=False,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    """RNN with tanh activation.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size) if batch_first=False
               or (batch, seq_len, input_size) if batch_first=True
        hx: Initial hidden state of shape (num_layers, batch, hidden_size)
        params: Tuple of weight tensors
        has_biases: Whether to use biases
        num_layers: Number of RNN layers
        dropout: Dropout rate (only used when num_layers > 1)
        train: Whether in training mode
        bidirectional: Whether to use bidirectional RNN
        batch_first: Whether input is (batch, seq, feature) format

    Returns:
        output: Output tensor
        hn: Final hidden state
    """
    logger.debug("GEMS RNN_TANH FORWARD")

    # Simple implementation using Python loop with torch operations
    # This leverages existing optimized kernels for matmul, add, tanh

    if bidirectional:
        raise NotImplementedError("Bidirectional RNN not supported yet")

    if num_layers > 1:
        raise NotImplementedError("Multi-layer RNN not supported yet")

    # Parse input shapes
    if batch_first:
        batch_size, seq_len, input_size = input.shape
    else:
        seq_len, batch_size, input_size = input.shape

    num_layers_out, batch_size_out, hidden_size = hx.shape

    # Unpack parameters for first layer
    # params: (weight_ih_l0, weight_hh_l0, [bias_ih_l0, bias_hh_l0] if has_biases)
    w_ih = params[0]
    w_hh = params[1]

    if has_biases and len(params) > 2:
        bias_ih = params[2]
        bias_hh = params[3]
    else:
        bias_ih = None
        bias_hh = None

    # Prepare output tensor
    if batch_first:
        output = torch.zeros(batch_size, seq_len, hidden_size, dtype=input.dtype, device=input.device)
    else:
        output = torch.zeros(seq_len, batch_size, hidden_size, dtype=input.dtype, device=input.device)

    # Initial hidden state
    h = hx[0]  # (batch, hidden_size)

    # Process each timestep
    for t in range(seq_len):
        # Get input at timestep t
        if batch_first:
            x_t = input[:, t, :]  # (batch, input_size)
        else:
            x_t = input[t, :, :]  # (batch, input_size)

        # Compute: h = tanh(x @ W_ih^T + h @ W_hh^T + b_ih + b_hh)
        # Using torch operations which will be intercepted by FlagGems
        h = torch.mm(x_t, w_ih.t())  # (batch, hidden_size)

        h = h + torch.mm(h, w_hh.t())  # (batch, hidden_size)

        if has_biases:
            h = h + bias_ih + bias_hh

        h = torch.tanh(h)

        # Store output
        if batch_first:
            output[:, t, :] = h
        else:
            output[t, :, :] = h

    # Final hidden state
    hn = h.unsqueeze(0)  # (1, batch, hidden_size)

    return output, hn


def rnn_tanh_(
    input,
    hx,
    params,
    has_biases=False,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    """In-place RNN with tanh activation (currently same as rnn_tanh)."""
    return rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)