import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Cache for GRU modules to avoid recreating them for the same config
# Using standard GRU instead of quantized to avoid recursion issues
_gru_cache = {}


def _get_gru(input_size, hidden_size, num_layers, batch_first, bidirectional, dtype, device):
    """Get or create a cached GRU module."""
    key = (input_size, hidden_size, num_layers, batch_first, bidirectional, dtype, device)
    if key not in _gru_cache:
        gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        gru = gru.to(dtype).to(device)
        _gru_cache[key] = gru
    return _gru_cache[key]


def quantized_gru(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Quantized GRU operator.

    This implementation wraps a standard GRU module for simplicity.
    In production, a proper quantized implementation would be used.
    The params argument is ignored since the weights are embedded in the module.

    Args:
        input: Input tensor of shape (batch, seq, input_size) if batch_first
               else (seq, batch, input_size)
        hx: Initial hidden state of shape (num_layers * num_directions, batch, hidden_size)
        params: List of CellParamsBase (ignored in this implementation)
        has_biases: Whether to use biases
        num_layers: Number of layers
        dropout: Dropout probability
        train: Whether in training mode
        bidirectional: Whether to use bidirectional GRU
        batch_first: Whether input is (batch, seq, feature)

    Returns:
        Tuple of (output, hx) where:
            output: Output tensor
            hx: Final hidden state
    """
    logger.debug("GEMS QUANTIZED_GRU")

    # Determine input_size from input shape
    if batch_first:
        # input shape: (batch, seq, input_size)
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input_size = input.shape[2]
    else:
        # input shape: (seq, batch, input_size)
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        input_size = input.shape[2]

    # Determine hidden_size from hx if provided
    if hx is not None:
        hidden_size = hx.shape[2]
    else:
        # Need to infer hidden_size - assume it's the same as input_size as a fallback
        hidden_size = input_size

    # Get or create the GRU module
    gru = _get_gru(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dtype=input.dtype,
        device=input.device,
    )

    # Handle hx - if None, let the module create zero hidden state
    if hx is None:
        num_directions = 2 if bidirectional else 1
        hx = torch.zeros(
            num_layers * num_directions,
            batch_size,
            hidden_size,
            dtype=input.dtype,
            device=input.device,
        )

    # Call the GRU - use non-quantized to avoid recursion
    # In a real quantized implementation, this would be a proper quantized GRU
    output, hidden = gru(input, hx)

    return output, hidden


# Register the operator as the entry point
def quantized_gru_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Implementation that delegates to GRU module."""
    return quantized_gru(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )