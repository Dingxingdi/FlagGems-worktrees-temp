import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64, "TILE_N": 64, "TILE_K": 64}),
        triton.Config({"TILE_M": 64, "TILE_N": 128, "TILE_K": 64}),
        triton.Config({"TILE_M": 128, "TILE_N": 64, "TILE_K": 64}),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 64}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _scaled_grouped_mm_kernel(
    mat1_ptr,
    mat2_ptr,
    output_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    batch_size,
    stride_mat1_batch,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_batch,
    stride_mat2_k,
    stride_mat2_n,
    stride_out_batch,
    stride_out_m,
    stride_out_n,
    stride_scale_a,
    stride_scale_b,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Get batch and position IDs
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_m = tl.cdiv(M, TILE_M)

    # Calculate batch index and position within batch
    num_pids_per_batch = num_pid_m * num_pid_n
    batch_idx = pid // num_pids_per_batch
    pid_in_batch = pid % num_pids_per_batch

    if batch_idx >= batch_size:
        return

    pid_m = pid_in_batch // num_pid_n
    pid_n = pid_in_batch % num_pid_n

    # Load scale factors for this batch
    scale_a = tl.load(scale_a_ptr + batch_idx * stride_scale_a)
    scale_b = tl.load(scale_b_ptr + batch_idx * stride_scale_b)
    combined_scale = scale_a * scale_b

    # Compute offsets
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_remaining = K - k * TILE_K

        # Compute valid mask for k dimension
        mask_k = k_remaining > offs_k

        # Compute pointers for mat1: (batch, M, K)
        # mat1[batch_idx, offs_m, offs_k]
        mat1_ptrs = (
            mat1_ptr
            + batch_idx * stride_mat1_batch
            + offs_m[:, None] * stride_mat1_m
            + (offs_k + k * TILE_K)[None, :] * stride_mat1_k
        )

        # Compute pointers for mat2: (batch, K, N)
        # mat2[batch_idx, offs_k, offs_n]
        mat2_ptrs = (
            mat2_ptr
            + batch_idx * stride_mat2_batch
            + (offs_k + k * TILE_K)[:, None] * stride_mat2_k
            + offs_n[None, :] * stride_mat2_n
        )

        # Load blocks with mask
        a = tl.load(mat1_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(mat2_ptrs, mask=mask_k[:, None], other=0.0)

        # Compute dot product
        acc += tl.dot(a, b)

    # Apply scaling
    acc = acc * combined_scale

    # Convert to output dtype
    acc = acc.to(output_ptr.dtype.element_ty)

    # Store result
    offs_out_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_out_n = pid_n * TILE_N + tl.arange(0, TILE_N)

    out_ptrs = (
        output_ptr
        + batch_idx * stride_out_batch
        + offs_out_m[:, None] * stride_out_m
        + offs_out_n[None, :] * stride_out_n
    )
    mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


def _scaled_grouped_mm_impl(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    scale_result: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    Triton implementation of scaled_grouped_mm.
    """
    logger.debug("GEMS SCALED_GROUPED_MM")

    # Determine if inputs are batched (3D) or not (2D)
    is_batched = mat1.dim() == 3

    if is_batched:
        batch_size = mat1.shape[0]
        M, K = mat1.shape[1], mat1.shape[2]
        _, N = mat2.shape[1], mat2.shape[2]

        # Ensure mat2 has batch dimension
        if mat2.dim() == 2:
            mat2 = mat2.unsqueeze(0)
            if scale_a.dim() == 1:
                scale_a = scale_a.unsqueeze(1)
            if scale_b.dim() == 1:
                scale_b = scale_b.unsqueeze(1)
    else:
        # Non-batched case: treat as single batch
        batch_size = 1
        M, K = mat1.shape
        _, N = mat2.shape

        # Add batch dimension
        mat1 = mat1.unsqueeze(0)
        mat2 = mat2.unsqueeze(0)
        if scale_a.dim() == 0:
            scale_a = scale_a.unsqueeze(0)
        if scale_b.dim() == 0:
            scale_b = scale_b.unsqueeze(0)

    # Determine output dtype
    if out_dtype is None:
        out_dtype = mat1.dtype

    # Allocate output tensor
    output = torch.empty((batch_size, M, N), dtype=out_dtype, device=mat1.device)

    # Launch kernel
    def grid(META):
        return (
            batch_size
            * triton.cdiv(M, META["TILE_M"])
            * triton.cdiv(N, META["TILE_N"]),
        )

    _scaled_grouped_mm_kernel[grid](
        mat1,
        mat2,
        output,
        scale_a,
        scale_b,
        M,
        N,
        K,
        batch_size,
        mat1.stride(0),
        mat1.stride(1),
        mat1.stride(2),
        mat2.stride(0),
        mat2.stride(1),
        mat2.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scale_a.stride(0) if scale_a.dim() > 0 else 0,
        scale_b.stride(0) if scale_b.dim() > 0 else 0,
    )

    # Apply scale_result if provided
    if scale_result is not None:
        if is_batched:
            for i in range(batch_size):
                sr = scale_result[i] if scale_result.dim() > 0 else scale_result
                output[i] = output[i] * sr
        else:
            if scale_result.dim() > 0:
                scale_result = scale_result.squeeze()
            output = output * scale_result

    # Return output with appropriate shape
    if not is_batched:
        output = output.squeeze(0)

    return output


# Main entry point
def scaled_grouped_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    scale_result: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    FlagGems implementation of scaled_grouped_mm.

    Performs group-wise matrix multiplication with per-group scaling factors.
    Each group (batch element) has its own scale factors.

    Args:
        mat1: Input tensor of shape (batch, M, K) or (M, K)
        mat2: Input tensor of shape (batch, K, N) or (K, N)
        scale_a: Scale factor for mat1, shape (batch,) or (batch, 1) or scalar
        scale_b: Scale factor for mat2, shape (batch,) or (batch, 1) or scalar
        scale_result: Optional scale factor for the result
        out_dtype: Optional output dtype
        use_fast_accum: Whether to use fast accumulation

    Returns:
        Output tensor of shape (batch, M, N) or (M, N)
    """
    logger.debug("GEMS SCALED_GROUPED_MM")
    return _scaled_grouped_mm_impl(
        mat1, mat2, scale_a, scale_b, scale_result, out_dtype, use_fast_accum
    )