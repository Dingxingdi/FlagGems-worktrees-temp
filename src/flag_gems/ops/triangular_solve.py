import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=2),
    ],
    key=["k"],
)
@triton.jit
def triangular_solve_kernel(
    b_ptr,
    a_ptr,
    solution_ptr,
    cloned_a_ptr,
    b_stride_b,
    b_stride_n,
    b_stride_k,
    a_stride_b,
    a_stride_n,
    solution_stride_b,
    solution_stride_n,
    solution_stride_k,
    cloned_a_stride_b,
    cloned_a_stride_n,
    n,
    k,
    batch_size,
    upper: tl.constexpr,
    transpose: tl.constexpr,
    unitriangular: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for triangular solve: AX = B
    Sequential substitution within each batch.
    Grid: (batch_size, ) - one thread block per batch element

    Each thread processes one batch, computing the triangular solve sequentially.
    """
    # Grid is (batch_size, )
    batch_idx = tl.program_id(0)

    if batch_idx >= batch_size:
        return

    # Pointers for this batch
    b_base = b_ptr + batch_idx * b_stride_b
    a_base = a_ptr + batch_idx * a_stride_b
    solution_base = solution_ptr + batch_idx * solution_stride_b
    cloned_a_base = cloned_a_ptr + batch_idx * cloned_a_stride_b

    # First, copy b to solution (initialize with b values)
    for i in range(n):
        for col in range(k):
            b_val = tl.load(b_base + i * b_stride_n + col)
            tl.store(solution_base + i * solution_stride_n + col, b_val)

    # Now do triangular solve
    if upper:
        # Backward substitution: process rows from n-1 down to 0
        for i in range(n - 1, -1, -1):
            # Load diagonal element (convert to float32 for computation)
            diag = tl.load(a_base + i * a_stride_n + i).to(tl.float32)

            # For each column k
            for col in range(k):
                # Load solution[i, col]
                val = tl.load(solution_base + i * solution_stride_n + col).to(tl.float32)

                # Compute sum: sum(A[i, j] * x[j]) for j > i
                sum_val = 0.0
                for j in range(i + 1, n):
                    a_ij = tl.load(a_base + i * a_stride_n + j).to(tl.float32)
                    xj = tl.load(solution_base + j * solution_stride_n + col).to(tl.float32)
                    sum_val = sum_val + a_ij * xj

                # Compute x[i, col]
                if unitriangular:
                    x_i = val - sum_val
                else:
                    x_i = (val - sum_val) / diag

                # Store solution
                tl.store(solution_base + i * solution_stride_n + col, x_i)
    else:
        # Forward substitution: process rows from 0 to n-1
        for i in range(n):
            # Load diagonal element
            diag = tl.load(a_base + i * a_stride_n + i).to(tl.float32)

            # For each column k
            for col in range(k):
                # Load solution[i, col]
                val = tl.load(solution_base + i * solution_stride_n + col).to(tl.float32)

                # Compute sum: sum(A[i, j] * x[j]) for j < i
                sum_val = 0.0
                for j in range(0, i):
                    a_ij = tl.load(a_base + i * a_stride_n + j).to(tl.float32)
                    xj = tl.load(solution_base + j * solution_stride_n + col).to(tl.float32)
                    sum_val = sum_val + a_ij * xj

                # Compute x[i, col]
                if unitriangular:
                    x_i = val - sum_val
                else:
                    x_i = (val - sum_val) / diag

                # Store solution
                tl.store(solution_base + i * solution_stride_n + col, x_i)

    # Copy A to cloned_A
    for i in range(n):
        for j in range(n):
            a_val = tl.load(a_base + i * a_stride_n + j)
            tl.store(cloned_a_base + i * cloned_a_stride_n + j, a_val)


def triangular_solve(b, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS triangular_solve")

    # Handle batch dimensions
    if A.dim() == 2:
        # Single matrix case: A is (n, n), b is (n, k)
        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = A.shape[0]
    n = A.shape[1]
    k = b.shape[2]

    # Output tensors
    solution = torch.empty_like(b)
    cloned_A = torch.empty_like(A)

    # Grid: one block per batch element
    grid_fn = lambda meta: (batch_size,)

    with torch_device_fn.device(A.device):
        triangular_solve_kernel[grid_fn](
            b,
            A,
            solution,
            cloned_A,
            b.stride(0),
            b.stride(1),
            b.stride(2),
            A.stride(0),
            A.stride(1),
            solution.stride(0),
            solution.stride(1),
            solution.stride(2),
            cloned_A.stride(0),
            cloned_A.stride(1),
            n,
            k,
            batch_size,
            upper,
            transpose,
            unitriangular,
        )

    if squeeze_output:
        solution = solution.squeeze(0)
        cloned_A = cloned_A.squeeze(0)

    # Return as a namedtuple-like object with .solution and .cloned_coefficient attributes
    # and also support indexing like a tuple
    class TriangularSolveResult:
        def __init__(self, solution, cloned_coefficient):
            self.solution = solution
            self.cloned_coefficient = cloned_coefficient

        def __getitem__(self, idx):
            if idx == 0:
                return self.solution
            elif idx == 1:
                return self.cloned_coefficient
            else:
                raise IndexError("TriangularSolveResult index out of range")

        def __len__(self):
            return 2

        def __iter__(self):
            yield self.solution
            yield self.cloned_coefficient

    return TriangularSolveResult(solution, cloned_A)