import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    code.newline()
    code.newline()
    return code


def generate_scatter_nd_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    # Heuristic for block size
    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    # The decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    indices_stride_vars = ",".join(f"'indices_stride_{i}'" for i in range(2))  # indices is always 2D
    values_stride_vars = ",".join(f"'values_stride_{i}'" for i in range(2))  # values is at least 2D
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['M','N',"
        f"{inp_stride_vars},{indices_stride_vars},{values_stride_vars},{shape_vars}])"
    )

    # Signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("input_ptr,")
        code.writeline("indices_ptr,")
        code.writeline("values_ptr,")

        # Input shape args
        for i in range(rank):
            code.writeline(f"shape_{i}: int,")

        # Indices shape (M, rank)
        code.writeline("indices_M: int,")

        # Values shape
        code.writeline("values_M: int,")
        code.writeline("values_N: int,")

        # Input stride args
        for i in range(rank):
            code.writeline(f"inp_stride_{i}: int,")

        # Indices stride args
        code.writeline("indices_stride_0: int,")
        code.writeline("indices_stride_1: int,")

        # Values stride args
        code.writeline("values_stride_0: int,")
        code.writeline("values_stride_1: int,")

        code.writeline("M,")
        code.writeline("N,")
        code.writeline("IS_ACCUMULATE: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("offsets = pid * BLOCK + tl.arange(0, BLOCK)")

        # Flattened index: m_idx * N + n_idx
        code.writeline("mask = offsets < M * N")
        code.writeline("m_idx = offsets // N")
        code.writeline("n_idx = offsets % N")

        # Load indices for each dimension
        for i in range(rank):
            code.writeline(f"idx_{i} = tl.load(indices_ptr + m_idx * indices_stride_0 + {i} * indices_stride_1, mask=mask, other=0)")
            code.writeline(f"idx_{i} = idx_{i}.to(tl.int64)")

        # Validate indices are in bounds
        code.writeline("index_valid = mask")
        for i in range(rank):
            code.writeline(f"index_valid = index_valid & (idx_{i} >= 0) & (idx_{i} < shape_{i})")

        code.writeline("load_mask = index_valid")

        # Compute offset in input tensor
        code.writeline("input_offset = 0")
        for i in range(rank):
            code.writeline(f"input_offset = input_offset + idx_{i} * inp_stride_{i}")

        # Compute offset in values tensor
        code.writeline("values_offset = m_idx * values_stride_0 + n_idx * values_stride_1")

        # Load and scatter
        code.writeline("cur_value = tl.load(values_ptr + values_offset, mask=load_mask, other=0)")

        code.writeline("if IS_ACCUMULATE:")
        with code.indent():
            code.writeline("tl.atomic_add(input_ptr + input_offset, cur_value, mask=load_mask)")
        code.writeline("else:")
        with code.indent():
            code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=load_mask)")

    code.newline()
    code.newline()
    return code


def generate_scatter_nd_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(f"def {wrapper_name}(input, indices, values, accumulate):")
    with code.indent():
        # Get shapes
        code.writeline("input_shape = list(input.shape)")
        code.writeline("indices_shape = list(indices.shape)")
        code.writeline("values_shape = list(values.shape)")

        # Get strides
        code.writeline("input_stride = list(input.stride())")
        code.writeline("indices_stride = list(indices.stride())")
        code.writeline("values_stride = list(values.stride())")

        code.writeline("M = indices_shape[0]")  # Number of indices
        code.writeline("N = values_shape[1] if len(values_shape) > 1 else 1")  # Elements per index

        code.newline()
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M * N, meta['BLOCK']),")
        code.writeline(")")

        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("input,")
            code.writeline("indices,")
            code.writeline("values,")

            # Input shape
            for i in range(rank):
                code.writeline(f"input_shape[{i}],")

            # Indices M
            code.writeline("indices_shape[0],")

            # Values shape
            code.writeline("values_shape[0],")
            code.writeline("values_shape[1] if len(values_shape) > 1 else 1,")

            # Input strides
            for i in range(rank):
                code.writeline(f"input_stride[{i}],")

            # Indices strides
            code.writeline("indices_stride[0],")
            code.writeline("indices_stride[1],")

            # Values strides
            code.writeline("values_stride[0],")
            code.writeline("values_stride[1] if len(values_stride) > 1 else 0,")

            code.writeline("M,")
            code.writeline("N,")
            code.writeline("accumulate==True,")
        code.writeline(")")
        code.writeline("return input")

    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: (input, indices, values)
    input = inputs[0]
    rank = input.ndim

    code = generate_imports(code)
    code = generate_scatter_nd_kernel(rank, kernel_name, code)
    code = generate_scatter_nd_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterNDFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_scatter_nd_wrapper",
                "_scatter_nd_jit_function",
                code,
            )

            file_name = f"scatter_nd_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_scatter_nd_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_nd_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        # Use input rank as key
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_nd_func = ScatterNDFunction()


def scatter_nd(input, indices, values, accumulate=False):
    """
    ScatterND operation: writes values to input at positions specified by indices.

    Args:
        input: The input tensor to scatter into
        indices: Integer tensor of shape (M, RANK) where RANK = input.ndim
                 Each row specifies an index into input
        values: Tensor of shape (M,) or (M, ...) containing values to scatter
        accumulate: If True, accumulate instead of overwrite

    Returns:
        The input tensor with values scattered at specified positions
    """
    logger.debug("GEMS SCATTER_ND")

    output = input.clone()
    return scatter_nd_(output, indices, values, accumulate)


def scatter_nd_(input, indices, values, accumulate=False):
    """
    In-place ScatterND operation.
    """
    logger.debug("GEMS SCATTER_ND_")

    # Validate input shapes
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got {indices.ndim}D")
    if indices.shape[1] != input.ndim:
        raise ValueError(
            f"indices.shape[1] ({indices.shape[1]}) must equal input.ndim ({input.ndim})"
        )

    # bfloat16 does not support atomic_add in Triton
    if accumulate and input.dtype == torch.bfloat16:
        raise ValueError(
            "Unsupported operation: scatter_nd accumulate on bfloat16 tensors."
        )

    # Ensure contiguous
    if not indices.is_contiguous():
        indices = indices.contiguous()
    if not values.is_contiguous():
        values = values.contiguous()

    # Handle different value shapes
    if values.ndim == 1:
        # values shape is (M,), reshape to (M, 1)
        values = values.unsqueeze(-1)

    _scatter_nd_func(input, indices, values, accumulate)

    return input