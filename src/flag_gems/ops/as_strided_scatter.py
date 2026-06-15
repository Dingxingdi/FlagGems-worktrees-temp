import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

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


def generate_as_strided_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics({\"BLOCK\": heur_block})")
    stride_vars = ",".join(f"'stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    size_vars = ",".join(f"'size_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','storage_offset',{stride_vars},{src_stride_vars},{size_vars}])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("src_strided,")
        code.writeline("inp,")
        code.writeline("out,")

        stride_args = ", ".join(f"stride_{i}: int" for i in range(rank))
        code.writeline(f"{stride_args}, # stride for as_strided view (output indexing)")

        stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
        code.writeline(f"{stride_args}, # stride for src")

        size_args = ", ".join(f"size_{i}: int" for i in range(rank))
        code.writeline(f"{size_args}, # size (shape of as_strided view)")
        code.writeline("storage_offset,")
        code.writeline("N,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * BLOCK + tl.arange(0, BLOCK)")
        code.writeline("mask = offsets < N")

        code.newline()
        code.writeline("if INT32_OFFSET:")
        with code.indent():
            code.writeline("storage_offset = storage_offset.to(tl.int32)")
            for i in range(rank):
                code.writeline(f"stride_{i} = stride_{i}.to(tl.int32)")
                code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                code.writeline(f"size_{i} = size_{i}.to(tl.int32)")
        code.newline()

        code.writeline("# Convert flat src index to multi-dim indices")
        code.writeline("cur_idx = offsets")
        for i in range(rank)[::-1]:
            code.writeline(f"src_idx_{i} = cur_idx % size_{i}")
            if i != 0:
                code.writeline(f"cur_idx = cur_idx // size_{i}")

        code.newline()
        code.writeline("# Compute src memory offset (for non-contiguous src)")
        code.writeline("src_offset = tl.zeros((BLOCK, ), dtype=tl.int64)")
        for i in range(rank):
            code.writeline(f"src_offset += src_idx_{i} * src_stride_{i}")
        # Note: src_stride_{i} corresponds to dimension i of src tensor

        code.newline()
        code.writeline("# Load from src")
        code.writeline("src_data = tl.load(src_strided + src_offset, mask=mask, other=0)")

        code.newline()
        code.writeline("# Compute target linear index in output")
        code.writeline("target_offset = storage_offset")
        for i in range(rank):
            code.writeline(f"target_offset += src_idx_{i} * stride_{i}")

        code.newline()
        code.writeline("# Store to output")
        code.writeline("tl.store(out + target_offset, src_data, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(f"def {wrapper_name}(src, inp, out, size, stride, storage_offset):")
    with code.indent():
        code.writeline("src_strides = src.stride()")
        code.writeline("N = src.numel()")

        code.writeline("int32_offset = True")

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(N, meta['BLOCK']), ")
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("src, inp, out,")
            for i in range(rank):
                code.writeline(f"stride[{i}], # user-provided stride for output indexing")
            for i in range(rank):
                code.writeline(f"src_strides[{i}],")
            for i in range(rank):
                code.writeline(f"size[{i}],")
            code.writeline("storage_offset,")
            code.writeline("N,")
            code.writeline("INT32_OFFSET=int32_offset,")
        code.writeline(")")
        code.writeline("return out")

    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs = (src, inp, out, size, stride, storage_offset)
    src, inp, out, size, stride, storage_offset = inputs
    rank = len(size)

    code = generate_imports(code)
    code = generate_as_strided_scatter_kernel(rank, kernel_name, code)
    code = generate_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class AsStridedScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_as_strided_scatter_wrapper",
                "_as_strided_scatter_jit",
                code,
            )

            file_name = f"as_strided_scatter_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_as_strided_scatter_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args, **kwargs):
        src, inp, out, size, stride, storage_offset = args
        rank = len(size)
        return f"{rank}"


_as_strided_scatter_func = AsStridedScatterFunction()


def as_strided_scatter(inp, src, size, stride, storage_offset=None):
    logger.debug("GEMS AS_STRIDED_SCATTER")

    if storage_offset is None:
        storage_offset = 0

    # Validate inputs
    src_view_shape = torch.as_strided(inp, size, stride, storage_offset).shape
    assert list(src.shape) == list(
        src_view_shape
    ), f"src shape {src.shape} must match as_strided view shape {src_view_shape}"

    # Check for overlapping positions in the as_strided view
    view = torch.as_strided(inp, size, stride, storage_offset)
    # If the view has overlapping memory positions, PyTorch would throw an error
    # We let PyTorch handle this validation

    # Create output tensor with fresh storage
    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    # Copy input to output first
    out.copy_(inp)

    # Make src contiguous for easier indexing
    src = src.contiguous()

    _as_strided_scatter_func(src, inp, out, size, stride, storage_offset)

    return out