import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_unsafe_index_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["out_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"out_stride{i}," for i in range(index_rank + inp_rank - indices_len)]
        args += [
            "M,",
            "N,",
            "BLOCK_SIZE0: tl.constexpr = 128,",
            "BLOCK_SIZE1: tl.constexpr = 1024,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = tle.program_id(axis=0)")
        code.writeline("pid1 = tle.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        else:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
        code.newline()
        code.writeline("cur_idx = offset1")
        for i in range(inp_rank - 1, indices_len - 1, -1):
            code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
            code.writeline(f"cur_idx = cur_idx // input_shape{i}")
        code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
            )
        code.newline()
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
        comp += [
            f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)
        ]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"indices_idx{i} * out_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * out_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(input_ptr + input_offset, mask=mask)")
        code.writeline("tl.store(out_ptr + out_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_unsafe_index_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']), ")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']), ")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"out_stride[{i}]," for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,"]
            code.writelines(args)
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
):
    inp_rank = inputs[0].ndim
    indices_len = len(inputs[1])
    if indices_len == 0:
        raise ValueError("At least one index tensor is required")
    index_rank = inputs[1][0].ndim
    code = generate_imports(code)
    generate_unsafe_index_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_unsafe_index_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class UnsafeIndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, out = args
        full_args = (inp, tensor_indices)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_unsafe_index_wrapper",
                "_unsafe_index_jit_function",
                code,
            )

            file_name = f"unsafe_index_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_unsafe_index_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices = args[0], args[1]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        index_rank = tensor_indices[0].ndim if indices_len > 0 else 0
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_unsafe_index_func = UnsafeIndexFunction()


def _unsafe_index(inp, indices):
    logger.debug("GEMS UNSAFE_INDEX")

    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    # Move indices to the same device as input
    indices = [
        index.to(inp.device) if index.device != inp.device else index
        for index in indices
    ]

    # Validate dtypes - only integer tensors allowed (no bool, no None)
    for i, index in enumerate(indices):
        if index.dtype not in [torch.int32, torch.int64, torch.long]:
            raise TypeError(
                f"tensors used as indices must be long, int, or int64, but got {index.dtype}"
            )

    # Broadcast all tensor indices together
    if len(indices) > 1:
        broadcasted = torch.broadcast_tensors(*indices)
    else:
        broadcasted = indices

    # Calculate output shape: broadcast_shape + input_shape[num_indices:]
    broadcast_shape = list(broadcasted[0].shape)
    out_shape = broadcast_shape + list(inp.shape[len(indices) :])

    # Create output tensor
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Handle empty case
    if inp.numel() == 0 or out.numel() == 0:
        return out

    # Call kernel with broadcasted indices
    _unsafe_index_func(inp, broadcasted, out)

    return out