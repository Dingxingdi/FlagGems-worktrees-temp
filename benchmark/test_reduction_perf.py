import random
from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from benchmark.performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmark4DOnly,
    SkipVersion,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)
from flag_gems.utils import shape_utils


class UnaryReductionBenchmark(Benchmark):
    """
    Base class for benchmarking reduction operations.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        more_shapes_1d = [
            (1025 * 1024,),
            (1024 * 1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 21, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 15, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                yield inp, 1
            else:
                yield inp,


forward_operations = [
    ("all", torch.all, FLOAT_DTYPES),
    ("any", torch.any, FLOAT_DTYPES),
    ("amax", torch.amax, FLOAT_DTYPES),
    ("argmax", torch.argmax, FLOAT_DTYPES),
    ("argmin", torch.argmin, FLOAT_DTYPES),
    ("max", torch.max, FLOAT_DTYPES),
    ("mean", torch.mean, FLOAT_DTYPES),
    ("min", torch.min, FLOAT_DTYPES),
    ("nanmedian", torch.nanmedian, FLOAT_DTYPES),
    ("prod", torch.prod, FLOAT_DTYPES),
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
    ("std", torch.std, FLOAT_DTYPES),
    ("sum", torch.sum, FLOAT_DTYPES),
    ("var_mean", torch.var_mean, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in forward_operations
    ],
)
def test_general_reduction_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


# Custom benchmark for nanmedian with smaller shapes
class NanmedianBenchmark(UnaryReductionBenchmark):
    """Custom benchmark for nanmedian with smaller shapes due to sorting complexity."""

    def set_more_shapes(self):
        # Use smaller shapes for nanmedian since it requires sorting
        more_shapes_1d = [
            (1024,),
            (1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 13, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 10, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmedian_perf(dtype):
    bench = NanmedianBenchmark(
        op_name="nanmedian",
        torch_op=torch.nanmedian,
        dtypes=[dtype],
    )
    bench.run()


backward_operations = [
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in backward_operations
    ],
)
def test_general_reduction_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()