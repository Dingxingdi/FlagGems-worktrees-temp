import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


def normal_inplace_input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


def geometric_input_fn(shape, cur_dtype, device):
    # geometric takes a tensor of uniform random values and a probability p
    inp = torch.rand(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield inp, p


def geometric_inplace_input_fn(shape, cur_dtype, device):
    # geometric_ is inplace, modifies the input tensor
    inp = torch.rand(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield inp, p


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "normal",
            torch.normal,
            normal_input_fn,
            marks=pytest.mark.normal,
        ),
        pytest.param(
            "normal_",
            torch.Tensor.normal_,
            normal_inplace_input_fn,
            marks=pytest.mark.normal_,
        ),
        pytest.param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            marks=pytest.mark.uniform_,
        ),
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            marks=pytest.mark.exponential_,
        ),
        pytest.param(
            "geometric",
            torch.ops.aten.geometric,
            geometric_input_fn,
            marks=pytest.mark.geometric,
        ),
        pytest.param(
            "geometric_",
            torch.ops.aten.geometric_,
            geometric_inplace_input_fn,
            marks=pytest.mark.geometric_,
        ),
    ],
)
def test_distribution_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
