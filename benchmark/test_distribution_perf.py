import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


def bernoulli_input_fn(shape, cur_dtype, device):
    # Generate probability values in [0, 1]
    yield torch.rand(shape, dtype=cur_dtype, device=device),


def bernoulli_inplace_input_fn(shape, cur_dtype, device):
    # For inplace bernoulli, the input tensor is modified in place
    yield torch.rand(shape, dtype=cur_dtype, device=device),


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


def normal_inplace_input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


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
            "bernoulli",
            torch.bernoulli,
            bernoulli_input_fn,
            marks=pytest.mark.bernoulli,
        ),
        pytest.param(
            "bernoulli_",
            torch.Tensor.bernoulli_,
            bernoulli_inplace_input_fn,
            marks=pytest.mark.bernoulli_,
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
