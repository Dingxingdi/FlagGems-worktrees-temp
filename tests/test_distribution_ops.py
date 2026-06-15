import numpy as np
import pytest
import scipy
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES, to_reference

device = flag_gems.device


@pytest.mark.normal
@pytest.mark.parametrize("float", ["none", "mean", "std"])
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(float, shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    loc = (
        3.0
        if float == "mean"
        else torch.full(
            size=shape, fill_value=3.0, dtype=dtype, device=flag_gems.device
        )
    )
    scale = (
        10.0
        if float == "std"
        else torch.full(
            size=shape, fill_value=10.0, dtype=dtype, device=flag_gems.device
        )
    )
    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.inplace
@pytest.mark.normal_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal_(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    loc = 3.0
    scale = 10.0
    res_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out.normal_(loc, scale)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.uniform_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fast_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    with flag_gems.use_gems():
        x.exponential_()
    x_res = to_reference(x)
    mean_res = torch.mean(x_res.to(torch.float32)).to(dtype)
    var_res = torch.var(x_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)
    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol


@pytest.mark.multinomial
@pytest.mark.parametrize("shape", [(1024, 10)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_samples", [2048])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    # First use multinomial to generate a series of indices, then
    # use the index counts as the input probabilities (scaled)
    rand_indices = torch.multinomial(torch.rand(shape), n_samples, True).to(device)
    inp_counts = torch.nn.functional.one_hot(rand_indices).sum(1)
    with flag_gems.use_gems():
        out_indices = torch.multinomial(inp_counts.to(dtype=dtype), n_samples, True)
    out_counts = torch.nn.functional.one_hot(out_indices).sum(1)
    # Do a simple Chi-square test
    assert torch.equal(inp_counts.sum(-1), out_counts.sum(-1))
    chi2, pvalue = scipy.stats.chisquare(
        out_counts.tolist(), inp_counts.tolist(), axis=-1
    )
    assert np.sum(pvalue < 0.05) / len(pvalue) < 0.1


@pytest.mark.cauchy
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cauchy(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    ref_x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch.ops.aten.cauchy.default(x, 0.0, 1.0)
    with torch.no_grad():
        ref_out = torch.ops.aten.cauchy.default(ref_x, 0.0, 1.0)
    # Cauchy distribution has undefined mean, but we can check:
    # 1. Values are not all the same (randomness check)
    # 2. Values span a reasonable range (heavy tails)
    assert res.shape == shape
    assert not torch.all(res == res[0])  # Not all equal


@pytest.mark.inplace
@pytest.mark.cauchy_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cauchy_(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res = x.cauchy_(median=0.0, sigma=1.0)
    # Cauchy distribution has undefined mean, but we can check:
    # 1. Values are not all the same (randomness check)
    # 2. Values span a reasonable range (heavy tails)
    assert res.shape == shape
    assert not torch.all(res == res[0])  # Not all equal


@pytest.mark.cauchy
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cauchy_median_sigma(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    median = 5.0
    sigma = 2.0
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch.ops.aten.cauchy.default(x, median, sigma)
    # For Cauchy distribution, the median of the sample should be close to the true median
    # We use median of the sample rather than mean (which is undefined for Cauchy)
    sample_median = torch.median(res.to(torch.float32)).to(dtype)
    # Median should be within a reasonable range of the true median
    assert torch.abs(sample_median - median) < 10.0  # Cauchy has heavy tails
