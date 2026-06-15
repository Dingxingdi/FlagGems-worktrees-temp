import os

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((64, 64, 64), (128, 64, 7)),
    # ((32, 16, 1024), (1024, 16, 8)),
    # ((32, 12, 9), (17, 12, 3)),
    # ((32, 6, 6), (64, 6, 2)),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d(shape, kernel, stride, padding, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv1d_padding
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d_padding(shape, kernel, stride, padding, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_CONV1D_DILATION = [
    ((32, 2, 16), (17, 2, 3)),
    ((32, 15, 32), (17, 15, 3)),
    ((64, 64, 64), (128, 64, 3)),
]


@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dilation", [1, 2, (1,), (2,)])
def test_accuracy_conv1d_dilation(shape, kernel, stride, padding, dtype, dilation):
    """Test conv1d with various dilation values, including tuple form.

    This specifically tests the fix where conv1d must properly convert dilation
    to a 2D tuple before delegating to conv2d. Previously, passing dilation as
    a single-element tuple (e.g., (1,)) would cause a ValueError in conv2d.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=dilation
    )
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    # ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    # ((18, 16, 4, 4), (16, 16, 2, 2), 1),
    # ((9, 16, 4, 4), (128, 4, 2, 2), 4),
    # ((32, 16, 8, 8), (32, 4, 4, 4), 4),
    # ((18, 16, 4, 4), (16, 8, 2, 2), 2),
    # ((9, 16, 4, 4), (128, 8, 2, 2), 2),
    # ((32, 8, 8, 8), (32, 8, 3, 3), 1),
    # ((18, 16, 5, 5), (16, 16, 3, 3), 1),
    # ((9, 16, 7, 7), (128, 4, 3, 3), 4),
    # ((32, 16, 9, 9), (32, 4, 5, 5), 4),
    # ((18, 16, 11, 11), (16, 8, 3, 3), 2),
    # ((9, 16, 6, 6), (128, 8, 3, 3), 2),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv2d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = to_reference(out_grad, True)
    if bias is not None:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        (res_in_grad, res_weight_grad) = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv2d_padding
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d_padding(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = to_reference(out_grad, True)
    if bias is not None:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        (res_in_grad, res_weight_grad) = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_CONV3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
    # ((2, 2, 3, 3, 3), (1, 2, 2, 2, 2), 1),
    # ((32, 8, 8, 8, 8), (32, 8, 2, 2, 2), 1),
    # ((18, 16, 4, 4, 4), (16, 16, 2, 2, 2), 1),
    # ((9, 16, 4, 4, 4), (128, 4, 2, 2, 2), 4),
    # ((32, 16, 8, 8, 8), (32, 4, 4, 4, 4), 4),
    # ((18, 16, 4, 4, 4), (16, 8, 2, 2, 2), 2),
    # ((9, 16, 4, 4, 4), (128, 8, 2, 2, 2), 2),
    # ((32, 8, 8, 8, 8), (32, 8, 3, 3, 3), 1),
    # ((18, 16, 5, 5, 5), (16, 16, 3, 3, 3), 1),
    # ((9, 16, 7, 7, 7), (128, 4, 3, 3, 3), 4),
    # ((32, 16, 9, 9, 9), (32, 4, 5, 5, 5), 4),
    # ((18, 16, 11, 11, 11), (16, 8, 3, 3, 3), 2),
    # ((9, 16, 6, 6, 6), (128, 8, 3, 3, 3), 2),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv3d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv3d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv3d_padding
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv3d_padding(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    # ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    # ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    # ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    # ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    # ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    # ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


# test for depthwise depends on specific device
@pytest.mark.skip("conv_depthwise2d introduces failures, disable it temporarily")
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight,kernel ", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dtype
):
    inp = torch.randn(
        shape_input, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    ref_out = torch._C._nn._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
    )

    res_out = flag_gems._conv_depthwise2d(
        inp, weight, kernel, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)


# Test for cudnnconvbwd - convolution backward
SHAPE_CUDNNCONVBWD = [
    ((1, 3, 16, 16), (8, 3, 3, 3)),
    ((2, 8, 8, 8), (16, 8, 3, 3)),
    ((4, 16, 4, 4), (32, 16, 2, 2)),
]


@pytest.mark.cudnnconvbwd
@pytest.mark.parametrize("shape_input, shape_weight", SHAPE_CUDNNCONVBWD)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_cudnnconvbwd(shape_input, shape_weight, stride, padding, dtype):
    """Test cudnnconvbwd - convolution backward pass."""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    # Create inputs
    input = torch.randn(shape_input, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device, requires_grad=True)

    # Forward pass reference
    ref_input = to_reference(input, True)
    ref_weight = to_reference(weight, True)
    torch.backends.cudnn.allow_tf32 = False

    # Compute forward pass
    ref_out = torch.nn.functional.conv2d(
        ref_input, ref_weight, bias=None, stride=stride, padding=padding
    )

    # Create grad_output with correct dtype
    grad_output = torch.randn_like(ref_out).to(dtype=dtype, device=flag_gems.device)
    ref_grad_output = to_reference(grad_output, True)

    # Reference: compute backward using PyTorch's convolution_backward
    ref_result = torch.ops.aten.convolution_backward(
        ref_grad_output,
        ref_input,
        ref_weight,
        None,
        [stride, stride],
        [padding, padding],
        [1, 1],
        False,
        [0, 0],
        1,
        [True, True, False],  # output_mask: input_grad, weight_grad, bias_grad
    )
    ref_input_grad = ref_result[0]
    ref_weight_grad = ref_result[1]

    # GEMS: compute backward using our cudnnconvbwd
    res_input = input.detach().requires_grad_(True)
    res_weight = weight.detach().requires_grad_(True)

    with flag_gems.use_gems():
        res_result = flag_gems.cudnnconvbwd(
            grad_output,
            res_input,
            res_weight,
            stride=stride,
            padding=padding,
        )
    res_input_grad = res_result[0]
    res_weight_grad = res_result[1]

    # Compare gradients - use slightly larger tolerance for float16 due to numerical precision
    if dtype == torch.float16:
        # For float16, use a larger tolerance as convolution backward is numerically sensitive
        gems_assert_close(res_input_grad, ref_input_grad, dtype, reduce_dim=weight.shape[2], atol=1e-3)
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0], atol=1e-3)
    else:
        gems_assert_close(res_input_grad, ref_input_grad, dtype, reduce_dim=weight.shape[2])
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0])

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
