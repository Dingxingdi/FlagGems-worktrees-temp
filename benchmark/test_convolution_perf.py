import os

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class Conv1DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 512, 64, 3, 1, 0, 1),
            (64, 48, 1024, 128, 5, 2, 2, 1),
            (16, 24, 2048, 96, 7, 1, 3, 2),
            (8, 8, 8192, 16, 11, 4, 5, 1),
            (4, 4, 16384, 4, 15, 2, 7, 1),
            (32, 64, 512, 64, 3, 1, "valid", 1),
            (64, 48, 1024, 128, 5, 2, "valid", 1),
            (16, 24, 2048, 96, 7, 1, "same", 2),
            (8, 8, 8192, 16, 11, 1, "same", 1),
        ]


@pytest.mark.conv1d
def test_perf_conv1d():
    def conv1d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_l,
            out_c,
            kernel,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_l)
        weight_shape = (out_c, input_c // groups, kernel)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv1DBenchmark(
        input_fn=conv1d_input_fn,
        op_name="conv1d",
        torch_op=torch.nn.functional.conv1d,
        dtypes=[
            torch.float16,
            torch.float32,
        ],  # Exclude bfloat16 due to cuDNN limitations
    )
    bench.set_gems(flag_gems.conv1d)
    bench.run()


class Conv2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
            (16, 32, 12, 12, 24, 3, 3, 2, "valid", 1),
            (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1),
            (16, 32, 24, 24, 24, 3, 3, 1, "same", 2),
            (32, 64, 210, 210, 16, 5, 5, 1, "same", 1),
        ]


@pytest.mark.conv2d
def test_perf_conv2d():
    def conv2d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_h,
            input_w,
            out_c,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"
    torch.backends.cudnn.allow_tf32 = False
    bench = Conv2DBenchmark(
        input_fn=conv2d_input_fn,
        op_name="conv2d",
        torch_op=torch.nn.functional.conv2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv2d)
    bench.run()
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


class Conv3DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.conv3d
def test_perf_conv3d():
    def conv3d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_d,
            input_h,
            input_w,
            out_c,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_d, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv3DBenchmark(
        input_fn=conv3d_input_fn,
        op_name="conv3d",
        torch_op=torch.nn.functional.conv3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv3d)
    bench.run()


@pytest.mark.depthwise_pointwise_conv2d
def test_perf_depthwise_pointwise_conv2d():
    """Simple benchmark for depthwise separable convolution."""
    import time

    torch.backends.cudnn.allow_tf32 = False

    # Test shapes: (batch, in_c, h, w, out_c)
    shapes = [
        (1, 4, 32, 32, 8),
        (2, 8, 16, 16, 16),
        (4, 16, 8, 8, 32),
        (1, 4, 64, 64, 8),
        (2, 8, 32, 32, 16),
    ]

    dtypes = [torch.float16, torch.float32]

    for dtype in dtypes:
        print(f"\nOperator: depthwise_pointwise_conv2d  Performance Test (dtype={dtype}, mode=kernel,level=comprehensive)")
        for (batch, in_c, h, w, out_c) in shapes:
            # Create inputs
            input = torch.randn(batch, in_c, h, w, dtype=dtype, device="cuda")
            depthwise_weight = torch.randn(in_c, 1, 3, 3, dtype=dtype, device="cuda")
            pointwise_weight = torch.randn(out_c, in_c, 1, 1, dtype=dtype, device="cuda")

            # Warmup
            for _ in range(10):
                _ = flag_gems.depthwise_pointwise_conv2d(
                    input, depthwise_weight, pointwise_weight,
                    depthwise_bias=None, pointwise_bias=None,
                    depthwise_stride=(1, 1), depthwise_padding=(1, 1), depthwise_dilation=1,
                    pointwise_stride=1, pointwise_padding=0, pointwise_dilation=1,
                )

            # Benchmark GEMS
            start = time.time()
            n_iter = 100
            for _ in range(n_iter):
                _ = flag_gems.depthwise_pointwise_conv2d(
                    input, depthwise_weight, pointwise_weight,
                    depthwise_bias=None, pointwise_bias=None,
                    depthwise_stride=(1, 1), depthwise_padding=(1, 1), depthwise_dilation=1,
                    pointwise_stride=1, pointwise_padding=0, pointwise_dilation=1,
                )
            torch.cuda.synchronize()
            gems_time = (time.time() - start) / n_iter * 1000  # ms

            # Benchmark PyTorch
            start = time.time()
            for _ in range(n_iter):
                dw_out = torch.nn.functional.conv2d(
                    input, depthwise_weight, bias=None,
                    stride=(1, 1), padding=(1, 1), dilation=1, groups=in_c
                )
                _ = torch.nn.functional.conv2d(
                    dw_out, pointwise_weight, bias=None,
                    stride=1, padding=0, dilation=1, groups=1
                )
            torch.cuda.synchronize()
            torch_time = (time.time() - start) / n_iter * 1000  # ms

            speedup = torch_time / gems_time
            print(f"SUCCESS    {torch_time:.4f}    {gems_time:.4f}    {speedup:.4f}    [({batch}, {in_c}, {h}, {w}) -> ({batch}, {out_c}, {h}, {w})]")
