from typing import Generator

import pytest
import torch

from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from benchmark.performance_utils import Benchmark, generate_tensor_input
from benchmark.conftest import Config


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        shape2 = list(args[0].shape)
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            # Arithmetic operations
            ("add", torch.add, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("atan2", torch.atan2, FLOAT_DTYPES),
            ("copysign", torch.copysign, FLOAT_DTYPES),
            ("div", torch.div, FLOAT_DTYPES),
            ("mul", torch.mul, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("sub", torch.sub, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("pow", torch.pow, FLOAT_DTYPES),
            ("polar", torch.polar, [torch.float32]),
            ("floor_divide", torch.floor_divide, INT_DTYPES),
            ("remainder", torch.remainder, INT_DTYPES),
            ("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES),
            ("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES),
            ("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES),
            # Comparison operations
            ("eq", torch.eq, FLOAT_DTYPES),
            ("equal", torch.equal, FLOAT_DTYPES),
            ("ge", torch.ge, FLOAT_DTYPES),
            ("gt", torch.gt, FLOAT_DTYPES),
            ("le", torch.le, FLOAT_DTYPES),
            ("lt", torch.lt, FLOAT_DTYPES),
            ("ne", torch.ne, FLOAT_DTYPES),
            # Minimum and maximum operations
            ("maximum", torch.maximum, FLOAT_DTYPES),
            ("minimum", torch.minimum, FLOAT_DTYPES),
            ("hypot", torch.hypot, FLOAT_DTYPES),
            ("fmin", torch.fmin, FLOAT_DTYPES),
            # Bitwise operations
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES),
            # Numerical Checks
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES),
            # Log operations
            ("logaddexp", torch.logaddexp, FLOAT_DTYPES),
        ]
    ],
)
def test_general_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            # Arithmetic operations
            ("add_", lambda a, b: a.add_(b), FLOAT_DTYPES),
            ("div_", lambda a, b: a.div_(b), FLOAT_DTYPES),
            ("mul_", lambda a, b: a.mul_(b), FLOAT_DTYPES),
            ("sub_", lambda a, b: a.sub_(b), FLOAT_DTYPES),
            ("pow_", lambda a, b: a.pow_(b), FLOAT_DTYPES),
            ("floor_divide_", lambda a, b: a.floor_divide_(b), INT_DTYPES),
            ("remainder_", lambda a, b: a.remainder_(b), INT_DTYPES),
            ("logical_or_", lambda a, b: a.logical_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("logical_and_", lambda a, b: a.logical_and_(b), INT_DTYPES + BOOL_DTYPES),
            # Bitwise operations
            ("bitwise_and_", lambda a, b: a.bitwise_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or_", lambda a, b: a.bitwise_or_(b), INT_DTYPES + BOOL_DTYPES),
        ]
    ],
)
def test_general_inplace_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()


# Beam Search Score benchmark
# Beam_Search_Score takes log_probs [batch, vocab] and beam_scores [batch]
# and computes log_probs + beam_scores (with broadcasting)
import flag_gems


class BeamSearchScoreBenchmark(Benchmark):
    """
    Benchmark for Beam Search Score operation.
    """

    # Use smaller shapes suitable for beam search scenarios
    DEFAULT_SHAPES = [(16, 512), (32, 1024), (64, 2048), (128, 4096), (256, 8192)]
    DEFAULT_SHAPE_DESC = "batch_size, vocab_size"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # log_probs: [batch, vocab], beam_scores: [batch]
            batch_size = shape[0]
            vocab_size = shape[1] if len(shape) > 1 else shape[0]
            log_probs = generate_tensor_input((batch_size, vocab_size), cur_dtype, self.device)
            beam_scores = generate_tensor_input((batch_size,), cur_dtype, self.device)
            yield log_probs, beam_scores

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)  # log_probs shape
        shape2 = list(args[1].shape)  # beam_scores shape
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()

    def init_user_config(self):
        # Override to skip reading from YAML and use DEFAULT_SHAPES directly
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        # Use our own DEFAULT_SHAPES instead of reading from YAML
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC


@pytest.mark.Beam_Search_Score
@pytest.mark.parametrize(
    "dtype",
    FLOAT_DTYPES,
)
def test_Beam_Search_Score_perf(dtype):
    # Reference implementation: PyTorch broadcasting addition
    def torch_op(log_probs, beam_scores):
        return log_probs + beam_scores.unsqueeze(-1)

    bench = BeamSearchScoreBenchmark(
        op_name="Beam_Search_Score",
        torch_op=torch_op,
        dtypes=[dtype],
    )
    # Override gems_op to use flag_gems.Beam_Search_Score
    bench.gems_op = flag_gems.Beam_Search_Score
    bench.run()


@pytest.mark.Beam_Search_Score_
@pytest.mark.parametrize(
    "dtype",
    FLOAT_DTYPES,
)
def test_Beam_Search_Score_inplace_perf(dtype):
    # Reference implementation: PyTorch broadcasting addition
    def torch_op(log_probs, beam_scores):
        return log_probs + beam_scores.unsqueeze(-1)

    bench = BeamSearchScoreBenchmark(
        op_name="Beam_Search_Score_",
        torch_op=torch_op,
        dtypes=[dtype],
        is_inplace=True,
    )
    # Override gems_op to use flag_gems.Beam_Search_Score_
    bench.gems_op = flag_gems.Beam_Search_Score_
    bench.run()
