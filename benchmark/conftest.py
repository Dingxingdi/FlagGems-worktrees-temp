import json
import logging
import os

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchMode,
    OperationAttribute,
    get_recommended_shapes,
)
from flag_gems.runtime import torch_device_fn

device = flag_gems.device
vendor_name = flag_gems.vendor_name
recordLogger = logging.getLogger("flag_gems_benchmark")
recordLogger.propagate = False


def emit_record_logger(message: str) -> None:
    if recordLogger.handlers:
        handler = recordLogger.handlers[0]
        if getattr(handler, "stream", None) is None:
            handler.acquire()
            try:
                handler.stream = handler._open()
            finally:
                handler.release()
    recordLogger.info(message)


class BenchConfig:
    def __init__(self):
        self.mode = BenchMode.KERNEL
        self.bench_level = BenchLevel.COMPREHENSIVE
        self.warm_up = DEFAULT_WARMUP_COUNT
        self.repetition = DEFAULT_ITER_COUNT
        if (
            vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            self.warm_up = 1
            self.repetition = 1
        self.record_log = False
        self.user_desired_dtypes = None
        self.user_desired_metrics = None
        self.shape_file = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        self.query = False
        self.parallel = 0


Config = BenchConfig()


def pytest_addoption(parser):
    parser.addoption(
        (
            "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args
        action="store",
        default="kernel",
        required=False,
        choices=["kernel", "operator", "wrapper"],
        help=(
            "Specify how to measure latency, 'kernel' for device kernel, "
            "'operator' for end2end operator or 'wrapper' for runtime wrapper."
        ),
    )

    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

    parser.addoption(
        "--warmup",
        default=DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs before benchmark run.",
    )

    parser.addoption(
        "--iter",
        default=DEFAULT_ITER_COUNT,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--query", action="store_true", default=False, help="Enable query mode"
    )

    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        choices=ALL_AVAILABLE_METRICS,
        help=(
            "Specify the metrics we want to benchmark. "
            "If not specified, the metric items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        choices=[
            str(ele).split(".")[-1]
            for ele in FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES + [torch.cfloat]
        ],
        help=(
            "Specify the data types for benchmarks. "
            "If not specified, the dtype items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--shape_file",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "core_shapes.yaml"),
        required=False,
        help="Specify the shape file name for benchmarks. If not specified, a default shape list will be used.",
    )

    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="Benchmark info recorded in log files or not",
    )

    parser.addoption(
        "--parallel",
        action="store",
        type=int,
        default=0,
        help=(
            "Enable multi-GPU parallel benchmark execution across shapes. "
            "Example: --parallel 8 means using GPU 0~7 in parallel. "
            "Default 0 means serial execution."
        ),
    )


def pytest_configure(config):
    global Config  # noqa: F824
    mode_value = config.getoption(
        "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
    )
    Config.mode = BenchMode(mode_value)

    Config.query = config.getoption("--query")

    level_value = config.getoption("--level")
    Config.bench_level = BenchLevel(level_value)

    warmup_value = config.getoption("--warmup")
    Config.warm_up = int(warmup_value)

    iter_value = config.getoption("--iter")
    Config.repetition = int(iter_value)

    types_str = config.getoption("--dtypes")
    dtypes = [getattr(torch, dtype) for dtype in types_str] if types_str else types_str
    Config.user_desired_dtypes = dtypes

    metrics = config.getoption("--metrics")
    Config.user_desired_metrics = metrics

    shape_file_str = config.getoption("--shape_file")
    Config.shape_file = shape_file_str

    Config.record_log = config.getoption("--record") == "log"
    Config.parallel = int(config.getoption("--parallel") or 0)
    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]

        log_file = "result_{}.log".format("_".join(cmd_args)).replace("_-", "-")

        for h in list(recordLogger.handlers):
            recordLogger.removeHandler(h)
            try:
                h.close()
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to close handler: {e}")

        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=False)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        recordLogger.addHandler(handler)
        recordLogger.setLevel(logging.INFO)
        emit_record_logger("Benchmark record logger enabled")


BUILTIN_MARKS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
}


@pytest.fixture(scope="session", autouse=True)
def setup_once(request):
    if request.config.getoption("--query"):
        print("\nThis is query mode; all benchmark functions will be skipped.")
    # else:
    #     note_info = (
    #         "\n\nNote: The 'size' field below is for backward compatibility with previous versions of the benchmark. "
    #         "\nThis field will be removed in a future release."
    #     )
    #     print(note_info)


@pytest.fixture(scope="function", autouse=True)
def clear_function_cache():
    yield
    empty_cache = getattr(torch_device_fn, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


@pytest.fixture(scope="module", autouse=True)
def clear_module_cache():
    yield
    empty_cache = getattr(torch_device_fn, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


@pytest.fixture()
def extract_and_log_op_attributes(request):
    print("")
    op_attributes = []

    # Extract the 'recommended_shapes' attribute from the pytest marker decoration.
    for mark in request.node.iter_markers():
        if mark.name in BUILTIN_MARKS:
            continue
        op_specified_shapes = mark.kwargs.get("recommended_shapes")
        shape_desc = mark.kwargs.get("shape_desc", "M, N")
        rec_core_shapes = get_recommended_shapes(mark.name, op_specified_shapes)

        if rec_core_shapes:
            attri = OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec_core_shapes,
                shape_desc=shape_desc,
            )
            print(attri)
            op_attributes.append(attri.to_dict())

    if request.config.getoption("--query"):
        # Skip the real benchmark functions
        pytest.skip("Skipping benchmark due to the query parameter.")

    yield
    if Config.record_log and op_attributes:
        emit_record_logger(json.dumps(op_attributes, indent=2))

# PTPU does not implement some factory/random kernels used by input setup.
# Build helper inputs on CPU first, then move them to the FlagGems device.
import os as _ptpu_patch_os
import torch as _ptpu_patch_torch
import flag_gems as _ptpu_patch_flag_gems

_ptpu_branch_op = _ptpu_patch_os.path.basename(
    _ptpu_patch_os.path.dirname(_ptpu_patch_os.path.dirname(__file__))
)
if _ptpu_branch_op.startswith("gen-"):
    _ptpu_branch_op = _ptpu_branch_op[4:]

_ptpu_orig_rand = _ptpu_patch_torch.rand
_ptpu_orig_rand_like = _ptpu_patch_torch.rand_like
_ptpu_orig_randn = _ptpu_patch_torch.randn
_ptpu_orig_randint = _ptpu_patch_torch.randint
_ptpu_orig_eye = _ptpu_patch_torch.eye
_ptpu_orig_uniform_ = _ptpu_patch_torch.Tensor.uniform_
_ptpu_orig_normal_ = _ptpu_patch_torch.Tensor.normal_
_ptpu_orig_bernoulli_ = _ptpu_patch_torch.Tensor.bernoulli_


def _ptpu_target_device(device):
    return str(device).startswith("ptpu")


def _ptpu_should_cpu_first(*op_names):
    return _ptpu_branch_op not in op_names


def _ptpu_cpu_first_factory(orig, op_names, *args, **kwargs):
    device = kwargs.get("device")
    if _ptpu_target_device(device) and _ptpu_should_cpu_first(*op_names):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return orig(*args, **kwargs).to(device)
    return orig(*args, **kwargs)


def _ptpu_cpu_first_like(orig, op_names, input, *args, **kwargs):
    device = kwargs.get("device", getattr(input, "device", None))
    if _ptpu_target_device(device) and _ptpu_should_cpu_first(*op_names):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        cpu_input = input.to("cpu") if isinstance(input, _ptpu_patch_torch.Tensor) else input
        return orig(cpu_input, *args, **kwargs).to(device)
    return orig(input, *args, **kwargs)


def _ptpu_cpu_first_rand(*args, **kwargs):
    return _ptpu_cpu_first_factory(_ptpu_orig_rand, ("rand",), *args, **kwargs)


def _ptpu_cpu_first_rand_like(input, *args, **kwargs):
    return _ptpu_cpu_first_like(_ptpu_orig_rand_like, ("rand_like",), input, *args, **kwargs)


def _ptpu_cpu_first_randn(*args, **kwargs):
    return _ptpu_cpu_first_factory(_ptpu_orig_randn, ("randn", "normal", "normal_"), *args, **kwargs)


def _ptpu_cpu_first_randint(*args, **kwargs):
    return _ptpu_cpu_first_factory(_ptpu_orig_randint, ("randint",), *args, **kwargs)


def _ptpu_cpu_first_eye(*args, **kwargs):
    return _ptpu_cpu_first_factory(_ptpu_orig_eye, ("eye",), *args, **kwargs)


def _ptpu_cpu_first_inplace_random(orig, op_names, self, *args, **kwargs):
    if _ptpu_target_device(getattr(self, "device", None)) and _ptpu_should_cpu_first(*op_names):
        cpu_self = _ptpu_patch_torch.empty_strided(
            tuple(self.shape), self.stride(), dtype=self.dtype, device="cpu"
        )
        orig(cpu_self, *args, **kwargs)
        self.copy_(cpu_self.to(self.device))
        return self
    return orig(self, *args, **kwargs)


def _ptpu_cpu_first_uniform_(self, *args, **kwargs):
    return _ptpu_cpu_first_inplace_random(_ptpu_orig_uniform_, ("uniform", "uniform_"), self, *args, **kwargs)


def _ptpu_cpu_first_normal_(self, *args, **kwargs):
    return _ptpu_cpu_first_inplace_random(_ptpu_orig_normal_, ("normal", "normal_"), self, *args, **kwargs)


def _ptpu_cpu_first_bernoulli_(self, *args, **kwargs):
    return _ptpu_cpu_first_inplace_random(_ptpu_orig_bernoulli_, ("bernoulli",), self, *args, **kwargs)


_ptpu_patch_torch.rand = _ptpu_cpu_first_rand
_ptpu_patch_torch.rand_like = _ptpu_cpu_first_rand_like
_ptpu_patch_torch.randn = _ptpu_cpu_first_randn
_ptpu_patch_torch.randint = _ptpu_cpu_first_randint
_ptpu_patch_torch.eye = _ptpu_cpu_first_eye
_ptpu_patch_torch.Tensor.uniform_ = _ptpu_cpu_first_uniform_
_ptpu_patch_torch.Tensor.normal_ = _ptpu_cpu_first_normal_
_ptpu_patch_torch.Tensor.bernoulli_ = _ptpu_cpu_first_bernoulli_
