import json
import logging
import os
from datetime import datetime

import pytest
import torch

import flag_gems

device = flag_gems.device

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_detail_and_result_{timestamp}.json"


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        (
            "--mode"
            if not (flag_gems.vendor_name == "kunlunxin" and torch.__version__ < "2.5")
            else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args,
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="run tests on normal or quick mode",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="tests function param recorded in log files or not",
    )


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"

    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"
    if RECORD_LOG:
        global RUNTEST_INFO, BUILTIN_MARKS, REGISTERED_MARKERS
        RUNTEST_INFO = {}
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
        REGISTERED_MARKERS = {
            marker.split(":")[0].strip() for marker in config.getini("markers")
        }
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return
    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKERS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


test_results = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    test_results[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    test_results[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    exclude_marks = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "inplace"}
    operator_marks = [mark for mark in all_marks if mark not in exclude_marks]
    test_results[item.nodeid]["opname"] = operator_marks


def get_skipped_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    elif isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    else:
        return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["result"] = "skipped"
            test_results[report.nodeid]["skipped_reason"] = reason

    elif report.when == "call":
        test_results[report.nodeid]["result"] = report.outcome
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["skipped_reason"] = reason
        else:
            test_results[report.nodeid]["skipped_reason"] = None


def pytest_terminal_summary(terminalreporter):
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(test_results)
    else:
        existing_data = test_results

    with open("result.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4, default=str)

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
_ptpu_orig_randn_like = _ptpu_patch_torch.randn_like
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


def _ptpu_cpu_first_randn_like(input, *args, **kwargs):
    return _ptpu_cpu_first_like(_ptpu_orig_randn_like, ("randn_like", "normal", "normal_"), input, *args, **kwargs)


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
_ptpu_patch_torch.randn_like = _ptpu_cpu_first_randn_like
_ptpu_patch_torch.randn = _ptpu_cpu_first_randn
_ptpu_patch_torch.randint = _ptpu_cpu_first_randint
_ptpu_patch_torch.eye = _ptpu_cpu_first_eye
_ptpu_patch_torch.Tensor.uniform_ = _ptpu_cpu_first_uniform_
_ptpu_patch_torch.Tensor.normal_ = _ptpu_cpu_first_normal_
_ptpu_patch_torch.Tensor.bernoulli_ = _ptpu_cpu_first_bernoulli_
