# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-role GPU memory report built from worker logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srtctl.backends.vllm import VLLMServerConfig
from srtctl.core.memory_report import (
    parse_worker_log,
    read_model_facts,
    record_memory_report,
    render_report,
)

# Trimmed from a real GB300 decode worker log: the lines the report reads, with
# the interleaved noise that makes parsing non-trivial kept in place.
WORKER_LOG = """\
(EngineCore_DP0 pid=1) INFO 08-13 11:22:28 [core.py:121] Initializing a V1 LLM engine with config: \
'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], \
'max_cudagraph_capture_size': 128
(Worker_DP1_EP1 pid=3) INFO 08-13 11:26:35 [gpu_model_runner.py:5483] Model loading took 999.99 GiB memory
(Worker_DP0_EP0 pid=2) INFO 08-13 11:26:35 [gpu_model_runner.py:5483] Model loading took 168.89 GiB memory and 232.8 s
(Worker_DP0_EP0 pid=2) INFO 08-13 11:26:35 [interface.py:911] Setting attention block size to 1056 tokens to \
ensure that attention page size is >= mamba page size.
(Worker_DP0_EP0 pid=2) INFO 08-13 11:26:35 [interface.py:935] Padding mamba page size by 0.19% to ensure that \
mamba page size and attention page size are exactly equal.
(Worker_DP0_EP0 pid=2) INFO 08-13 11:29:30 [flashinfer.py:826] FlashInfer resolved query dtypes: \
prefill=torch.bfloat16, decode=torch.bfloat16, kv_cache_dtype=torch.bfloat16, arch=sm103
(Worker_DP0_EP0 pid=2) INFO 08-13 11:29:30 [gpu_model_runner.py:6756] Profiling CUDA graph memory: \
PIECEWISE=19 (largest=128), FULL=11 (largest=64)
(Worker_DP0_EP0 pid=2) INFO 08-13 11:29:38 [gpu_worker.py:564] Available KV cache memory: 46.50 GiB
(EngineCore_DP0 pid=1) INFO 08-13 11:29:38 [kv_cache_utils.py:1882] GPU KV cache size: 394,633 tokens, \
Maximum concurrency for 10,240 tokens per request: 38.54x
(Worker_DP0_EP0 pid=2) INFO 08-13 11:31:42 [gpu_worker.py:727] CUDA graph pool memory: 8.57 GiB (actual), \
33.06 GiB (estimated), difference: 24.49 GiB (285.9%).
(Worker_DP0_EP0 pid=2) INFO 08-13 11:31:42 [gpu_worker.py:790] Free memory on device (274.27/276.62 GiB) on \
startup. Desired GPU memory utilization is (0.92, 254.49 GiB). Actual usage is 171.25 GiB for consumed memory \
(weights + non-torch), 36.74 GiB for peak activation, and 8.57 GiB for CUDAGraph memory. Replace \
gpu_memory_utilization config with `--kv-cache-memory=40574291354` (37.79 GiB) to fit into requested memory, or \
`--kv-cache-memory=61814666240` (57.57 GiB) to fully utilize gpu memory. Current kv cache memory in use is 46.5 GiB.
"""

# The same startup, killed by the Mamba cudagraph guard: vLLM never reached the
# lines it prints after capture, so only these two carry the budget. 0.8005 is
# 0.92 - 33.06/276.62, which is how the real advisory line is computed.
EARLY_EXIT_LOG = (
    "\n".join(
        line
        for line in WORKER_LOG.splitlines()
        if "CUDA graph pool" not in line and "Free memory on device" not in line
    )
    + """
(Worker_DP0_EP0 pid=2) INFO 08-13 11:29:38 [gpu_model_runner.py:6881] Estimated CUDA graph memory: 33.06 GiB total
(Worker_DP0_EP0 pid=2) INFO 08-13 11:29:38 [gpu_worker.py:579] CUDA graph memory profiling is enabled (default \
since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.8005 \
without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase \
--gpu-memory-utilization to 0.9607.
"""
)

MODEL_CONFIG = {
    "dtype": "bfloat16",
    "num_hidden_layers": 92,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "layer_types": ["linear_attention"] * 69 + ["full_attention"] * 23,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "bfloat16",
}


@pytest.fixture
def worker_log(tmp_path) -> Path:
    path = tmp_path / "node01_decode_w0.out"
    path.write_text(WORKER_LOG)
    return path


@pytest.fixture
def early_exit_log(tmp_path) -> Path:
    path = tmp_path / "node09_decode_w0.out"
    path.write_text(EARLY_EXIT_LOG)
    return path


@pytest.fixture
def model_dir(tmp_path) -> Path:
    path = tmp_path / "my-model-fp4"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(MODEL_CONFIG))
    return path


def _report(worker_log: Path, model_dir: Path, **overrides) -> str:
    kwargs = {
        "role": "decode",
        "layout": "DEP16, 4xGB300 per node",
        "nodes": 4,
        "engines": 16,
        "concurrency": 1024,
        "isl": 8192,
        "osl": 1024,
    }
    kwargs.update(overrides)
    return render_report(
        parse_worker_log(worker_log),
        read_model_facts(model_dir, "my-model-fp4"),
        **kwargs,
    )


# Rows whose numbers have to read as one column. None of the labels holds a
# digit, so the first number on the line is always the row's value.
ALIGNED_LABELS = (
    "  total on device",
    "  in use before vLLM starts",
    "  free when vLLM took its snapshot",
    "  reserved from the estimate",
    "  actually used after capture",
    "  reserved and never used",
    "    - model weights",
    "    - non-torch",
    "    - peak activation",
    "    - CUDA graph reservation",
    "    = KV cache",
)


def value_ends(report: str) -> dict[str, int]:
    """The column each aligned row's value ends at, keyed by the row's label."""
    ends = {}
    for line in report.splitlines():
        label = next((name for name in ALIGNED_LABELS if line.startswith(name)), None)
        if label is None:
            continue
        match = re.search(r"-?\d+\.\d\d|\?", line)
        assert match is not None, f"no value on an aligned row: {line}"
        ends[line[: match.start()].rstrip()] = match.end()
    return ends


class TestParsing:
    def test_values_are_read_from_the_log(self, worker_log):
        memory = parse_worker_log(worker_log)

        assert memory.total.value == 276.62
        assert memory.budget.value == 254.49
        assert memory.weights.value == 168.89
        assert memory.kv_memory.value == 46.50
        assert memory.graphs_estimated.value == 33.06
        assert memory.graphs_actual.value == 8.57
        assert memory.kv_dtype.value == "bfloat16"
        assert memory.block_size.value == 1056

    def test_only_the_reported_engine_is_used(self, worker_log):
        """Every DP rank logs the same lines; mixing them would be nonsense."""
        memory = parse_worker_log(worker_log)

        assert memory.weights.value == 168.89  # not 999.99 from DP1
        assert memory.engines == 2  # DP0 and DP1 appear in this log

    def test_citations_match_grep_line_numbers(self, worker_log):
        """The report claims file:line as proof, so it has to be the real line."""
        lines = worker_log.read_text().splitlines()
        memory = parse_worker_log(worker_log)

        assert "Available KV cache memory" in lines[memory.kv_memory.line - 1]
        assert "Model loading took 168.89" in lines[memory.weights.line - 1]

    def test_carriage_returns_do_not_shift_line_numbers(self, tmp_path):
        """vLLM logs carry progress-bar \\r; Python would split lines on them."""
        path = tmp_path / "node01_decode_w0.out"
        path.write_text("first\rprogress\n" + WORKER_LOG)

        memory = parse_worker_log(path)
        # Counted the way grep -n does: only \n starts a new line.
        with path.open(newline="\n") as handle:
            lines = handle.read().split("\n")

        assert "Available KV cache memory" in lines[memory.kv_memory.line - 1]

    def test_derived_terms_split_what_vllm_merges(self, worker_log):
        memory = parse_worker_log(worker_log)

        assert memory.non_torch == pytest.approx(2.36, abs=0.01)  # 171.25 - 168.89
        assert memory.peak_activation == pytest.approx(3.68, abs=0.01)  # 36.74 - 33.06


class TestModelFacts:
    def test_layer_composition_is_counted(self, model_dir):
        facts = read_model_facts(model_dir, "my-model-fp4")

        assert (facts.layers, facts.full_attn, facts.gdn) == (92, 23, 69)
        assert (facts.kv_heads, facts.head_dim) == (4, 256)

    def test_missing_config_is_reported_not_guessed(self, tmp_path):
        facts = read_model_facts(tmp_path / "nowhere", "hf-model")

        assert facts.problems and "config.json" in facts.problems[0]


class TestReport:
    def test_kv_arithmetic_is_shown_step_by_step(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        assert "( 4  x  256  x  2 )  x  2   =  4096 B  =  4 KiB" in report
        assert "ceil(10240 / 1056) = 10 blocks per-layer" in report
        assert "10 x 23 layers     = 230 blocks" in report
        assert "1 x 69 layers     = 69 blocks" in report
        assert "299 x 4.125 MiB = 1233 MiB per request" in report

    def test_capacity_is_expressed_in_requests(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        assert "38.54 requests = 394,633 tokens" in report
        assert "x 16 engines            = 616 requests for the decode side" in report

    def test_graph_overshoot_is_flagged_with_the_vllm_ratio(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        # 24.49 / 8.57 = 286%, the same number vLLM prints in its own line.
        assert "286% over" in report
        assert "CUDA graph estimate overshoots by 286%" in report
        assert "--kv-cache-memory=61814666240" in report

    def test_capacity_shortfall_is_flagged(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        assert "recipe concurrency 1024 needs 1024 / 16 = 64 requests per engine" in report
        assert "will not produce a usable result" in report

    def test_a_fitting_concurrency_is_marked_ok(self, worker_log, model_dir):
        report = _report(worker_log, model_dir, concurrency=512)

        assert "ok concurrency 512 needs 32 per engine" in report
        assert "will not produce a usable result" not in report

    def test_capture_sizes_are_grouped_by_step(self, worker_log, model_dir):
        """A wall of 19 numbers is unreadable; the head and the run are split."""
        report = _report(worker_log, model_dir)

        assert "PIECEWISE  19 graphs   1 2 4" in report
        assert "8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128" in report

    def test_missing_pattern_is_named_instead_of_guessed(self, tmp_path, model_dir):
        """A changed vLLM log must produce a gap that is obvious, not a wrong number."""
        path = tmp_path / "node01_decode_w0.out"
        path.write_text("\n".join(line for line in WORKER_LOG.splitlines() if "CUDA graph pool" not in line))

        report = _report(path, model_dir)

        assert "NOT FOUND" in report
        assert "CUDA graph pool memory" in report
        assert "?? not derived, missing from the log: graph pool" in report


class TestGdnState:
    """The GDN state is per request, so its size decides how many requests fit."""

    @staticmethod
    def _model_dir(tmp_path: Path, **overrides) -> Path:
        path = tmp_path / "fp32-ssm-model"
        path.mkdir()
        (path / "config.json").write_text(json.dumps({**MODEL_CONFIG, **overrides}))
        return path

    def test_both_tensors_are_spelled_out(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        assert "( 128  x  16  x  2  +  128  x  128 )  x  3  x  2  =  122,880 B  =  120 KiB" in report
        assert "( 128  x  128  x  128 )  x  2  =  4,194,304 B  =  4.0000 MiB" in report
        assert "conv + ssm = 4.1172 MiB per GDN layer" in report
        # 69 layers, each in a padded 4.125 MiB page, is what a request pays.
        assert "x 69 GDN layers = 284.6 MiB per request" in report

    def test_the_arrows_name_which_term_is_which(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        assert "q and k, both this size" in report
        assert "v: linear_num_value_heads" in report
        assert "linear_conv_kernel_dim - 1" in report

    def test_the_result_is_checked_against_the_logged_padding(self, worker_log, model_dir):
        report = _report(worker_log, model_dir)

        # 4.1172 MiB padded by 0.19% is the 4.125 MiB page the log reports.
        assert "the log pads that to the 4.125 MiB page by 0.19%, so the two agree" in report

    def test_a_recipe_override_is_preferred_over_config_json(self, tmp_path, model_dir):
        """The checkpoint asks for float32; the recipe's bfloat16 is what runs."""
        path = tmp_path / "node01_decode_w0.out"
        path.write_text(
            WORKER_LOG + "(Worker_DP0_EP0 pid=2) WARNING 08-13 11:22:28 [config.py:775] Qwen3.5 model "
            "specifies mamba_ssm_dtype='float32' in its config, but "
            "--mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value.\n"
        )

        report = _report(path, self._model_dir(tmp_path, mamba_ssm_dtype="float32"))

        assert "conv + ssm = 4.1172 MiB per GDN layer" in report
        assert "overriding config.json mamba_ssm_dtype = float32" in report

    def test_a_dtype_that_contradicts_the_log_is_flagged(self, worker_log, tmp_path):
        """Reading float32 for the ssm state doubles it, and the log disagrees."""
        report = _report(worker_log, self._model_dir(tmp_path, mamba_ssm_dtype="float32"))

        # 4 MiB ssm becomes 8 MiB, conv stays bfloat16 at 120 KiB.
        assert "8.1172 MiB per GDN layer" in report
        assert "not 8.1172: the dtypes above are not" in report
        # Flagged in place, not quietly dropped into the list of missing values.
        assert not any("GDN state" in line for line in report.splitlines() if line.startswith("  ??"))

    def test_a_model_without_the_linear_keys_says_so(self, worker_log, tmp_path):
        path = tmp_path / "no-linear-keys"
        path.mkdir()
        config = {key: value for key, value in MODEL_CONFIG.items() if not key.startswith("linear_")}
        (path / "config.json").write_text(json.dumps(config))

        report = _report(worker_log, path)

        assert "config.json has no linear_* keys: state size not derived" in report
        assert "missing from the log: " in report


class TestAlignment:
    """A report is read by eye, so the numbers have to sit in one column."""

    def test_every_value_ends_in_the_same_column(self, worker_log, model_dir):
        ends = value_ends(_report(worker_log, model_dir))

        assert len(ends) >= 8
        assert set(ends.values()) == {55}, ends

    def test_the_derived_report_keeps_that_column(self, early_exit_log, model_dir):
        ends = value_ends(_report(early_exit_log, model_dir))

        assert len(ends) >= 5
        assert set(ends.values()) == {55}, ends

    def test_a_digit_more_or_less_does_not_shift_a_row(self, tmp_path, model_dir):
        """Padding written per row holds only for the log it was measured against."""
        path = tmp_path / "node01_decode_w0.out"
        path.write_text(
            WORKER_LOG.replace("168.89", "68.89")  # weights lose a digit
            .replace("46.50", "146.50")  # KV cache gains one
            .replace("33.06", "3.06")  # the graph estimate loses one
        )

        ends = value_ends(_report(path, model_dir))

        assert set(ends.values()) == {55}, ends


class TestStartupThatDiedBeforeCapture:
    """A guard can kill the engine before vLLM prints its memory summary.

    Two lines survive: the pre-capture graph estimate and the utilization
    advisory. Together they still pin down the device size and the budget.
    """

    def test_device_size_is_derived_from_the_utilization_advisory(self, early_exit_log):
        memory = parse_worker_log(early_exit_log)

        assert memory.total is None
        # The full log of the same startup says 276.62 GiB.
        assert memory.derived_total.value == pytest.approx(276.62, abs=0.1)
        assert memory.graph_estimate.value == 33.06

    def test_budget_is_rebuilt_from_the_lines_that_are_left(self, early_exit_log, model_dir):
        report = _report(early_exit_log, model_dir)

        assert "BUDGET  = total x gpu-memory-utilization = 276.65 x 0.92 = 254.52 GiB" in report
        assert "(derived)" in report
        # 254.52 - 168.89 weights - 33.06 graphs - 46.50 KV, which vLLM would
        # otherwise print split into non-torch and peak activation.
        assert "- non-torch + peak activation" in report
        assert "6.07" in report

    def test_the_missing_graph_actual_is_named_not_guessed(self, early_exit_log, model_dir):
        report = _report(early_exit_log, model_dir)

        assert "reserved from the estimate" in report
        assert "actually used after capture" in report
        assert "286% over" not in report
        assert "CUDA graph estimate overshoots" not in report
        assert "graph pool (actual)" in report

    def test_capacity_still_comes_out(self, early_exit_log, model_dir):
        """The KV numbers are printed before capture, so they must survive."""
        report = _report(early_exit_log, model_dir)

        assert "38.54 requests = 394,633 tokens" in report
        assert "299 x 4.125 MiB = 1233 MiB per request" in report

    def test_a_log_without_either_line_still_says_not_found(self, tmp_path, model_dir):
        path = tmp_path / "node09_decode_w0.out"
        path.write_text(
            "\n".join(
                line
                for line in EARLY_EXIT_LOG.splitlines()
                if "Estimated CUDA graph memory" not in line and "is equivalent to" not in line
            )
        )

        report = _report(path, model_dir)

        assert "Free memory on device (.../... GiB) on startup" in report
        assert "Desired GPU memory utilization is (util, N GiB)" in report


class TestRecording:
    @staticmethod
    def _runtime(tmp_path, model_dir):
        return SimpleNamespace(log_dir=tmp_path, model_path=model_dir, gpus_per_node=4)

    @staticmethod
    def _config(vllm_config=None):
        # The real schema type, not a dict: vllm_config is a dataclass with one
        # flag dict per mode, and recipes may spell keys with either separator.
        if vllm_config is None:
            vllm_config = VLLMServerConfig(
                prefill={"data_parallel_size": 16, "enable_expert_parallel": True},
                decode={"data-parallel-size": 16, "enable-expert-parallel": True},
            )
        return SimpleNamespace(
            served_model_name="my-model-fp4",
            benchmark=SimpleNamespace(concurrencies="512x1024", isl=8192, osl=1024),
            resources=SimpleNamespace(gpu_type="gb300"),
            backend=SimpleNamespace(vllm_config=vllm_config),
        )

    def test_one_report_per_role(self, tmp_path, model_dir):
        for name in ("node01_decode_w0.out", "node02_decode_w0.out", "node03_prefill_w0.out"):
            (tmp_path / name).write_text(WORKER_LOG)

        written = record_memory_report(self._config(), self._runtime(tmp_path, model_dir))

        assert {path.name for path in written} == {"decode.out", "prefill.out"}
        assert (tmp_path / "memory" / "decode.out").is_file()

    def test_engine_count_spans_the_role(self, tmp_path, model_dir):
        """Two logs with two ranks each is a four-engine decode side."""
        for name in ("node01_decode_w0.out", "node02_decode_w0.out"):
            (tmp_path / name).write_text(WORKER_LOG)

        record_memory_report(self._config(), self._runtime(tmp_path, model_dir))

        assert "all 4 identical" in (tmp_path / "memory" / "decode.out").read_text()

    def test_layout_names_the_parallelism(self, tmp_path, model_dir):
        (tmp_path / "node01_decode_w0.out").write_text(WORKER_LOG)

        record_memory_report(self._config(), self._runtime(tmp_path, model_dir))

        assert "DEP16, 4xGB300 per node" in (tmp_path / "memory" / "decode.out").read_text()

    def test_underscored_recipe_keys_are_understood(self, tmp_path, model_dir):
        """A recipe may write data_parallel_size instead of data-parallel-size."""
        (tmp_path / "node01_prefill_w0.out").write_text(WORKER_LOG)

        record_memory_report(self._config(), self._runtime(tmp_path, model_dir))

        assert "DEP16" in (tmp_path / "memory" / "prefill.out").read_text()

    def test_an_undescribable_layout_still_yields_a_report(self, tmp_path, model_dir):
        """The layout is one cosmetic header line; it must not cost the report."""
        (tmp_path / "node01_decode_w0.out").write_text(WORKER_LOG)

        record_memory_report(self._config(vllm_config="not a config"), self._runtime(tmp_path, model_dir))

        report = (tmp_path / "memory" / "decode.out").read_text()
        assert "layout unknown" in report
        assert "38.54 requests" in report

    def test_nothing_written_without_worker_logs(self, tmp_path, model_dir):
        assert record_memory_report(self._config(), self._runtime(tmp_path, model_dir)) == []

    def test_a_failure_never_breaks_the_run(self, tmp_path, model_dir):
        (tmp_path / "node01_decode_w0.out").write_text(WORKER_LOG)

        with patch("srtctl.core.memory_report.render_report", side_effect=RuntimeError("boom")):
            assert record_memory_report(self._config(), self._runtime(tmp_path, model_dir)) == []
