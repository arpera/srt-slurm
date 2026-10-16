# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explain where a vLLM worker's GPU memory went, and whether KV cache fits.

vLLM reports its memory decisions across half a dozen log lines printed minutes
apart, in units that need converting before they mean anything: "GPU KV cache
size: 394,633 tokens" only becomes useful after dividing by max-model-len, and
the number it prints as "peak activation" silently includes the CUDA graph
reservation. Meanwhile the question a benchmark run actually needs answered is
"can one engine hold concurrency / engines requests?".

This module reads those lines back out of the worker logs, adds the KV cache
arithmetic derived from the model config, and writes one report per role to
``logs/memory/{prefill,decode,agg}.out``. Every value carries the log line it
came from so the numbers can be rechecked by hand; anything that could not be
found is printed as NOT FOUND rather than guessed, because a missing pattern
means the vLLM version or the log wording changed.

vLLM prints the line carrying the whole budget only after CUDA graph capture, so
a startup that dies before that (an OOM, or the guard on max-num-seqs versus
available Mamba blocks) would leave the report almost empty. For those runs the
budget is rebuilt from the two lines printed before capture and labelled
(derived), which is the one place this module computes instead of quoting.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

MEMORY_DIRNAME = "memory"

# Reported per engine, so the report is only meaningful for the rank it names.
ENGINE = 0

GIB = 1024**3
MIB = 1024**2

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# One vLLM line carries most of the budget: totals, utilization, the three
# usage terms and both --kv-cache-memory suggestions.
_BUDGET = re.compile(
    r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup\. "
    r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)\. "
    r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory\. "
    r".*?--kv-cache-memory=(\d+)` \(([\d.]+) GiB\) to fit into requested memory, "
    r"or `--kv-cache-memory=(\d+)` \(([\d.]+) GiB\) to fully utilize"
)
_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB memory")
_GRAPH_POOL = re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)")
# Printed before capture, unlike _GRAPH_POOL, so this is what a startup that died
# during capture leaves behind.
_GRAPH_ESTIMATE = re.compile(r"Estimated CUDA graph memory: ([\d.]+) GiB total")
# The advisory next to the KV cache size. The two utilizations differ by exactly
# the graph estimate's share of the device, which is the only way to recover the
# device size before vLLM prints it.
_UTIL_ADVISORY = re.compile(
    r"--gpu-memory-utilization=([\d.]+) is equivalent to "
    r"--gpu-memory-utilization=([\d.]+) without CUDA graph memory profiling"
)
_GRAPH_PLAN = re.compile(r"Profiling CUDA graph memory: (.+)$")
_GRAPH_PLAN_MODE = re.compile(r"(\w+)=(\d+) \(largest=(\d+)\)")
_CAPTURE_SIZES = re.compile(r"'cudagraph_capture_sizes':\s*\[([\d,\s]+)\]")
_KV_MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_KV_TOKENS = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
)
_BLOCK_SIZE = re.compile(r"Setting attention block size to (\d+) tokens")
_MAMBA_PAD = re.compile(r"Padding mamba page size by ([\d.]+)% ")
# Only logged when the recipe overrides what the checkpoint asks for, which is
# the one case where config.json alone would give the wrong SSM dtype.
_SSM_DTYPE = re.compile(r"--mamba-ssm-cache-dtype='(\w+)' was passed")
_KV_DTYPE = re.compile(r"kv_cache_dtype=torch\.(\w+)")
_ENGINE_RANK = re.compile(r"\((?:Worker|EngineCore)_DP(\d+)")

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float8_e4m3fn": 1, "fp8": 1, "uint8": 1}


@dataclass
class Cited:
    """A value together with the log line it was read from."""

    value: float | int | str | tuple
    line: int

    def cite(self) -> str:
        return f":{self.line}"


@dataclass
class WorkerMemory:
    """Everything the report needs from one worker log, for one engine."""

    log_name: str
    total: Cited | None = None
    free: Cited | None = None
    utilization: Cited | None = None
    budget: Cited | None = None
    consumed: Cited | None = None
    peak_activation_reported: Cited | None = None
    graphs_actual: Cited | None = None
    graphs_estimated: Cited | None = None
    graphs_estimate_early: Cited | None = None
    util_pair: Cited | None = None
    kv_to_budget: Cited | None = None
    kv_to_gpu: Cited | None = None
    weights: Cited | None = None
    graph_plan: Cited | None = None
    capture_sizes: Cited | None = None
    kv_memory: Cited | None = None
    kv_tokens: Cited | None = None
    kv_max_model_len: Cited | None = None
    kv_concurrency: Cited | None = None
    block_size: Cited | None = None
    block_size_text: str = ""
    mamba_pad: Cited | None = None
    mamba_pad_text: str = ""
    mamba_pad_line: int = 0
    ssm_dtype: Cited | None = None
    kv_dtype: Cited | None = None
    engines: int = 0

    @property
    def non_torch(self) -> float | None:
        """vLLM prints weights and non-torch as one number; split them back."""
        if self.consumed is None or self.weights is None:
            return None
        return float(self.consumed.value) - float(self.weights.value)

    @property
    def peak_activation(self) -> float | None:
        """The real activation peak, without the graph reservation vLLM folds in."""
        if self.peak_activation_reported is None or self.graphs_estimated is None:
            return None
        return float(self.peak_activation_reported.value) - float(self.graphs_estimated.value)

    @property
    def graph_estimate(self) -> Cited | None:
        """The graph reservation, preferring the post-capture line over the early one."""
        return self.graphs_estimated or self.graphs_estimate_early

    @property
    def derived_total(self) -> Cited | None:
        """Device size implied by the utilization advisory, for logs that stop early.

        vLLM says utilization A behaves like B once the graph estimate is taken out
        of the budget, so the estimate is (A - B) of the device. Both numbers are
        rounded in the log, which puts the result within about a GiB of the truth.
        """
        if self.util_pair is None or self.graph_estimate is None:
            return None
        utilization, without_graphs = self.util_pair.value
        if utilization <= without_graphs:
            return None
        return Cited(float(self.graph_estimate.value) / (utilization - without_graphs), self.util_pair.line)


@dataclass
class ModelFacts:
    """The parts of config.json the KV arithmetic depends on."""

    name: str
    source: str = "config.json"
    layers: int | None = None
    full_attn: int | None = None
    gdn: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    dtype: str | None = None
    num_experts: int | None = None
    experts_per_tok: int | None = None
    # GDN (linear attention) state, one tensor pair per layer per request.
    k_heads: int | None = None
    v_heads: int | None = None
    k_head_dim: int | None = None
    v_head_dim: int | None = None
    conv_kernel: int | None = None
    mamba_ssm_dtype: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def has_gdn_dims(self) -> bool:
        return None not in (self.k_heads, self.v_heads, self.k_head_dim, self.v_head_dim, self.conv_kernel)


def parse_worker_log(path: Path) -> WorkerMemory:
    """Read one worker log, keeping the line number of every value found."""
    memory = WorkerMemory(log_name=path.name)
    ranks: set[int] = set()

    # newline="\n" so progress-bar carriage returns do not split lines: the
    # cited numbers must match what grep -n and an editor show.
    with path.open(errors="replace", newline="\n") as handle:
        for number, raw in enumerate(handle, start=1):
            line = _ANSI.sub("", raw)

            rank = _ENGINE_RANK.search(line)
            if rank:
                ranks.add(int(rank.group(1)))
                if int(rank.group(1)) != ENGINE:
                    continue

            if (match := _BUDGET.search(line)) and memory.budget is None:
                memory.free = Cited(float(match.group(1)), number)
                memory.total = Cited(float(match.group(2)), number)
                memory.utilization = Cited(float(match.group(3)), number)
                memory.budget = Cited(float(match.group(4)), number)
                memory.consumed = Cited(float(match.group(5)), number)
                memory.peak_activation_reported = Cited(float(match.group(6)), number)
                memory.kv_to_budget = Cited((int(match.group(8)), float(match.group(9))), number)
                memory.kv_to_gpu = Cited((int(match.group(10)), float(match.group(11))), number)
            elif (match := _WEIGHTS.search(line)) and memory.weights is None:
                memory.weights = Cited(float(match.group(1)), number)
            elif (match := _GRAPH_POOL.search(line)) and memory.graphs_actual is None:
                memory.graphs_actual = Cited(float(match.group(1)), number)
                memory.graphs_estimated = Cited(float(match.group(2)), number)
            elif (match := _GRAPH_ESTIMATE.search(line)) and memory.graphs_estimate_early is None:
                memory.graphs_estimate_early = Cited(float(match.group(1)), number)
            elif (match := _UTIL_ADVISORY.search(line)) and memory.util_pair is None:
                memory.util_pair = Cited((float(match.group(1)), float(match.group(2))), number)
            elif (match := _GRAPH_PLAN.search(line)) and memory.graph_plan is None:
                modes = tuple(
                    (name, int(count), int(largest))
                    for name, count, largest in _GRAPH_PLAN_MODE.findall(match.group(1))
                )
                memory.graph_plan = Cited(modes, number)
            elif (match := _CAPTURE_SIZES.search(line)) and memory.capture_sizes is None:
                sizes = tuple(int(size) for size in match.group(1).split(",") if size.strip())
                memory.capture_sizes = Cited(sizes, number)
            elif (match := _KV_MEMORY.search(line)) and memory.kv_memory is None:
                memory.kv_memory = Cited(float(match.group(1)), number)
            elif (match := _KV_TOKENS.search(line)) and memory.kv_tokens is None:
                memory.kv_tokens = Cited(int(match.group(1).replace(",", "")), number)
                memory.kv_max_model_len = Cited(int(match.group(2).replace(",", "")), number)
                memory.kv_concurrency = Cited(float(match.group(3)), number)
            elif (match := _BLOCK_SIZE.search(line)) and memory.block_size is None:
                memory.block_size = Cited(int(match.group(1)), number)
                memory.block_size_text = line.split("] ", 1)[-1].strip()
            elif (match := _MAMBA_PAD.search(line)) and not memory.mamba_pad_text:
                memory.mamba_pad = Cited(float(match.group(1)), number)
                memory.mamba_pad_text = line.split("] ", 1)[-1].strip()
                memory.mamba_pad_line = number
            elif (match := _SSM_DTYPE.search(line)) and memory.ssm_dtype is None:
                memory.ssm_dtype = Cited(match.group(1), number)
            elif (match := _KV_DTYPE.search(line)) and memory.kv_dtype is None:
                memory.kv_dtype = Cited(match.group(1), number)

    memory.engines = len(ranks)
    return memory


def read_model_facts(model_path: Path, name: str) -> ModelFacts:
    """Pull the KV-relevant fields out of the model's config.json."""
    facts = ModelFacts(name=name)
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        facts.problems.append(f"config.json not readable at {config_path}")
        return facts

    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        facts.problems.append(f"config.json unreadable: {error}")
        return facts

    text = raw.get("text_config", raw)
    facts.layers = text.get("num_hidden_layers")
    facts.kv_heads = text.get("num_key_value_heads")
    facts.head_dim = text.get("head_dim")
    facts.dtype = text.get("dtype") or text.get("torch_dtype")
    facts.num_experts = text.get("num_experts")
    facts.experts_per_tok = text.get("num_experts_per_tok")
    facts.k_heads = text.get("linear_num_key_heads")
    facts.v_heads = text.get("linear_num_value_heads")
    facts.k_head_dim = text.get("linear_key_head_dim")
    facts.v_head_dim = text.get("linear_value_head_dim")
    facts.conv_kernel = text.get("linear_conv_kernel_dim")
    facts.mamba_ssm_dtype = text.get("mamba_ssm_dtype")

    layer_types = text.get("layer_types")
    if isinstance(layer_types, list):
        facts.full_attn = sum(1 for layer in layer_types if layer == "full_attention")
        facts.gdn = sum(1 for layer in layer_types if layer == "linear_attention")
    else:
        facts.problems.append("config.json has no layer_types: Full Attn / GDN split unknown")

    return facts


def _missing(log_name: str, what: str) -> str:
    return f"NOT FOUND   {log_name}, pattern for {what}"


def _gib(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "   ?  "


# Every value in the GPU MEMORY, BUDGET and CUDA GRAPHS blocks ends at the same
# column, and the notes start at the same one, so the decimal points read as a
# column. Padding baked into each literal cannot do that: it holds only for the
# digit count of whatever log it was written against.
_VALUE_END = 55
_NOTE_START = _VALUE_END + 8


def _row(label: str, value: str, note: str = "", unit: str = "") -> str:
    """One label/value line of the memory blocks, with an optional note column."""
    row = f"{label}{value:>{_VALUE_END - len(label)}}{unit}"
    return f"{row:<{_NOTE_START}}{note}" if note else row


def _factors(tokens: Sequence[tuple[str, str | None]], tail: str) -> list[str]:
    """An arithmetic line plus an arrow naming every labelled factor.

    Tokens are written out in order, so operators come in unlabelled. Arrow
    columns are measured from the rendered line rather than written by hand,
    which is what keeps them under the right factor when a model has wider
    numbers than the one this was first read against.
    """
    line = ""
    columns: list[tuple[int, str]] = []
    for token, label in tokens:
        if label is not None:
            # Centred under the factor, the way the KV-per-token block reads.
            columns.append((len(line) + len(token) // 2, label))
        line += token

    out = [line + tail]
    text_column = columns[-1][0] + 5

    def stems(width: int) -> str:
        row = [" "] * width
        for column, _ in columns:
            if column < width:
                row[column] = "|"
        return "".join(row)

    out.append(stems(columns[-1][0] + 1))
    for column, label in reversed(columns):
        out.append(f"{stems(column)}+{'-' * (text_column - column - 2)} {label}")
    return out


def _gdn_state_section(memory: WorkerMemory, facts: ModelFacts, page_mib: float | None) -> tuple[list[str], list[str]]:
    """The two state tensors of one GDN layer, with the arithmetic spelled out.

    The sizes come from config.json and the resolved dtypes, so they are a
    computation rather than a quote. The log's own padding percentage is used to
    check the result: a wrong dtype is otherwise invisible here and would
    misprice every request downstream.
    """
    out = ["GDN STATE PER REQUEST, one GDN layer   (fixed size, it does not grow with tokens)", ""]

    if not facts.has_gdn_dims:
        out += ["  !! config.json has no linear_* keys: state size not derived", ""]
        return out, ["GDN state"]

    conv_dtype = str(facts.dtype)
    ssm_dtype = str(memory.ssm_dtype.value) if memory.ssm_dtype else (facts.mamba_ssm_dtype or conv_dtype)
    conv_width = _DTYPE_BYTES.get(conv_dtype)
    ssm_width = _DTYPE_BYTES.get(ssm_dtype)
    if not conv_width or not ssm_width:
        out += [f"  !! unknown dtype (conv {conv_dtype}, ssm {ssm_dtype}): state size not derived", ""]
        return out, ["GDN state"]

    channels = facts.k_head_dim * facts.k_heads * 2 + facts.v_head_dim * facts.v_heads
    conv_bytes = channels * (facts.conv_kernel - 1) * conv_width
    ssm_bytes = facts.v_heads * facts.v_head_dim * facts.k_head_dim * ssm_width

    out += [
        "  conv state   the last kernel-1 inputs of the causal conv, one filter per",
        "               channel, where the channels are q, k and v concatenated",
        "",
    ]
    out += _factors(
        [
            ("      ( ", None),
            (str(facts.k_head_dim), "q,k: linear_key_head_dim"),
            ("  x  ", None),
            (str(facts.k_heads), "q,k: linear_num_key_heads"),
            ("  x  ", None),
            ("2", "q and k, both this size"),
            ("  +  ", None),
            (str(facts.v_head_dim), "v: linear_value_head_dim"),
            ("  x  ", None),
            (str(facts.v_heads), "v: linear_num_value_heads"),
            (" )", None),
            ("  x  ", None),
            (str(facts.conv_kernel - 1), "linear_conv_kernel_dim - 1"),
            ("  x  ", None),
            (str(conv_width), f"sizeof({conv_dtype})"),
        ],
        f"  =  {conv_bytes:,} B  =  {conv_bytes / 1024:g} KiB",
    )
    out += ["", "  ssm state    the recurrent matrix the whole context is folded into", ""]
    out += _factors(
        [
            ("      ( ", None),
            (str(facts.v_heads), "linear_num_value_heads"),
            ("  x  ", None),
            (str(facts.v_head_dim), "linear_value_head_dim"),
            ("  x  ", None),
            (str(facts.k_head_dim), "linear_key_head_dim"),
            (" )", None),
            ("  x  ", None),
            (str(ssm_width), f"sizeof({ssm_dtype})"),
        ],
        f"  =  {ssm_bytes:,} B  =  {ssm_bytes / MIB:.4f} MiB",
    )

    state_mib = (conv_bytes + ssm_bytes) / MIB
    out += ["", f"  conv + ssm = {state_mib:.4f} MiB per GDN layer"]
    if memory.ssm_dtype:
        out.append(
            f"  ssm dtype is {ssm_dtype} from the recipe {memory.ssm_dtype.cite()}, "
            f"overriding config.json mamba_ssm_dtype = {facts.mamba_ssm_dtype}"
        )
    else:
        out.append(f"  ssm dtype {ssm_dtype} from config.json, conv dtype {conv_dtype} = model dtype")

    if page_mib and memory.mamba_pad:
        pad = float(memory.mamba_pad.value)
        implied = page_mib / (1 + pad / 100)
        if abs(implied - state_mib) <= 0.01 * state_mib:
            out.append(
                f"  the log pads that to the {page_mib:.3f} MiB page by {pad}%, so the two agree"
                f"    :{memory.mamba_pad.line}"
            )
        else:
            out += [
                f"  !! the log pads the GDN page by {pad}% (:{memory.mamba_pad.line}), which implies",
                f"     {implied:.4f} MiB per layer, not {state_mib:.4f}: the dtypes above are not",
                "     what vLLM allocated, so every size derived from them is wrong",
            ]
    if page_mib:
        out.append(
            f"  x {facts.gdn} GDN layers = {facts.gdn * page_mib:.1f} MiB per request, charged once whatever the length"
        )
    out.append("")
    return out, []


def render_report(
    memory: WorkerMemory,
    facts: ModelFacts,
    *,
    role: str,
    layout: str,
    nodes: int,
    engines: int,
    concurrency: int | None,
    isl: int | None,
    osl: int | None,
) -> str:
    """Render the report exactly as documented, one section at a time."""
    out: list[str] = []
    add = out.append
    unknown: list[str] = []

    title = f"logs/{MEMORY_DIRNAME}/{role}.out"
    add(title)
    add("=" * len(title))
    add("")
    add(f"model    {facts.name:<30} layout  {layout}, {nodes} nodes")
    add(f"role     {role:<30} numbers are per engine (all {engines} identical)")
    add(f"source   {memory.log_name}, engine DP{ENGINE}        <file>:<line> below = proof")
    add("")

    add("GPU MEMORY")
    if memory.total and memory.free:
        add(_row("  total on device", _gib(memory.total.value), memory.total.cite(), unit=" GiB"))
        add(
            _row(
                "  in use before vLLM starts (CUDA context, driver)",
                f"{float(memory.total.value) - float(memory.free.value):.2f}",
                f"= {_gib(memory.total.value)} - {_gib(memory.free.value)}",
            )
        )
        add(_row("  free when vLLM took its snapshot", _gib(memory.free.value), memory.free.cite()))
    elif memory.derived_total and memory.util_pair and memory.graph_estimate:
        derived = memory.derived_total
        utilization, without_graphs = memory.util_pair.value
        add(_row("  total on device", _gib(derived.value), f"{derived.cite()} (derived)", unit=" GiB"))
        add(
            f"    = {_gib(memory.graph_estimate.value)} GiB graph estimate / ({utilization} - {without_graphs})"
            " utilization, that line's own arithmetic"
        )
        add("  free on startup and driver overhead are not in this log: vLLM prints them")
        add("  only after graph capture, and this run stopped before that")
        unknown.append("free memory on startup")
    else:
        add(f"  {_missing(memory.log_name, 'Free memory on device (.../... GiB) on startup')}")
        unknown.append("device memory")
    add("")

    if memory.budget and memory.utilization and memory.total:
        add(
            f"BUDGET  = total x gpu-memory-utilization = {_gib(memory.total.value)} x "
            f"{memory.utilization.value} = {_gib(memory.budget.value)} GiB    {memory.budget.cite()}"
        )
        add("  note: the budget is a share of *total*, not of free, so the memory already")
        add("  in use above is spent twice on paper, and actual usage can exceed the budget.")
        add("")
        if memory.weights:
            add(_row("    - model weights", _gib(memory.weights.value), memory.weights.cite()))
        else:
            add(_row("    - model weights", "?"))
        non_torch = memory.non_torch
        if non_torch is not None and memory.consumed and memory.weights:
            add(
                _row(
                    "    - non-torch (NCCL buffers, allocator)",
                    f"{non_torch:.2f}",
                    f"= {_gib(memory.consumed.value)} - {_gib(memory.weights.value)}",
                )
            )
        activation = memory.peak_activation
        if activation is not None and memory.peak_activation_reported and memory.graphs_estimated:
            add(
                _row(
                    "    - peak activation (eager dummy forward)",
                    f"{activation:.2f}",
                    f"= {_gib(memory.peak_activation_reported.value)} - {_gib(memory.graphs_estimated.value)}",
                )
            )
        if memory.graphs_estimated:
            add(
                _row(
                    "    - CUDA graph reservation (an estimate)",
                    _gib(memory.graphs_estimated.value),
                    f'{memory.graphs_estimated.cite()} ("estimated")',
                )
            )
        add("    ----------------------------------------------------")
        if memory.kv_memory:
            add(_row("    = KV cache", _gib(memory.kv_memory.value), memory.kv_memory.cite()))
        add("")
        if memory.peak_activation_reported and memory.graphs_estimated:
            add(f'  the log prints {memory.peak_activation_reported.value} as "peak activation" because vLLM folds')
            add("  the graph estimate into that counter (gpu_worker.py:528); the two lines above split it")
    elif memory.derived_total and memory.util_pair and memory.graph_estimate:
        total = float(memory.derived_total.value)
        estimate = memory.graph_estimate
        utilization = memory.util_pair.value[0]
        budget = total * utilization
        add(
            f"BUDGET  = total x gpu-memory-utilization = {_gib(total)} x {utilization} = "
            f"{_gib(budget)} GiB    {memory.derived_total.cite()} (derived)"
        )
        add("")
        if memory.weights:
            add(_row("    - model weights", _gib(memory.weights.value), memory.weights.cite()))
        if memory.weights and memory.kv_memory:
            rest = budget - float(memory.weights.value) - float(estimate.value) - float(memory.kv_memory.value)
            add(
                _row(
                    "    - non-torch + peak activation",
                    f"{rest:.2f}",
                    f"= {_gib(budget)} - {_gib(memory.weights.value)}"
                    f" - {_gib(estimate.value)} - {_gib(memory.kv_memory.value)}",
                )
            )
        add(_row("    - CUDA graph reservation (an estimate)", _gib(estimate.value), estimate.cite()))
        add("    ----------------------------------------------------")
        if memory.kv_memory:
            add(_row("    = KV cache", _gib(memory.kv_memory.value), memory.kv_memory.cite()))
        add("")
        add("  non-torch and peak activation share one line: splitting them needs the")
        add("  post-capture summary, which this run never printed")
        unknown.append("non-torch / activation split")
    else:
        add(f"BUDGET  {_missing(memory.log_name, 'Desired GPU memory utilization is (util, N GiB)')}")
        unknown.append("budget")
    add("")

    add("CUDA GRAPHS")
    if memory.graphs_actual and memory.graphs_estimated:
        estimated = float(memory.graphs_estimated.value)
        actual = float(memory.graphs_actual.value)
        # Same ratio vLLM prints in that line: how far the miss is above actual.
        over = ((estimated - actual) / actual * 100) if actual else 0.0
        add(_row("  reserved from the estimate", _gib(estimated), memory.graphs_estimated.cite(), unit=" GiB"))
        add(_row("  actually used after capture", _gib(actual), memory.graphs_actual.cite(), unit=" GiB"))
        add(
            _row(
                "  reserved and never used",
                _gib(estimated - actual),
                f"{over:.0f}% over  <-- see VERDICT",
                unit=" GiB",
            )
        )
    elif memory.graph_estimate:
        estimate = memory.graph_estimate
        add(_row("  reserved from the estimate", _gib(estimate.value), estimate.cite(), unit=" GiB"))
        add(_row("  actually used after capture", "?", "printed only after capture", unit=" GiB"))
        add("    this run stopped before that, so the overshoot cannot be checked")
        unknown.append("graph pool (actual)")
    else:
        add(f"  {_missing(memory.log_name, 'CUDA graph pool memory: N GiB (actual), N GiB (estimated)')}")
        unknown.append("graph pool")
    add("")

    if memory.graph_plan:
        sizes = list(memory.capture_sizes.value) if memory.capture_sizes else []
        for name, count, largest in memory.graph_plan.value:
            mode_sizes = [size for size in sizes if size <= largest] or None
            add(f"  {name:<10} {count} graphs   {_format_sizes(mode_sizes)}")
        add(f"{'':<62}{memory.graph_plan.cite()} (counts)")
    else:
        add(f"  {_missing(memory.log_name, 'Profiling CUDA graph memory: MODE=N (largest=N)')}")
    add("")
    add("  how the estimate is built (vllm/v1/worker/gpu_model_runner.py, profile_cudagraph_memory):")
    add("    only the 2 largest graphs per mode are captured, then")
    add("      estimate = max(first_capture) + SUM per_graph x (N - 1)")
    add("    so a graph for 1 token is priced as a graph for the second-largest size.")
    add("")

    add("MODEL  (all values from config.json, key names given so they can be rechecked)")
    add(f"  layers            {_num(facts.layers):>6}        num_hidden_layers")
    add(f'    Full Attn       {_num(facts.full_attn):>6}        layer_types == "full_attention"')
    add(f'    GDN             {_num(facts.gdn):>6}        layer_types == "linear_attention"')
    if facts.num_experts:
        add(
            f"    MoE             {_num(facts.layers):>6}        num_experts {facts.num_experts}, "
            f"num_experts_per_tok {facts.experts_per_tok}"
        )
    add(f"  KV heads          {_num(facts.kv_heads):>6}        num_key_value_heads")
    add(f"  head dim          {_num(facts.head_dim):>6}        head_dim")
    dtype = memory.kv_dtype.value if memory.kv_dtype else None
    if dtype:
        add(f"  KV dtype        {dtype:>8}        resolved at runtime, kv_cache_dtype=torch.{dtype}")
        add(f"                                  {memory.kv_dtype.cite()}; config.json dtype = {facts.dtype}")
    else:
        add(f"  KV dtype        {_missing(memory.log_name, 'kv_cache_dtype=torch.<dtype>')}")
        unknown.append("KV dtype")
    for problem in facts.problems:
        add(f"  !! {problem}")
    add("")

    dtype_bytes = _DTYPE_BYTES.get(str(dtype))
    per_token = None
    if dtype_bytes and facts.kv_heads and facts.head_dim:
        per_token = facts.kv_heads * facts.head_dim * dtype_bytes * 2
        add("KV PER TOKEN, one Full Attn layer")
        add("")
        out.extend(
            _factors(
                [
                    ("      ( ", None),
                    (str(facts.kv_heads), "num_key_value_heads"),
                    ("  x  ", None),
                    (str(facts.head_dim), "head_dim"),
                    ("  x  ", None),
                    (str(dtype_bytes), f"sizeof({dtype})"),
                    (" )", None),
                    ("  x  ", None),
                    ("2", "K and V"),
                ],
                f"   =  {per_token} B  =  {per_token / 1024:g} KiB",
            )
        )
        add("")

    page_mib = int(memory.block_size.value) * per_token / MIB if memory.block_size and per_token else None
    if facts.gdn:
        gdn_lines, gdn_unknown = _gdn_state_section(memory, facts, page_mib)
        out.extend(gdn_lines)
        unknown.extend(gdn_unknown)

    if memory.block_size and per_token:
        block = int(memory.block_size.value)
        add(f"BLOCK SIZE  {block} tokens, the same for Full Attn and GDN")
        add(f'  "{memory.block_size_text}"')
        add(f"{'':<50}{memory.log_name}:{memory.block_size.line}")
        if memory.mamba_pad_text:
            add(f'  "{memory.mamba_pad_text}"')
            add(f"{'':<50}{memory.log_name}:{memory.mamba_pad_line}")
        add("")
        add(f"  page = {block} tokens x {per_token / 1024:g} KiB = {page_mib:.3f} MiB   (one block of one layer)")
        add("")
    else:
        add(f"BLOCK SIZE  {_missing(memory.log_name, 'Setting attention block size to N tokens')}")
        unknown.append("block size")
        add("")

    per_request_mib = None
    max_model_len = int(memory.kv_max_model_len.value) if memory.kv_max_model_len else None
    if page_mib and max_model_len and facts.full_attn is not None and facts.gdn is not None:
        blocks_per_layer = math.ceil(max_model_len / int(memory.block_size.value))
        full_blocks = blocks_per_layer * facts.full_attn
        gdn_blocks = facts.gdn
        total_blocks = full_blocks + gdn_blocks
        per_request_mib = total_blocks * page_mib
        add(f"PER REQUEST at max-model-len {max_model_len}")
        add(
            f"  Full Attn   state grows per token   ceil({max_model_len} / {memory.block_size.value})"
            f" = {blocks_per_layer} blocks per-layer"
        )
        add(f"{'':<38}{blocks_per_layer} x {facts.full_attn} layers{'':<5}= {full_blocks} blocks")
        add("  GDN         state is per request     1 block per-layer")
        add(f"{'':<39}1 x {facts.gdn} layers{'':<5}= {gdn_blocks} blocks")
        add(f"{'':<38}{full_blocks} + {gdn_blocks}{'':<11}= {total_blocks} blocks")
        add("")
        head = f"  {total_blocks} x {page_mib:.3f} MiB = {per_request_mib:.0f} MiB per request"
        add(f"{head}      {full_blocks * page_mib:.0f} MiB Full Attn")
        add(f"{'':<{len(head) + 6}}{gdn_blocks * page_mib:.0f} MiB GDN state")
        add("")

    add("CAPACITY")
    computed = None
    if per_request_mib and memory.kv_memory:
        computed = float(memory.kv_memory.value) * 1024 / per_request_mib
        add(
            f"  {_gib(memory.kv_memory.value)} GiB / {per_request_mib:.0f} MiB"
            f"   = {computed:.1f} requests per engine (from the arithmetic above)"
        )
    if memory.kv_concurrency and memory.kv_tokens:
        add(
            f"  vLLM reports             {memory.kv_concurrency.value} requests"
            f" = {memory.kv_tokens.value:,} tokens    {memory.kv_concurrency.cite()}"
        )
        add(
            f"  x {engines} engines{'':<12}= {int(float(memory.kv_concurrency.value) * engines)}"
            f" requests for the {role} side"
        )
    else:
        add(f"  {_missing(memory.log_name, 'GPU KV cache size: N tokens, Maximum concurrency ...')}")
        unknown.append("capacity")
    add("")

    add("VERDICT")
    verdicts = _verdicts(
        memory,
        role=role,
        engines=engines,
        concurrency=concurrency,
        isl=isl,
        osl=osl,
        max_model_len=max_model_len,
        per_request_mib=per_request_mib,
    )
    out.extend(verdicts)
    if unknown:
        out.append(f"  ?? not derived, missing from the log: {', '.join(unknown)}")

    return "\n".join(out) + "\n"


ROLE_PATTERNS = {"prefill": "*_prefill_w*.out", "decode": "*_decode_w*.out", "agg": "*_agg_w*.out"}
_ROLE_CONFIG_KEY = {"prefill": "prefill", "decode": "decode", "agg": "aggregated"}


def record_memory_report(config: SrtConfig, runtime: RuntimeContext) -> list[Path]:
    """Write one memory report per worker role, never failing the job."""
    try:
        return _record(config, runtime)
    except Exception as error:  # noqa: BLE001
        logger.warning("Failed to write memory report: %s", error)
        return []


def _record(config: SrtConfig, runtime: RuntimeContext) -> list[Path]:
    facts = read_model_facts(runtime.model_path, config.served_model_name)
    concurrency = _max_concurrency(config.benchmark.concurrencies)
    written: list[Path] = []

    for role, pattern in ROLE_PATTERNS.items():
        logs = sorted(runtime.log_dir.glob(pattern))
        if not logs:
            continue

        memory = parse_worker_log(logs[0])
        report = render_report(
            memory,
            facts,
            role=role,
            # Cosmetic: never let the header line cost us the whole report.
            layout=_safe_layout(config, role, runtime),
            nodes=len(logs),
            engines=max(memory.engines, 1) * len(logs),
            concurrency=concurrency,
            isl=config.benchmark.isl,
            osl=config.benchmark.osl,
        )

        path = runtime.log_dir / MEMORY_DIRNAME / f"{role}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
        written.append(path)
        logger.info("Wrote memory report: %s", path)

    return written


def _safe_layout(config: SrtConfig, role: str, runtime: RuntimeContext) -> str:
    try:
        return _layout(config, role, runtime)
    except Exception as error:  # noqa: BLE001
        logger.debug("Could not describe the %s layout: %s", role, error)
        return "layout unknown"


def _mode_flags(backend: object, role: str) -> dict[str, object]:
    """The recipe's vLLM flags for one role, keyed the way the CLI spells them.

    ``backend.vllm_config`` is a dataclass with a dict per mode, and a recipe may
    write either ``data_parallel_size`` or ``data-parallel-size``.
    """
    vllm_config = getattr(backend, "vllm_config", None)
    flags = getattr(vllm_config, _ROLE_CONFIG_KEY[role], None)
    if not isinstance(flags, dict):
        return {}
    return {str(key).replace("_", "-"): value for key, value in flags.items()}


def _layout(config: SrtConfig, role: str, runtime: RuntimeContext) -> str:
    """Name the parallel layout the way the recipe means it: DEP16, TP8, ..."""
    mode = _mode_flags(config.backend, role)
    dp = mode.get("data-parallel-size")
    tp = mode.get("tensor-parallel-size")
    gpu = (config.resources.gpu_type or "gpu").upper()
    parts = []
    if dp and dp > 1:
        parts.append(f"{'DEP' if mode.get('enable-expert-parallel') else 'DP'}{dp}")
    if tp and tp > 1:
        parts.append(f"TP{tp}")
    if parts:
        layout = ", ".join(parts)
    elif mode:
        layout = "single GPU"  # flags for this role exist, none of them parallel
    else:
        layout = "layout unknown"  # the recipe says nothing about this role
    return f"{layout}, {runtime.gpus_per_node}x{gpu} per node"


def _max_concurrency(concurrencies: list[int] | str | None) -> int | None:
    if concurrencies is None:
        return None
    if isinstance(concurrencies, str):
        values = [int(part) for part in concurrencies.split("x") if part.strip().isdigit()]
    else:
        values = [int(value) for value in concurrencies]
    return max(values) if values else None


def _num(value: int | None) -> str:
    return str(value) if value is not None else "?"


def _format_sizes(sizes: list[int] | None, width: int = 55, indent: int = 25) -> str:
    """Group capture sizes into runs of a constant step, so the list reads at a glance.

    The small irregular head (1 2 4) lands on its own line and each constant-step
    run gets its own, which is how the capture list is actually generated.
    """
    if not sizes:
        return "(capture sizes not in the log)"

    groups: list[list[int]] = []
    head: list[int] = []
    index = 0
    while index < len(sizes):
        run = 1
        if index + 1 < len(sizes):
            step = sizes[index + 1] - sizes[index]
            while index + run < len(sizes) and sizes[index + run] - sizes[index + run - 1] == step:
                run += 1
        # Three points make a run worth its own line; anything shorter is part
        # of the irregular head (1 2 4).
        if run >= 3:
            if head:
                groups.append(head)
                head = []
            groups.append(sizes[index : index + run])
            index += run
        else:
            head.append(sizes[index])
            index += 1
    if head:
        groups.append(head)

    lines = []
    for group in groups:
        text = " ".join(str(size) for size in group)
        while len(text) > width:
            cut = text.rfind(" ", 0, width)
            lines.append(text[:cut])
            text = text[cut + 1 :]
        lines.append(text)
    pad = "\n" + " " * indent
    return pad.join(lines)


def _verdicts(
    memory: WorkerMemory,
    *,
    role: str,
    engines: int,
    concurrency: int | None,
    isl: int | None,
    osl: int | None,
    max_model_len: int | None,
    per_request_mib: float | None,
) -> list[str]:
    lines: list[str] = []

    if memory.graphs_actual and memory.graphs_estimated:
        actual = float(memory.graphs_actual.value)
        estimated = float(memory.graphs_estimated.value)
        if actual > 0 and estimated / actual >= 1.5:
            lost = estimated - actual
            lines.append(
                f"  !! CUDA graph estimate overshoots by {(estimated - actual) / actual * 100:.0f}% "
                f"({estimated:.2f} reserved, {actual:.2f} used)."
            )
            if per_request_mib:
                lines.append(
                    f"     The {lost:.2f} GiB lost is {lost * 1024 / per_request_mib:.0f} requests per engine."
                )
            if memory.kv_to_gpu:
                flag, gib = memory.kv_to_gpu.value
                extra = f" -> {gib * 1024 / per_request_mib:.1f} req/engine" if per_request_mib else ""
                lines.append(f"     Recover with --kv-cache-memory={flag} ({gib:.2f} GiB{extra}), which skips")
                lines.append("     the estimate entirely, or by capturing fewer graph sizes (max-num-seqs).")

    if concurrency and engines and memory.kv_concurrency:
        needed = concurrency / engines
        capacity = float(memory.kv_concurrency.value)
        if needed > capacity:
            short = (needed - capacity) / needed * 100
            lines.append(
                f"  !! recipe concurrency {concurrency} needs {concurrency} / {engines} = {needed:.0f} "
                f"requests per engine,"
            )
            lines.append(f"     capacity is {capacity}. {short:.0f}% of the requests cannot be admitted: they queue")
            lines.append("     while prefill holds their KV. This point will not produce a usable result.")
        else:
            lines.append(f"  ok concurrency {concurrency} needs {needed:.0f} per engine, capacity {capacity}")

    if max_model_len and isl and osl:
        if max_model_len >= isl + osl:
            lines.append(f"  ok max-model-len {max_model_len} covers isl {isl} + osl {osl}")
        else:
            lines.append(f"  !! max-model-len {max_model_len} is below isl {isl} + osl {osl}")

    return lines or ["  (nothing to flag)"]
