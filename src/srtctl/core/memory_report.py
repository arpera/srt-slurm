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
"""

from __future__ import annotations

import json
import logging
import math
import re
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
_GRAPH_PLAN = re.compile(r"Profiling CUDA graph memory: (.+)$")
_GRAPH_PLAN_MODE = re.compile(r"(\w+)=(\d+) \(largest=(\d+)\)")
_CAPTURE_SIZES = re.compile(r"'cudagraph_capture_sizes':\s*\[([\d,\s]+)\]")
_KV_MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_KV_TOKENS = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
)
_BLOCK_SIZE = re.compile(r"Setting attention block size to (\d+) tokens")
_MAMBA_PAD = re.compile(r"Padding mamba page size by ([\d.]+)% ")
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
    mamba_pad_text: str = ""
    mamba_pad_line: int = 0
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
    problems: list[str] = field(default_factory=list)


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
                memory.mamba_pad_text = line.split("] ", 1)[-1].strip()
                memory.mamba_pad_line = number
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
        add(f"  total on device                                {_gib(memory.total.value)} GiB    {memory.total.cite()}")
        add(
            f"  in use before vLLM starts (CUDA context, driver)  "
            f"{float(memory.total.value) - float(memory.free.value):.2f}"
            f"        = {_gib(memory.total.value)} - {_gib(memory.free.value)}"
        )
        add(f"  free when vLLM took its snapshot               {_gib(memory.free.value)}        {memory.free.cite()}")
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
        weights = f"{_gib(memory.weights.value)}        {memory.weights.cite()}" if memory.weights else "   ?"
        add(f"    - model weights                              {weights}")
        non_torch = memory.non_torch
        if non_torch is not None and memory.consumed and memory.weights:
            add(
                f"    - non-torch (NCCL buffers, allocator)          {non_torch:.2f}"
                f"        = {_gib(memory.consumed.value)} - {_gib(memory.weights.value)}"
            )
        activation = memory.peak_activation
        if activation is not None and memory.peak_activation_reported and memory.graphs_estimated:
            add(
                f"    - peak activation (eager dummy forward)        {activation:.2f}"
                f"        = {_gib(memory.peak_activation_reported.value)}"
                f" - {_gib(memory.graphs_estimated.value)}"
            )
        if memory.graphs_estimated:
            add(
                f"    - CUDA graph reservation (an estimate)        {_gib(memory.graphs_estimated.value)}"
                f'        {memory.graphs_estimated.cite()} ("estimated")'
            )
        add("    ----------------------------------------------------")
        if memory.kv_memory:
            add(
                f"    = KV cache                                    {_gib(memory.kv_memory.value)}"
                f"        {memory.kv_memory.cite()}"
            )
        add("")
        if memory.peak_activation_reported and memory.graphs_estimated:
            add(f'  the log prints {memory.peak_activation_reported.value} as "peak activation" because vLLM folds')
            add("  the graph estimate into that counter (gpu_worker.py:528); the two lines above split it")
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
        add(
            f"  reserved from the estimate                     {estimated:>6.2f} GiB    {memory.graphs_estimated.cite()}"
        )
        add(f"  actually used after capture                    {actual:>6.2f} GiB    {memory.graphs_actual.cite()}")
        add(
            f"  reserved and never used                        {estimated - actual:>6.2f} GiB"
            f"    {over:.0f}% over  <-- see VERDICT"
        )
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
        add(
            f"      ( {facts.kv_heads}  x  {facts.head_dim}  x  {dtype_bytes} )  x  2   "
            f"=  {per_token} B  =  {per_token / 1024:g} KiB"
        )
        add("        |      |      |       |")
        add("        |      |      |       +--- K and V")
        add(f"        |      |      +----------- sizeof({dtype})")
        add("        |      +------------------ head_dim")
        add("        +------------------------- num_key_value_heads")
        add("")

    page_mib = None
    if memory.block_size and per_token:
        block = int(memory.block_size.value)
        page_mib = block * per_token / MIB
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
            layout=_layout(config, role, runtime),
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


def _layout(config: SrtConfig, role: str, runtime: RuntimeContext) -> str:
    """Name the parallel layout the way the recipe means it: DEP16, TP8, ..."""
    vllm_config = getattr(config.backend, "vllm_config", None) or {}
    mode = vllm_config.get(_ROLE_CONFIG_KEY[role], {}) or {}
    dp = mode.get("data-parallel-size")
    tp = mode.get("tensor-parallel-size")
    gpu = (config.resources.gpu_type or "gpu").upper()
    parts = []
    if dp and dp > 1:
        parts.append(f"{'DEP' if mode.get('enable-expert-parallel') else 'DP'}{dp}")
    if tp and tp > 1:
        parts.append(f"TP{tp}")
    layout = ", ".join(parts) if parts else "single GPU"
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
