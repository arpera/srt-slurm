# Monitoring

## Table of Contents

- [Live Dashboard (srtctl monitor)](#live-dashboard-srtctl-monitor)
- [Checking Job Status](#checking-job-status)
- [Log Directory](#log-directory)
- [Log Structure](#log-structure)
- [Key Files](#key-files)
- [Common Commands](#common-commands)
- [Connecting to Running Jobs](#connecting-to-running-jobs)

---

## Live Dashboard (srtctl monitor)

`srtctl monitor` is a live terminal dashboard that brings everything into one place: SLURM queue state, job lifecycle stage, worker readiness, and benchmark metrics — all auto-refreshing without juggling `squeue` and `tail -f`.

```bash
srtctl monitor                          # Active + recently completed jobs
srtctl monitor --all                    # Also include older jobs from outputs/
srtctl monitor --outputs /path/to/out   # Override outputs directory
srtctl monitor --interval 10            # Refresh interval in seconds (default: 5)
srtctl monitor --once                   # Print snapshot and exit
srtctl monitor --resume KEY             # Resume a previous session
```

The outputs directory is auto-detected from `./outputs/` or `../outputs/`.

### Columns

| Column | Description |
|--------|-------------|
| Job ID | SLURM job ID (`▶` marks the selected row) |
| Slurm | Queue state: RUNNING / PENDING / ENDED … |
| Stage | Lifecycle stage inferred from the sweep log |
| Workers | Live readiness, e.g. `2/4P  4/4D` |
| Time | Elapsed wall time |
| Config | GPU type, topology, benchmark type, ISL/OSL |
| Metrics | Throughput (tok/s), TTFT, TPOT |

**Lifecycle stages:** Starting → Starting Infra → Head Ready → Starting Workers → Awaiting Workers → Starting Frontend → Benchmarking → Completed / Failed / Killed / Timed Out

### Keybindings

**Main view**

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate jobs |
| `↵` | Open detail view |
| `y` | Open `config.yaml` in vim |
| `d` | Delete output dir (finished) or cancel job (active) — prompts to confirm |
| `c` | Toggle last vs all concurrencies in Metrics |
| `a` | Toggle active-only vs all jobs |
| `q` | Quit |

**Detail view** (`↵` on a job — sweep log left, worker + benchmark logs right)

| Key | Action |
|-----|--------|
| `↑` / `↓` | Cycle panels (sweep / worker / benchmark) |
| `←` / `→` | Cycle worker files or benchmark concurrency sections |
| `↵` | Open current log in vim |
| `r` | Toggle auto-refresh |
| `ESC` | Back to job list |

### Session Resume

On exit, a session key is printed:

```
To resume this session, use  srtctl monitor --resume abc123def456
```

Sessions are saved to `/tmp/srt-dash-<user>.json` and restore the full set of tracked job IDs, including completed jobs.

---

## Checking Job Status

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

## Log Directory

After submission, `srtctl` tells you where logs are stored:

```
Submitted batch job 4459
Logs: logs/4459_4P_1D_20251122_041341/
```

The directory name follows the pattern: `{job_id}_{prefill}P_{decode}D_{timestamp}`

## Log Structure

```
logs/4459_4P_1D_20251122_041341/
│
├── config.yaml                              # Resolved job configuration
├── sglang_config.yaml                       # SGLang worker configuration
├── sbatch_script.sh                         # Generated SLURM script
├── nginx.conf                               # Load balancer configuration
├── 4459.json                                # Job metadata
│
├── log.out                                  # Main orchestration stdout
├── log.err                                  # Main orchestration stderr
├── benchmark.out                            # Benchmark results
├── benchmark.err                            # Benchmark errors
│
├── {node}_prefill_w{n}.out                  # Prefill worker stdout
├── {node}_prefill_w{n}.err                  # Prefill worker stderr (SGLang logs)
├── {node}_decode_w{n}.out                   # Decode worker stdout
├── {node}_decode_w{n}.err                   # Decode worker stderr (SGLang logs)
├── {node}_frontend_{n}.out                  # Frontend stdout
├── {node}_frontend_{n}.err                  # Frontend stderr
├── {node}_nginx.out                         # Nginx stdout
├── {node}_nginx.err                         # Nginx stderr
├── {node}_config.json                       # Per-node SGLang config dump
│
├── memory/                                  # Where each worker's GPU memory went
│   ├── prefill.out                          # Budget, CUDA graphs, KV capacity
│   └── decode.out
│
├── cached_assets/                           # Cached model assets
└── sa-bench_isl_1024_osl_1024/              # Benchmark results
    ├── isl_1024_osl_1024_concurrency_128_req_rate_inf.json
    ├── isl_1024_osl_1024_concurrency_512_req_rate_inf.json
    └── ...
```

## Key Files

### log.out

The main orchestration log showing node assignments, worker launches, and the frontend URL:

```
Node 0: watchtower-aqua-cn01
Node 1: watchtower-aqua-cn02
...
Master IP address (node 1): 10.30.1.49
Nginx node (node 0): watchtower-aqua-cn01
...
Prefill worker 0 leader: watchtower-aqua-cn01 (10.30.1.163)
Launching prefill worker 0, node 0 (local_rank 0): watchtower-aqua-cn01
...
Decode worker 0 leader: watchtower-aqua-cn05 (10.30.1.153)
...
Frontend available at: http://watchtower-aqua-cn01:8000
```

### benchmark.out

Shows benchmark progress and results:

```
Polling http://localhost:8000/health every 5 seconds...
Model is not ready, waiting for 4 prefills and 1 decodes to spin up.
Model is ready.

Warming up model with concurrency 128
============ Serving Benchmark Result ============
Successful requests:                     640
Benchmark duration (s):                  93.97
Request throughput (req/s):              6.81
Output token throughput (tok/s):         6278.02
---------------Time to First Token----------------
Mean TTFT (ms):                          1924.07
Median TTFT (ms):                        342.39
P99 TTFT (ms):                           13652.77
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.78
Median TPOT (ms):                        15.48
P99 TPOT (ms):                           22.36
==================================================
```

### Worker Logs ({node}\_prefill_w0.err, {node}\_decode_w0.err)

SGLang worker logs showing model loading, memory allocation, and runtime info. Check these for debugging CUDA errors, OOM issues, or NCCL failures.

Once the workers exit, srtctl rewrites these logs for reading: the colour
escapes the dynamo logger emits are stripped, and every progress bar is reduced
to the frame a terminal would have been left showing. A checkpoint load stops
being 102 near-identical lines:

```
(Worker_DP0_EP0 pid=942741) Loading safetensors checkpoint shards: 100% Completed | 100/100 [03:31<00:00,  2.11s/it]
(Worker_DP0_EP0 pid=942741) [AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%| 23/23 [01:20<00:00,  3.5s/profile]
```

Everything else is left byte for byte as the worker wrote it, and the surviving
frame keeps its place in the file, so timestamps around it still line up.

### memory/prefill.out, memory/decode.out

Where a worker's GPU memory went and how many requests one engine can hold,
written once the workers are up and before the benchmark starts. One file per
role, numbers are per engine:

```
BUDGET  = total x gpu-memory-utilization = 276.62 x 0.92 = 254.49 GiB    :1890

    - model weights                              168.89        :1342
    - non-torch (NCCL buffers, allocator)          2.36        = 171.25 - 168.89
    - peak activation (eager dummy forward)        3.68        = 36.74 - 33.06
    - CUDA graph reservation (an estimate)        33.06        :1889 ("estimated")
    ----------------------------------------------------
    = KV cache                                    46.50        :1429
```

The report then derives the KV cost of one request from the model config (bytes
per token, page size, blocks for attention plus one page per linear-attention
layer), converts the KV budget into requests per engine, and ends with a VERDICT
block that flags the two failures worth catching before a benchmark burns hours:

For hybrid models the two GDN state tensors are broken out the same way, with an
arrow naming every factor, because that state is charged per request rather than
per token and is usually the part that decides how many requests fit. Its size
depends on the resolved mamba dtypes, so the result is cross-checked against the
padding percentage vLLM logs: if the two disagree, the report says so instead of
pricing every request from a dtype the engine never used.

```
VERDICT
  !! CUDA graph estimate overshoots by 286% (33.06 reserved, 8.57 used).
     The 24.49 GiB lost is 20 requests per engine.
     Recover with --kv-cache-memory=61814666240 (57.57 GiB -> 47.8 req/engine)
  !! recipe concurrency 1024 needs 1024 / 16 = 64 requests per engine,
     capacity is 38.54. 40% of the requests cannot be admitted
```

`:1890` is the line in the worker log the value was read from, so every number
can be rechecked with `sed -n 1890p <node>_decode_w0.out`. Values whose log line
is missing are printed as `NOT FOUND` together with the pattern that was looked
for, and anything derived from them is left out — a missing pattern usually
means the vLLM version changed its wording, and the report should not guess.

The budget line above is printed by vLLM only after CUDA graph capture, so a
startup that dies earlier (an OOM, or the guard that rejects `max-num-seqs`
above the available Mamba blocks) leaves it out entirely. In that case the
report rebuilds the same block from the two lines vLLM prints before capture,
marks the header `(derived)`, and reports non-torch and peak activation as one
remainder, since only the missing line splits them:

```
GPU MEMORY
  total on device                                276.76 GiB    :204 (derived)
    = 10.60 GiB graph estimate / (0.92 - 0.8817) utilization, that line's own arithmetic
```

The device size comes out within about a GiB, because both utilizations are
rounded in the log. The graph reservation is still shown, but the actual pool
size after capture is left as `?`: without it the overshoot cannot be checked.

### config.yaml

The fully resolved configuration showing exactly what ran, with all aliases expanded and defaults applied.

## Common Commands

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Watch logs
tail -f logs/4459_*/*_prefill_*.err logs/4459_*/*_decode_*.err

# Watch benchmark progress
tail -f logs/4459_*/benchmark.out
```

## Connecting to Running Jobs

The `log.out` file includes commands to connect to running nodes
