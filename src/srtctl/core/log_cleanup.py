# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn raw worker logs into something a person can read.

Redirected to a file, a worker log keeps everything a terminal would have
consumed and thrown away: colour escapes from the dynamo logger, and every frame
of every progress bar. One checkpoint load leaves 102 near-identical lines, and
the FlashInfer autotuner packs dozens of frames into a single line separated by
carriage returns, block-drawing glyphs included.

Cleaning keeps what the terminal would have shown: escapes removed, each bar
reduced to its final frame. Nothing else is touched, so the surviving lines are
byte-for-byte what the worker wrote.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Colour and cursor control from the dynamo logger: ESC [ ... <letter>.
_ANSI = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")

# srun labels every line with the task that wrote it; kept so a collapsed bar
# still says which rank it belongs to.
_PREFIX = re.compile(r"^(\([^)]*pid=\d+\)\s)")

# A progress frame is anything that reports a percentage the way tqdm and the
# checkpoint loader do. The text before the percentage identifies the bar.
_PROGRESS = re.compile(r"^(?P<label>.*?)\s*(?P<percent>\d+)%(?P<rest>[|\s].*)$")

# tqdm draws the bar itself between two pipes: "  74%|███████▍  | 17/23 [...]".
_BAR = re.compile(r"\|[^|]*\|")

# The tail of a frame whose write was split across a line break: bar glyphs and
# the counter, with no label or percentage left to identify it.
_BAR_TAIL = re.compile(r"^[^\w]*\|\s*\d+/\d+\s*\[[^\]]*\]\s*$")


def clean_lines(lines: Iterable[str]) -> Iterator[str]:
    """Strip escapes and reduce each progress bar to its final frame.

    The surviving frame stays where the bar finished, so the log keeps its
    order. Bars from several ranks interleave in one file, so frames are tracked
    per (rank, bar); a percentage that goes backwards starts a new bar, which is
    how a second run of the same operation keeps its own line.
    """
    prepared: list[tuple[str, str, tuple[str, str] | None, int] | None] = []
    last_prefix = ""
    for line in lines:
        entry = _prepare(line, last_prefix)
        if entry is not None and entry[2] is not None:
            last_prefix = entry[0]
        prepared.append(entry)

    last_frame: dict[tuple[str, str, int], int] = {}
    runs: dict[tuple[str, str], tuple[int, int]] = {}
    for index, entry in enumerate(prepared):
        if entry is None:
            continue
        _, _, key, percent = entry
        if key is None:
            continue
        run, previous = runs.get(key, (0, -1))
        if percent < previous:
            run += 1
        runs[key] = (run, percent)
        last_frame[(*key, run)] = index

    keep = set(last_frame.values())
    for index, entry in enumerate(prepared):
        if entry is None:
            continue
        prefix, body, key, _ = entry
        if key is None or index in keep:
            yield prefix + body


def _prepare(line: str, last_prefix: str) -> tuple[str, str, tuple[str, str] | None, int] | None:
    """Clean one line, or return None when it is only a leftover of a bar.

    A progress frame also reports which bar it belongs to, so the caller can
    keep just the last one.
    """
    prefix, body = _split_prefix(line)
    body = _ANSI.sub("", _last_frame(body))

    if _BAR_TAIL.match(body):
        return None

    progress = _PROGRESS.match(body)
    if not progress or not _looks_like_bar(progress):
        return prefix, body, None, 0

    # A frame that starts mid-line carries no srun prefix of its own: the write
    # was split across a line break, so it continues the previous rank's bar.
    prefix = prefix or last_prefix
    return prefix, _strip_bar(body), (prefix, progress.group("label")), int(progress.group("percent"))


def _split_prefix(line: str) -> tuple[str, str]:
    match = _PREFIX.match(line)
    return (match.group(1), line[match.end() :]) if match else ("", line)


def _last_frame(body: str) -> str:
    """Keep what the terminal would still be showing after the redraws."""
    if "\r" not in body:
        return body
    frames = [frame for frame in body.split("\r") if frame.strip()]
    return frames[-1] if frames else ""


def _looks_like_bar(progress: re.Match[str]) -> bool:
    """Guard against eating ordinary lines that merely contain a percentage."""
    rest = progress.group("rest")
    return "|" in rest or "Completed" in rest


def _strip_bar(body: str) -> str:
    """Drop the drawn bar, which is block glyphs that read as mojibake in a file."""
    return _BAR.sub("| ", body, count=1)


def clean_worker_logs(
    log_dir: Path, patterns: Iterable[str] = ("*_prefill_w*.out", "*_decode_w*.out", "*_agg_w*.out")
) -> list[Path]:
    """Rewrite worker logs in place. Only safe once the workers have exited."""
    cleaned: list[Path] = []
    for pattern in patterns:
        for path in sorted(log_dir.glob(pattern)):
            try:
                if _clean_file(path):
                    cleaned.append(path)
            except Exception as error:  # noqa: BLE001 - a tidy log is never worth a failed job
                logger.warning("Could not clean %s: %s", path.name, error)
    if cleaned:
        logger.info("Cleaned %d worker log(s): escapes stripped, progress bars collapsed", len(cleaned))
    return cleaned


def _clean_file(path: Path) -> bool:
    text = path.read_bytes().decode("utf-8", errors="replace")
    # Split on \n only: carriage returns are progress frames, not line breaks.
    lines = text.split("\n")
    trailing_newline = lines and lines[-1] == ""
    if trailing_newline:
        lines.pop()

    cleaned = list(clean_lines(lines))
    if cleaned == lines:
        return False

    # Publish atomically so a reader never sees a half-written log.
    tmp = path.with_name(f"{path.name}.clean.{os.getpid()}")
    tmp.write_text("\n".join(cleaned) + ("\n" if trailing_newline else ""))
    os.replace(tmp, path)
    return True
