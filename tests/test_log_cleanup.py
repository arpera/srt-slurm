# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for making worker logs readable after the workers have exited."""

from __future__ import annotations

from srtctl.core.log_cleanup import clean_lines, clean_worker_logs

PREFIX = "(Worker_DP0_EP0 pid=942741) "

# The checkpoint loader redraws one bar 100 times, one \n-terminated line each.
LOADER = [
    f"{PREFIX}\rLoading safetensors checkpoint shards:   0% Completed | 0/100 [00:00<?, ?it/s]",
    f"{PREFIX}\rLoading safetensors checkpoint shards:  49% Completed | 49/100 [01:52<01:40,  1.97s/it]",
    f"{PREFIX}\rLoading safetensors checkpoint shards: 100% Completed | 100/100 [03:31<00:00,  2.11s/it]",
]

# The autotuner packs many frames into one line, separated by carriage returns,
# and draws the bar with block glyphs.
AUTOTUNER = (
    f"{PREFIX}\r[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:   0%|          | 0/23 [00:00<?, ?profile/s]"
    "\r[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:   4%|\u258d         | 1/23 [00:03<01:10,  3.2s/profile]"
    "\r[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%|\u2588\u2588\u2588\u2588| 23/23 [01:20<00:00,  3.5s/profile]"
)

DYNAMO = (
    "\x1b[2m2026-08-13T08:33:28.674517Z\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mloggers.log\x1b[0m\x1b[2m:\x1b[0m "
    "Engine 000: Avg prompt throughput: 5644.3 tokens/s, Running: 1 reqs"
)


class TestCleaning:
    def test_ansi_escapes_are_removed(self):
        (line,) = clean_lines([DYNAMO])

        assert "\x1b" not in line
        assert line.startswith("2026-08-13T08:33:28.674517Z  INFO loggers.log: Engine 000:")

    def test_a_redrawn_bar_becomes_its_final_line(self):
        cleaned = list(clean_lines(LOADER))

        assert cleaned == [
            f"{PREFIX}Loading safetensors checkpoint shards: 100% Completed | 100/100 [03:31<00:00,  2.11s/it]"
        ]

    def test_frames_packed_into_one_line_collapse_too(self):
        (line,) = clean_lines([AUTOTUNER])

        assert line.startswith(f"{PREFIX}[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%")
        assert "23/23 [01:20<00:00,  3.5s/profile]" in line
        assert "\u2588" not in line and "\r" not in line

    def test_the_bar_keeps_its_position_in_the_log(self):
        """The surviving frame stays where the bar finished, not at the end."""
        cleaned = list(clean_lines([*LOADER, "after the load"]))

        assert cleaned[-1] == "after the load"
        assert "100/100" in cleaned[0]

    def test_bars_from_different_ranks_are_kept_apart(self):
        other = "(Worker_DP1_EP1 pid=7) \rLoading safetensors checkpoint shards: 100% Completed | 100/100 [03:30<00:00]"
        cleaned = list(clean_lines([LOADER[0], other, LOADER[2]]))

        assert len(cleaned) == 2
        assert any(line.startswith("(Worker_DP1_EP1 pid=7)") for line in cleaned)

    def test_a_second_run_of_the_same_bar_keeps_its_own_line(self):
        """A percentage going backwards means a new bar, not a redraw."""
        cleaned = list(clean_lines([LOADER[2], *LOADER]))

        assert len(cleaned) == 2

    def test_a_split_frame_leaves_no_debris(self):
        """A frame cut by a line break leaves a tail with no label; drop it."""
        cleaned = list(clean_lines([LOADER[2], "\ufffd\u2588| 23/23 [01:20<00:00,  3.50s/profile]"]))

        assert len(cleaned) == 1

    def test_ordinary_lines_are_untouched(self):
        lines = [
            f"{PREFIX}INFO 08-13 11:29:38 [gpu_worker.py:564] Available KV cache memory: 46.5 GiB",
            "GPU KV cache size: 394,633 tokens, Maximum concurrency for 10,240 tokens per request: 38.54x",
            "  Bandwidth                     : Full",
        ]

        assert list(clean_lines(lines)) == lines

    def test_a_percentage_in_prose_is_not_a_progress_bar(self):
        line = f"{PREFIX}INFO GPU KV cache usage: 2.3%, Prefix cache hit rate: 0.0%"

        assert list(clean_lines([line])) == [line]


class TestCleaningFiles:
    def _log(self, tmp_path, name):
        path = tmp_path / name
        path.write_text("\n".join([DYNAMO, *LOADER, "done"]) + "\n")
        return path

    def test_worker_logs_are_rewritten_in_place(self, tmp_path):
        path = self._log(tmp_path, "node01_decode_w0.out")

        assert clean_worker_logs(tmp_path) == [path]

        text = path.read_text()
        assert "\x1b" not in text and "\r" not in text
        assert text.count("Loading safetensors") == 1
        assert text.endswith("done\n")

    def test_other_logs_are_left_alone(self, tmp_path):
        frontend = self._log(tmp_path, "node01_frontend_0.out")
        before = frontend.read_text()

        clean_worker_logs(tmp_path)

        assert frontend.read_text() == before

    def test_a_clean_log_is_not_rewritten(self, tmp_path):
        path = tmp_path / "node01_decode_w0.out"
        path.write_text("nothing to clean here\n")

        assert clean_worker_logs(tmp_path) == []

    def test_an_unreadable_log_does_not_stop_the_others(self, tmp_path, monkeypatch):
        good = self._log(tmp_path, "node02_decode_w0.out")
        bad = self._log(tmp_path, "node01_decode_w0.out")
        original = type(bad).read_bytes

        def explode(self):
            if self.name == bad.name:
                raise OSError("gone")
            return original(self)

        monkeypatch.setattr(type(bad), "read_bytes", explode)

        assert clean_worker_logs(tmp_path) == [good]
