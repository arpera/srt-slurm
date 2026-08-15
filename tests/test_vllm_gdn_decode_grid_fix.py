# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GDN packed-decode Triton grid hotfix."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
PATCHER = REPO_ROOT / "configs/patches/vllm_gdn_decode_grid_fix.py"
WRAPPER = REPO_ROOT / "configs/patches/vllm-gdn-decode-grid-fix.sh"

# The chunked prefill kernel already uses three axes and must survive untouched.
CHUNKED_KERNEL = """\
def fused_recurrent_gated_delta_rule_fwd_kernel(A_log, dt_bias):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
"""

UNPATCHED_SOURCE = (
    CHUNKED_KERNEL
    + """\


def fused_recurrent_gated_delta_rule_packed_decode_kernel(A_log, dt_bias):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)


def fused_recurrent_gated_delta_rule_fwd(q, k, v):
    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](q)


def fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv):
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](mixed_qkv)
"""
)


def run_patcher(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCHER), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_patches_grid_and_program_ids_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "fused_recurrent.py"
    target.write_text(UNPATCHED_SOURCE)

    first = run_patcher(target)

    assert first.returncode == 0
    assert "Moved the decode batch onto grid axis 2" in first.stderr
    patched = target.read_text()
    assert "    grid = (NV, HV, B)" in patched
    assert "B * HV" not in patched
    assert "    i_v, i_hv, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n" in patched
    assert "    i_h = i_hv // (HV // H)" in patched
    assert CHUNKED_KERNEL in patched

    second = run_patcher(target)

    assert second.returncode == 0
    assert "Already patched, skipping" in second.stderr
    assert target.read_text() == patched


def test_leaves_no_staging_files_behind(tmp_path: Path) -> None:
    target = tmp_path / "fused_recurrent.py"
    target.write_text(UNPATCHED_SOURCE)

    run_patcher(target)

    assert [path.name for path in tmp_path.iterdir()] == ["fused_recurrent.py"]


def test_rejects_drifted_source_without_modifying_it(tmp_path: Path) -> None:
    target = tmp_path / "fused_recurrent.py"
    drifted = UNPATCHED_SOURCE.replace("grid = (NV, B * HV)", "grid = (NV, B * HV, 1)")
    target.write_text(drifted)

    result = run_patcher(target)

    assert result.returncode == 1
    assert "flash-linear-attention source may have drifted" in result.stderr
    assert target.read_text() == drifted


def test_rejects_ambiguous_anchors_without_modifying_source(tmp_path: Path) -> None:
    target = tmp_path / "fused_recurrent.py"
    ambiguous = UNPATCHED_SOURCE + UNPATCHED_SOURCE
    target.write_text(ambiguous)

    result = run_patcher(target)

    assert result.returncode == 1
    assert "found old=(2, 2)" in result.stderr
    assert target.read_text() == ambiguous


def test_wrapper_runs_the_patcher() -> None:
    assert "python3 /configs/patches/vllm_gdn_decode_grid_fix.py" in WRAPPER.read_text()
