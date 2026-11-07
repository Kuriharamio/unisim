from __future__ import annotations

import pytest

from unisim.backend.mjwarp.dependencies import _supports_mujoco_warp_version


@pytest.mark.parametrize("version", ("3.11.0", "3.11.4", "3.17.0", "3.17.2"))
def test_supported_mujoco_warp_runtime_lines(version: str) -> None:
    assert _supports_mujoco_warp_version(version)


@pytest.mark.parametrize("version", ("3.10.0", "3.12.0", "3.16.0", "4.0.0"))
def test_unsupported_mujoco_warp_runtime_lines(version: str) -> None:
    assert not _supports_mujoco_warp_version(version)
