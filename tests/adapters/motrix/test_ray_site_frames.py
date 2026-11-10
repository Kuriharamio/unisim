from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.scene import SceneCfg


def _source_backend(path: Path) -> MotrixBackend:
    backend = MotrixBackend.__new__(MotrixBackend)
    backend._closed = False
    backend._portable_faulted = False
    backend._portable_uniform_mesh_variants = False
    backend._portable_variant_assignment = None
    backend._scene = SceneCfg(model_file=str(path))
    backend._scene_model_file = str(path)
    return backend


def test_site_frames_are_projected_as_named_body_local_wxyz(tmp_path: Path) -> None:
    model_path = tmp_path / "site.xml"
    model_path.write_text(
        """<mujoco model='site-frame'>
          <worldbody>
            <body name='pad' pos='0 0 1'>
              <site name='gel' pos='.1 .2 .3' quat='.7071067811865476 0 .7071067811865476 0'/>
            </body>
          </worldbody>
        </mujoco>""",
        encoding="utf-8",
    )
    backend = _source_backend(model_path)

    (frame,) = backend.get_ray_site_frames(("gel",))

    assert frame.site_name == "gel"
    assert frame.body_name == "pad"
    np.testing.assert_allclose(frame.pos, (0.1, 0.2, 0.3))
    np.testing.assert_allclose(frame.quat, (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0))
    with pytest.raises(KeyError, match="missing"):
        backend.get_ray_site_frames(("missing",))
