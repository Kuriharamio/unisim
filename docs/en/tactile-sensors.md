# Tactile sensor plugin contract

[English](tactile-sensors.md) | [中文](../zh/tactile-sensors.md)

UniSim provides a backend-neutral lifecycle for tactile computation. A plugin owns its model, solver, device buffers, and output layout; UniSim owns only the detached scene, mounted frames, pose inputs, lifecycle, and field metadata. Importing `unisim` does not import a tactile plugin, Torch, CuPy, or a CUDA runtime.

## Scene and geometry

`TactileSceneDescription` reuses the public `RayCollisionDescription` for collision geometry and `RaySiteFrameDescription` for each sensor's local mount. `object_body_names` selects bodies; `object_geom_names` optionally narrows perception to an exact set of collision geoms and defaults to all geoms on those bodies. `geometry_state` can carry the aligned per-environment snapshot from `SimBackend.get_ray_geometry_state()`. Use `SimBackend.get_ray_collision_description(body_names=...)`, `SimBackend.get_ray_geometry_state()`, and `SimBackend.get_ray_site_frames(site_names)` on the cold path to build it. Plugins can reject geometry randomization they cannot represent. A geometry change requires a fresh scene and `rebuild()`; a plugin may reject per-environment topology or geometry differences. Names and arrays are detached from the engine. No solver-specific value such as an SDF, FEM matrix, marker lattice, or image calibration belongs in this description.

Motrix projects named site frames from source MJCF scenes without generated terrain, appended fragments, or fixed variants. Its ray collision projection still rejects mesh and height-field geoms. Motrix uses convex-hull mesh collision, so source mesh triangles do not establish the exact runtime contact surface ([MJCF mesh reference](https://motrixsim.readthedocs.io/en/latest/user_guide/getting_started/mjcf_reference.html)).

## Lifecycle

```python
from unisim import TactileSceneDescription
from unisim.factory import create_tactile_sensor

sensor = create_tactile_sensor(
    "unitac",
    model="hydroshear",
    num_envs=64,
    config=model_config,
)
sensor.materialize(scene)
sensor.update_pose(body_pos, body_quat)
sensor.advance()
observation = sensor.read()
sensor.reset([0, 3])
sensor.close()
```

`"unitac"` identifies the plugin package. `model="hydroshear"` selects an algorithm inside that package. UniSim passes `config` through without parsing model-specific keys.

| Operation | Semantics |
| --- | --- |
| `materialize(scene)` | Binds collision geometry and sensor mounts once. Cached body poses start at zero translation with identity `wxyz` rotation; callers must synchronize actual poses with `update_pose()` before the first `advance()`. |
| `update_pose(pos, quat, env_ids=None)` | Replaces cached world poses for the selected rows; accepts `(rows, bodies, 3)` positions and `(rows, bodies, 4)` unit `wxyz` quaternions. It does not advance history. |
| `advance()` | Computes one observation from the latest cached poses and advances temporal state exactly once. Repeated calls advance repeatedly, even if poses did not change. |
| `read()` | Returns the latest observation without changing state. Reading after `materialize()` but before `advance()` or `reset()` is an implementation error; `reset()` makes its cleared values readable before the first `advance()`. |
| `reset(env_ids=None)` | Clears temporal state and observation values for all or selected environments, while retaining the cached poses. |
| `rebuild(scene)` | Rebinds static geometry only when `supports_geometry_rebuild` is declared. A successful rebuild clears history, invalidates prior output views, and requires a new pose update before `advance()`. |
| `close()` | Releases resources; it is idempotent. Other operations after close fail. |

The implementation declares accepted NumPy/Torch/CuPy input profiles, output profiles, selected-environment support, geometry rebuild support, and CUDA stream ownership through `TactileSensorCapabilities`. Unsupported profiles fail closed.

## Observations

`sensor.read()` returns a mapping from field name to its array or tensor. `sensor.output_specs` describes every field with its full shape (including environment rows), dtype, units, coordinate frame, dimension names, buffer owner, and reuse lifetime. A dimension may be `None` when its size varies. Plugins choose their own output structures; the contract does not require a regular marker grid or any fixed channel set.

## Plugin boundary

`create_tactile_sensor(plugin_type, model=..., num_envs=..., config=..., **kwargs)` imports only the selected package. The plugin exposes `create_tactile_sensor(model=..., num_envs=..., config=..., **kwargs) -> TactileSensor`. The manifest records package identity only; adding an algorithm within a plugin does not require a UniSim registration change.

The base package remains NumPy-only. Tactile plugins declare and load their own optional compute libraries when the selected model is created or materialized.
