# MJWarp device tensor contract

[English](mjwarp-device-contract.md) | [中文](../zh/mjwarp-device-contract.md)

MJWarp exposes its device-resident lifecycle through `SimBackend` tensor methods. Callers must negotiate capabilities and use the exact declared CUDA device; an unsupported operation or device fails closed. This contract does not route device data through the NumPy host bridge.

## Batched frame reads

Bind a cold read layout with `TensorIOSpec` and `bind_tensor_read_plan()`. A `TensorReadPlan.read()` returns requested state fields, named sensors, tracked bodies, site poses, and site Jacobians together in a `TensorFrameRead`. This gives a caller one declared frame boundary rather than a sequence of per-field Python getters.

Site positions use world coordinates; quaternions use wxyz order. Site Jacobians are world-frame `jacp` and `jacr` tensors shaped `(num_envs, 3, num_dofs)` in requested public qvel-column order. `get_site_kinematics_for_state_tensor()` evaluates candidate selected-row qpos for reset IK without publishing that candidate into the live simulation.

```python
import torch
from unisim import TensorIOSpec

device = torch.device("cuda:0")
plan = backend.bind_tensor_read_plan(
    TensorIOSpec(
        state_fields=("qpos", "qvel", "ctrl"),
        sensor_names=("touch",),
        body_names=("tcp",),
        body_fields=("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"),
        site_names=("tcp_site",),
        jacobian_site_names=("tcp_site",),
        device=device,
    )
)
result = backend.step_tensor(control, nsteps=4)
result["completion"].wait_before_read(torch.cuda.current_stream(device))
frame = plan.read()
```

`step_tensor()` returns an event-backed completion receipt. Consumers reading step outputs on another Torch stream must pass that stream to `wait_before_read()` before accessing the views. MJWarp owns completion of its physics and derived-state work; the caller owns the Torch stream. Backend-owned views are reusable and are valid only until the next step or reset. Copy a view before retaining its contents across that boundary. Candidate-state kinematics views are valid until the next state-conditioned kinematics call.

## Selected reset and force inputs

`set_state_tensor()` accepts device-resident selected rows. `set_state_tensor_mask()` accepts a full-width CUDA bool mask and full-width qpos/qvel tables; false rows remain unchanged. `bind_packed_reset()` composes task state, model randomization, and mocap writes before one public commit. `write_mocap_pose()` stages selected rows, while `write_mocap_pose_mask()` stages full-width poses under a device mask. MJWarp validates the combined payload on device and publishes mocap-derived kinematics as part of the commit. Production mode accumulates finite checks for the configured rollout boundary; strict mode resolves them during commit. The packed-reset composer does not read CUDA validation scalars before that backend boundary.

`set_generalized_force_tensor()` accepts contiguous float32 CUDA tensors: `joint_f` has shape `(num_envs, nv)` in public qvel order, and `body_f` has shape `(num_envs, num_bodies, 6)` in public body order. These inputs persist until replaced; pass zeros to clear them. They are control inputs and survive state resets.

## Capability and runtime diagnostics

`get_tensor_capabilities()` declares `DEVICE_RESIDENT`, `DIRECT` storage, accepted Torch devices, stream/event ownership, and operation-level flags including `frame_read_plan`, `site_pose_views`, `site_jacobian_views`, `site_state_kinematics`, `packed_reset_mocap`, and `generalized_force_injection`. `get_tensor_runtime_diagnostics()` reports whether requested runtime features are enabled and why a requested feature was disabled. `get_tensor_transfer_diagnostics()` reports backend transfer, scalar-verdict synchronization, and reset-commit counts. These counters cover backend operations; callers should also collect counters from task, sensor, and rollout layers.
