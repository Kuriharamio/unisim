# MJWarp 设备 tensor 契约

[English](../en/mjwarp-device-contract.md) | [中文](mjwarp-device-contract.md)

MJWarp 通过 `SimBackend` tensor 方法公开设备驻留生命周期。调用方必须先读取 capability，并使用声明的准确 CUDA device；不支持的操作或 device 会快速失败。本契约不会把设备数据送入 NumPy host bridge。

## 批量读取一帧

使用 `TensorIOSpec` 和 `bind_tensor_read_plan()` 绑定冷路径读取布局。`TensorReadPlan.read()` 会在一个 `TensorFrameRead` 中返回请求的 state 字段、具名 sensor、tracked body、site pose 与 site Jacobian。调用方因此使用一个已声明的 frame 边界，而不是逐字段调用 Python getter。

Site position 使用世界坐标，四元数顺序为 wxyz。Site Jacobian 的世界系 `jacp` 与 `jacr` 形状为 `(num_envs, 3, num_dofs)`，最后一维按请求的公开 qvel 列顺序排列。`get_site_kinematics_for_state_tensor()` 可针对 reset IK 计算候选 selected-row qpos，不会把候选状态发布到活动仿真中。

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

`step_tensor()` 返回基于 event 的完成凭据。若消费者在另一个 Torch stream 上读取 step 输出，必须先把该 stream 传给 `wait_before_read()`。MJWarp 负责完成物理与派生状态计算；Torch stream 由调用方所有。Backend-owned view 会复用，仅在下一次 step 或 reset 前有效。若要跨过该生命周期保留内容，调用方应先 copy。候选状态运动学 view 在下一次 state-conditioned kinematics 调用前有效。

## 选中行 reset 与 force 输入

`set_state_tensor()` 接受设备驻留的 selected rows。`set_state_tensor_mask()` 接受全宽 CUDA bool mask 和全宽 qpos/qvel 表；false 行保持不变。`bind_packed_reset()` 可在一个公开 commit 前组合 task state、模型随机化与 mocap 写入。`write_mocap_pose()` stage selected rows，`write_mocap_pose_mask()` 则按设备 mask stage 全宽 pose。MJWarp 在设备上校验合并后的 payload，并在 commit 中发布 mocap 派生运动学。生产模式会将 finite 检查累计到配置的 rollout 边界；strict 模式在 commit 时立即解析。packed-reset composer 不会在 backend 校验边界前读取 CUDA 标量。

`set_generalized_force_tensor()` 接受 contiguous float32 CUDA tensor：`joint_f` 形状为 `(num_envs, nv)`，列顺序为公开 qvel 顺序；`body_f` 形状为 `(num_envs, num_bodies, 6)`，顺序为公开 body 顺序。输入会一直保留到被替换；传入零值即可清除。它们属于控制输入，会在 state reset 后保留。

## Capability 与运行时诊断

`get_tensor_capabilities()` 声明 `DEVICE_RESIDENT`、`DIRECT` 存储、支持的 Torch device、stream/event 所有权，以及 `frame_read_plan`、`site_pose_views`、`site_jacobian_views`、`site_state_kinematics`、`packed_reset_mocap` 和 `generalized_force_injection` 等逐操作标志。`get_tensor_runtime_diagnostics()` 报告请求的运行时特性是否启用，以及未启用的原因。`get_tensor_transfer_diagnostics()` 报告 backend 传输、scalar-verdict 同步与 reset commit 次数。这些计数覆盖 backend 操作；调用方还应收集 task、sensor 与 rollout 层的计数。
