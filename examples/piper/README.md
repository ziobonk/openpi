# Piper 机械臂 openpi 集成

本目录提供将 openpi VLA 模型部署到 Piper 机械臂上的脚本。

## 文件说明

| 文件 | 用途 |
|------|------|
| `collect_demos.py` | 数据采集脚本 — 录制 Piper 关节/夹爪/图像/指令为 LeRobot 格式 |
| `inference.py` | 推理脚本 — 连接 openpi 策略服务器控制机械臂 |
| `../../src/openpi/policies/piper_policy.py` | 数据映射层 — Piper 格式 ↔ openpi Observation 格式 |

## 快速开始

### 1. 安装依赖

```bash
# openpi (在项目根目录)
uv sync
uv pip install -e .

# openpi-client (轻量客户端，可在机器人端单独安装)
cd packages/openpi-client && pip install -e .

# Piper SDK
cd piper_sdk && pip install .

# 相机支持 (可选)
pip install opencv-python
```

### 2. 激活 CAN 总线

```bash
# 在 piper_sdk 目录下
bash can_activate.sh can0 1000000
# 验证: ifconfig | grep can0
```

### 3. 确保机械臂工作在从臂模式

运行一次以下代码（或使用 SDK demo）:
```python
from piper_sdk import C_PiperInterface_V2
piper = C_PiperInterface_V2("can0")
piper.ConnectPort()
piper.MasterSlaveConfig(0xFC, 0, 0, 0)  # 0xFC = 运动输出臂 (从臂)
```

---

## 数据采集

```bash
# 无相机采集
python examples/piper/collect_demos.py --repo_id your_hf_username/piper_data

# 带相机 (cam_ids: 第一个=基座相机, 第二个=腕部相机)
python examples/piper/collect_demos.py \
    --repo_id your_hf_username/piper_data \
    --cam_ids 0 2

# 推送到 HuggingFace Hub
python examples/piper/collect_demos.py \
    --repo_id your_hf_username/piper_data \
    --push_to_hub
```

操作流程:
1. 按 **Enter** — 开始新 episode
2. 输入任务指令（如 "pick up the red block"）
3. 手动操作机械臂完成一次完整的任务演示
4. 按 **s** — 停止并保存当前 episode
5. 重复 1-4 采集多条数据
6. 按 **q** — 退出

数据将保存为 LeRobot 格式，位于 `~/.cache/huggingface/lerobot/<repo_id>/`。

---

## 推理 (策略服务器模式)

### 步骤 1: 启动策略服务器 (GPU 机器)

```bash
# 使用预训练 LIBERO checkpoint （零样本测试）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

# 使用自己微调的 Piper checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_piper \
    --policy.dir=checkpoints/pi05_piper/my_experiment/20000
```

服务器默认监听 `0.0.0.0:8000`。

### 步骤 2: 运行推理 (机器人端)

```bash
# 基础推理
python examples/piper/inference.py --host <GPU_SERVER_IP> --port 8000

# 带相机
python examples/piper/inference.py --host 192.168.1.100 --port 8000 --cam_ids 0

# 交互模式 (每次推理前换指令)
python examples/piper/inference.py --host localhost --interactive

# 调整动作参数
python examples/piper/inference.py \
    --host 192.168.1.100 \
    --action_horizon 10 \
    --exec_horizon 3 \
    --prompt "pick the fork"
```

操作流程:
1. 按 **Enter** — 开始推理循环
2. 机械臂开始执行模型输出的动作
3. 按 **s** — 暂停
4. 按 **r** — 回到初始位姿
5. 按 **q** — 退出

### 推理参数说明

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--action_horizon` | 模型一次输出多少步动作 | 10 (与模型 config 一致) |
| `--exec_horizon` | 执行多少步后重新推理 | 3~5 (值越小越"谨慎") |
| `--prompt` | 默认语言指令 | 根据任务设定 |

`exec_horizon` 越小，模型重新推理的频率越高，动作执行越连贯但延迟越大。
建议从 3 开始，根据实际效果调整。

---

## 微调自己的 Piper 模型

1. 用 `collect_demos.py` 采集足够的演示数据 (建议 ≥ 50 个 episode)

2. 在 `src/openpi/training/config.py` 的 `_CONFIGS` 中添加 Piper 训练配置:

```python
TrainConfig(
    name="pi05_piper",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,          # 6 joints + 1 gripper
        action_horizon=10,
    ),
    data=LeRobotPiperDataConfig(
        repo_id="your_hf_username/piper_data",
        assets=AssetsConfig(asset_id="piper"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[piper_policy.PiperInputs(model_type=model.model_type)],
            outputs=[piper_policy.PiperOutputs(piper_action_dim=7)],
        ),
    ),
    ...
)
```

3. 计算归一化统计量并训练:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piper
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper \
    --exp-name=piper_finetune --overwrite
```

---

## 安全注意事项

- **首次运行前**：确认 `JOINT_LIMITS_RAD` 中的关节限位与你的 Piper 型号匹配
- **速度控制**：初次测试请使用较低速度 `DEFAULT_SPEED_PCT=20~30`
- **急停**：推理过程中按 `s` 可立即暂停控制
- **测试顺序**：建议先在无相机、无策略服务器的 dry-run 模式下验证机械臂通信正常
