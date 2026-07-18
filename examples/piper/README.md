# Piper 机械臂 openpi 集成

将 openpi VLA 模型部署到 Piper 机械臂的完整方案：数据采集 → 训练 → 推理。

## 文件说明

| 文件 | 用途 |
|------|------|
| `collect_demos.py` | 数据采集 — 录制 Piper 关节/夹爪/图像/指令为 LeRobot 格式 |
| `inference.py` | 推理 — 连接 openpi 策略服务器，执行 receding horizon 控制 |
| `../../src/openpi/policies/piper_policy.py` | 数据映射 — Piper ↔ openpi Observation/Actions 格式转换 |
| `../../src/openpi/training/config.py` | 训练配置 — `pi05_piper` / `pi0_piper` 两个预设 |

---

## 环境准备

```bash
# openpi 依赖
uv sync && uv pip install -e .

# openpi-client (机器人端可单独安装)
cd packages/openpi-client && pip install -e .

# Piper SDK
cd piper_sdk && pip install .

# 相机 (可选)
pip install opencv-python

# 激活 CAN 总线 (在 piper_sdk 目录下)
bash can_activate.sh can0 1000000
```

机械臂切从臂模式（一次性）:
```python
from piper_sdk import C_PiperInterface_V2
piper = C_PiperInterface_V2("can0")
piper.ConnectPort()
piper.MasterSlaveConfig(0xFC, 0, 0, 0)
```

---

## 相机配置 (RealSense D435i + D405)

### 安装依赖

```bash
pip install pyrealsense2
```

### 查找序列号

```bash
cd examples/piper
python camera_utils.py --list
```

输出示例：
```
发现 2 个 RealSense 设备:
  serial=128422272318  name=Intel RealSense D435I  usb=3.2
  serial=218722271368  name=Intel RealSense D405   usb=3.2
```

D435i 做基座/外部相机，D405 做腕部相机。记下对应序列号。

### 在采集/推理脚本中使用

```bash
# 数据采集
python examples/piper/collect_demos.py \
    --repo_id your_hf_username/piper_data \
    --rs2_base 128422272318 --rs2_wrist 218722271368

# 推理
python examples/piper/inference.py \
    --host localhost \
    --rs2_base 128422272318 --rs2_wrist 218722271368
```

RealSense 在代码中自动配置为 **640×480@30fps** 采集，**等比缩放+居中填充**到 **224×224** 后输入模型，与训练时的 `ResizeImages` transform 行为一致。

### 只用单相机

```bash
# 只有 D435i (基座)，没有腕部相机
python examples/piper/inference.py --host localhost --rs2_base 128422272318

# 只有 D405 (腕部)，没有基座相机
python examples/piper/inference.py --host localhost --rs2_wrist 218722271368
```

缺失的相机会自动填零（`piper_policy.PiperInputs` 中 `right_wrist_0_rgb` 已设为 `np.zeros` 并 `image_mask=False`）。

### OpenCV webcam 回退

如果没装 `pyrealsense2` 或想用普通 USB 相机：
```bash
python examples/piper/inference.py --host localhost --cam_ids 0 2
```

---

## 训练流程总览

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│  采集演示数据  │ →  │  LeRobot 数据集 │ →  │  compute_norm    │
│  collect_demos │    │  ~/.cache/    │    │  _stats.py       │
│  .py           │    │  huggingface/ │    │  → norm_stats    │
└──────────────┘    │  lerobot/      │    │  .json           │
                    └──────────────┘    └──────┬──────────┘
                                              │
         ┌────────────────────────────────────┘
         ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  train.py    │ →  │  serve_policy.py  │ →  │  inference.py    │
│  (JAX/PyTorch)│   │  (GPU 服务器)      │    │  (机器人端)       │
│              │    │  :8000            │    │  WebsocketClient │
└──────────────┘    └──────────────────┘    └──────────────────┘
```

### 数据格式链路

```
collect_demos.py              LeRobot Dataset         config.py                 Policy Pipeline
───────────────────────────────────────────────────────────────────────────────────────────────
state    (7,) float32    →    state              →    Repack("observation/   →  Normalize
  j1..j6 (rad)                                            state" = "state")      TokenizePrompt
  gripper (raw 0.001mm)                                                           PiperInputs
actions  (7,) float32    →    actions            →    DeltaActions(joints)   →  BaseModel
  (同 state)                                             → model Observation      .compute_loss()
image    (224,224,3)     →    image              →    Repack("observation/
  uint8                                                     image" = "image")
wrist_image (224,224,3)  →    wrist_image        →    Repack("observation/
  uint8                                                     wrist_image" = ...)
task     str             →    task               →    prompt_from_task=True  →  TokenizePrompt
  ("pick the block")                                      → "prompt"             → prompt tokens
```

---

## 第一步：采集数据

```bash
python examples/piper/collect_demos.py \
    --repo_id your_hf_username/piper_data \
    --cam_ids 0 2
```

**操作流程：**
1. 按 **Enter** → 输入任务指令（如 "pick up the red block"）
2. 手动拖臂完成一次完整的任务演示
3. 按 **s** → 停止并保存当前 episode
4. 重复 1-3 采集多条数据
5. 按 **q** → 退出

**建议：** ≥ 50 个 episode，每个 ≥ 100 帧。数据越多、多样性越高，模型效果越好。

数据保存到 `~/.cache/huggingface/lerobot/<repo_id>/`。

---

## 第二步：修改配置

`src/openpi/training/config.py` 中已预设 `pi05_piper` 和 `pi0_piper`，只需把 `repo_id` 改成你的：

```python
TrainConfig(
    name="pi05_piper",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,           # 6 关节 + 1 夹爪
        action_horizon=10,      # 预测 10 步
        discrete_state_input=False,  # 连续值状态
    ),
    data=LeRobotPiperDataConfig(
        repo_id="your_hf_username/piper_data",  # ← 改成你的
        assets=AssetsConfig(asset_id="piper"),
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,   # 关节做 delta，夹爪保持 absolute
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"  # 从预训练基座开始
    ),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=2e-5, decay_steps=50_000, decay_lr=2e-6,
    ),
    ema_decay=0.999,
    num_train_steps=30_000,
    save_interval=2000,
),
```

**关键参数说明：**

| 参数 | 含义 | 值 |
|------|------|-----|
| `action_dim` | 模型输出维度 | 7 (6关节+1夹爪) |
| `action_horizon` | 预测的动作序列长度 | 10 (可减到 5 提推理速度) |
| `use_delta_joint_actions=True` | 关节角做 delta (夹爪保持 absolute) | **Piper 采集的是绝对位姿，必须开** |
| `discrete_state_input=False` | 状态是连续值非离散 token | Piper 关节角是连续值 |
| `weight_loader` | 从 π₀.₅ 基座加载预训练权重 | 大幅提升小数据集效果 |
| `peak_lr=2e-5` | 峰值学习率 | 小数据集可提高到 5e-5 |
| `batch_size` | 批大小 | 按 GPU 显存调整 (22GB: 32-64) |

---

## 第三步：计算归一化统计量

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piper
```

从 LeRobot 数据集读取数据，计算 state/action 的 `mean/std/q01/q99`，保存到 `assets/pi05_piper/piper/norm_stats.json`。训练时会自动从这里加载。

---

## 第四步：训练

```bash
# JAX 训练 (推荐，官方路径)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper \
    --exp-name=piper_exp1 --overwrite

# PyTorch 单卡
uv run scripts/train_pytorch.py pi05_piper \
    --exp_name piper_exp1 --save_interval 2000

# PyTorch 多卡 DDP
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_piper --exp_name piper_exp1
```

检查点自动保存到 `checkpoints/pi05_piper/<exp_name>/<step>/`。

**恢复训练：**

```bash
# JAX
uv run scripts/train.py pi05_piper --exp-name=piper_exp1 --resume

# PyTorch
uv run scripts/train_pytorch.py pi05_piper --exp_name piper_exp1 --resume
```

---

## 第五步：推理

### 启动策略服务器 (GPU 机器)

```bash
# 用自己的微调模型
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_piper \
    --policy.dir=checkpoints/pi05_piper/piper_exp1/20000

# 或用 LIBERO 预训练模型零样本测试 (效果不保证)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

服务器监听 `0.0.0.0:8000`。

### 运行推理 (机器人端)

```bash
# 基础用法
python examples/piper/inference.py --host <GPU_SERVER_IP> --port 8000

# 带相机
python examples/piper/inference.py --host 192.168.1.100 --cam_ids 0

# 交互模式 (手动输入指令)
python examples/piper/inference.py --host localhost --interactive
```

**控制：** Enter 开始推理 | s 暂停 | r 回初始位姿 | q 退出

**执行模式 — receding horizon control：** 模型一次输出 `action_horizon` 步动作块，只执行前 `exec_horizon` 步就重新推理，保证动作连贯性和对环境的响应。

| 参数 | 含义 | 建议 |
|------|------|------|
| `--action_horizon` | 模型一次输出步数 | 与训练 config 一致 (10) |
| `--exec_horizon` | 执行几步后重推理 | 3~5 (越小越"谨慎") |
| `--prompt` | 语言指令 | 按任务设定 |

---

## PyTorch 训练补充说明

如果用 PyTorch，需要先将 transformers 补丁覆盖到虚拟环境：

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
```

PyTorch 训练不支持：π₀-FAST、混合精度、FSDP、LoRA、EMA。

---

## 安全注意事项

- **首次运行前**：确认 `JOINT_LIMITS_RAD` 限位值与你的 Piper 型号匹配
- **速度**：初次测试用 `DEFAULT_SPEED_PCT=20~30`，确认正常后再调高
- **急停**：推理中按 `s` 立即暂停
- **测试顺序**：先 dry-run 验证机械臂通信 → 再带策略服务器 → 最后接相机
