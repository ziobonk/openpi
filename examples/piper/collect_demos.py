#!/usr/bin/env python3
"""
Piper 机械臂数据采集脚本。

通过键盘控制录制会话（episode），将 Piper 关节角、夹爪、相机图像和语言指令
保存为 LeRobot 格式，用于后续 fine-tune openpi 模型。

用法:
    # 保存到本地目录 (推荐)
    python examples/piper/collect_demos.py --data_dir ./piper_data

    # HuggingFace 模式 (需要联网)
    python examples/piper/collect_demos.py --repo_id your_hf_username/piper_task

    # RealSense D435i/D405 相机
    python examples/piper/collect_demos.py --data_dir ./piper_data \
        --rs2_base 231122071797 --rs2_wrist 260322279175

    # OpenCV webcam 回退
    python examples/piper/collect_demos.py --data_dir ./piper_data \
        --cam_ids 0 2

    # 覆盖已有数据集
    python examples/piper/collect_demos.py --data_dir ./piper_data --overwrite

    # 分批上传，每 5 个 episode 上传一次
    python examples/piper/collect_demos.py --repo_id your_hf_username/piper_task \
        --stream_hub --stream_batch 5

    # 采集完成后一次性推送到 HuggingFace Hub
    python examples/piper/collect_demos.py --repo_id your_hf_username/piper_task \
        --push_to_hub

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电并处于从臂模式
    3. piper_sdk 已安装:  pip install piper_sdk
    4. openpi 依赖已安装: uv sync

键盘控制:
    Enter   — 开始新的 episode（会提示输入指令语）
    s       — 停止当前 episode 并保存
    d       — 丢弃当前 episode（不保存，确认后清除）
    q       — 退出程序 (机械臂保持使能)
    ESC     — 同 q

相机预览窗口自动显示 base + wrist 画面并排，关闭:
    --no_preview  禁用预览
"""

import dataclasses
import datetime
import json
import os
from pathlib import Path
import select
import shutil
import sys
import threading
import time
from typing import Optional

import numpy as np

# --- 相机工具 (支持 RealSense D435i/D405 和 OpenCV) ---
from camera_utils import RealSenseCameras, OpenCVCameras, create_cameras

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Piper SDK 导入
# ---------------------------------------------------------------------------
# 注意: 需要先将 piper_sdk 安装到当前 Python 环境:
#   cd piper_sdk && pip install .
# 或将 piper_sdk 目录加入 sys.path:
PIPER_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "piper_sdk")
PIPER_EXAMPLES_PATH = os.path.dirname(__file__)
if PIPER_SDK_PATH not in sys.path:
    sys.path.insert(0, PIPER_SDK_PATH)
if PIPER_EXAMPLES_PATH not in sys.path:
    sys.path.insert(0, PIPER_EXAMPLES_PATH)

try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore[import-untyped]
except ImportError:
    print("[ERROR] 无法导入 piper_sdk。请先安装:")
    print("  cd piper_sdk && pip install .")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LeRobot 导入
# ---------------------------------------------------------------------------
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError:
    print("[ERROR] 无法导入 lerobot。请确认 openpi 依赖已安装:")
    print("  uv sync")
    sys.exit(1)

# ============================================================================
# 常量
# ============================================================================

# Piper 关节角原始单位: 0.001 度 → 弧度的转换
DEGREE_RAW_TO_RAD = np.pi / 180.0 / 1000.0
# Piper 夹爪原始单位: 0.001 mm → 米
GRIPPER_RAW_TO_M = 1e-6
# 采集频率 (Hz)
COLLECT_FPS = 50
# 默认图像分辨率
# 默认图像分辨率
DEFAULT_IMAGE_SIZE = (224, 224)


# ============================================================================
# Piper 机械臂接口封装
# ============================================================================

class PiperRobot:
    """封装 Piper SDK 的读取 / 控制接口。"""

    def __init__(self, can_name: str = "can0"):
        self._piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=False,  # 允许非官方 CAN 模块
            can_auto_init=True,
            dh_is_offset=1,  # 根据固件版本选择: 0 = 旧版, 1 = 新版(S-V1.6-3+)
        )
        self._piper.ConnectPort()
        time.sleep(0.1)
        self._enabled = False

    # ---- 使能 / 去使能 ----

    def enable(self) -> bool:
        """使能机械臂。返回是否成功。"""
        print("[Piper] 正在使能...")
        deadline = time.monotonic() + 5.0
        while not self._piper.EnablePiper():
            if time.monotonic() > deadline:
                print("[Piper] 使能超时！请确认机械臂已上电且处于从臂模式。")
                return False
            time.sleep(0.01)
        self._enabled = True
        print("[Piper] 使能成功")
        return True

    def disable(self):
        """去使能机械臂。"""
        print("[Piper] 正在去使能...")
        self._piper.DisableArm()
        self._enabled = False
        time.sleep(0.1)
        print("[Piper] 已去使能")

    # ---- 读取状态 ----

    def get_joints_rad(self) -> np.ndarray:
        """读取6个关节角，单位: 弧度。shape = (6,), float32。"""
        joint_msg = self._piper.GetArmJointMsgs()
        js = joint_msg.joint_state  # ArmMsgFeedBackJointStates
        raw = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float32,
        )
        return raw * DEGREE_RAW_TO_RAD

    def get_gripper_raw(self) -> np.ndarray:
        """读取夹爪位置（原始值 0.001mm）。shape = (1,), float32。"""
        gripper_msg = self._piper.GetArmGripperMsgs()
        return np.array([gripper_msg.gripper_state.grippers_angle], dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """读取完整状态: [j1..j6(rad), gripper(raw)]，shape = (7,), float32。"""
        return np.concatenate([self.get_joints_rad(), self.get_gripper_raw()])

    def get_state_velocity(self) -> np.ndarray:
        """读取完整状态 + 关节速度。shape = (14,), float32。

        前7维: [j1..j6(rad), gripper(raw)]
        后7维: [j1_vel..j6_vel(rad/s), 0] (Piper 不直接提供关节速度，填充0)
        """
        state = self.get_state()
        # Piper SDK 不直接暴露速度，用上一帧差分近似 — 上层负责处理
        vel = np.zeros(7, dtype=np.float32)
        return np.concatenate([state, vel])

    # ---- 控制 ----

    def set_joint_mode(self, speed_pct: int = 50):
        """切换到关节空间控制模式。

        Args:
            speed_pct: 速度百分比 (0-100)。
        """
        self._piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)

    def set_end_pose_mode(self, speed_pct: int = 50):
        """切换到末端位姿控制模式。"""
        self._piper.MotionCtrl_2(0x01, 0x00, speed_pct, 0x00)

    def send_joint_command(self, joints_rad: np.ndarray):
        """发送关节角指令 (弧度)。joints_rad.shape = (6,)。"""
        raw = (joints_rad / DEGREE_RAW_TO_RAD).astype(int)
        self._piper.JointCtrl(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])

    def send_gripper_command(self, position_raw: int, effort: int = 1000):
        """发送夹爪指令。position_raw 单位 0.001mm。"""
        self._piper.GripperCtrl(int(position_raw), effort, 0x01, 0)

    def send_command(self, action: np.ndarray, speed_pct: int = 50):
        """发送完整动作指令。

        action.shape = (7,): [j1..j6(rad), gripper(raw)]
        前6维为关节目标角(弧度)，第7维为夹爪目标(原始值 0.001mm)。
        """
        self.set_joint_mode(speed_pct)
        self.send_joint_command(action[:6])
        self.send_gripper_command(action[6])

    # ---- 属性 ----

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def raw_interface(self):
        """返回底层 Piper 接口，用于直接调用高级方法。"""
        return self._piper


# ============================================================================
# 数据采集器
# ============================================================================

@dataclasses.dataclass(frozen=True)
class CollectConfig:
    """数据采集配置。

    两种模式:
      - 本地模式: 指定 data_dir，数据保存到本地任意目录
      - HF 模式: 指定 repo_id，数据保存到 HF_LEROBOT_HOME 下
    """

    # 本地保存目录 (如 ./piper_data)。与 repo_id 二选一至少指定一个。
    data_dir: Optional[str] = None
    # LeRobot 数据集名称 (HF 模式, 如 your_hf_username/piper_data)
    repo_id: Optional[str] = None
    # CAN 端口名称
    can_name: str = "can0"
    # 采集频率
    fps: int = COLLECT_FPS
    # RealSense 相机序列号 (D435i 基座, D405 腕部)
    rs2_base_serial: Optional[str] = None
    rs2_wrist_serial: Optional[str] = None
    # OpenCV 相机设备 ID 回退 (base, wrist)
    cv_base_id: Optional[int] = None
    cv_wrist_id: Optional[int] = None
    # 是否在采集完成后推送到 HuggingFace Hub (仅在 HF 模式下有效)
    push_to_hub: bool = False
    # 示教模式: 记录从臂位姿作为 state 和 action
    teach_mode: bool = False
    # 是否关闭相机预览窗口
    no_preview: bool = False
    # 覆盖已有数据集 (否则默认追加 episode)
    overwrite: bool = False
    # 每保存一个 episode 后立即上传到 HF Hub（仅在 HF 模式有效）
    stream_hub: bool = False
    # stream_hub 模式下每 N 个 episode 上传一次
    stream_batch: int = 5
    # HF 分支/版本 (默认 main)
    hf_revision: str = "main"


class DemoCollector:
    """Piper 机械臂数据采集器。

    使用方式:
        collector = DemoCollector(CollectConfig(repo_id="my_piper_data"))
        collector.run()
        # 按 Enter 开始录制 → 输入指令语 → 机械臂操作 → 按 s 停止保存
    """

    def __init__(self, config: CollectConfig):
        self._config = config
        self._robot = PiperRobot(config.can_name)

        # 初始化相机 (优先 RealSense，回退 OpenCV)
        self._camera = create_cameras(
            base_serial=config.rs2_base_serial,
            wrist_serial=config.rs2_wrist_serial,
            base_cv_id=config.cv_base_id,
            wrist_cv_id=config.cv_wrist_id,
        )
        self._has_base_cam = bool(config.rs2_base_serial or config.cv_base_id is not None)
        self._has_wrist_cam = bool(config.rs2_wrist_serial or config.cv_wrist_id is not None)

        # LeRobot 数据集（延迟创建，因为需要先确认 features）
        self._dataset: Optional[LeRobotDataset] = None

        # 预览窗口 (OpenCV GUI)
        self._preview_active = cv2 is not None and self._camera is not None and not config.no_preview

        # 运行状态
        self._running = False
        self._recording = False
        self._episode_count = 0  # 当前采集的第几个 episode
        self._pending_uploads: list[threading.Thread] = []  # 后台上传线程
        self._stream_unsaved = 0  # stream_hub 批量: 攒了几个 episode 没上传

    # ======================== 运行入口 ========================

    def run(self):
        """启动数据采集。"""
        print("=" * 60)
        print("Piper 机械臂数据采集器")
        print("=" * 60)

        # 使能机械臂
        # if not self._robot.enable():
        #     self._robot.disable()
        #     return

        # # 切换为关节模式（低速率以策安全）
        # self._robot.set_joint_mode(speed_pct=30)

        # 启动相机
        if self._camera:
            self._camera.start()
            cam_list = []
            if self._has_base_cam:
                cam_list.append("base(D435i)" if self._config.rs2_base_serial else "base(CV)")
            if self._has_wrist_cam:
                cam_list.append("wrist(D405)" if self._config.rs2_wrist_serial else "wrist(CV)")
            print(f"[Camera] 已启动: {', '.join(cam_list)}")

        # 初始化 LeRobot 数据集
        self._init_dataset()

        # 启动相机预览窗口 (OpenCV 线程)
        if self._preview_active:
            self._start_preview()

        # 打印操作提示
        print("\n操作提示:")
        print("  [Enter]  开始新 episode")
        print("  [s]      保存当前 episode")
        print("  [d]      丢弃当前 episode (不保存)")
        print("  [q]      退出\n")

        self._running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 收到中断信号，正在退出...")
        finally:
            self._shutdown()

    # ======================== 主循环 ========================

    def _main_loop(self):
        """主循环：select 非阻塞读 stdin，管理录制状态。"""
        fps = self._config.fps
        period = 1.0 / fps

        recording = False
        step = 0

        while self._running:
            loop_start = time.monotonic()

            # 非阻塞检查键盘输入 (select 超时 0)
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.readline().strip().lower()
                if ch == "":  # Enter — 仅在未录制时有效
                    if recording:
                        print("  [提示] 录制中，按 s 保存 / d 丢弃")
                        continue
                    recording = True
                    self._recording = True
                    step = 0
                    self._episode_count += 1
                    task = self._prompt_for_task()
                    ds_features = list(getattr(self._dataset, "features", {}).keys())
                    print(f"\n[Recording] Episode #{self._episode_count} 开始, 指令: '{task}'")
                    print(f"[Recording] features: {ds_features}")
                    print("[Recording] 按 's' 保存 | 'd' 丢弃")
                elif ch == "s":
                    if recording:
                        self._dataset.save_episode()
                        print(f"\n[Recording] Episode #{self._episode_count} 已保存 (约 {step} 帧)")
                        if self._config.stream_hub and self._config.repo_id:
                            self._stream_unsaved += 1
                            if self._stream_unsaved >= self._config.stream_batch:
                                self._flush_stream_batch()
                    recording = False
                    self._recording = False
                    step = 0
                elif ch == "d":
                    if recording:
                        confirm = input(f"  确认丢弃 Episode #{self._episode_count}? [y/N]: ").strip().lower()
                        if confirm == "y":
                            self._dataset.clear_episode_buffer()
                            print(f"\n[Recording] Episode #{self._episode_count} 已丢弃 (约 {step} 帧未保存)")
                            self._episode_count -= 1
                            recording = False
                            self._recording = False
                            step = 0
                        else:
                            print("  已取消，继续录制")
                elif ch == "q":
                    self._running = False
                    break

            if recording:
                self._record_frame(task, step)
                step += 1

            # 控制帧率
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    def _record_frame(self, task: str, step: int):
        """录制一帧数据。"""
        frame: dict = {}

        # 状态
        frame["state"] = self._robot.get_state().astype(np.float32)  # (7,)

        # 动作
        frame["actions"] = self._robot.get_state().astype(np.float32)  # (7,) — 同 state

        # 语言指令
        frame["task"] = task

        # 图像（仅当数据集支持这些 feature 时才添加）
        if self._camera:
            ds_features = getattr(self._dataset, "features", {})
            if self._has_base_cam and "image" in ds_features:
                frame["image"] = self._camera.get_base()
            if self._has_wrist_cam and "wrist_image" in ds_features:
                frame["wrist_image"] = self._camera.get_wrist()

        self._dataset.add_frame(frame)

        # 进度提示
        if step % 50 == 0 and step > 0:
            joints = frame["state"]
            j_str = ", ".join(f"{j:.3f}" for j in joints[:6])
            print(f"  [Ep#{self._episode_count} step={step:5d}] joints(rad)=[{j_str}]")

    # ======================== 数据集初始化 ========================

    def _init_dataset(self):
        """创建或打开 LeRobot 数据集。

        支持两种模式:
          - 本地模式 (data_dir): 直接保存到指定目录，不经过 HuggingFace
          - HF 模式 (repo_id): 使用 HF_LEROBOT_HOME 默认路径
        """
        # 构建 features 描述
        features: dict = {
            "state": {
                "dtype": "float32",
                "shape": (7,),  # 6 joints(rad) + 1 gripper(raw)
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "action_1",
                    "action_2",
                    "action_3",
                    "action_4",
                    "action_5",
                    "action_6",
                    "action_gripper",
                ],
            },
        }

        if self._has_base_cam:
            features["image"] = {
                "dtype": "image",
                "shape": (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            }
        if self._has_wrist_cam:
            features["wrist_image"] = {
                "dtype": "image",
                "shape": (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3),
                "names": ["height", "width", "channel"],
            }

        # 确定 root 和 repo_id
        if self._config.data_dir:
            data_path = os.path.abspath(self._config.data_dir)
            root = data_path
            repo_id = os.path.basename(data_path)
            dataset_label = data_path
        else:
            root = None
            repo_id = self._config.repo_id
            dataset_label = repo_id

        # 检测已有数据集
        output_path = root if root else os.path.join(str(HF_LEROBOT_HOME), repo_id)
        already_exists = os.path.exists(output_path)

        # HF 模式：从 Hub 拉 meta 来数已有 episode（即使本地无缓存）
        if not self._config.data_dir and repo_id:
            self._fetch_hf_meta(repo_id, output_path, revision=self._config.hf_revision)
            already_exists = os.path.exists(output_path)

        if already_exists and self._config.overwrite:
            resp = input(f"[WARNING] 将覆盖已有数据集 {dataset_label}。确认? [y/N]: ")
            if resp.lower() == "y":
                shutil.rmtree(output_path)
                already_exists = False
            else:
                print("[INFO] 已取消覆盖，将追加 episode")

        if already_exists:
            # 加载已有数据集（HF 模式会自动下载缺失的 parquet）
            try:
                kwargs = {}
                if not self._config.data_dir:
                    kwargs["revision"] = self._config.hf_revision
                self._dataset = LeRobotDataset(repo_id, root=root, **kwargs)
                ep_file = Path(output_path) / "meta" / "episodes.jsonl"
                existing = sum(1 for _ in ep_file.read_text().strip().split("\n") if _.strip()) if ep_file.exists() else 0
                self._episode_count = existing
                print(f"[Dataset] 已连接: {output_path} (已有 {existing} 个 episode)")
                return
            except Exception as e:
                err = str(e)
                if "version" in err.lower() or "revision" in err.lower():
                    print(f"[WARNING] 本地缓存版本不匹配，请指定 --hf_revision 或清缓存重试")
                    print(f"  错误: {e}")
                else:
                    print(f"[WARNING] 加载已有数据集失败 ({e})，重建本地缓存")
                shutil.rmtree(output_path, ignore_errors=True)

        self._dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._config.fps,
            root=root,
            robot_type="piper",
            features=features,
            use_videos=False,  # 直接存图片，不生成视频
            image_writer_threads=4,
            image_writer_processes=2,
        )

        # 打印保存路径
        actual_path = root if root else os.path.join(HF_LEROBOT_HOME, repo_id)
        print(f"[Dataset] 数据集已初始化: {actual_path}")

    # ======================== HF 工具 ========================

    @staticmethod
    def _fetch_hf_meta(repo_id: str, local_dir: str, revision: str = "main"):
        """从 HuggingFace Hub 下载 meta 文件到本地（仅 meta，不含 parquet）。"""
        try:
            from huggingface_hub import hf_hub_download, whoami

            try:
                whoami()
            except Exception:
                print("[Dataset] 未登录 HuggingFace，无法检测 HF 已有 episode。")
                print("  运行 huggingface-cli login 后重试")
                return

            Path(local_dir).mkdir(parents=True, exist_ok=True)
            for f in ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
                try:
                    hf_hub_download(repo_id, f, repo_type="dataset", local_dir=local_dir, revision=revision)
                except Exception:
                    pass
        except Exception:
            pass

    # ======================== 上传 ========================

    def _flush_stream_batch(self):
        """后台上传攒下的所有 episode，上传完成后清理本地 parquet 缓存。"""
        count = self._stream_unsaved
        if count == 0:
            return
        self._stream_unsaved = 0
        first_ep = self._episode_count - count + 1
        last_ep = self._episode_count

        def _upload():
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                root = self._dataset.root
                repo_id = self._config.repo_id

                # 收集要上传的文件
                files = []
                for pf in sorted((root / "data" / "chunk-000").glob("*.parquet")):
                    files.append(("parquet", pf))
                for mf in sorted((root / "meta").glob("*.json*")):
                    files.append(("meta", mf))

                print(f"[Hub] 后台上传 Episode #{first_ep}~#{last_ep} ({count} 个, {len(files)} 个文件)...")

                uploaded_parquets = []
                for i, (kind, f) in enumerate(files, 1):
                    print(f"  [{i}/{len(files)}] {f.name}")
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f"data/chunk-000/{f.name}" if kind == "parquet" else f"meta/{f.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    if kind == "parquet":
                        uploaded_parquets.append(f)

                print(f"[Hub] Episode #{first_ep}~#{last_ep} 上传完成")

                # 清理已上传的 parquet（meta 保留给 LeRobot 用）
                for pf in uploaded_parquets:
                    pf.unlink(missing_ok=True)
                print(f"[Hub] 已清理 {len(uploaded_parquets)} 个本地 parquet 缓存")
            except Exception as e:
                print(f"[Hub] 批量上传失败: {e}")

        t = threading.Thread(target=_upload, daemon=True)
        t.start()
        self._pending_uploads.append(t)

    # ======================== 清理 ========================

    def _shutdown(self):
        """清理资源。退出时不失能机械臂。"""
        self._running = False
        if self._camera:
            self._camera.stop()
        if self._preview_active:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        # 上传攒下的剩余 episode
        if self._stream_unsaved > 0:
            self._flush_stream_batch()

        # 等待后台上传完成
        if self._pending_uploads:
            print(f"[Hub] 等待 {len(self._pending_uploads)} 个后台上传完成...")
            for t in self._pending_uploads:
                t.join(timeout=120)
            self._pending_uploads.clear()

        # 最终推送：如果用了 stream_hub，做一次 push_to_hub 收尾
        if self._dataset and self._config.repo_id:
            if self._config.push_to_hub or self._config.stream_hub:
                print("[Hub] 正在同步数据集到 HuggingFace...")
                try:
                    self._dataset.push_to_hub(
                        tags=["piper", "robot", "manipulation"],
                        private=True,
                        push_videos=False,
                        license="apache-2.0",
                    )
                    print("[Hub] 同步完成")
                except Exception as e:
                    print(f"[Hub] 同步失败: {e}")

        print("[INFO] 采集器已退出 (机械臂保持使能)")

    # ======================== 相机预览 ========================

    def _start_preview(self):
        """启动 OpenCV 相机预览窗口 (base | wrist 并排显示)。"""

        def _preview_loop():
            cv2.namedWindow("Piper Camera (base | wrist)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Piper Camera (base | wrist)", 896, 448)

            while self._running:
                base = np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), dtype=np.uint8)
                wrist = np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), dtype=np.uint8)
                if self._camera:
                    if self._has_base_cam:
                        base = self._camera.get_base()
                    if self._has_wrist_cam:
                        wrist = self._camera.get_wrist()
                    elif self._has_base_cam:
                        wrist = base.copy()

                # 左右拼接 + 标签 (RGB → BGR for OpenCV)
                canvas = np.hstack([base[..., ::-1], wrist[..., ::-1]])
                cv2.putText(canvas, "base  D435i", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(canvas, "wrist D405", (224 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # 录制状态指示
                if self._recording:
                    cv2.putText(canvas, "REC", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Piper Camera (base | wrist)", canvas)
                key = cv2.waitKey(33) & 0xFF
                if key in (27, ord("q")):  # ESC or q
                    self._running = False
                    break

            cv2.destroyAllWindows()

        thread = threading.Thread(target=_preview_loop, daemon=True)
        thread.start()
        print("[Preview] 相机预览窗口已打开 (按 ESC 或 q 退出)")

    # ======================== 交互工具 ========================

    def _prompt_for_task(self) -> str:
        """提示用户输入任务指令。"""
        print()
        task = input("请输入任务指令 (如 'pick up the red block'): ").strip()
        if not task:
            task = "do something"
        return task



# ============================================================================
# 便捷: 示教模式数据采集 (主从臂联动)
# ============================================================================

class TeachModeCollector(DemoCollector):
    """示教模式采集器：主臂拖动 → 从臂跟随 → 记录从臂的关节角作为 state 和 action。

    适用于双 Piper 臂的主从示教场景。如果只有单臂，使用 DemoCollector 即可。
    """

    def _record_frame(self, task: str, step: int):
        """从臂当前位姿即目标位姿（跟随主臂）。"""
        frame: dict = {}
        state = self._robot.get_state()
        frame["state"] = state.astype(np.float32)
        frame["actions"] = state.astype(np.float32)  # 从臂位姿 = 主臂示教的目标
        frame["task"] = task

        if self._camera:
            ds_features = getattr(self._dataset, "features", {})
            if self._has_base_cam and "image" in ds_features:
                frame["image"] = self._camera.get_base()
            if self._has_wrist_cam and "wrist_image" in ds_features:
                frame["wrist_image"] = self._camera.get_wrist()

        self._dataset.add_frame(frame)

        if step % 50 == 0 and step > 0:
            j_str = ", ".join(f"{j:.3f}" for j in state[:6])
            print(f"  [Ep#{self._episode_count} step={step:5d}] joints(rad)=[{j_str}]")


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> CollectConfig:
    """解析命令行参数。"""
    import argparse

    p = argparse.ArgumentParser(description="Piper 机械臂数据采集脚本")
    p.add_argument("--data_dir", default=None, help="本地保存目录 (如 ./piper_data)。与 --repo_id 二选一")
    p.add_argument("--repo_id", default=None, help="HF 数据集名称 (如 your_hf_username/piper_data)。与 --data_dir 二选一")
    p.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")
    p.add_argument("--fps", type=int, default=COLLECT_FPS, help=f"采集频率 (默认: {COLLECT_FPS})")
    p.add_argument(
        "--rs2_base",
        default=None,
        help="D435i 基座/外部相机序列号。先运行 'python camera_utils.py --list' 查看",
    )
    p.add_argument(
        "--rs2_wrist",
        default=None,
        help="D405 腕部相机序列号。先运行 'python camera_utils.py --list' 查看",
    )
    p.add_argument(
        "--cam_ids",
        type=int,
        nargs="*",
        default=[],
        help="OpenCV 相机设备 ID 回退。第一个=基座，第二个=腕部。如: --cam_ids 0 2",
    )
    p.add_argument("--push_to_hub", action="store_true", help="采集完成后一次性推送到 HuggingFace Hub (需 --repo_id)")
    p.add_argument("--stream_hub", action="store_true", help="分批上传到 Hub (需 --repo_id)")
    p.add_argument("--stream_batch", type=int, default=5, help="stream_hub 每 N 个 episode 上传一次 (默认 5)")
    p.add_argument("--hf_revision", default="main", help="HF 数据集分支 (默认 main，如 v2.1)")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有数据集 (默认追加新 episode)")
    p.add_argument(
        "--no_preview",
        action="store_true",
        help="关闭相机预览窗口 (默认在有相机时自动开启)",
    )
    p.add_argument(
        "--teach_mode",
        action="store_true",
        help="示教模式: 记录从臂位姿作为 state 和 action (用于主从示教场景)",
    )
    args = p.parse_args()

    # 校验: --data_dir 和 --repo_id 至少指定一个
    if not args.data_dir and not args.repo_id:
        p.error("必须指定 --data_dir 或 --repo_id 之一")

    # 推送到 Hub 只在 HF 模式下有效
    if args.push_to_hub and not args.repo_id:
        p.error("--push_to_hub 需要同时指定 --repo_id")

    return CollectConfig(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        can_name=args.can_name,
        fps=args.fps,
        rs2_base_serial=args.rs2_base,
        rs2_wrist_serial=args.rs2_wrist,
        cv_base_id=args.cam_ids[0] if len(args.cam_ids) > 0 else None,
        cv_wrist_id=args.cam_ids[1] if len(args.cam_ids) > 1 else None,
        push_to_hub=args.push_to_hub,
        stream_hub=args.stream_hub,
        stream_batch=args.stream_batch,
        hf_revision=args.hf_revision,
        teach_mode=args.teach_mode,
        no_preview=args.no_preview,
        overwrite=args.overwrite,
    )


def main():
    config = _parse_args()

    if config.teach_mode:
        collector = TeachModeCollector(config)
    else:
        collector = DemoCollector(config)

    collector.run()


if __name__ == "__main__":
    main()
