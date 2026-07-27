#!/usr/bin/env python3
"""
Piper 机械臂数据集回放脚本。

从 LeRobot 格式的本地数据集读取 action，逐帧回放到 Piper 机械臂上。

用法:
    python examples/piper/replay_dataset.py --data_dir /media/rhr/Doc/ubuntu/data
    python examples/piper/replay_dataset.py --data_dir /media/rhr/Doc/ubuntu/data --episode 2
    python examples/piper/replay_dataset.py --data_dir /media/rhr/Doc/ubuntu/data --speed 0.5 --loop

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电并处于从臂模式
    3. piper_sdk 已安装:  pip install piper_sdk
    4. pandas 已安装:  pip install pandas

键盘控制:
    Enter   — 开始回放当前 episode
    Space   — 暂停/继续
    r       — 重置到当前 episode 开头
    n       — 下一个 episode
    p       — 上一个 episode
    +/-     — 加速/减速 (0.25x ~ 4x)
    q       — 退出（机械臂保持使能）
"""

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 可选依赖
# ---------------------------------------------------------------------------
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Piper SDK 导入
# ---------------------------------------------------------------------------
PIPER_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "piper_sdk")
PIPER_EXAMPLES_PATH = os.path.dirname(__file__)
if PIPER_SDK_PATH not in sys.path:
    sys.path.insert(0, PIPER_SDK_PATH)
if PIPER_EXAMPLES_PATH not in sys.path:
    sys.path.insert(0, PIPER_EXAMPLES_PATH)

try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore[import-untyped]
except ImportError:
    print("[ERROR] 无法导入 piper_sdk，请先安装:")
    print("  cd piper_sdk && pip install .")
    sys.exit(1)


# ===========================================================================
# 常量
# ===========================================================================

DEFAULT_DATA_DIR = "/media/rhr/Doc/ubuntu/data"

# Piper 关节角转换
RAW_TO_RAD = np.pi / 180.0 / 1000.0
RAD_TO_RAW = 180.0 * 1000.0 / np.pi

# 默认回放速度（百分比），回放时较低速以保安全
DEFAULT_REPLAY_SPEED_PCT = 30

# 关节角限位 (弧度)
JOINT_LIMITS_RAD = np.array(
    [
        [-2.8, 2.8],  # J1
        [-1.5, 1.5],  # J2
        [-2.8, 2.8],  # J3
        [-2.8, 2.8],  # J4
        [-1.5, 1.5],  # J5
        [-2.8, 2.8],  # J6
    ]
)

# 夹爪控制力矩
GRIPPER_EFFORT = 1000


# ===========================================================================
# Piper 机械臂控制封装
# ===========================================================================


class PiperController:
    """封装 Piper SDK 的控制接口，用于回放。"""

    def __init__(self, can_name: str = "can0"):
        self._piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=False,
            can_auto_init=True,
            dh_is_offset=1,
        )
        self._piper.ConnectPort()
        time.sleep(0.1)
        self._enabled = False

    def enable(self) -> bool:
        print("[Piper] 正在使能...")
        deadline = time.monotonic() + 5.0
        while not self._piper.EnablePiper():
            if time.monotonic() > deadline:
                print("[Piper] 使能超时! 检查机械臂状态。")
                return False
            time.sleep(0.01)
        self._enabled = True
        self.set_joint_mode(DEFAULT_REPLAY_SPEED_PCT)
        print("[Piper] 使能成功")
        return True

    def disable(self):
        print("[Piper] 正在去使能...")
        self._piper.DisableArm()
        self._enabled = False
        time.sleep(0.1)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_joints_rad(self) -> np.ndarray:
        joint_msg = self._piper.GetArmJointMsgs()
        js = joint_msg.joint_state
        raw = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float32,
        )
        return raw * RAW_TO_RAD

    def get_gripper_raw(self) -> np.ndarray:
        gripper_msg = self._piper.GetArmGripperMsgs()
        return np.array([gripper_msg.gripper_state.grippers_angle], dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.get_joints_rad(), self.get_gripper_raw()])

    def set_joint_mode(self, speed_pct: int = DEFAULT_REPLAY_SPEED_PCT):
        self._piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)

    def send_joint_command(self, joints_rad: np.ndarray):
        # clipped = np.clip(
        #     joints_rad[:6],
        #     JOINT_LIMITS_RAD[:, 0],
        #     JOINT_LIMITS_RAD[:, 1],
        # )
        raw = (joints_rad[:6] * RAD_TO_RAW).astype(int)
        self._piper.JointCtrl(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])

    def send_gripper_command(self, pos_raw: float, effort: int = GRIPPER_EFFORT):
        self._piper.GripperCtrl(int(pos_raw), effort, 0x01, 0)

    def execute_action(self, action: np.ndarray, speed_pct: int = DEFAULT_REPLAY_SPEED_PCT):
        """执行单个动作。action.shape = (7,): [j1..j6(rad), gripper(raw)]。"""
        self.set_joint_mode(speed_pct)
        self.send_joint_command(action[:6])
        self.send_gripper_command(action[6])

    def go_to_init_pose(self):
        """回到初始位姿。"""
        print("[Piper] 回到初始位姿...")
        init_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(100):
            self.set_joint_mode(20)
            self.send_joint_command(init_joints)
            time.sleep(0.02)
        print("[Piper] 已到初始位姿")


# ===========================================================================
# 数据集加载
# ===========================================================================


def load_dataset(data_dir: str) -> tuple[dict, dict]:
    """加载 LeRobot 数据集的元信息和按 episode 分组的 DataFrame。

    Returns:
        (meta, frames_per_episode)
        meta: 元信息 dict (info, episodes, tasks)
        frames_per_episode: {episode_index: DataFrame}
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    if not HAS_PANDAS:
        raise ImportError("需要 pandas: pip install pandas")

    # 加载 meta
    meta: dict = {}
    for meta_file in ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
        p = root / "meta" / meta_file
        if not p.exists():
            continue
        key = meta_file.replace(".jsonl", "").replace(".json", "")
        if meta_file.endswith(".jsonl"):
            meta[key] = [json.loads(line) for line in p.read_text().strip().split("\n") if line]
        else:
            meta[key] = json.loads(p.read_text())

    # 加载 parquet
    data_path = root / "data"
    parquet_files = sorted(data_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {data_path}")

    frames_per_episode: dict = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episodes = df["episode_index"].unique()
        for ep in episodes:
            ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
            frames_per_episode[int(ep)] = ep_df

    return meta, frames_per_episode


def get_dataset_fps(meta: dict) -> float:
    """从 meta 中读取 fps。"""
    info = meta.get("info", {})
    return float(info.get("fps", 50))


def get_episode_task(meta: dict, ep_idx: int) -> str:
    """获取 episode 的任务指令字符串。"""
    episodes_meta = meta.get("episodes", [])
    if ep_idx < len(episodes_meta):
        tasks = episodes_meta[ep_idx].get("tasks", [])
        return ", ".join(tasks) if tasks else "(无)"
    return "(无)"


# ===========================================================================
# 回放引擎
# ===========================================================================


class DatasetReplayer:
    """从数据集读取 action 并回放到机械臂。"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        can_name: str = "can0",
        speed: float = 1.0,
        loop: bool = False,
    ):
        self._data_dir = data_dir
        self._can_name = can_name
        self._speed = speed
        self._loop = loop

        # 加载数据
        self._meta, self._frames = load_dataset(data_dir)
        self._dataset_fps = get_dataset_fps(self._meta)
        self._episode_indices = sorted(self._frames.keys())
        if not self._episode_indices:
            raise ValueError("数据集中没有 episode")

        # 机械臂
        self._robot = PiperController(can_name)

        # 状态
        self._current_ep = self._episode_indices[0]
        self._playing = False
        self._running = True
        self._frame_idx = 0
        self._episode_len = 0

    # ---- 运行入口 ----

    def run(self):
        print("=" * 60)
        print("Piper 数据集回放")
        print(f"数据集: {self._data_dir}")
        print(f"Episodes: {len(self._episode_indices)} 个")
        print(f"FPS:      {self._dataset_fps}")
        print("=" * 60)

        # 使能
        if not self._robot.enable():
            print("[ERROR] 机械臂使能失败，退出。")
            return

        # 打印首个 episode 信息
        self._show_episode_info(self._current_ep)
        self._show_help()

        # 键盘监听（后台线程）
        import threading
        input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        input_thread.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            print("[INFO] 已退出 (机械臂保持使能)")

    # ---- 主循环 ----

    def _main_loop(self):
        """主循环：管理回放状态，按帧率发送 action。"""
        period = 1.0 / self._dataset_fps / self._speed

        while self._running:
            if not self._playing:
                time.sleep(0.05)
                continue

            loop_start = time.monotonic()

            # 检查是否播完
            if self._frame_idx >= self._episode_len:
                self._on_episode_end()
                if not self._playing:
                    continue

            # 发送当前帧 action
            try:
                action = self._get_action(self._current_ep, self._frame_idx)
                self._robot.execute_action(action)
                self._frame_idx += 1

                # 进度
                if self._frame_idx % 50 == 0:
                    j_str = ", ".join(f"{action[i]:.3f}" for i in range(6))
                    bar = self._progress_bar(self._frame_idx, self._episode_len)
                    print(
                        f"\r[Ep{self._current_ep}] {bar} {self._frame_idx}/{self._episode_len} | "
                        f"joints=[{j_str}] grip={action[6]:.0f} | {self._speed:.2f}x",
                        end="",
                        flush=True,
                    )
            except Exception as e:
                print(f"\n[ERROR] 执行 action 失败: {e}")
                self._playing = False

            # 帧率控制
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    def _on_episode_end(self):
        """Episode 播放完毕。"""
        print(f"\n[Replay] Episode {self._current_ep} 回放完成 ({self._episode_len} 帧)")

        if self._loop:
            self._frame_idx = 0
            print(f"[Replay] 循环回放 Episode {self._current_ep}...")
            return

        self._playing = False

        # 自动播放下一个
        current_pos = self._episode_indices.index(self._current_ep)
        if current_pos + 1 < len(self._episode_indices):
            self._current_ep = self._episode_indices[current_pos + 1]
            self._frame_idx = 0
            self._episode_len = self._get_episode_len(self._current_ep)
            self._show_episode_info(self._current_ep)
            print("按 Enter 继续回放")
        else:
            print("[Replay] 所有 episode 已播完。按 Enter 重新开始或按 q 退出。")

    # ---- 数据访问 ----

    def _get_action(self, ep: int, frame: int) -> np.ndarray:
        df = self._frames[ep]
        return np.asarray(df["actions"].iloc[frame], dtype=np.float32)

    def _get_state(self, ep: int, frame: int) -> np.ndarray:
        df = self._frames[ep]
        return np.asarray(df["state"].iloc[frame], dtype=np.float32)

    def _get_episode_len(self, ep: int) -> int:
        return len(self._frames[ep])

    # ---- 显示 ----

    def _show_episode_info(self, ep: int):
        """展示 episode 概要。"""
        df = self._frames[ep]
        task = get_episode_task(self._meta, ep)
        n = len(df)
        state = np.stack(df["state"].values)
        actions = np.stack(df["actions"].values)

        print(f"\n{'─' * 40}")
        print(f"  Episode:     {ep}")
        print(f"  任务指令:    {task}")
        print(f"  帧数:        {n}")
        if "timestamp" in df.columns:
            duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
            print(f"  时长:        {duration:.1f}s")
        print(f"  关节范围 (rad):")
        for i in range(6):
            print(f"    J{i+1}: {state[:, i].min():7.3f} ~ {state[:, i].max():7.3f}")
        print(f"  夹爪范围:     {state[:, 6].min():6.0f} ~ {state[:, 6].max():6.0f}")
        print(f"  当前速度:     {self._speed:.2f}x")
        print(f"{'─' * 40}")

    def _show_help(self):
        print("""
操作提示:
  [Enter]  开始/继续回放
  [Space]  暂停
  [r]      重置到当前 episode 开头
  [n]      下一个 episode
  [p]      上一个 episode
  [=/-]    加速/减速 (0.25x ~ 4x)
  [q]      退出
""")

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 40) -> str:
        ratio = min(current / max(total, 1), 1.0)
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    # ---- 键盘交互 ----

    def _keyboard_listener(self):
        """后台监听键盘事件。"""
        while self._running:
            try:
                ch = sys.stdin.readline().strip().lower()
                self._handle_key(ch)
            except (EOFError, OSError):
                break

    def _handle_key(self, ch: str):
        if ch == "":  # Enter
            if not self._playing:
                self._episode_len = self._get_episode_len(self._current_ep)
                if self._frame_idx >= self._episode_len:
                    self._frame_idx = 0
                print(f"\n[Replay] 开始回放 Episode {self._current_ep}")
                self._playing = True
        elif ch == " ":
            if self._playing:
                print(f"\n[Replay] 暂停 (frame {self._frame_idx})")
                self._playing = False
            else:
                print(f"\n[Replay] 继续 (frame {self._frame_idx})")
                self._playing = True
        elif ch == "r":
            was_playing = self._playing
            self._playing = False
            self._frame_idx = 0
            print(f"\n[Replay] 重置到 Episode {self._current_ep} 开头")
            if was_playing:
                self._episode_len = self._get_episode_len(self._current_ep)
                print("按 Enter 重新开始")
        elif ch == "n":
            was_playing = self._playing
            self._playing = False
            current_pos = self._episode_indices.index(self._current_ep)
            if current_pos + 1 < len(self._episode_indices):
                self._current_ep = self._episode_indices[current_pos + 1]
                self._frame_idx = 0
                self._show_episode_info(self._current_ep)
                print("按 Enter 开始回放")
            else:
                print(f"[Replay] 已是最后一个 episode ({self._current_ep})")
        elif ch == "p":
            was_playing = self._playing
            self._playing = False
            current_pos = self._episode_indices.index(self._current_ep)
            if current_pos > 0:
                self._current_ep = self._episode_indices[current_pos - 1]
                self._frame_idx = 0
                self._show_episode_info(self._current_ep)
                print("按 Enter 开始回放")
            else:
                print(f"[Replay] 已是第一个 episode ({self._current_ep})")
        elif ch in ("=", "+"):
            self._speed = min(self._speed * 2, 4.0)
            print(f"[Replay] 速度: {self._speed:.2f}x")
        elif ch == "-":
            self._speed = max(self._speed / 2, 0.25)
            print(f"[Replay] 速度: {self._speed:.2f}x")
        elif ch == "q":
            print("[Replay] 退出...")
            self._running = False
            self._playing = False


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args():
    p = argparse.ArgumentParser(
        description="Piper 机械臂数据集回放",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python replay_dataset.py
  python replay_dataset.py --data_dir /media/rhr/Doc/ubuntu/data --episode 2
  python replay_dataset.py --speed 0.5 --loop
        """,
    )
    p.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=f"数据集目录 (默认: {DEFAULT_DATA_DIR})",
    )
    p.add_argument("--episode", type=int, default=0, help="起始 episode (默认: 0)")
    p.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (默认: 1.0)")
    p.add_argument("--loop", action="store_true", help="循环回放当前 episode")
    p.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")
    p.add_argument("--info_only", action="store_true", help="仅显示数据集信息，不回放")
    return p.parse_args()


def main():
    args = _parse_args()

    if not HAS_PANDAS:
        print("[ERROR] 需要 pandas: pip install pandas")
        sys.exit(1)

    # 仅查看信息模式
    if args.info_only:
        meta, frames = load_dataset(args.data_dir)
        print(f"数据集: {args.data_dir}")
        print(f"Episodes: {sorted(frames.keys())}")
        print(f"FPS:      {get_dataset_fps(meta)}")
        for ep in sorted(frames.keys()):
            df = frames[ep]
            task = get_episode_task(meta, ep)
            state = np.stack(df["state"].values)
            print(f"\n  Episode {ep}: {len(df)} 帧 | 任务: {task}")
            for i in range(6):
                print(f"    J{i+1}: {state[:, i].min():.3f} ~ {state[:, i].max():.3f} rad")
        return

    replayer = DatasetReplayer(
        data_dir=args.data_dir,
        can_name=args.can_name,
        speed=args.speed,
        loop=args.loop,
    )

    # 如果指定了非零 episode，切换到该 episode
    if args.episode in replayer._episode_indices:
        replayer._current_ep = args.episode
    else:
        print(f"[WARNING] Episode {args.episode} 不存在，使用第一个 episode (0)")

    replayer.run()


if __name__ == "__main__":
    main()
