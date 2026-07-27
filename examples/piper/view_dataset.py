#!/usr/bin/env python3
"""
Piper LeRobot 数据集可视化工具。

显示内容:
  - 关节角轨迹图 (state 和 actions)
  - 同步的相机图像回放
  - Episode 统计信息
  pick up the pen and place it into the cup

用法:
    python examples/piper/view_dataset.py --data_dir ./piper_data
    python examples/piper/view_dataset.py --data_dir ./piper_data --episode 0
    python examples/piper/view_dataset.py --data_dir ./piper_data --gif output.gif
"""

import argparse
import io
import json
import os
import struct
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

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# 数据加载
# ============================================================================


def load_dataset(data_dir: str):
    """加载 LeRobot 数据集的元信息和 parquet 文件列表。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    # 加载 meta
    meta = {}
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
    data_dir_path = root / "data"
    parquet_files = sorted(data_dir_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {data_dir_path}")

    frames_per_episode = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episodes = df["episode_index"].unique()
        for ep in episodes:
            ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
            frames_per_episode[int(ep)] = ep_df

    return meta, frames_per_episode


def _decode_image(row, key: str) -> np.ndarray:
    """从 parquet row 解码 PNG/二进制图像为 numpy uint8 RGB。"""
    val = row[key]
    if isinstance(val, dict):
        data = val.get("bytes", val.get("path", None))
    elif isinstance(val, bytes):
        data = val
    elif isinstance(val, np.ndarray):
        return val
    else:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    if isinstance(data, bytes):
        if data[:4] == b"\x89PNG" or data[:2] == b"\xff\xd8":
            # PNG or JPEG → decode via cv2
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(data) == 224 * 224 * 3:
            return np.frombuffer(data, np.uint8).reshape(224, 224, 3)
        else:
            # raw RGB
            return np.frombuffer(data, np.uint8).reshape(224, 224, 3)

    return np.zeros((224, 224, 3), dtype=np.uint8)


# ============================================================================
# 可视化
# ============================================================================


def print_dataset_info(meta: dict, frames_per_episode: dict):
    """打印数据集概览。"""
    info = meta.get("info", {})
    episodes_meta = meta.get("episodes", [])
    tasks_meta = meta.get("tasks", [])

    print("=" * 60)
    print("数据集概览")
    print("=" * 60)
    print(f"  机器人类型: {info.get('robot_type', 'unknown')}")
    print(f"  代码版本:   {info.get('codebase_version', 'unknown')}")
    print(f"  Episode 数:  {len(frames_per_episode)}")

    for ep, df in frames_per_episode.items():
        n = len(df)
        ep_info = episodes_meta[ep] if ep < len(episodes_meta) else {}
        task = ep_info.get("tasks", [])
        task_str = ", ".join(task) if task else "(无)"
        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        print(f"\n  Episode {ep}:")
        print(f"    帧数:   {n}")
        print(f"    任务:   {task_str}")
        print(f"    时长:   {duration:.1f}s")

        # 关节范围
        state = np.stack(df["state"].values)
        actions = np.stack(df["actions"].values)
        for label, arr in [("state", state), ("actions", actions)]:
            j_min = arr[:, :6].min(axis=0)
            j_max = arr[:, :6].max(axis=0)
            g_min, g_max = arr[:, 6].min(), arr[:, 6].max()
            print(f"    {label}: joints=({j_min[0]:.2f}~{j_max[0]:.2f}, ..., "
                  f"{j_min[5]:.2f}~{j_max[5]:.2f}) rad | gripper={g_min:.0f}~{g_max:.0f}")


def _label_image(img: np.ndarray, label: str) -> np.ndarray:
    """在图像左上角打标签。"""
    out = img.copy()
    cv2.putText(out, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def show_episode_gui(df, ep_idx: int, task: str = ""):
    """交互式 GUI 播放 episode。"""
    if not HAS_MPL or not HAS_CV2:
        print("[ERROR] 需要 matplotlib 和 opencv-python")
        return

    state = np.stack(df["state"].values)
    actions = np.stack(df["actions"].values)
    n_frames = len(df)
    timestamps = df["timestamp"].values

    # 检查是否有图像
    has_image = "image" in df.columns
    has_wrist = "wrist_image" in df.columns
    has_wrist_right = "wrist_image_right" in df.columns
    n_cams = sum([has_image, has_wrist, has_wrist_right])

    # ---- 创建界面 ----
    fig = plt.figure("Piper Dataset Viewer", figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3, wspace=0.25)

    # 图像: base + wrist + wrist_right (水平拼接)
    canvas_w = 224 * max(n_cams, 1)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_title("Cameras")
    ax_img.axis("off")
    img_show = ax_img.imshow(np.zeros((224, canvas_w, 3), dtype=np.uint8))

    # 关节轨迹
    ax_joints = fig.add_subplot(gs[0, 1])
    ax_joints.set_title("Joint Angles (state)")
    ax_joints.set_xlabel("frame")
    ax_joints.set_ylabel("rad")
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    joint_lines = []
    for i, (name, c) in enumerate(zip(joint_names, colors)):
        (line,) = ax_joints.plot([], [], label=name, color=c, linewidth=0.8)
        joint_lines.append(line)
    ax_joints.legend(loc="upper right", fontsize=7, ncol=3)
    ax_joints.set_xlim(0, n_frames)
    j_all = state[:, :6]
    margin = 0.1
    ax_joints.set_ylim(j_all.min() - margin, j_all.max() + margin)
    cursor_line = ax_joints.axvline(0, color="red", alpha=0.5, linewidth=1)

    # 夹爪 + action diff
    ax_gripper = fig.add_subplot(gs[1, 0])
    ax_gripper.set_title("Gripper & Action Error")
    ax_gripper.set_xlabel("frame")
    (grip_line,) = ax_gripper.plot([], [], "b-", label="gripper state", linewidth=0.8)
    (diff_line,) = ax_gripper.plot([], [], "r-", label="state-action diff", linewidth=0.8)
    ax_gripper.legend(fontsize=7)
    ax_gripper.set_xlim(0, n_frames)
    ax_gripper.set_ylim(state[:, 6].min() - 50, state[:, 6].max() + 50)
    cursor_grip = ax_gripper.axvline(0, color="red", alpha=0.5, linewidth=1)

    # 统计信息文本
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes, fontsize=9, fontfamily="monospace", va="top"
    )

    fig.suptitle(f"Episode {ep_idx}: {task}", fontsize=12, fontweight="bold")

    # ---- 滑块 + 播放控制 ----
    ax_slider = fig.add_axes([0.12, 0.02, 0.72, 0.03])
    slider = matplotlib.widgets.Slider(ax_slider, "", 0, n_frames - 1, valinit=0, valfmt="%d")

    playing = [False]
    play_timer = [None]

    def update(frame_idx):
        """刷新所有子图到指定帧。"""
        idx = int(frame_idx)
        # 图像
        if has_image or has_wrist or has_wrist_right:
            row = df.iloc[idx]
            frames = []
            labels = ["base", "wrist", "wrist_r"]
            if has_image:
                img = _decode_image(row, "image")
                frames.append(_label_image(img, labels[0]))
            if has_wrist:
                img = _decode_image(row, "wrist_image")
                frames.append(_label_image(img, labels[1]))
            if has_wrist_right:
                img = _decode_image(row, "wrist_image_right")
                frames.append(_label_image(img, labels[2]))
            canvas = np.hstack(frames) if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            img_show.set_data(canvas)

        # 关节线
        for i, line in enumerate(joint_lines):
            line.set_data(range(idx + 1), state[: idx + 1, i])
        cursor_line.set_xdata([idx, idx])

        # 夹爪
        grip_line.set_data(range(idx + 1), state[: idx + 1, 6])
        diff_line.set_data(range(idx + 1), (state[: idx + 1, 6] - actions[: idx + 1, 6]))
        cursor_grip.set_xdata([idx, idx])

        # 信息
        s = state[idx]
        a = actions[idx]
        j_str = "  ".join(f"{name}:{s[i]:7.3f}" for i, name in enumerate(joint_names))
        info = (
            f"Frame: {idx}/{n_frames - 1}\n"
            f"Time:  {timestamps[idx]:.2f}s\n\n"
            f"State joints (rad):\n  {j_str}\n"
            f"State gripper: {s[6]:.0f}\n\n"
            f"Action joints (rad):\n"
            f"  {'  '.join(f'{name}:{a[i]:7.3f}' for i, name in enumerate(joint_names))}\n"
            f"Action gripper: {a[6]:.0f}"
        )
        info_text.set_text(info)

        return [img_show, *joint_lines, cursor_line, grip_line, diff_line, cursor_grip, info_text]

    def on_slider(val):
        update(val)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, n_frames - 1))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))
        elif event.key == " ":
            playing[0] = not playing[0]
            if playing[0]:
                _auto_play()
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(n_frames - 1)

    def _auto_play():
        if not playing[0]:
            return
        next_val = slider.val + 1
        if next_val >= n_frames:
            playing[0] = False
            return
        slider.set_val(next_val)
        play_timer[0] = fig.canvas.new_timer(interval=30)
        play_timer[0].single_shot = True
        play_timer[0].add_callback(_auto_play)
        play_timer[0].start()

    fig.canvas.mpl_connect("key_press_event", on_key)

    update(0)
    plt.show()


def export_gif(df, output_path: str, task: str = "", fps: int = 15, max_frames: int = 300):
    """导出 episode 为 GIF 动图。"""
    if not HAS_MPL or not HAS_CV2:
        print("[ERROR] 需要 matplotlib 和 opencv-python")
        return

    n = min(len(df), max_frames)

    # 只看图像 + 关节的简化布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_img, ax_j = axes
    ax_img.axis("off")
    ax_j.set_title("Joint Angles")
    ax_j.set_xlabel("frame")
    ax_j.set_ylabel("rad")
    ax_j.set_xlim(0, n)

    state = np.stack(df["state"].values)
    ax_j.set_ylim(state[:, :6].min() - 0.1, state[:, :6].max() + 0.1)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    imgs = []
    for idx in range(n):
        ax_img.clear()
        ax_img.axis("off")

        row = df.iloc[idx]
        base = _decode_image(row, "image") if "image" in df.columns else np.zeros((224, 224, 3), dtype=np.uint8)
        wrist = (
            _decode_image(row, "wrist_image")
            if "wrist_image" in df.columns
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        canvas = np.hstack([base, wrist])
        ax_img.imshow(canvas)
        ax_img.set_title(f"Episode — {task}", fontsize=10)

        ax_j.clear()
        ax_j.set_title(f"Joint Angles (frame {idx}/{n})")
        for i, c in enumerate(colors):
            ax_j.plot(range(idx + 1), state[: idx + 1, i], color=c, linewidth=0.6)
        ax_j.set_xlim(0, n)
        ax_j.set_ylim(state[:, :6].min() - 0.1, state[:, :6].max() + 0.1)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        imgs.append(data[..., :3])

        if idx % 50 == 0:
            print(f"  GIF frame {idx}/{n}")

    # 写 GIF
    print(f"  导出 GIF → {output_path} ({len(imgs)} frames)")
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=fps)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.axis("off")
    ani = FuncAnimation(fig2, lambda i: [ax2.imshow(imgs[i]), ax2.set_title(f"frame {i}")], frames=len(imgs))
    ani.save(output_path, writer=writer, dpi=100)
    plt.close("all")
    print(f"  完成: {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    p = argparse.ArgumentParser(description="Piper LeRobot 数据集可视化")
    p.add_argument("--data_dir", required=True, help="数据集目录 (如 ./piper_data)")
    p.add_argument("--episode", type=int, default=None, help="指定 episode (默认显示第一个)")
    p.add_argument("--gif", default=None, help="导出为 GIF 文件 (如 output.gif)")
    p.add_argument("--info_only", action="store_true", help="仅打印数据集信息，不播放")
    args = p.parse_args()

    if not HAS_PANDAS:
        print("[ERROR] 需要 pandas: pip install pandas")
        return
    if not HAS_MPL:
        print("[ERROR] 需要 matplotlib: pip install matplotlib")
        return

    meta, frames = load_dataset(args.data_dir)

    print_dataset_info(meta, frames)

    if args.info_only:
        return

    # 选 episode
    if args.episode is not None:
        ep_idx = args.episode
    else:
        ep_idx = sorted(frames.keys())[0]

    if ep_idx not in frames:
        print(f"[ERROR] Episode {ep_idx} 不存在。可用: {sorted(frames.keys())}")
        return

    df = frames[ep_idx]
    ep_meta = meta.get("episodes", [])
    task = ""
    if ep_idx < len(ep_meta):
        task = ", ".join(ep_meta[ep_idx].get("tasks", ["(无)"]))

    if args.gif:
        print(f"\n导出 Episode {ep_idx}/{len(frames)} — 任务: {task or '(无)'}")
        export_gif(df, args.gif, task=task)
    else:
        total = len(frames)
        available = sorted(frames.keys())
        print(f"\n正在播放 Episode {ep_idx}/{total} (可用: {available}) — 任务: {task or '(无)'}")
        show_episode_gui(df, ep_idx, task=task)


if __name__ == "__main__":
    main()
