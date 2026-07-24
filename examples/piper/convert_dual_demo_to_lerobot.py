#!/usr/bin/env python3
"""
将 diffusion_policy_piper 的 zarr replay buffer 转换为 LeRobot 格式数据集。

输入: data/dual_demo/replay_buffer.zarr
输出: data/dual_piper_eef_lerobot/

State/action 格式 (14维 EEF):
    [left_x,  left_y,  left_z,  left_rx,  left_ry,  left_rz,  left_gripper,
     right_x, right_y, right_z, right_rx, right_ry, right_rz, right_gripper]

相机:
    image              ← 全局相机 (img[:, 0])
    wrist_image        ← 左腕部相机 (img[:, 1])
    wrist_image_right  ← 右腕部相机 (img[:, 2])

用法:
    python examples/piper/convert_dual_demo_to_lerobot.py
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ZARR_PATH = "data/dual_demo/replay_buffer.zarr"
OUTPUT_DIR = "data/dual_piper_eef_lerobot"
FPS = 50
TASK_TEXT = "pick up the block and place it in the bin"
ROBOT_TYPE = "piper_dual_eef"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """将 uint8 (H, W, 3) RGB 图像编码为 JPEG 字节。"""
    import cv2

    # cv2.imencode 需要 BGR
    bgr = img[..., ::-1]
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------


def main():
    root = Path(ZARR_PATH)
    if not root.exists():
        print(f"[ERROR] 找不到 {ZARR_PATH}")
        sys.exit(1)

    print(f"[INFO] 读取 {ZARR_PATH} ...")
    z = zarr.open(str(root), mode="r")

    ep_ends = z["meta/episode_ends"][:]
    n_episodes = len(ep_ends)
    total_frames = int(ep_ends[-1])

    # 读取所有数据
    left_eef = z["data/left_robot_eef_pose"][:]    # (N, 6) float64
    right_eef = z["data/right_robot_eef_pose"][:]   # (N, 6) float64
    left_grip = z["data/left_gripper_angle"][:]      # (N,)  float64
    right_grip = z["data/right_gripper_angle"][:]    # (N,)  float64
    imgs = z["data/img"][:]                           # (N, 3, 224, 224, 3) uint8
    timestamps = z["data/timestamp"][:]               # (N,)  float64

    # actions: 直接用 EEF 当前值 (采集时 action = 当前状态)
    # 如果 action 列存在且是 delta 格式，这里用 state 代替
    action_arr = z["data/action"][:]  # (N, 14)

    print(f"[INFO] 总帧数: {total_frames}, Episodes: {n_episodes}")
    print(f"[INFO] 各 episode 帧数: {[int(ep_ends[i] - (ep_ends[i-1] if i > 0 else 0)) for i in range(n_episodes)]}")

    # ---- 组装 14 维 EEF state/action ----
    # state = [left_eef(6), left_gripper(1), right_eef(6), right_gripper(1)]
    state = np.zeros((total_frames, 14), dtype=np.float32)
    state[:, 0:6] = left_eef.astype(np.float32)
    state[:, 6] = left_grip.astype(np.float32)
    state[:, 7:13] = right_eef.astype(np.float32)
    state[:, 13] = right_grip.astype(np.float32)

    # action: 用 action 列的数据 (已经是 14 维 EEF 格式)
    actions = action_arr.astype(np.float32)

    # ---- 创建输出目录 ----
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    data_dir = out / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐 episode 写 parquet ----
    episode_info = []
    task_idx = 0

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 需要 pandas: pip install pandas")
        sys.exit(1)

    ep_start = 0
    for ep_idx in range(n_episodes):
        ep_end = int(ep_ends[ep_idx])
        n_frames = ep_end - ep_start

        print(f"[INFO] Episode {ep_idx}: {n_frames} frames (frames {ep_start}..{ep_end-1})")

        frames = []
        for i in range(ep_start, ep_end):
            # 编码三路相机为 JPEG
            img_global_jpg = encode_jpeg(imgs[i, 0])  # (H,W,3) → JPEG bytes
            img_wrist_jpg = encode_jpeg(imgs[i, 1])
            img_wrist_right_jpg = encode_jpeg(imgs[i, 2])

            frames.append({
                "state": state[i],
                "actions": actions[i],
                "timestamp": np.array([timestamps[i]], dtype=np.float32),
                "frame_index": np.array([i - ep_start], dtype=np.int64),
                "episode_index": np.array([ep_idx], dtype=np.int64),
                "index": np.array([i], dtype=np.int64),
                "task_index": np.array([task_idx], dtype=np.int64),
                "image": img_global_jpg,
                "wrist_image": img_wrist_jpg,
                "wrist_image_right": img_wrist_right_jpg,
            })

        df = pd.DataFrame(frames)
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow")
        print(f"  → {parquet_path} ({len(frames)} rows)")

        episode_info.append({
            "episode_index": ep_idx,
            "tasks": [TASK_TEXT],
            "length": n_frames,
        })
        ep_start = ep_end

    # ---- 写 meta 文件 ----
    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "state": {
                "dtype": "float32",
                "shape": [14],
                "names": [
                    "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
                    "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
                ],
            },
            "actions": {
                "dtype": "float32",
                "shape": [14],
                "names": [
                    "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
                    "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
                ],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "image": {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"]},
            "wrist_image_right": {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"]},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # episodes.jsonl
    with open(out / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episode_info:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # tasks.jsonl
    tasks = [{"task_index": 0, "task": TASK_TEXT}]
    (out / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )

    # episodes_stats.jsonl (LeRobot 加载需要)
    stats_entries = []
    ep_start = 0
    for ep in episode_info:
        n = ep["length"]
        s = state[ep_start : ep_start + n]
        a = actions[ep_start : ep_start + n]
        stats_entries.append({
            "episode_index": ep["episode_index"],
            "state_mean": s.mean(axis=0).tolist(),
            "state_std": s.std(axis=0).tolist(),
            "state_min": s.min(axis=0).tolist(),
            "state_max": s.max(axis=0).tolist(),
            "action_mean": a.mean(axis=0).tolist(),
            "action_std": a.std(axis=0).tolist(),
            "action_min": a.min(axis=0).tolist(),
            "action_max": a.max(axis=0).tolist(),
        })
        ep_start += n

    with open(out / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for entry in stats_entries:
            f.write(json.dumps(entry) + "\n")

    # stats.json (全局聚合) — LeRobot 可选但推荐
    global_stats = {
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": state.std(axis=0).tolist(),
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
    }
    (out / "meta" / "stats.json").write_text(json.dumps(global_stats, indent=2), encoding="utf-8")

    print(f"\n[DONE] 数据集已输出到 {OUTPUT_DIR}/")
    print(f"  Episodes: {n_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / FPS:.1f}s @ {FPS}fps")


if __name__ == "__main__":
    main()
