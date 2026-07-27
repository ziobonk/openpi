#!/usr/bin/env python3
"""
将 diffusion_policy_piper 的 zarr replay buffer 转换为 LeRobot 格式数据集。

两种模式:
    eef   — 末端位姿 (笛卡尔空间)
    joint — 关节角 (关节空间)

用法:
    # EEF 模式 (默认)
    python examples/piper/convert_dual_demo_to_lerobot.py

    # 关节模式
    python examples/piper/convert_dual_demo_to_lerobot.py --mode joint

    # 指定输入输出
    python examples/piper/convert_dual_demo_to_lerobot.py \
        --input /home/rhr/diffusion_policy_piper/data/dual_demo/replay_buffer.zarr \
        --output data/dual_piper_joint_lerobot \
        --mode joint
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
FPS = 50
TASK_TEXT = "pick up the block and place it in the bin"
MODE_EEF = "eef"
MODE_JOINT = "joint"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """将 uint8 (H, W, 3) RGB 图像编码为 JPEG 字节。"""
    import cv2

    bgr = img[..., ::-1]
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def joint_state_names() -> list[str]:
    return [
        "left_joint_1", "left_joint_2", "left_joint_3",
        "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3",
        "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
    ]


def eef_state_names() -> list[str]:
    return [
        "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
        "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
    ]

# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Convert dual-piper zarr to LeRobot format")
    p.add_argument("--input", default="data/dual_demo/replay_buffer.zarr",
                   help="输入 zarr 路径 (默认: data/dual_demo/replay_buffer.zarr)")
    p.add_argument("--output", default=None,
                   help="输出目录 (默认: data/dual_piper_{mode}_lerobot)")
    p.add_argument("--mode", choices=[MODE_EEF, MODE_JOINT], default=MODE_EEF,
                   help="控制模式: eef (末端位姿) | joint (关节角)")
    p.add_argument("--fps", type=int, default=FPS, help=f"帧率 (默认: {FPS})")
    p.add_argument("--task", default=TASK_TEXT, help="任务指令文本")
    args = p.parse_args()

    zarr_path = Path(args.input)
    if not zarr_path.exists():
        print(f"[ERROR] 找不到 {zarr_path}")
        sys.exit(1)

    mode = args.mode
    output_dir = args.output or f"data/dual_piper_{mode}_lerobot"
    robot_type = f"piper_dual_{mode}"

    print(f"[INFO] 模式: {mode}")
    print(f"[INFO] 输入: {zarr_path}")
    print(f"[INFO] 输出: {output_dir}")

    z = zarr.open(str(zarr_path), mode="r")

    ep_ends = z["meta/episode_ends"][:]
    n_episodes = len(ep_ends)
    total_frames = int(ep_ends[-1])

    # 读取所有数据
    if mode == MODE_EEF:
        left_pose = z["data/left_robot_eef_pose"][:]    # (N, 6)
        right_pose = z["data/right_robot_eef_pose"][:]   # (N, 6)
        action_arr = z["data/action"][:]                  # (N, 14) EEF actions
    else:  # joint
        left_pose = z["data/left_robot_joint"][:]         # (N, 6) rad
        right_pose = z["data/right_robot_joint"][:]       # (N, 6) rad
        # joint 模式没有预存的 action 列，用当前关节角作为 actions
        action_arr = None

    left_grip = z["data/left_gripper_angle"][:]           # (N,)
    right_grip = z["data/right_gripper_angle"][:]         # (N,)
    imgs = z["data/img"][:]                                # (N, 3, 224, 224, 3) uint8
    timestamps = z["data/timestamp"][:]                    # (N,)

    print(f"[INFO] 总帧数: {total_frames}, Episodes: {n_episodes}")
    ep_lengths = [int(ep_ends[i] - (ep_ends[i - 1] if i > 0 else 0)) for i in range(n_episodes)]
    print(f"[INFO] 各 episode 帧数: {ep_lengths}")

    # ---- 组装 14 维 state ----
    state = np.zeros((total_frames, 14), dtype=np.float32)
    state[:, 0:6] = left_pose.astype(np.float32)
    state[:, 6] = left_grip.astype(np.float32)
    state[:, 7:13] = right_pose.astype(np.float32)
    state[:, 13] = right_grip.astype(np.float32)

    # ---- actions ----
    if action_arr is not None:
        # EEF: 用 action 列
        actions = action_arr.astype(np.float32)
    else:
        # Joint: 用当前关节角作为 actions (teleop 采集时 action ≈ state)
        actions = state.astype(np.float32)

    # ---- 特征名 ----
    if mode == MODE_JOINT:
        s_names = joint_state_names()
        a_names = joint_state_names()
    else:
        s_names = eef_state_names()
        a_names = eef_state_names()

    # ---- 创建输出目录 ----
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 需要 pandas: pip install pandas")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    data_dir = out / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐 episode 写 parquet ----
    episode_info = []
    task_idx = 0
    ep_start = 0

    for ep_idx in range(n_episodes):
        ep_end = int(ep_ends[ep_idx])
        n_frames = ep_end - ep_start
        print(f"[INFO] Episode {ep_idx}: {n_frames} frames (frames {ep_start}..{ep_end - 1})")

        frames = []
        for i in range(ep_start, ep_end):
            frames.append({
                "state": state[i],
                "actions": actions[i],
                "timestamp": np.float32(timestamps[i]),
                "frame_index": i - ep_start,
                "episode_index": ep_idx,
                "index": i,
                "task_index": task_idx,
                "image": encode_jpeg(imgs[i, 0]),
                "wrist_image": encode_jpeg(imgs[i, 1]),
                "wrist_image_right": encode_jpeg(imgs[i, 2]),
            })

        df = pd.DataFrame(frames)
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow")
        print(f"  → {parquet_path} ({len(frames)} rows)")

        episode_info.append({
            "episode_index": ep_idx,
            "tasks": [args.task],
            "length": n_frames,
        })
        ep_start = ep_end

    # ---- 写 meta 文件 ----
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
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
            "state": {"dtype": "float32", "shape": [14], "names": s_names},
            "actions": {"dtype": "float32", "shape": [14], "names": a_names},
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
    tasks = [{"task_index": 0, "task": args.task}]
    (out / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )

    # episodes_stats.jsonl
    stats_entries = []
    ep_start = 0
    for ep in episode_info:
        n = ep["length"]
        s = state[ep_start: ep_start + n]
        a = actions[ep_start: ep_start + n]
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

    # stats.json
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

    print(f"\n[DONE] 数据集已输出到 {output_dir}/")
    print(f"  Mode:         {mode}")
    print(f"  Robot type:   {robot_type}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration:     {total_frames / args.fps:.1f}s @ {args.fps}fps")


if __name__ == "__main__":
    main()
