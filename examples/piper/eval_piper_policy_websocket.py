#!/usr/bin/env python3
"""
Piper 机械臂 WebSocket 推理脚本 — 使用 openpi 策略服务器。

完全仿照 /home/rhr/diffusion_policy_piper/eval_piper_policy_diffcontrol.py
的多进程架构，但推理引擎替换为 openpi WebSocket 策略服务器 (pi05_piper_lora)。

架构:
    SharedMemoryManager (顶层)
    ├── MultiRealsense → SingleRealsense × N (每个相机一个独立 mp.Process)
    └── PiperJointController (独立 mp.Process, 关节空间控制)

用法:
    # 启动策略服务器 (GPU 机器)
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi05_piper_lora \\
        --policy.dir=<checkpoint_path>

    # 直接模式 (无插值, 最快响应)
    python examples/piper/eval_piper_policy_websocket.py \\
        --host <GPU_IP> --port 8000 --can_name can0

    # 插值模式 (平滑运动)
    python examples/piper/eval_piper_policy_websocket.py \\
        --host <GPU_IP> --port 8000 --can_name can0 \\
        --max_joint_speed 1.0

    # 指定相机序列号
    python examples/piper/eval_piper_policy_websocket.py \\
        --host <GPU_IP> --port 8000 \\
        --camera_serials 128422272318 218722271368

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电、处于从臂模式
    3. piper_sdk 已安装:  pip install piper_sdk
    4. openpi-client 已安装:  cd packages/openpi-client && pip install -e .
    5. diffusion_policy_piper 在 /home/rhr/diffusion_policy_piper

键盘控制:
    c       — 开始推理
    s       — 停止当前 episode
    r       — 重置机械臂到初始关节角
    q       — 退出
"""

import os
import sys
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import click
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径设置 — 复用 diffusion_policy_piper 的组件
# ---------------------------------------------------------------------------
_DIFFUSION_POLICY_ROOT = "/home/rhr/diffusion_policy_piper"
if _DIFFUSION_POLICY_ROOT not in sys.path:
    sys.path.insert(0, _DIFFUSION_POLICY_ROOT)

# Piper SDK
_PIPER_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "piper_sdk")
if _PIPER_SDK_PATH not in sys.path:
    sys.path.insert(0, _PIPER_SDK_PATH)

# openpi-client
_OPENPI_CLIENT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "packages", "openpi-client", "src"
)
if _OPENPI_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _OPENPI_CLIENT_PATH)

# ---------------------------------------------------------------------------
# 复用 diffusion_policy_piper 组件
# ---------------------------------------------------------------------------
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.precise_sleep import precise_wait

# ---------------------------------------------------------------------------
# openpi-client
# ---------------------------------------------------------------------------
try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("[ERROR] 无法导入 openpi_client，请先安装:")
    print("  cd packages/openpi-client && pip install -e .")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 自定义关节控制器
# ---------------------------------------------------------------------------
from piper_joint_controller import PiperJointController

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 模型期望图像尺寸 (服务器端 ResizeImages 也会 resize，但提前 resize 减少带宽)
MODEL_IMAGE_SIZE = (224, 224)

# 默认 action 参数 (与 pi05_piper_lora config 一致)
DEFAULT_ACTION_HORIZON = 10
DEFAULT_ACTION_DIM = 7  # 6 joints + 1 gripper

# 默认控制频率 (策略调用频率)
DEFAULT_FREQUENCY = 10  # Hz
DEFAULT_EXEC_HORIZON = 8

# 录制默认参数
DEFAULT_VIDEO_CAPTURE_FPS = 30
DEFAULT_VIDEO_CAPTURE_RESOLUTION = (640, 480)

# 初始关节位姿 (安全位姿, 弧度)
DEFAULT_INIT_JOINTS = np.array(
    [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)


# ===========================================================================
# 主推理类
# ===========================================================================


class PiperWebsocketInference:
    """Piper 推理主循环。

    管理相机进程 (MultiRealsense)、关节控制器进程 (PiperJointController)、
    WebSocket 策略客户端，以及控制循环的时序和键盘交互。
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 8000,
        can_name: str = "can0",
        camera_serials: Optional[list[str]] = None,
        frequency: float = DEFAULT_FREQUENCY,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        exec_horizon: int = DEFAULT_EXEC_HORIZON,
        max_duration: float = 60.0,
        max_joint_speed: Optional[float] = None,
        speed_pct: int = 40,
        prompt: str = "do something",
        output_dir: Optional[str] = None,
        camera_resolution: tuple[int, int] = DEFAULT_VIDEO_CAPTURE_RESOLUTION,
        video_capture_fps: int = DEFAULT_VIDEO_CAPTURE_FPS,
        no_camera: bool = False,
        vis_camera_idx: int = 0,
        init_joints: np.ndarray = DEFAULT_INIT_JOINTS,
    ):
        self.host = host
        self.port = port
        self.can_name = can_name
        self.frequency = frequency
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.max_duration = max_duration
        self.max_joint_speed = max_joint_speed
        self.speed_pct = speed_pct
        self.prompt = prompt
        self.no_camera = no_camera
        self.vis_camera_idx = vis_camera_idx
        self.init_joints = init_joints
        self.dt = 1.0 / frequency

        cam_w, cam_h = camera_resolution

        # ---- 录制输出 ----
        self._recording = output_dir is not None
        if self._recording:
            output_path = pathlib.Path(output_dir)
            assert output_path.parent.is_dir(), f"输出目录不存在: {output_path.parent}"
            output_path.mkdir(parents=True, exist_ok=True)
            self.output_dir = output_path
            self.video_dir = output_path / "videos"
            self.video_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            self.video_dir = None

        # ---- 图像变换: 采集分辨率 → 模型分辨率 ----
        color_tf = get_image_transform(
            input_res=(cam_w, cam_h),
            output_res=MODEL_IMAGE_SIZE,
            bgr_to_rgb=True,  # SingleRealsense 输出 BGR → 转为 RGB
        )

        def _transform(data: dict) -> dict:
            data["color"] = color_tf(data["color"])
            return data

        # ---- 相机 ----
        if not no_camera:
            if camera_serials is None:
                camera_serials = SingleRealsense.get_connected_devices_serial()
                print(f"[Camera] 自动检测到相机: {camera_serials}")

            if not camera_serials:
                print("[Camera] 未检测到相机，将使用无相机模式")
                self._camera_serials = []
            else:
                self._camera_serials = camera_serials
        else:
            self._camera_serials = []

        self._has_cameras = len(self._camera_serials) > 0

        # ---- 视频录制器 ----
        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = "bgr24"

        self.video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
            crf=21,
            thread_type="FRAME",
            thread_count=2,
        )

        # ---- SharedMemoryManager (顶层) ----
        self._own_shm_manager = True
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # ---- MultiRealsense ----
        if self._has_cameras:
            self.realsense = MultiRealsense(
                serial_numbers=self._camera_serials,
                shm_manager=self.shm_manager,
                resolution=(cam_w, cam_h),
                capture_fps=video_capture_fps,
                put_fps=video_capture_fps,
                put_downsample=False,
                record_fps=recording_fps,
                enable_color=True,
                enable_depth=False,
                enable_infrared=False,
                get_max_k=30,
                transform=_transform,
                vis_transform=None,
                recording_transform=recording_transform,
                video_recorder=self.video_recorder,
                verbose=False,
            )
        else:
            self.realsense = None

        # ---- PiperJointController ----
        self.robot = PiperJointController(
            shm_manager=self.shm_manager,
            can_name=can_name,
            frequency=50,  # 内部控制循环 50Hz
            max_joint_speed=max_joint_speed,
            speed_pct=speed_pct,
            launch_timeout=10,
            init_joints=None,
            soft_real_time=False,
            verbose=False,
            get_max_k=128,
        )

        # ---- WebSocket 策略客户端 ----
        self._policy_client: Optional[
            websocket_client_policy.WebsocketClientPolicy
        ] = None

        # ---- 运行时状态 ----
        self._running = False
        self._inferring = False
        self._action_cache: Optional[np.ndarray] = None  # (action_horizon, 7)
        self._cache_step = 0
        self._last_obs: Optional[dict] = None
        self._episode_start_time: Optional[float] = None

        # ---- 录制状态 ----
        self._n_episodes = 0
        self._recording_active = False

    # ======================== 运行入口 ========================

    def run(self):
        print("=" * 60)
        print("Piper WebSocket 推理客户端")
        print(f"  策略服务器:   ws://{self.host}:{self.port}")
        print(f"  CAN 接口:     {self.can_name}")
        print(f"  控制频率:     {self.frequency} Hz")
        print(f"  Action块长:   {self.action_horizon}")
        print(f"  执行步数:     {self.exec_horizon}")
        if self.max_joint_speed:
            print(f"  关节插值:     启用 (max_speed={self.max_joint_speed} rad/s)")
        else:
            print(f"  关节插值:     关闭 (直接控制)")
        print(f"  相机数量:     {len(self._camera_serials)}")
        if self._camera_serials:
            for i, sn in enumerate(self._camera_serials):
                print(f"    camera_{i}: {sn}")
        print(f"  录制:         {'开' if self._recording else '关'}")
        print("=" * 60)

        cv2.setNumThreads(1)

        # 1. 启动机械臂控制器进程
        print("[Robot] 正在启动关节控制器...")
        self.robot.start(wait=True)
        print("[Robot] 关节控制器就绪")

        # 2. 启动相机
        if self.realsense is not None:
            print("[Camera] 正在启动相机...")
            self.realsense.start(wait=True)
            # 设置曝光和白平衡
            self.realsense.set_exposure(exposure=200, gain=0)
            self.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)
            print("[Camera] 相机就绪")

        # 3. 连接策略服务器
        self._connect_policy()

        # 4. 预热推理
        print("[Policy] 预热推理 (首次可能包含 JIT/编译)...")
        self._warmup_inference()

        self._running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            # 停止录制
            if self._recording_active:
                self._stop_recording()

            # 停止相机
            if self.realsense is not None:
                self.realsense.stop(wait=True)

            # 停止机械臂 (保持使能)
            self.robot.stop(wait=True)

            print("[INFO] 已退出 (机械臂保持使能)")

    # ======================== 策略服务器连接 ========================

    def _connect_policy(self):
        print(f"[Policy] 正在连接 ws://{self.host}:{self.port} ...")
        self._policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host,
            port=self.port,
        )
        meta = self._policy_client.get_server_metadata()
        print(f"[Policy] 已连接. 服务器元数据: {meta}")

    def _warmup_inference(self):
        """预热推理：发送一次假 observation 触发 JIT/编译。"""
        fake_obs = {
            "observation/state": np.zeros(7, dtype=np.float32),
            "observation/image": np.zeros(
                (MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8
            ),
            "observation/wrist_image": np.zeros(
                (MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8
            ),
            "prompt": self.prompt,
        }
        try:
            result = self._policy_client.infer(fake_obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            print(f"[Policy] 预热完成. action shape={actions.shape}")
        except Exception as e:
            print(f"[Policy] 预热失败: {e}")
            raise

    # ======================== 主循环 ========================

    def _main_loop(self):
        """主循环：idle loop + policy loop。"""
        command_latency = 0.01
        frame_latency = 1.0 / 50

        while self._running:
            # ============ idle loop ============
            t_start = time.monotonic()
            iter_idx = 0

            while True:
                # 取 obs
                if self._has_cameras:
                    obs = self._get_raw_obs()
                else:
                    obs = self._get_robot_state_only()

                t_cycle_end = t_start + (iter_idx + 1) * self.dt
                t_sample = t_cycle_end - command_latency

                # 显示相机画面
                if self._has_cameras and not self.no_camera:
                    vis_img = self._get_vis_image(obs)
                    cv2.putText(
                        vis_img,
                        f"Ep:{self._n_episodes} | Press C to start, Q to quit",
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("eval", vis_img[..., ::-1])  # RGB → BGR for cv2

                # 键盘
                key = cv2.pollKey() & 0xFF
                if key == ord("c"):
                    print("[Control] 开始推理...")
                    break
                elif key == ord("q"):
                    print("[Control] 退出.")
                    self._running = False
                    return
                elif key == ord("r"):
                    self._reset_robot()

                precise_wait(t_sample)
                precise_wait(t_cycle_end)
                iter_idx += 1

            # ============ policy loop ============
            try:
                start_delay = 1.0
                eval_t_start = time.time() + start_delay
                t_start = time.monotonic() + start_delay

                # 开始录制 episode
                if self._recording:
                    self._start_recording(eval_t_start)

                precise_wait(eval_t_start - frame_latency, time_func=time.time)
                print(
                    f"[Episode {self._n_episodes}] 开始! "
                    f"按 's' 停止, 'q' 退出"
                )

                iter_idx = 0
                iter_cycle = 0
                steps_per_inference_cycle = self.exec_horizon
                self._action_cache = None
                self._cache_step = 0

                while True:
                    t_cycle_end = (
                        t_start + (iter_cycle + steps_per_inference_cycle) * self.dt
                    )

                    # ---- 取 obs ----
                    if self._has_cameras:
                        raw_obs = self._get_raw_obs()
                    else:
                        raw_obs = self._get_robot_state_only()

                    obs_timestamp = (
                        raw_obs["timestamp"][-1]
                        if "timestamp" in raw_obs
                        else time.time()
                    )

                    # ---- 推理 ----
                    t_infer_start = time.time()
                    obs_msg = self._build_observation(raw_obs)

                    try:
                        result = self._policy_client.infer(obs_msg)
                        self._action_cache = np.asarray(
                            result["actions"], dtype=np.float32
                        )
                        infer_ms = (time.time() - t_infer_start) * 1000

                        server_timing = result.get("server_timing", {})
                        policy_timing = result.get("policy_timing", {})
                        timing_parts = [f"client_infer={infer_ms:.0f}ms"]
                        if policy_timing:
                            timing_parts.append(
                                f"model={policy_timing.get('infer_ms', 0):.0f}ms"
                            )
                        if server_timing:
                            timing_parts.append(
                                f"srv_total={server_timing.get('prev_total_ms', 0):.0f}ms"
                            )
                        print(f"  [Policy] {' | '.join(timing_parts)}")
                    except Exception as e:
                        print(f"[Policy] 推理失败: {e}")
                        time.sleep(1)
                        try:
                            self._connect_policy()
                            print("[Policy] 重连成功")
                        except Exception:
                            print("[Policy] 重连失败")
                        break

                    # ---- 计算 action timestamps ----
                    actions = self._action_cache  # (action_horizon, 7)
                    n_steps = len(actions)

                    action_timestamps = (
                        np.arange(n_steps, dtype=np.float64) * self.dt
                        + obs_timestamp
                    )

                    # ---- 过滤已过期的时间戳 ----
                    curr_time = time.time()
                    action_exec_latency = 0.01
                    is_new = action_timestamps > (curr_time + action_exec_latency)

                    if np.sum(is_new) == 0:
                        # 全部过期，发送最后一步
                        target_actions = actions[[-1]]
                        next_step = int(np.ceil((curr_time - eval_t_start) / self.dt))
                        target_timestamps = np.array(
                            [eval_t_start + next_step * self.dt]
                        )
                        print("[WARNING] Over budget")
                    else:
                        target_actions = actions[is_new]
                        target_timestamps = action_timestamps[is_new]

                    # ---- 重叠窗口: 第一周期发送全部，后续周期跳过重叠部分 ----
                    if iter_cycle == 0:
                        pass  # 发送全部
                    else:
                        skip = self.action_horizon - self.exec_horizon
                        if skip > 0 and len(target_actions) > skip:
                            target_actions = target_actions[skip:]
                            target_timestamps = target_timestamps[skip:]

                    # ---- 执行动作 ----
                    for i in range(len(target_actions)):
                        action = target_actions[i]
                        target_time = target_timestamps[i]
                        joints = action[:6].astype(np.float64)   # rad
                        gripper = float(action[6])                # raw 0.001mm

                        self.robot.schedule_waypoint(
                            joints=joints,
                            target_time=target_time,
                            gripper=gripper,
                            gripper_effort=1.5,
                        )

                    gripper_str = f"{target_actions[0, 6]:.0f}"
                    joints_str = ", ".join(
                        f"{target_actions[0, i]:.3f}" for i in range(6)
                    )
                    print(
                        f"  [Action] {len(target_actions)} steps | "
                        f"joints=[{joints_str}] grip={gripper_str}"
                    )

                    # ---- 可视化 + 键盘 ----
                    stop_episode = False
                    if self._has_cameras and not self.no_camera:
                        vis_img = self._get_vis_image(raw_obs)
                        t_elapsed = time.monotonic() - t_start
                        text = (
                            f"Ep:{self._n_episodes} "
                            f"T:{t_elapsed:.1f}s | S=stop Q=quit"
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255),
                        )
                        cv2.imshow("eval", vis_img[..., ::-1])
                        key = cv2.pollKey() & 0xFF
                        if key == ord("s"):
                            print("[Control] 用户停止 episode")
                            stop_episode = True
                        elif key == ord("q"):
                            print("[Control] 退出.")
                            self._running = False
                            if self._recording_active:
                                self._stop_recording()
                            return
                    else:
                        key = cv2.pollKey() & 0xFF
                        if key == ord("s"):
                            print("[Control] 用户停止 episode")
                            stop_episode = True
                        elif key == ord("q"):
                            print("[Control] 退出.")
                            self._running = False
                            if self._recording_active:
                                self._stop_recording()
                            return

                    # ---- 自动终止 ----
                    if time.monotonic() - t_start > self.max_duration:
                        print(f"[Episode {self._n_episodes}] 超时 ({self.max_duration}s)")
                        stop_episode = True

                    if stop_episode:
                        if self._recording_active:
                            self._stop_recording()
                        print(f"[Episode {self._n_episodes}] 结束")
                        break

                    # ---- 频率控制 ----
                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += self.action_horizon
                    iter_cycle += steps_per_inference_cycle

            except KeyboardInterrupt:
                print("\n[INFO] 中断信号")
                if self._recording_active:
                    self._stop_recording()
                break

    # ======================== Observation 处理 ========================

    def _get_raw_obs(self) -> dict:
        """从 MultiRealsense 和 PiperJointController 获取原始 observation。

        Returns
        -------
        dict 包含:
            camera_0, camera_1, ... : 各相机最新帧 (H, W, 3) uint8 RGB
            joint_angles: (6,) float64 — 当前关节角 (rad)
            gripper_angle: float — 当前夹爪位置 (raw 0.001mm)
            timestamp: float — 时间戳
        """
        # 相机: 取最新帧
        k = max(1, int(self.dt * 30))  # 取覆盖 dt 时间窗口的帧
        camera_data = self.realsense.get(k=k)

        # 机器人状态: 取最新
        robot_state = self.robot.get_state()

        obs = {}
        for cam_idx, cam_dict in camera_data.items():
            # camera_dict['color'] shape: (T, H, W, 3) uint8 RGB (经过 transform)
            obs[f"camera_{cam_idx}"] = cam_dict["color"][-1]  # 最新帧

        obs["joint_angles"] = robot_state["joint_angles"]        # (6,) rad
        obs["gripper_angle"] = robot_state["gripper_angle"]      # raw 0.001mm
        obs["timestamp"] = time.time()

        return obs

    def _get_robot_state_only(self) -> dict:
        """无相机时仅返回机器人状态。"""
        robot_state = self.robot.get_state()
        return {
            "joint_angles": robot_state["joint_angles"],
            "gripper_angle": robot_state["gripper_angle"],
            "timestamp": time.time(),
        }

    def _get_vis_image(self, obs: dict) -> np.ndarray:
        """获取可视化图像。"""
        key = f"camera_{self.vis_camera_idx}"
        if key in obs:
            return obs[key].copy()
        # fallback to first camera
        for k in obs:
            if k.startswith("camera_"):
                return obs[k].copy()
        return np.zeros((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8)

    def _build_observation(self, raw_obs: dict) -> dict:
        """将 raw obs 转换为 openpi WebSocket 服务器期望的格式。

        raw_obs 中的相机图像已经过 transform (BGR→RGB + resize 224×224)。
        """
        # 基座相机 (camera_0 = D435i)
        base_img = raw_obs.get("camera_0")
        if base_img is None:
            base_img = np.zeros(
                (MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8
            )

        # 腕部相机 (camera_1 = D405)
        wrist_img = raw_obs.get("camera_1")
        if wrist_img is None:
            wrist_img = np.zeros(
                (MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8
            )

        # 状态: [j1..j6(rad), gripper(raw)]
        joints = np.asarray(raw_obs["joint_angles"], dtype=np.float32)  # (6,) rad
        gripper = np.float32(raw_obs["gripper_angle"])                   # raw
        state = np.concatenate([joints, [gripper]]).astype(np.float32)   # (7,)

        return {
            "observation/state": state,
            "observation/image": np.asarray(base_img, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_img, dtype=np.uint8),
            "prompt": self.prompt,
        }

    # ======================== 机器人控制 ========================

    def _reset_robot(self):
        """重置机械臂到初始关节角。"""
        print("[Robot] 重置到初始关节位姿...")
        was_inferring = self._inferring
        self._inferring = False

        for _ in range(100):
            self.robot.schedule_waypoint(
                joints=self.init_joints,
                target_time=time.time() + 0.05,
                gripper=0,
                gripper_effort=1.5,
            )
            time.sleep(0.02)

        self._inferring = was_inferring
        print("[Robot] 重置完成")

    # ======================== 录制管理 ========================

    def _start_recording(self, start_time: float):
        """开始录制当前 episode。"""
        if not self._recording or self.realsense is None:
            return

        n_cameras = self.realsense.n_cameras
        video_paths = [
            str(self.video_dir / f"{self._n_episodes}_{i}.mp4")
            for i in range(n_cameras)
        ]

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)
        self._recording_active = True
        self._episode_start_time = start_time

    def _stop_recording(self):
        """停止录制。"""
        if not self._recording_active:
            return

        if self.realsense is not None:
            self.realsense.stop_recording()

        self._recording_active = False
        self._n_episodes += 1
        print(f"[Record] Episode {self._n_episodes - 1} 录制完成")


# ===========================================================================
# CLI
# ===========================================================================


@click.command()
@click.option("--host", default="localhost", help="策略服务器地址 (默认: localhost)")
@click.option("--port", type=int, default=8000, help="策略服务器端口 (默认: 8000)")
@click.option("--can_name", default="can0", help="CAN 接口名称 (默认: can0)")
@click.option(
    "--camera_serials",
    "-cs",
    multiple=True,
    default=None,
    help="相机序列号 (可多次指定). 不指定则自动检测. "
    "camera_0=D435i(全局), camera_1=D405(腕部1), camera_2=D405(腕部2)",
)
@click.option(
    "--frequency", "-f", type=float, default=DEFAULT_FREQUENCY, help="控制频率 Hz (默认: 10)"
)
@click.option(
    "--action_horizon",
    type=int,
    default=DEFAULT_ACTION_HORIZON,
    help=f"动作块长度 (默认: {DEFAULT_ACTION_HORIZON})",
)
@click.option(
    "--exec_horizon",
    type=int,
    default=DEFAULT_EXEC_HORIZON,
    help=f"每次执行步数再重新推理 (默认: {DEFAULT_EXEC_HORIZON})",
)
@click.option(
    "--max_duration",
    "-md",
    type=float,
    default=60.0,
    help="单次 episode 最长秒数 (默认: 60)",
)
@click.option(
    "--max_joint_speed",
    type=float,
    default=None,
    help="关节空间最大速度 rad/s (可选). 设置后启用插值平滑运动.",
)
@click.option(
    "--speed_pct",
    type=int,
    default=40,
    help="机械臂速度百分比 0-100 (默认: 40)",
)
@click.option(
    "--prompt",
    default="pick up the object",
    help="语言指令",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="录制输出目录 (可选, 设置后启用录制)",
)
@click.option(
    "--camera_resolution",
    "-cr",
    default="640x480",
    help="采集分辨率 WxH (默认: 640x480)",
)
@click.option(
    "--video_capture_fps",
    type=int,
    default=DEFAULT_VIDEO_CAPTURE_FPS,
    help=f"相机采集帧率 (默认: {DEFAULT_VIDEO_CAPTURE_FPS})",
)
@click.option(
    "--no_camera",
    is_flag=True,
    default=False,
    help="无相机模式",
)
@click.option(
    "--vis_camera_idx",
    type=int,
    default=0,
    help="可视化相机索引 (默认: 0)",
)
def main(
    host,
    port,
    can_name,
    camera_serials,
    frequency,
    action_horizon,
    exec_horizon,
    max_duration,
    max_joint_speed,
    speed_pct,
    prompt,
    output,
    camera_resolution,
    video_capture_fps,
    no_camera,
    vis_camera_idx,
):
    cam_w, cam_h = [int(x) for x in camera_resolution.split("x")]

    # camera_serials 是 tuple，None 表示自动检测
    serials = list(camera_serials) if camera_serials else None

    inference = PiperWebsocketInference(
        host=host,
        port=port,
        can_name=can_name,
        camera_serials=serials,
        frequency=frequency,
        action_horizon=action_horizon,
        exec_horizon=exec_horizon,
        max_duration=max_duration,
        max_joint_speed=max_joint_speed,
        speed_pct=speed_pct,
        prompt=prompt,
        output_dir=output,
        camera_resolution=(cam_w, cam_h),
        video_capture_fps=video_capture_fps,
        no_camera=no_camera,
        vis_camera_idx=vis_camera_idx,
    )
    inference.run()


if __name__ == "__main__":
    main()
