#!/usr/bin/env python3
"""
Piper 机械臂推理脚本 — 通过策略服务器控制机械臂。

本脚本连接到 openpi 策略服务器 (WebsocketPolicyServer)，读取 Piper 的关节角、
夹爪状态和相机图像，发送给模型推理，接收动作块并执行。

用法:
    # 启动策略服务器 (在 GPU 机器上)
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

    # 在机器人端运行推理
    python examples/piper/inference.py --host <GPU_SERVER_IP> --port 8000

    # RealSense D435i/D405 相机
    python examples/piper/inference.py --host <GPU_SERVER_IP> --port 8000 \
        --rs2_base 128422272318 --rs2_wrist 218722271368

    # OpenCV webcam 回退
    python examples/piper/inference.py --host 192.168.1.100 --port 8000 \
        --cam_ids 0 2

    # 交互模式：每次推理前输入新的 prompt
    python examples/piper/inference.py --host localhost --interactive

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电、处于从臂模式
    3. piper_sdk 已安装:  pip install piper_sdk
    4. openpi-client 已安装: cd packages/openpi-client && pip install -e .

控制流程:
    模型每次返回一个 action chunk (action_horizon, action_dim)。
    我们执行 chunk 的前 `exec_horizon` 步，然后重新查询模型。
    这类似于 receding horizon control / action chunking with temporal ensemble。

键盘控制:
    Enter   — 开始/继续推理
    s       — 暂停推理
    q       — 退出
    r       — 重置机械臂到初始位置
"""

import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# ===========================================================================
# 导入
# ===========================================================================

# --- Piper SDK ---
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

# --- openpi-client (策略服务器客户端) ---
try:
    from openpi_client import image_tools as _image_tools
    from openpi_client import websocket_client_policy
except ImportError:
    print("[ERROR] 无法导入 openpi_client，请先安装:")
    print("  cd packages/openpi-client && pip install -e .")
    sys.exit(1)

# --- 相机工具 (支持 RealSense D435i/D405 和 OpenCV) ---
from camera_utils import create_cameras


# ===========================================================================
# 常量 — 根据你的模型和任务调整
# ===========================================================================

# 默认动作块长度 (与模型 config 保持一致)
DEFAULT_ACTION_HORIZON = 50
# 每次执行多少步后再重新查询模型 (≤ action_horizon)
DEFAULT_EXEC_HORIZON = 25
# Piper 关节角: raw (0.001度) ↔ rad
RAW_TO_RAD = np.pi / 180.0 / 1000.0
RAD_TO_RAW = 180.0 * 1000.0 / np.pi
# 图像尺寸 (必须与模型训练时一致)
IMAGE_SIZE = (224, 224)
# 默认控制频率 (Hz)
CONTROL_FREQ = 50
# 默认速度百分比
DEFAULT_SPEED_PCT = 40
# 默认最大关节速度 (rad/s), 3 rad/s ≈ 100% speed
DEFAULT_MAX_JOINT_SPEED = 3.0
# 默认插值频率 (Hz), None=不插值, 建议 50-200
DEFAULT_INTERP_FREQ = 50
# 夹爪控制力矩
GRIPPER_EFFORT = 1000


# ===========================================================================
# 关节空间插值工具 (基于 demo_dual_replay.py _send_joint_cmd)
# ===========================================================================

def _shortest_delta(target: np.ndarray, start: np.ndarray) -> np.ndarray:
    """最短弧差 (处理 ±π 缠绕)。"""
    return (target - start + np.pi) % (2 * np.pi) - np.pi


def _joint_arrived(cur: float, target: float, step: float) -> bool:
    """检查关节是否已到达或将 overshoot 目标。"""
    if abs(step) < 1e-10:
        return True
    return (step > 0 and cur + step >= target) or \
           (step < 0 and cur + step <= target)


# ===========================================================================
# Piper 控制封装
# ===========================================================================

class PiperController:
    """封装 Piper SDK 的控制循环。"""

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

    # ---- 使能 ----

    def enable(self, speed_pct: int = DEFAULT_SPEED_PCT, max_acc: int = 200) -> bool:
        print("[Piper] 正在使能...")
        deadline = time.monotonic() + 10.0
        while not self._piper.EnablePiper():
            if time.monotonic() > deadline:
                print("[Piper] 使能超时! 检查机械臂状态。")
                return False
            time.sleep(0.01)
        self._enabled = True
        self.set_joint_mode(speed_pct)
        # 设置各关节最大加速度 (写入 flash，需延迟)
        for j in range(1, 7):
            self._piper.JointMaxAccConfig(j, max_acc)
            time.sleep(0.05)
        # 夹爪初始化
        self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x02, 0)
        time.sleep(0.05)
        self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x01, 0)
        self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x02, 0)
        time.sleep(0.05)
        self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x01, 0)
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

    # ---- 读取 ----

    def get_joints_rad(self) -> np.ndarray:
        """读取6个关节角 (弧度), shape=(6,), float32。"""
        joint_msg = self._piper.GetArmJointMsgs()
        js = joint_msg.joint_state  # ArmMsgFeedBackJointStates
        raw = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float32,
        )
        return raw * RAW_TO_RAD

    def get_gripper_raw(self) -> np.ndarray:
        """读取夹爪原始值 (0.001mm), shape=(1,), float32。"""
        gripper_msg = self._piper.GetArmGripperMsgs()
        return np.array([gripper_msg.gripper_state.grippers_angle], dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """完整状态: [j1..j6(rad), gripper(raw)], shape=(7,), float32。"""
        return np.concatenate([self.get_joints_rad(), self.get_gripper_raw()])

    # ---- 控制 ----

    def set_joint_mode(self, speed_pct: int = DEFAULT_SPEED_PCT):
        self._piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)

    def set_speed(self, speed_pct: int):
        """动态调整速度百分比 (20-100)。"""
        self._piper.MotionCtrl_2(0x01, 0x01, max(20, min(100, speed_pct)), 0x00)

    def send_joint_command(self, joints_rad: np.ndarray):
        """发送关节角指令，同时做限位保护。joints_rad.shape=(6,) 弧度。"""
        # clipped = np.clip(
        #     joints_rad[:6],
        #     JOINT_LIMITS_RAD[:, 0],
        #     JOINT_LIMITS_RAD[:, 1],
        # )
        raw = (joints_rad[:6] * RAD_TO_RAW).astype(int)
        self._piper.JointCtrl(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])

    def send_gripper_command(self, pos_raw: float, effort: int = GRIPPER_EFFORT):
        """发送夹爪指令。pos_raw 单位 0.001mm。"""
        self._piper.GripperCtrl(int(pos_raw), effort, 0x01, 0)

    def execute_action(self, action: np.ndarray, speed_pct: int = DEFAULT_SPEED_PCT):
        """执行单个动作。

        action.shape = (7,): [j1..j6(rad), gripper(raw_0.001mm)]
        """
        # self.set_joint_mode(speed_pct)
        self.send_joint_command(action[:6])
        self.send_gripper_command(action[6])

    def go_to_init_pose(self, max_joint_speed: float = DEFAULT_MAX_JOINT_SPEED,
                         interp_freq: float = DEFAULT_INTERP_FREQ):
        """使用插值平滑回到初始关节位姿 [-pi/2, 0, 0, 0, 0, 0]."""
        init_joints = np.array([-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cur = self.get_joints_rad().astype(np.float64)
        print(f"[Piper] 回到初始位姿: {np.array2string(init_joints, precision=3)}  "
              f"当前: {np.array2string(cur, precision=3)}")

        if interp_freq is not None:
            inner_dt = 1.0 / interp_freq
            delta = _shortest_delta(init_joints, cur)
            max_delta = np.max(np.abs(delta))
            if max_delta < 1e-6:
                print("[Piper] 已在初始位姿")
                return
            joint_spd = min(max_delta * 2.0, max_joint_speed)  # ~0.5s to target
            angle_per_tick_rad = joint_spd * inner_dt
            tick_frac = np.where(np.abs(delta) > 0, delta / max_delta, 0.0)
            speed_pct = int(joint_spd / 3.0 * 100)
            self.set_speed(speed_pct)
            while True:
                done = True
                step = tick_frac * angle_per_tick_rad
                for j in range(6):
                    if not _joint_arrived(cur[j], init_joints[j], step[j]):
                        cur[j] += step[j]
                        done = False
                    else:
                        cur[j] = init_joints[j]
                self.send_joint_command(cur)
                if done:
                    break
                time.sleep(inner_dt)
        else:
            for _ in range(50):
                self.set_joint_mode(20)
                self.send_joint_command(init_joints)
                time.sleep(0.02)

        self.send_gripper_command(0)
        print("[Piper] 已到初始位姿")


# ===========================================================================
# 推理引擎
# ===========================================================================

class PiperInference:
    """Piper 推理主循环。

    连接策略服务器，读取机器人状态和图像，执行 receding horizon control。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        can_name: str = "can0",
        rs2_base_serial: Optional[str] = None,
        rs2_wrist_serial: Optional[str] = None,
        cv_base_id: Optional[int] = None,
        cv_wrist_id: Optional[int] = None,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        exec_horizon: int = DEFAULT_EXEC_HORIZON,
        max_joint_speed: float = DEFAULT_MAX_JOINT_SPEED,
        interp_freq: Optional[float] = DEFAULT_INTERP_FREQ,
        default_prompt: str = "pick up the walnut and place it into the cup",
        interactive: bool = False,
    ):
        self._action_horizon = action_horizon
        self._exec_horizon = exec_horizon
        self._max_joint_speed = max_joint_speed
        self._interp_freq = interp_freq
        self._use_interp = interp_freq is not None
        self._default_prompt = default_prompt
        self._interactive = interactive
        self._period = 1.0 / CONTROL_FREQ

        # 连接
        self._robot = PiperController(can_name)
        self._policy_client: Optional[websocket_client_policy.WebsocketClientPolicy] = None
        self._policy_host = host
        self._policy_port = port

        # 相机 (优先 RealSense，回退 OpenCV)
        self._camera = create_cameras(
            base_serial=rs2_base_serial,
            wrist_serial=rs2_wrist_serial,
            base_cv_id=cv_base_id,
            wrist_cv_id=cv_wrist_id,
        )
        self._has_base_cam = bool(rs2_base_serial or cv_base_id is not None)
        self._has_wrist_cam = bool(rs2_wrist_serial or cv_wrist_id is not None)

        # 状态
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._inferring = False
        self._action_cache: Optional[np.ndarray] = None  # (action_horizon, action_dim)
        self._cache_step = 0

    # ======================== 运行 ========================

    def run(self):
        print("=" * 60)
        print("Piper 推理客户端 (策略服务器模式)")
        print(f"服务器:         {self._policy_host}:{self._policy_port}")
        print(f"控制频率:       {CONTROL_FREQ} Hz")
        if self._use_interp:
            print(f"关节插值:       启用 (max_speed={self._max_joint_speed:.1f} rad/s, freq={self._interp_freq} Hz)")
        else:
            print(f"关节插值:       关闭 (直接控制)")
        print("=" * 60)

        # 1. 使能机械臂
        if not self._robot.enable():
            return

        # 2. 启动相机
        if self._camera:
            self._camera.start()
            print(f"[Camera] 已启动")

        # 3. 连接策略服务器
        self._connect_policy()

        self._running = True

        # 4. 键盘监听
        input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        input_thread.start()

        try:
            self._control_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            if self._camera:
                self._camera.stop()
            cv2.destroyAllWindows()
            print("[INFO] 已退出 (机械臂保持使能)")

    def _connect_policy(self):
        """连接到策略服务器。"""
        print(f"[Policy] 正在连接 ws://{self._policy_host}:{self._policy_port} ...")
        self._policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._policy_host,
            port=self._policy_port,
        )
        meta = self._policy_client.get_server_metadata()
        print(f"[Policy] 已连接. 服务器元数据: {meta}")

    # ======================== 控制循环 ========================

    def _control_loop(self):
        """主控制循环 — 使用 joint-space 插值执行动作。"""
        fps_counter = _FPSCounter("control")
        # 当前关节角 (用于插值的起点)
        cur_joints: Optional[np.ndarray] = None
        # 数据步长: 1/50s 对应模型 50Hz 数据频率
        data_dt = 1.0 / CONTROL_FREQ

        while self._running:
            loop_start = time.monotonic()

            # 处理键盘事件
            try:
                cmd = self._input_queue.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                pass

            if not self._inferring:
                self._display_cameras()
                time.sleep(0.01)
                continue

            # --- 推理 + 动作执行 ---

            # 决定是否需要重新查询模型
            if self._action_cache is None or self._cache_step >= self._exec_horizon:
                self._query_policy()
                self._cache_step = 0
                # 新 chunk 开始时，从机械臂读取当前关节角
                cur_joints = self._robot.get_joints_rad().astype(np.float64)

            if self._action_cache is not None:
                # 从 action chunk 中取当前步
                idx = min(self._cache_step, self._action_cache.shape[0] - 1)
                action = self._action_cache[idx]

                target_joints = action[:6].astype(np.float64)
                gripper = float(action[6])

                # 使用插值发送关节指令 (如果 cur_joints 为 None 则从机械臂读取)
                if cur_joints is None:
                    cur_joints = self._robot.get_joints_rad().astype(np.float64)
                self._send_joint_cmd(cur_joints, target_joints, gripper, data_dt)
                cur_joints = target_joints.copy()

                self._cache_step += 1

                fps_counter.tick()

                # 每 5 步显示一次相机画面
                if fps_counter.count % 5 == 0:
                    self._display_cameras()

                # 定期打印状态
                if fps_counter.count % 100 == 0:
                    state = self._robot.get_state()
                    j_str = ", ".join(f"{s:.3f}" for s in state[:6])
                    g_str = f"{state[6]:.0f}"
                    a_str = ", ".join(f"{a:.3f}" for a in action[:6])
                    mode_str = "interp" if self._use_interp else "direct"
                    print(
                        f"[{fps_counter.count:5d}] fps={fps_counter.fps:.1f} | "
                        f"joints=[{j_str}] grip={g_str} | "
                        f"cmd=[{a_str}] grip={action[6]:.0f} | {mode_str}"
                    )

            # 控制频率
            elapsed = time.monotonic() - loop_start
            if elapsed < self._period:
                time.sleep(self._period - elapsed)

    # ======================== 关节空间控制 (基于 demo_dual_replay.py) ========================

    def _send_joint_cmd(
        self,
        cur_joints: np.ndarray,
        target_joints: np.ndarray,
        gripper: float,
        data_dt: float,
    ):
        """发送一个关节角目标，可选带步间插值。

        算法 (与 demo_dual_replay.py 相同):
            1. 计算 delta = target - cur (最短弧)
            2. joint_spd = min(|delta| * data_freq, max_joint_speed)
            3. 动态调整 MotionCtrl_2 速度百分比
            4. While 循环生成中间角度，每 tick 发送一次

        Parameters
        ----------
        cur_joints : ndarray (6,) float64 — 当前关节角 (rad)
        target_joints : ndarray (6,) float64 — 目标关节角 (rad)
        gripper : float — 目标夹爪位置 (raw 0.001mm)
        data_dt : float — 数据步长 (s)
        """
        delta = _shortest_delta(target_joints, cur_joints)
        max_delta = np.max(np.abs(delta))

        # 无插值模式: 直接发送
        if not self._use_interp:
            self._robot.send_joint_command(target_joints)
            self._robot.send_gripper_command(gripper)
            time.sleep(data_dt)
            return

        # 插值模式
        if max_delta < 1e-6:
            self._robot.send_gripper_command(gripper)
            return

        inner_dt = 1.0 / self._interp_freq

        # joint_spd = delta_rad * data_freq (rad/s), capped
        joint_spd = min(max_delta / data_dt, self._max_joint_speed)

        # 动态速度百分比 (3 rad/s ≈ 100%)
        speed_pct = max(20, min(100, int(joint_spd / 3.0 * 100)))
        self._robot.set_speed(speed_pct)

        # angle_per_tick in radians for the fastest joint
        angle_per_tick_rad = joint_spd * inner_dt

        # 各关节按比例推进
        tick_frac = np.where(np.abs(delta) > 0, delta / max_delta, 0.0)

        cur = cur_joints.copy()
        while True:
            done = True
            step = tick_frac * angle_per_tick_rad
            for j in range(6):
                if not _joint_arrived(cur[j], target_joints[j], step[j]):
                    cur[j] += step[j]
                    done = False
                else:
                    cur[j] = target_joints[j]
            self._robot.send_joint_command(cur)
            if done:
                break
            time.sleep(inner_dt)

        self._robot.send_gripper_command(gripper)

    def _display_cameras(self):
        """将所有相机画面拼接显示。"""
        if not self._camera:
            return

        frames = []
        if self._has_base_cam:
            img = self._camera.get_base()
            cv2.putText(img, "base", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img)
        if self._has_wrist_cam:
            img = self._camera.get_wrist()
            cv2.putText(img, "wrist", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img)

        if frames:
            vis = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
            cv2.imshow("Piper Inference", vis[..., ::-1])  # RGB → BGR
            cv2.waitKey(1)

    def _query_policy(self):
        """向策略服务器查询动作块。"""
        try:
            obs = self._build_observation()
            result = self._policy_client.infer(obs)
            self._action_cache = np.asarray(result["actions"], dtype=np.float32)
            # 打印服务器端耗时
            timings = result.get("policy_timing", {})
            server_timings = result.get("server_timing", {})
            if timings or server_timings:
                parts = []
                if timings:
                    parts.append(f"infer={timings.get('infer_ms', 0):.0f}ms")
                if server_timings:
                    parts.append(f"server={server_timings.get('total_ms', 0):.0f}ms")
                print(f"  [Policy] {' | '.join(parts)}")
        except Exception as e:
            print(f"[Policy] 查询失败: {e}")
            # 如果是连接断开，尝试重连
            time.sleep(1)
            try:
                self._connect_policy()
                print("[Policy] 重连成功")
            except Exception:
                print("[Policy] 重连失败，暂停推理")
                self._inferring = False

    def _build_observation(self) -> dict:
        """构建发送给策略服务器的 observation。"""
        state = self._robot.get_state()

        # 相机图像
        base_image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
        wrist_image = base_image.copy()
        if self._camera:
            if self._has_base_cam:
                base_image = self._camera.get_base()
            if self._has_wrist_cam:
                wrist_image = self._camera.get_wrist()

        # Prompt
        if self._interactive:
            prompt = input("指令: ").strip() or self._default_prompt
        else:
            prompt = self._default_prompt

        # 格式必须匹配策略服务器端配置的 transforms。
        # 对于 LIBERO 格式的模型: observation/state, observation/image, observation/wrist_image
        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "prompt": prompt,
        }

    # ======================== 键盘交互 ========================

    def _handle_command(self, cmd: str):
        if cmd == "start":
            if not self._inferring:
                print("[Control] 开始推理...")
                self._action_cache = None
                self._cache_step = 0
                self._inferring = True
        elif cmd == "stop":
            print("[Control] 暂停推理")
            self._inferring = False
            self._action_cache = None
            self._cache_step = 0
        elif cmd == "quit":
            self._running = False
        elif cmd == "reset":
            was_inferring = self._inferring
            self._inferring = False
            self._robot.go_to_init_pose(
                max_joint_speed=self._max_joint_speed,
                interp_freq=self._interp_freq,
            )
            self._inferring = was_inferring

    def _keyboard_listener(self):
        """后台键盘监听。"""
        print("\n操作提示:")
        print("  [Enter]  开始推理")
        print("  [s]      暂停")
        print("  [r]      重置到初始位姿")
        print("  [q]      退出\n")

        while self._running:
            try:
                ch = sys.stdin.readline().strip().lower()
                if ch == "":
                    self._input_queue.put("start")
                elif ch == "s":
                    self._input_queue.put("stop")
                elif ch == "q":
                    self._input_queue.put("quit")
                elif ch == "r":
                    self._input_queue.put("reset")
            except (EOFError, OSError):
                break


# ===========================================================================
# FPS 计数器
# ===========================================================================

class _FPSCounter:
    def __init__(self, label: str = ""):
        self.label = label
        self.count = 0
        self._last = time.monotonic()
        self.fps = 0.0

    def tick(self):
        self.count += 1
        now = time.monotonic()
        if now - self._last >= 1.0:
            self.fps = self.count / (now - self._last)
            self.count = 0
            self._last = now


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Piper 策略服务器推理客户端")
    p.add_argument("--host", default="localhost", help="策略服务器地址 (默认: localhost)")
    p.add_argument("--port", type=int, default=8000, help="策略服务器端口 (默认: 8000)")
    p.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")
    p.add_argument(
        "--rs2_base", default=None, help="D435i 基座相机序列号。先运行 'python camera_utils.py --list' 查看"
    )
    p.add_argument(
        "--rs2_wrist", default=None, help="D405 腕部相机序列号。先运行 'python camera_utils.py --list' 查看"
    )
    p.add_argument(
        "--cam_ids", type=int, nargs="*", default=[], help="OpenCV 设备 ID 回退。第一个=基座，第二个=腕部"
    )
    p.add_argument(
        "--action_horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=f"动作块长度 (默认: {DEFAULT_ACTION_HORIZON})",
    )
    p.add_argument(
        "--exec_horizon",
        type=int,
        default=DEFAULT_EXEC_HORIZON,
        help=f"每次执行步数再重新推理 (默认: {DEFAULT_EXEC_HORIZON})",
    )
    p.add_argument(
        "--prompt",
        default="pick up the pen and place it into the cup",
        help="默认语言指令",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式: 每次推理前手动输入指令",
    )
    p.add_argument(
        "--max_joint_speed",
        type=float,
        default=DEFAULT_MAX_JOINT_SPEED,
        help=f"最大关节速度 rad/s (默认: {DEFAULT_MAX_JOINT_SPEED})",
    )
    p.add_argument(
        "--interp_freq",
        type=float,
        default=DEFAULT_INTERP_FREQ,
        help=f"插值频率 Hz (默认: {DEFAULT_INTERP_FREQ}). 设为 0 禁用插值",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    inference = PiperInference(
        host=args.host,
        port=args.port,
        can_name=args.can_name,
        rs2_base_serial=args.rs2_base,
        rs2_wrist_serial=args.rs2_wrist,
        cv_base_id=args.cam_ids[0] if len(args.cam_ids) > 0 else None,
        cv_wrist_id=args.cam_ids[1] if len(args.cam_ids) > 1 else None,
        action_horizon=args.action_horizon,
        exec_horizon=args.exec_horizon,
        max_joint_speed=args.max_joint_speed,
        interp_freq=args.interp_freq if args.interp_freq > 0 else None,
        default_prompt=args.prompt,
        interactive=args.interactive,
    )
    inference.run()


if __name__ == "__main__":
    main()
