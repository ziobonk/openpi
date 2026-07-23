"""
Piper arm controller using piper_sdk (C_PiperInterface_V2) — joint-space version.

Runs a control loop in a separate process. Mirrors PiperInterpolationController's
architecture exactly (SharedMemoryRingBuffer + SharedMemoryQueue + mp.Process),
but uses joint-space control (JointCtrl + GripperCtrl) instead of Cartesian
(EndPoseCtrl).

Optionally supports joint-space trajectory interpolation via
JointTrajectoryInterpolator for smooth motion when `max_joint_speed` is set.

Commands (via SharedMemoryQueue):
    SCHEDULE_WAYPOINT  — schedule a joint-space waypoint at absolute wall time
    SET_GRIPPER        — set gripper width / effort
    STOP               — exit the control loop

State (via SharedMemoryRingBuffer):
    joint_angles, gripper_angle, robot_receive_timestamp
"""

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# diffusion_policy shared-memory primitives (reused from diffusion_policy_piper)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, "/home/rhr/diffusion_policy_piper")

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue,
    Empty,
)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

# Optional: joint-space trajectory interpolator
try:
    from diffusion_policy.common.joint_trajectory_interpolator import (
        JointTrajectoryInterpolator,
    )
    HAS_JOINT_INTERP = True
except ImportError:
    HAS_JOINT_INTERP = False

# ---------------------------------------------------------------------------
# Piper SDK
# ---------------------------------------------------------------------------
try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("piper_sdk 未安装. 请: cd piper_sdk && pip install .")

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
RAD_TO_001DEG = 1000.0 * 180.0 / np.pi   # rad → 0.001 deg (piper internal)
_001DEG_TO_RAD = np.pi / (1000.0 * 180.0)  # 0.001 deg → rad
M_TO_001MM = 1000000.0                     # m → 0.001 mm
_001MM_TO_M = 1e-6                         # 0.001 mm → m


class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    SET_GRIPPER = 2


class PiperJointController(mp.Process):
    """Piper joint-space controller in a separate process.

    Mirrors PiperInterpolationController's process architecture exactly:
    - Inherits mp.Process
    - SharedMemoryRingBuffer for streaming state output
    - SharedMemoryQueue for receiving commands
    - mp.Event for ready signal

    Uses piper_sdk's JointCtrl + GripperCtrl (joint-space control).
    Optionally applies JointTrajectoryInterpolator for smooth motion.

    Parameters
    ----------
    shm_manager : SharedMemoryManager
        Started SharedMemoryManager for inter-process communication.
    can_name : str
        CAN interface name (e.g. 'can0').
    frequency : float
        Control loop frequency in Hz (default: 50).
    max_joint_speed : float or None
        If set, enables JointTrajectoryInterpolator with this max speed (rad/s L2).
        If None, commands are sent directly without interpolation.
    speed_pct : int
        Joint motion speed percentage (0-100).
    launch_timeout : float
        Process startup timeout in seconds.
    init_joints : list[float] or None
        Initial joint angles in radians for homing (optional).
    soft_real_time : bool
        Enable SCHED_RR scheduling.
    verbose : bool
    get_max_k : int
        Ring buffer capacity for state history.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        can_name: str = "can0",
        frequency: float = 50,
        max_joint_speed: Optional[float] = None,
        speed_pct: int = 40,
        launch_timeout: float = 10,
        init_joints: Optional[list[float]] = None,
        soft_real_time: bool = False,
        verbose: bool = False,
        get_max_k: int = 128,
    ):
        assert 0 < frequency <= 200
        if max_joint_speed is not None:
            assert max_joint_speed > 0
            assert HAS_JOINT_INTERP, (
                "JointTrajectoryInterpolator 未找到，"
                "请确认 diffusion_policy_piper 路径正确"
            )
        if init_joints is not None:
            init_joints = np.array(init_joints)
            assert init_joints.shape == (6,)

        super().__init__(name="PiperJointController")
        self.can_name = can_name
        self.frequency = frequency
        self.max_joint_speed = max_joint_speed
        self.speed_pct = speed_pct
        self.launch_timeout = launch_timeout
        self.init_joints = init_joints
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # ---- input queue ----
        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "joints": np.zeros((6,), dtype=np.float64),
            "gripper": np.float64(0.0),       # raw 0.001mm units
            "gripper_effort": np.float64(1.5),
            "target_time": np.float64(0.0),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256,
        )

        # ---- ring buffer (state output) ----
        example = {
            "joint_angles": np.zeros(6, dtype=np.float64),
            "gripper_angle": np.float64(0.0),
            "robot_receive_timestamp": np.float64(time.time()),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # =========== launch / stop ===========

    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperJointController] 进程已启动, PID={self.pid}")

    def stop(self, wait: bool = True):
        self.input_queue.put({"cmd": Command.STOP.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # =========== command API (called from main process) ===========

    def schedule_waypoint(
        self,
        joints: np.ndarray,
        target_time: float,
        gripper: Optional[float] = None,
        gripper_effort: float = 1.5,
    ):
        """Schedule a joint-space waypoint at an absolute wall-clock time.

        Parameters
        ----------
        joints : ndarray (6,) float
            Target joint angles in radians.
        target_time : float
            Absolute wall-clock arrival time (time.time() frame).
        gripper : float or None
            If set, also command gripper to this raw position (0.001mm).
        gripper_effort : float
            Gripper effort in Newtons.
        """
        assert self.is_alive()
        joints = np.asarray(joints, dtype=np.float64)
        assert joints.shape == (6,)
        self.input_queue.put({
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "joints": joints,
            "gripper": np.float64(gripper) if gripper is not None else np.float64(0.0),
            "gripper_effort": np.float64(gripper_effort),
            "target_time": np.float64(target_time),
        })

    def set_gripper(self, angle: float, effort: float = 1.5):
        """Set gripper width immediately.

        Parameters
        ----------
        angle : float
            Gripper width in raw 0.001mm units.
        effort : float
            Gripper force in Newtons.
        """
        assert self.is_alive()
        self.input_queue.put({
            "cmd": Command.SET_GRIPPER.value,
            "joints": np.zeros(6, dtype=np.float64),
            "gripper": np.float64(angle),
            "gripper_effort": np.float64(effort),
            "target_time": np.float64(0.0),
        })

    # =========== receive API (called from main process) ===========

    def get_state(self, k: Optional[int] = None, out=None) -> dict:
        """Get latest state(s) from ring buffer."""
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self) -> dict:
        """Get all available state history."""
        return self.ring_buffer.get_all()

    # =========== main loop (runs in child process) ===========

    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        piper = C_PiperInterface_V2(can_name=self.can_name)
        piper.ConnectPort()

        try:
            if self.verbose:
                print(f"[PiperJointController] 连接 {self.can_name} ...")

            # Wait for connection
            connect_timeout = time.monotonic() + 10
            while not piper.get_connect_status():
                if time.monotonic() > connect_timeout:
                    print("[PiperJointController] WARNING: 连接超时")
                    break
                time.sleep(0.01)

            # Enable arm
            if self.verbose:
                print("[PiperJointController] 使能中...")
            enable_timeout = time.monotonic() + 10
            while not piper.EnablePiper():
                if time.monotonic() > enable_timeout:
                    print("[PiperJointController] WARNING: 使能超时")
                    break
                time.sleep(0.01)

            # Joint control mode: CAN control (0x01), joint mode (0x01), speed %
            piper.MotionCtrl_2(0x01, 0x01, self.speed_pct, 0x00)

            # Init joints if specified
            if self.init_joints is not None:
                factor = RAD_TO_001DEG
                joint_cmds = [int(round(j * factor)) for j in self.init_joints]
                piper.JointCtrl(*joint_cmds)
                time.sleep(2.0)

            # Enable gripper
            piper.GripperCtrl(0, 1000, 0x02, 0)  # disable + clear errors
            time.sleep(0.1)
            piper.GripperCtrl(0, 1000, 0x01, 0)  # enable

            # Read initial state
            init_timeout = time.monotonic() + 5
            state = None
            while state is None:
                state = self._read_state(piper)
                if state is None:
                    if time.monotonic() > init_timeout:
                        raise RuntimeError("无法读取初始关节角")
                    time.sleep(0.05)

            current_joints = state["joint_angles"].copy()
            current_gripper = self._read_gripper_raw(piper)

            # Optional interpolator
            interp: Optional[JointTrajectoryInterpolator] = None
            if self.max_joint_speed is not None:
                curr_t = time.monotonic()
                interp = JointTrajectoryInterpolator(
                    times=np.array([curr_t]),
                    joint_positions=current_joints.reshape(1, -1),
                )
                self._last_waypoint_time = curr_t

            dt = 1.0 / self.frequency
            iter_idx = 0
            keep_running = True

            while keep_running:
                t_loop_start = time.perf_counter()
                t_now = time.monotonic()

                # ---- evaluate interpolator or hold ----
                if interp is not None:
                    cmd_joints = interp(t_now)
                    self._send_joint_command(piper, cmd_joints)
                else:
                    cmd_joints = current_joints

                # ---- read state ----
                state = self._read_state(piper)
                if state is not None:
                    current_joints = state["joint_angles"]
                    current_gripper = self._read_gripper_raw(piper)

                    ring_state = {
                        "joint_angles": current_joints,
                        "gripper_angle": current_gripper,
                        "robot_receive_timestamp": time.time(),
                    }
                    self.ring_buffer.put(ring_state)

                # ---- process commands ----
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_joints = command["joints"]
                        target_time = float(command["target_time"])
                        gripper_val = float(command["gripper"])
                        gripper_effort = float(command["gripper_effort"])

                        # Convert target_time from time.time() to time.monotonic()
                        target_time_mono = time.monotonic() - time.time() + target_time

                        if interp is not None:
                            # Use interpolator for smooth motion
                            curr_time = t_now + dt
                            interp = interp.schedule_waypoint(
                                joints=target_joints,
                                time=target_time_mono,
                                max_joint_speed=self.max_joint_speed,
                                curr_time=curr_time,
                                last_waypoint_time=getattr(
                                    self, "_last_waypoint_time", curr_time
                                ),
                            )
                            self._last_waypoint_time = target_time_mono
                        else:
                            # Direct control — send immediately
                            self._send_joint_command(piper, target_joints)
                            current_joints = target_joints

                        # Gripper (always direct, no interpolation)
                        if gripper_val > 0:
                            effort_001nm = int(round(gripper_effort * 1e3))
                            piper.GripperCtrl(
                                int(round(gripper_val)),
                                effort_001nm,
                                0x01,
                                0,
                            )

                    elif cmd == Command.SET_GRIPPER.value:
                        angle = float(command["gripper"])
                        effort = float(command["gripper_effort"])
                        effort_001nm = int(round(effort * 1e3))
                        piper.GripperCtrl(
                            int(round(angle)), effort_001nm, 0x01, 0
                        )

                # ---- frequency regulation ----
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose and iter_idx % 100 == 0:
                    actual_freq = 1.0 / max(time.perf_counter() - t_loop_start, 1e-6)
                    print(
                        f"[PiperJointController] freq={actual_freq:.1f}Hz "
                        f"mode={'interp' if interp else 'direct'}"
                    )

        finally:
            # Keep arm enabled on exit
            piper.DisconnectPort()
            self.ready_event.set()
            if self.verbose:
                print(f"[PiperJointController] 已断开 {self.can_name}")

    # =========== static helpers ===========

    @staticmethod
    def _read_state(piper) -> Optional[dict]:
        """Read joint angles from piper, return dict or None."""
        joint_msgs = piper.GetArmJointMsgs()
        if joint_msgs is None:
            return None

        js = joint_msgs.joint_state
        joint_angles = np.array(
            [
                js.joint_1,
                js.joint_2,
                js.joint_3,
                js.joint_4,
                js.joint_5,
                js.joint_6,
            ],
            dtype=np.float64,
        )
        # Convert 0.001 deg → rad
        joint_angles *= _001DEG_TO_RAD

        return {"joint_angles": joint_angles}

    @staticmethod
    def _read_gripper_raw(piper) -> float:
        """Read gripper position in raw 0.001mm units."""
        gripper_msgs = piper.GetArmGripperMsgs()
        if gripper_msgs is None:
            return 0.0
        return float(gripper_msgs.gripper_state.grippers_angle)

    @staticmethod
    def _send_joint_command(piper, joints_rad: np.ndarray):
        """Send joint angles (rad) as JointCtrl.

        Parameters
        ----------
        joints_rad : ndarray (6,) float64
            Joint angles in radians.
        """
        raw = (joints_rad * RAD_TO_001DEG).astype(int)
        piper.JointCtrl(
            int(raw[0]), int(raw[1]), int(raw[2]),
            int(raw[3]), int(raw[4]), int(raw[5]),
        )
