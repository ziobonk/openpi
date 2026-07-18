"""
Piper 相机工具 — 支持 RealSense (D435i / D405) 和 OpenCV 回退。

通过相机序列号区分外部相机和腕部相机:

    D435i  → base (外部 / 基座相机)    serial: 如 "128422272318"
    D405   → wrist (腕部相机)          serial: 如 "218722271368"

用法:
    from camera_utils import RealSenseCameras

    cams = RealSenseCameras(
        base_serial="128422272318",   # D435i
        wrist_serial="218722271368",  # D405
    )
    cams.start()
    base_img = cams.get_base()      # (224, 224, 3) uint8 RGB
    wrist_img = cams.get_wrist()    # (224, 224, 3) uint8 RGB
    cams.stop()
"""

import threading
import time
from typing import Optional

import numpy as np

# ---- 可选依赖 ----
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    HAS_RS2 = True
except ImportError:
    HAS_RS2 = False


# 模型需要的输入分辨率
TARGET_SIZE = (224, 224)
# RealSense 采集分辨率 (640x480 是通用选择，再 resize 到 224x224)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30


def _resize_pad_bgr(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """BGR 图像 → 等比缩放 + 居中黑边填充 → BGR。"""
    import cv2

    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pw = (target_w - new_w) // 2
    ph = (target_h - new_h) // 2
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[ph : ph + new_h, pw : pw + new_w] = resized
    return padded


# ============================================================================
# RealSense 后端
# ============================================================================


class RealSenseCameras:
    """RealSense 双相机抓取器 (D435i + D405)。

    Args:
        base_serial: 外部/基座相机 (D435i) 序列号。None 表示不使用。
        wrist_serial: 腕部相机 (D405) 序列号。None 表示不使用。
        width: 采集分辨率宽度 (默认 640)。
        height: 采集分辨率高度 (默认 480)。
        fps: 采集帧率。
    """

    def __init__(
        self,
        base_serial: Optional[str] = None,
        wrist_serial: Optional[str] = None,
        *,
        width: int = CAPTURE_WIDTH,
        height: int = CAPTURE_HEIGHT,
        fps: int = CAPTURE_FPS,
    ):
        if not HAS_RS2:
            raise ImportError("需要 pyrealsense2: pip install pyrealsense2")

        self._base_serial = base_serial
        self._wrist_serial = wrist_serial
        self._width = width
        self._height = height
        self._fps = fps

        self._pipelines: dict[str, rs.pipeline] = {}
        self._configs: dict[str, rs.config] = {}
        self._latest: dict[str, np.ndarray] = {}  # BGR images
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---- 启动 / 停止 ----

    def start(self):
        """启动所有相机管线。"""
        serials: list[tuple[str, str]] = []
        if self._base_serial:
            serials.append(("base", self._base_serial))
        if self._wrist_serial:
            serials.append(("wrist", self._wrist_serial))

        if not serials:
            print("[Camera] 未配置任何 RealSense 相机")
            return

        # 列出已连接设备
        ctx = rs.context()
        connected_serials = {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}
        print(f"[Camera] 已连接 RealSense 设备: {connected_serials}")

        for name, sn in serials:
            if sn not in connected_serials:
                print(f"[Camera] ⚠ 序列号 {sn} ({name}) 未找到，跳过")
                continue

            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(sn)
            cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

            profile = pipe.start(cfg)
            device = profile.get_device()
            print(
                f"[Camera] {name}: serial={sn} "
                f"name={device.get_info(rs.camera_info.name)} "
                f"{self._width}x{self._height}@{self._fps}fps"
            )

            self._pipelines[name] = pipe
            self._configs[name] = cfg
            # 初始化占位帧
            self._latest[name] = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        # 等几帧稳定
        print("[Camera] 等待首帧...")
        for _ in range(30):
            self._grab_once()
            time.sleep(1 / self._fps)

        # 启动后台抓取线程
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        print("[Camera] 后台抓取已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        for pipe in self._pipelines.values():
            pipe.stop()
        self._pipelines.clear()
        print("[Camera] 已停止")

    # ---- 获取图像 ----

    def get_base(self) -> np.ndarray:
        """获取基座相机 (D435i) 最新帧。返回 (224, 224, 3) uint8 RGB。"""
        return self._get("base")

    def get_wrist(self) -> np.ndarray:
        """获取腕部相机 (D405) 最新帧。返回 (224, 224, 3) uint8 RGB。"""
        return self._get("wrist")

    def get_base_bgr(self) -> np.ndarray:
        """获取基座相机原始 BGR 帧 (采集分辨率)。"""
        return self._get_raw("base")

    def get_wrist_bgr(self) -> np.ndarray:
        """获取腕部相机原始 BGR 帧 (采集分辨率)。"""
        return self._get_raw("wrist")

    def _get(self, name: str) -> np.ndarray:
        """获取已 resize 的 RGB 帧。"""
        bgr = self._get_raw(name)
        # BGR → RGB
        rgb = bgr[..., ::-1].copy()
        return _resize_pad_bgr(rgb, *TARGET_SIZE)

    def _get_raw(self, name: str) -> np.ndarray:
        with self._lock:
            return self._latest.get(name, np.zeros((self._height, self._width, 3), dtype=np.uint8))

    # ---- 内部 ----

    def _grab_once(self):
        """读取所有已启动管线的当前帧。"""
        for name, pipe in list(self._pipelines.items()):
            try:
                frameset = pipe.wait_for_frames(timeout_ms=100)
                color_frame = frameset.get_color_frame()
                if color_frame:
                    bgr = np.asanyarray(color_frame.get_data())
                    with self._lock:
                        self._latest[name] = bgr
            except Exception:
                pass  # 丢帧，下次再试

    def _grab_loop(self):
        while self._running:
            self._grab_once()
            time.sleep(1 / (self._fps * 2))  # 比帧率稍快

    # ---- 工具 ----

    @staticmethod
    def list_devices():
        """列出所有连接的 RealSense 设备。"""
        if not HAS_RS2:
            print("pyrealsense2 未安装")
            return
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            print("未检测到 RealSense 设备")
            return
        print(f"发现 {len(devices)} 个 RealSense 设备:")
        for d in devices:
            print(
                f"  serial={d.get_info(rs.camera_info.serial_number)}  "
                f"name={d.get_info(rs.camera_info.name)}  "
                f"usb={d.get_info(rs.camera_info.usb_type_descriptor)}"
            )


# ============================================================================
# OpenCV 回退后端 (简单的 webcam)
# ============================================================================


class OpenCVCameras:
    """OpenCV webcam 双相机抓取器 (无 RealSense 时的回退方案)。

    Args:
        base_id: 基座相机 OpenCV 设备 ID。
        wrist_id: 腕部相机 OpenCV 设备 ID。
    """

    def __init__(self, base_id: Optional[int] = None, wrist_id: Optional[int] = None):
        if not HAS_CV2:
            raise ImportError("需要 opencv-python: pip install opencv-python")

        self._caps: dict[str, cv2.VideoCapture] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        for name, dev_id in [("base", base_id), ("wrist", wrist_id)]:
            if dev_id is None:
                continue
            cap = cv2.VideoCapture(dev_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            self._caps[name] = cap
            self._latest[name] = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

        # 如果只有一个相机，复制为 wrist 也指向同一个设备
        if "base" in self._caps and "wrist" not in self._caps:
            self._caps["wrist"] = self._caps["base"]

    def start(self):
        if not self._caps:
            print("[Camera] 未配置 OpenCV 相机")
            return
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] OpenCV 已启动: {list(self._caps.keys())}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        seen = set()
        for cap in self._caps.values():
            if id(cap) not in seen:
                seen.add(id(cap))
                cap.release()

    def get_base(self) -> np.ndarray:
        return self._get("base")

    def get_wrist(self) -> np.ndarray:
        return self._get("wrist")

    def _get(self, name: str) -> np.ndarray:
        with self._lock:
            bgr = self._latest.get(name, np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _resize_pad_bgr(rgb, *TARGET_SIZE)

    def _grab_loop(self):
        while self._running:
            seen = set()
            for name, cap in self._caps.items():
                cid = id(cap)
                if cid in seen:
                    continue
                seen.add(cid)
                ret, frame = cap.read()
                if ret:
                    for n, c in self._caps.items():
                        if id(c) == cid:
                            with self._lock:
                                self._latest[n] = frame.copy()
            time.sleep(0.005)


# ============================================================================
# 统一工厂
# ============================================================================


def create_cameras(
    base_serial: Optional[str] = None,
    wrist_serial: Optional[str] = None,
    base_cv_id: Optional[int] = None,
    wrist_cv_id: Optional[int] = None,
):
    """创建一个统一接口的相机对象。

    优先使用 RealSense (如果提供了序列号)，回退到 OpenCV (如果提供了设备 ID)。

    Args:
        base_serial: D435i 基座相机序列号。
        wrist_serial: D405 腕部相机序列号。
        base_cv_id: OpenCV 基座相机设备 ID (回退方案)。
        wrist_cv_id: OpenCV 腕部相机设备 ID (回退方案)。

    Returns:
        RealSenseCameras | OpenCVCameras | None
    """
    if base_serial or wrist_serial:
        if not HAS_RS2:
            print("[Camera] pyrealsense2 未安装，回退到 OpenCV")
        else:
            return RealSenseCameras(base_serial=base_serial, wrist_serial=wrist_serial)

    if base_cv_id is not None or wrist_cv_id is not None:
        return OpenCVCameras(base_id=base_cv_id, wrist_id=wrist_cv_id)

    print("[Camera] 未配置任何相机")
    return None


# ============================================================================
# CLI 工具: 列出 RealSense 设备
# ============================================================================

if __name__ == "__main__":
    import sys

    if "--list" in sys.argv:
        RealSenseCameras.list_devices()
    else:
        print("用法: python camera_utils.py --list")
        print("  列出所有连接的 RealSense 设备序列号")
