#!/usr/bin/env python3
"""
检查策略推理服务器状态的脚本。

通过 WebSocket 连接到 WebsocketPolicyServer，检查其健康状态、元数据、
推理延迟和吞吐量，帮助排查推理服务是否正常运行。

用法:
    # 基本检查
    python examples/piper/check_server.py --host localhost --port 8000

    # 指定环境 (影响生成的测试数据格式)
    python examples/piper/check_server.py --host 192.168.1.100 --port 8000 --env libero

    # 性能基准测试
    python examples/piper/check_server.py --host localhost --port 8000 --benchmark --repeat 50

输出:
    1. ✓/✗ 连接状态
    2. 服务器元数据 (模型名、checkpoint、action_dim 等)
    3. 健康检查 (/healthz)
    4. 推理延迟 (单次温启动 + 平均/最小/最大)
    5. 吞吐量 (指定重复次数)
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# --- openpi-client ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "openpi-client", "src"))
try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("[ERROR] 无法导入 openpi_client，请先安装:")
    print("  cd packages/openpi-client && pip install -e .")
    sys.exit(1)

# ===========================================================================
# 常量
# ===========================================================================

_SMALL_IMAGE = 224  # 训练默认图像尺寸
_STATE_DIM = 7       # Piper: 6 关节角 + 1 夹爪


def fake_observation(env: str) -> dict:
    """生成一条假的 observation 字典，用于测试服务器推理。"""
    if env == "libero":
        return {
            "observation/state": np.random.randn(_STATE_DIM).astype(np.float32),
            "observation/image": np.random.randint(0, 255, (_SMALL_IMAGE, _SMALL_IMAGE, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (_SMALL_IMAGE, _SMALL_IMAGE, 3), dtype=np.uint8),
            "prompt": "pick up the red block",
        }
    elif env == "aloha":
        # ALOHA 通常有双机械臂状态
        return {
            "observation/state": np.random.randn(14).astype(np.float32),
            "observation/image": np.random.randint(0, 255, (_SMALL_IMAGE, _SMALL_IMAGE, 3), dtype=np.uint8),
            "prompt": "fold the towel",
        }
    elif env == "droid":
        return {
            "observation/state": np.random.randn(_STATE_DIM).astype(np.float32),
            "observation/image": np.random.randint(0, 255, (_SMALL_IMAGE, _SMALL_IMAGE, 3), dtype=np.uint8),
            "prompt": "open the drawer",
        }
    else:
        # 通用格式
        return {
            "observation/state": np.random.randn(_STATE_DIM).astype(np.float32),
            "observation/image": np.random.randint(0, 255, (_SMALL_IMAGE, _SMALL_IMAGE, 3), dtype=np.uint8),
            "prompt": "do something",
        }


def _color(c: str, text: str) -> str:
    """终端着色。"""
    codes = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "bold": "1", "dim": "2"}
    code = codes.get(c, "0")
    return f"\033[{code}m{text}\033[0m"


def ok(text: str) -> str:
    return _color("green", f"  ✓ {text}")


def fail(text: str) -> str:
    return _color("red", f"  ✗ {text}")


def warn(text: str) -> str:
    return _color("yellow", f"  ⚠ {text}")


def info(text: str) -> str:
    return _color("dim", f"    {text}")


def section(title: str) -> str:
    return _color("bold", f"\n── {title} ──")


# ===========================================================================
# 检查项目
# ===========================================================================


class ServerChecker:
    def __init__(self, host: str, port: int, env: str = "libero"):
        self.host = host
        self.port = port
        self.env = env
        self._policy: Optional[websocket_client_policy.WebsocketClientPolicy] = None

    # ---- 1. 连接 ----

    def check_connection(self) -> bool:
        """检查 WebSocket 连接是否成功，并获取元数据。"""
        print(section("连接"))
        try:
            self._policy = websocket_client_policy.WebsocketClientPolicy(
                host=self.host, port=self.port
            )
            meta = self._policy.get_server_metadata()
            print(ok(f"已连接到 ws://{self.host}:{self.port}"))
            print("")
            print(f"  服务器元数据:")
            if meta:
                for k, v in meta.items():
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "..."
                    print(f"    {k}: {v_str}")
            else:
                print(info("(无元数据 — 服务器未设置 policy_metadata，可通过推理自动检测)"))
            return True
        except ConnectionRefusedError:
            print(fail(f"连接被拒绝 ws://{self.host}:{self.port} — 服务器未启动?"))
            return False
        except OSError as e:
            print(fail(f"无法连接到 ws://{self.host}:{self.port}: {e}"))
            return False
        except Exception as e:
            print(fail(f"连接失败: {e}"))
            return False

    # ---- 2. 健康检查 ----

    def check_health(self) -> bool:
        """通过 HTTP 请求 /healthz 端点检查服务器是否存活。"""
        print(section("健康检查"))
        import http.client

        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/healthz")
            resp = conn.getresponse()
            body = resp.read().decode().strip()
            if resp.status == 200 and body == "OK":
                print(ok(f"GET /healthz → 200 OK"))
            else:
                print(warn(f"GET /healthz → {resp.status} {body}"))
            conn.close()
            return resp.status == 200
        except ConnectionRefusedError:
            print(fail("无法连接 — 服务器未启动或端口错误"))
            return False
        except Exception as e:
            print(warn(f"健康检查失败: {e}"))
            return False

    # ---- 3. 推理功能 ----

    def check_infer(self) -> bool:
        """发送一条测试 observation 并检查返回结果。"""
        print(section("推理测试"))
        if self._policy is None:
            print(fail("未连接 — 请先运行 check_connection"))
            return False

        obs = fake_observation(self.env)
        try:
            # 预热 (可能有 JIT 编译)
            t0 = time.monotonic()
            result = self._policy.infer(obs)
            warmup_ms = (time.monotonic() - t0) * 1000

            # 验证返回格式
            if "actions" not in result:
                print(fail("返回结果缺少 'actions' 字段"))
                return False

            actions = np.asarray(result["actions"])
            print(ok(f"推理成功!"))
            print("")

            # 自动检测模型信息
            if actions.ndim == 2:
                action_horizon, action_dim = actions.shape
                print(f"  action_horizon: {action_horizon}")
                print(f"  action_dim:     {action_dim}")
                if action_dim == 7:
                    print(info("  → 猜测: Piper (6 关节角 + 1 夹爪)"))
                elif action_dim == 14:
                    print(info("  → 猜测: ALOHA 双臂 (7+7)"))
            else:
                print(f"  action shape:  {actions.shape}")

            print(f"  action dtype:   {actions.dtype}")
            print(f"  action 范围:    [{actions.min():.4f}, {actions.max():.4f}]")

            # state 输出 (如果有)
            if "state" in result:
                state = np.asarray(result["state"])
                print(f"  state shape:    {state.shape}")
                print(f"  state 范围:     [{state.min():.4f}, {state.max():.4f}]")

            print(f"  预热耗时:       {warmup_ms:.0f} ms (含可能的 JIT 编译)")

            # 服务端耗时
            server_timing = result.get("server_timing", {})
            policy_timing = result.get("policy_timing", {})
            timings = {**server_timing, **policy_timing}
            if timings:
                parts = []
                if "infer_ms" in timings:
                    parts.append(f"model={timings['infer_ms']:.0f}ms")
                if "prev_total_ms" in timings:
                    parts.append(f"prev_total={timings['prev_total_ms']:.0f}ms")
                if parts:
                    print(info(f"服务端耗时: {', '.join(parts)}"))

            # 保存检测到的信息供后续使用
            self._detected_info = {
                "action_shape": actions.shape,
                "action_dtype": str(actions.dtype),
                "warmup_ms": warmup_ms,
            }

            return True
        except Exception as e:
            print(fail(f"推理失败: {e}"))
            import traceback
            print(_color("red", traceback.format_exc()))
            return False

    # ---- 4. 性能基准测试 ----

    def check_benchmark(self, repeat: int = 50) -> dict:
        """重复推理并统计延迟。"""
        print(section(f"性能基准 ({repeat} 次推理)"))

        if self._policy is None:
            print(fail("未连接"))
            return {}

        latencies = []
        errors = 0

        for i in range(repeat):
            try:
                obs = fake_observation(self.env)
                t0 = time.monotonic()
                result = self._policy.infer(obs)
                lat = (time.monotonic() - t0) * 1000  # ms
                latencies.append(lat)
            except Exception:
                errors += 1

        if not latencies:
            print(fail("所有推理均失败"))
            return {}

        lat = np.array(latencies)
        print(ok(f"完成 {len(latencies)}/{repeat} 次推理 (失败 {errors})"))
        print("")
        print(f"  延迟 (往返, ms):")
        print(f"    平均值:  {lat.mean():.1f}")
        print(f"    中位数:  {np.median(lat):.1f}")
        print(f"    最小值:  {lat.min():.1f}")
        print(f"    最大值:  {lat.max():.1f}")
        print(f"    P95:     {np.percentile(lat, 95):.1f}")
        print(f"    P99:     {np.percentile(lat, 99):.1f}")
        print(f"    标准差:  {lat.std():.1f}")
        print("")
        throughput = 1000.0 / lat.mean()
        print(f"  估算吞吐量: {throughput:.1f} fps (单连接串行)")

        return {
            "n": len(latencies),
            "errors": errors,
            "mean_ms": float(lat.mean()),
            "median_ms": float(np.median(lat)),
            "min_ms": float(lat.min()),
            "max_ms": float(lat.max()),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "std_ms": float(lat.std()),
        }

    # ---- 5. 汇总 ----

    def run_all(self, *, benchmark: bool = False, repeat: int = 50):
        """运行所有检查并打印汇总。"""
        print(_color("bold", "\n" + "=" * 60))
        print(_color("bold", "策略推理服务器检查"))
        print(_color("bold", f"目标: ws://{self.host}:{self.port}"))
        print(_color("bold", f"环境: {self.env}"))
        print(_color("bold", "=" * 60))

        results = {}

        # 1. 连接
        results["connection"] = self.check_connection()
        if not results["connection"]:
            print(fail("\n✗ 无法连接到服务器，退出。"))
            return 1

        # 2. 健康检查
        results["health"] = self.check_health()

        # 3. 推理
        results["infer"] = self.check_infer()
        if not results["infer"]:
            print(fail("\n✗ 推理失败，跳过基准测试。"))
            return 1

        # 4. 基准
        if benchmark:
            results["benchmark"] = self.check_benchmark(repeat=repeat)

        # 5. 最终汇总
        print(section("汇总"))
        all_ok = all(results.values())
        if all_ok:
            print(ok("所有检查通过"))
        else:
            print(fail("部分检查未通过:"))
            for name, passed in results.items():
                status = ok("通过") if passed else fail("失败")
                print(f"  {name}: {status}")

        return 0 if all_ok else 1


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args():
    p = argparse.ArgumentParser(
        description="检查策略推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python check_server.py --host localhost --port 8000
  python check_server.py --host 192.168.1.100 --port 8000 --env aloha
  python check_server.py --host localhost --port 8000 --benchmark --repeat 100
        """,
    )
    p.add_argument("--host", default="localhost", help="策略服务器地址 (默认: localhost)")
    p.add_argument("--port", type=int, default=8000, help="策略服务器端口 (默认: 8000)")
    p.add_argument(
        "--env",
        default="libero",
        choices=["libero", "aloha", "droid"],
        help="环境名称，决定测试数据格式 (默认: libero)",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="运行性能基准测试",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=50,
        help="基准测试重复次数 (默认: 50)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    checker = ServerChecker(host=args.host, port=args.port, env=args.env)
    sys.exit(checker.run_all(benchmark=args.benchmark, repeat=args.repeat))


if __name__ == "__main__":
    main()
