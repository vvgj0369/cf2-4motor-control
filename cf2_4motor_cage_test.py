"""
cf2_4motor_cage_test.py

专门用于测试带 simplified spherical cage 的四旋翼模型。

功能：
1. 默认加载 scene_4motor_cage.xml；
2. 自动从 XML 中读取 cf2 body 的真实质量；
3. 使用原来的 FourMotorController 控制器；
4. 自动保存 CSV 到 result_cage/rectangle_log.csv；
5. 自动生成 8 张结果图到 result_cage 文件夹。

运行方式：
    cd /home/yikai/py/cf2_4motor
    source /home/yikai/py/.venv/bin/activate
    python cf2_4motor_cage_test.py

如果想加速仿真：
    python cf2_4motor_cage_test.py --no-realtime

如果不想打开 MuJoCo viewer：
    python cf2_4motor_cage_test.py --no-viewer
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cf2_4motor_control import QuadParams, FourMotorController


# ============================================================
# 1. 一些辅助函数
# ============================================================

def get_body_mass_from_xml_model(model: mujoco.MjModel, body_name: str = "cf2") -> float:
    """
    从 MuJoCo model 中读取指定 body 的总质量。

    对你的模型来说：
    - baseline XML 中 cf2 的 mass 应该是 0.027 kg；
    - cage XML 中 cf2 的 mass 应该是 0.042 kg。

    使用这个函数以后，就不需要手动修改 Python 里的 QuadParams.mass。
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")

    mass = float(model.body_subtreemass[body_id])
    return mass


def safe_vec(value: object, length: int) -> np.ndarray:
    """
    从 controller.last_info 中安全读取向量。
    如果没有对应数据，就返回 NaN，避免写 CSV 时报错。
    """

    if value is None:
        return np.full(length, np.nan, dtype=float)

    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size < length:
        out = np.full(length, np.nan, dtype=float)
        out[:arr.size] = arr
        return out

    return arr[:length]


def write_csv_header(writer: csv.writer) -> None:
    """
    写入 CSV 表头。
    表头和 plot_results() 里面读取的列名保持一致。
    """

    writer.writerow(
        [
            "t",
            "x", "y", "z",
            "x_ref", "y_ref", "z_ref",
            "vx", "vy", "vz",
            "roll", "pitch", "yaw",
            "roll_ref", "pitch_ref", "yaw_ref",
            "acc_x_cmd", "acc_y_cmd", "acc_z_cmd",
            "thrust_total",
            "tau_x", "tau_y", "tau_z",
            "f1", "f2", "f3", "f4",
        ]
    )


def write_csv_row(
    writer: csv.writer,
    t: float,
    info: dict[str, object],
    motor_cmd: np.ndarray,
) -> None:
    """
    每个仿真步写入一行数据。
    """

    pos = safe_vec(info.get("pos"), 3)
    ref_pos = safe_vec(info.get("ref_pos"), 3)
    vel = safe_vec(info.get("vel"), 3)
    rpy = safe_vec(info.get("rpy"), 3)
    rpy_ref = safe_vec(info.get("rpy_ref"), 3)
    acc_cmd = safe_vec(info.get("acc_cmd"), 3)
    tau = safe_vec(info.get("tau"), 3)
    motor_cmd = safe_vec(motor_cmd, 4)

    thrust_total = info.get("thrust_total", np.nan)

    writer.writerow(
        [
            f"{t:.6f}",
            f"{pos[0]:.9f}", f"{pos[1]:.9f}", f"{pos[2]:.9f}",
            f"{ref_pos[0]:.9f}", f"{ref_pos[1]:.9f}", f"{ref_pos[2]:.9f}",
            f"{vel[0]:.9f}", f"{vel[1]:.9f}", f"{vel[2]:.9f}",
            f"{rpy[0]:.9f}", f"{rpy[1]:.9f}", f"{rpy[2]:.9f}",
            f"{rpy_ref[0]:.9f}", f"{rpy_ref[1]:.9f}", f"{rpy_ref[2]:.9f}",
            f"{acc_cmd[0]:.9f}", f"{acc_cmd[1]:.9f}", f"{acc_cmd[2]:.9f}",
            f"{float(thrust_total):.9f}",
            f"{tau[0]:.12f}", f"{tau[1]:.12f}", f"{tau[2]:.12f}",
            f"{motor_cmd[0]:.9f}", f"{motor_cmd[1]:.9f}",
            f"{motor_cmd[2]:.9f}", f"{motor_cmd[3]:.9f}",
        ]
    )


# ============================================================
# 2. 运行 cage 仿真
# ============================================================

def run_cage_simulation(
    xml_path: Path,
    duration: float,
    out_dir: Path,
    realtime: bool = True,
    use_viewer: bool = True,
) -> Path:
    """
    运行带 cage 的仿真，并保存 CSV。

    返回：
        csv_path
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rectangle_log.csv"

    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 XML 文件：{xml_path}")

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 从 XML 模型里读取实际质量
    mass_from_xml = get_body_mass_from_xml_model(model, "cf2")

    # 创建参数，并强制使用 XML 里的质量
    params = QuadParams()
    params.mass = mass_from_xml

    # 创建控制器
    controller = FourMotorController(
        model=model,
        data=data,
        params=params,
    )

    # 再设置一次，防止原控制器 __init__ 里有其他覆盖逻辑
    controller.params.mass = mass_from_xml

    hover_per_motor = controller.params.mass * controller.params.gravity / 4.0

    print(f"Loaded cage model: {xml_path}")
    print(f"Initial position: {controller.initial_pos}")
    print(f"Initial yaw: {controller.initial_yaw:.4f} rad")
    print(f"Controller mass from XML: {controller.params.mass:.5f} kg")
    print(f"Nominal hover thrust per motor: {hover_per_motor:.5f} N")
    print(f"Logging CSV to: {csv_path}")

    log_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    write_csv_header(writer)

    last_print_time = -1.0

    def step_once() -> None:
        nonlocal last_print_time

        t = float(data.time)

        # 控制器计算电机命令，并写入 data.ctrl
        motor_cmd = controller.update(t)

        # 记录 CSV
        write_csv_row(
            writer=writer,
            t=t,
            info=controller.last_info,
            motor_cmd=motor_cmd,
        )

        # 每约 1 秒打印一次
        if t - last_print_time >= 1.0 or last_print_time < 0.0:
            last_print_time = t

            info = controller.last_info
            pos = safe_vec(info.get("pos"), 3)
            ref_pos = safe_vec(info.get("ref_pos"), 3)
            vel = safe_vec(info.get("vel"), 3)
            acc_cmd = safe_vec(info.get("acc_cmd"), 3)

            print(
                f"t={t:6.2f} s | "
                f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | "
                f"ref=({ref_pos[0]:+.3f}, {ref_pos[1]:+.3f}, {ref_pos[2]:+.3f}) | "
                f"vel=({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) | "
                f"acc=({acc_cmd[0]:+.3f}, {acc_cmd[1]:+.3f}, {acc_cmd[2]:+.3f}) | "
                f"ctrl=[{motor_cmd[0]:.7f}, {motor_cmd[1]:.7f}, "
                f"{motor_cmd[2]:.7f}, {motor_cmd[3]:.7f}]"
            )

        # MuJoCo 推进一步
        mujoco.mj_step(model, data)

    try:
        if use_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                wall_start = time.time()
                sim_start = float(data.time)

                while viewer.is_running() and data.time < duration:
                    step_once()
                    viewer.sync()

                    if realtime:
                        target_wall_time = wall_start + (float(data.time) - sim_start)
                        sleep_time = target_wall_time - time.time()
                        if sleep_time > 0.0:
                            time.sleep(sleep_time)
        else:
            wall_start = time.time()
            sim_start = float(data.time)

            while data.time < duration:
                step_once()

                if realtime:
                    target_wall_time = wall_start + (float(data.time) - sim_start)
                    sleep_time = target_wall_time - time.time()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

    finally:
        log_file.close()

    print(
        f"Simulation finished. "
        f"Sim time = {data.time:.2f} s"
    )

    return csv_path


# ============================================================
# 3. 画图函数
# ============================================================

def plot_results(csv_path: Path, out_dir: Path) -> None:
    """
    读取 CSV，并生成 8 张图。
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    t = df["t"].to_numpy()

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    x_ref = df["x_ref"].to_numpy()
    y_ref = df["y_ref"].to_numpy()
    z_ref = df["z_ref"].to_numpy()

    roll = df["roll"].to_numpy()
    pitch = df["pitch"].to_numpy()
    yaw = df["yaw"].to_numpy()

    roll_ref = df["roll_ref"].to_numpy()
    pitch_ref = df["pitch_ref"].to_numpy()
    yaw_ref = df["yaw_ref"].to_numpy()

    acc_x_cmd = df["acc_x_cmd"].to_numpy()
    acc_y_cmd = df["acc_y_cmd"].to_numpy()
    acc_z_cmd = df["acc_z_cmd"].to_numpy()

    thrust_total = df["thrust_total"].to_numpy()

    tau_x = df["tau_x"].to_numpy()
    tau_y = df["tau_y"].to_numpy()
    tau_z = df["tau_z"].to_numpy()

    f1 = df["f1"].to_numpy()
    f2 = df["f2"].to_numpy()
    f3 = df["f3"].to_numpy()
    f4 = df["f4"].to_numpy()

    # 位置误差
    ex = x_ref - x
    ey = y_ref - y
    ez = z_ref - z

    # 电机差分推力
    motor_mean = (f1 + f2 + f3 + f4) / 4.0
    df1 = f1 - motor_mean
    df2 = f2 - motor_mean
    df3 = f3 - motor_mean
    df4 = f4 - motor_mean

    # ------------------------------------------------------------
    # 图 1：XY 平面轨迹
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(x_ref, y_ref, "--", label="Reference trajectory")
    plt.plot(x, y, label="Actual trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Cage Model: XY Plane Rectangle Trajectory Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_xy_trajectory.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 2：位置跟踪
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_ref, "--", label="x_ref")
    plt.plot(t, x, label="x")
    plt.plot(t, y_ref, "--", label="y_ref")
    plt.plot(t, y, label="y")
    plt.plot(t, z_ref, "--", label="z_ref")
    plt.plot(t, z, label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Cage Model: Position Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_xyz_tracking.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 3：位置误差
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, ex, label="x error")
    plt.plot(t, ey, label="y error")
    plt.plot(t, ez, label="z error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.title("Cage Model: Position Tracking Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_position_error.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 4：姿态跟踪
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.rad2deg(roll_ref), "--", label="roll_ref")
    plt.plot(t, np.rad2deg(roll), label="roll")
    plt.plot(t, np.rad2deg(pitch_ref), "--", label="pitch_ref")
    plt.plot(t, np.rad2deg(pitch), label="pitch")
    plt.plot(t, np.rad2deg(yaw_ref), "--", label="yaw_ref")
    plt.plot(t, np.rad2deg(yaw), label="yaw")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Cage Model: Attitude Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_attitude_tracking.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 5：电机差分推力
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, df1, label="motor 1 delta")
    plt.plot(t, df2, label="motor 2 delta")
    plt.plot(t, df3, label="motor 3 delta")
    plt.plot(t, df4, label="motor 4 delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Differential motor thrust (N)")
    plt.title("Cage Model: Motor Differential Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_motor_commands.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 6：外环加速度指令
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, acc_x_cmd, label="acc_x_cmd")
    plt.plot(t, acc_y_cmd, label="acc_y_cmd")
    plt.plot(t, acc_z_cmd, label="acc_z_cmd")
    plt.xlabel("Time (s)")
    plt.ylabel("Commanded acceleration (m/s²)")
    plt.title("Cage Model: Outer-loop Acceleration Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_acceleration_commands.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 7：姿态内环力矩指令
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, tau_x, label="tau_x")
    plt.plot(t, tau_y, label="tau_y")
    plt.plot(t, tau_z, label="tau_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque command (N·m)")
    plt.title("Cage Model: Attitude-loop Torque Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_torque_commands.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 图 8：总推力
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, thrust_total, label="thrust_total")
    plt.xlabel("Time (s)")
    plt.ylabel("Total thrust command (N)")
    plt.title("Cage Model: Total Thrust Command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_total_thrust.png", dpi=300)
    plt.close()

    print(f"8 张 cage 结果图已保存到：{out_dir}")


# ============================================================
# 4. 主函数
# ============================================================

def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run rectangular trajectory test for caged quadrotor model."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=script_dir / "scene_4motor_cage.xml",
        help="cage 场景 XML 文件路径，默认 scene_4motor_cage.xml。",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="仿真时间，单位秒。",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "result_cage",
        help="结果输出文件夹，默认 result_cage。",
    )

    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不按照真实时间运行，尽可能快地仿真。",
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="不打开 MuJoCo viewer，只在后台运行并保存 CSV/图片。",
    )

    args = parser.parse_args()

    xml_path = args.xml
    if not xml_path.is_absolute():
        xml_path = script_dir / xml_path

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    csv_path = run_cage_simulation(
        xml_path=xml_path,
        duration=args.duration,
        out_dir=out_dir,
        realtime=not args.no_realtime,
        use_viewer=not args.no_viewer,
    )

    plot_results(
        csv_path=csv_path,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()