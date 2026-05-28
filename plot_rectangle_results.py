"""
plot_rectangle_results.py

读取矩形轨迹仿真 CSV 日志，并将 8 张结果图保存到 result 文件夹中。

推荐运行流程：

1. 进入项目文件夹：
   cd /home/yikai/py/cf2_4motor

2. 激活虚拟环境：
   source /home/yikai/py/.venv/bin/activate

3. 运行仿真并保存 CSV 到 result 文件夹：
   python cf2_4motor_control.py --duration 50 --log-csv result/rectangle_log.csv

4. 运行本脚本画图：
   python plot_rectangle_results.py

输出文件：
result/
├── rectangle_log.csv
├── fig_xy_trajectory.png
├── fig_xyz_tracking.png
├── fig_position_error.png
├── fig_attitude_tracking.png
├── fig_motor_commands.png
├── fig_acceleration_commands.png
├── fig_torque_commands.png
└── fig_total_thrust.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # ============================================================
    # 1. 路径设置
    # ============================================================

    # 当前脚本所在目录，例如 /home/yikai/py/cf2_4motor
    script_dir = Path(__file__).resolve().parent

    # result 文件夹路径
    result_dir = script_dir / "result"

    # 如果 result 文件夹不存在，就自动创建
    result_dir.mkdir(exist_ok=True)

    # 优先读取 result/rectangle_log.csv
    csv_path = result_dir / "rectangle_log.csv"

    # 如果 result 文件夹里没有 CSV，就尝试读取当前目录下的 rectangle_log.csv
    fallback_csv_path = script_dir / "rectangle_log.csv"

    if not csv_path.exists():
        if fallback_csv_path.exists():
            csv_path = fallback_csv_path
            print(f"没有在 result 文件夹中找到 CSV，改为读取当前目录下的文件：{csv_path}")
        else:
            raise FileNotFoundError(
                "找不到 rectangle_log.csv。\n\n"
                "请先运行下面命令生成 CSV：\n"
                "cd /home/yikai/py/cf2_4motor\n"
                "python cf2_4motor_control.py --duration 50 --log-csv result/rectangle_log.csv\n"
            )

    print(f"读取 CSV 文件：{csv_path}")
    print(f"所有图片将保存到：{result_dir}")

    # ============================================================
    # 2. 读取 CSV 数据
    # ============================================================

    df = pd.read_csv(csv_path)

    # 时间
    t = df["t"].to_numpy()

    # 实际位置
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    # 参考位置
    x_ref = df["x_ref"].to_numpy()
    y_ref = df["y_ref"].to_numpy()
    z_ref = df["z_ref"].to_numpy()

    # 实际姿态角，单位 rad
    roll = df["roll"].to_numpy()
    pitch = df["pitch"].to_numpy()
    yaw = df["yaw"].to_numpy()

    # 参考姿态角，单位 rad
    roll_ref = df["roll_ref"].to_numpy()
    pitch_ref = df["pitch_ref"].to_numpy()
    yaw_ref = df["yaw_ref"].to_numpy()

    # 外环期望线加速度
    acc_x_cmd = df["acc_x_cmd"].to_numpy()
    acc_y_cmd = df["acc_y_cmd"].to_numpy()
    acc_z_cmd = df["acc_z_cmd"].to_numpy()

    # 总推力和三轴力矩
    thrust_total = df["thrust_total"].to_numpy()
    tau_x = df["tau_x"].to_numpy()
    tau_y = df["tau_y"].to_numpy()
    tau_z = df["tau_z"].to_numpy()

    # 四个电机推力
    f1 = df["f1"].to_numpy()
    f2 = df["f2"].to_numpy()
    f3 = df["f3"].to_numpy()
    f4 = df["f4"].to_numpy()

    # ============================================================
    # 3. 计算误差和电机差分
    # ============================================================

    # 各方向位置误差：参考位置 - 实际位置
    ex = x_ref - x
    ey = y_ref - y
    ez = z_ref - z

    # 四个电机的平均推力
    motor_mean = (f1 + f2 + f3 + f4) / 4.0

    # 电机差分推力：每个电机相对平均推力的差值
    # 这个图比直接画 f1/f2/f3/f4 更能看出姿态控制时的微小电机差异。
    df1 = f1 - motor_mean
    df2 = f2 - motor_mean
    df3 = f3 - motor_mean
    df4 = f4 - motor_mean

    # ============================================================
    # 4. 图 1：XY 平面矩形轨迹
    # ============================================================

    plt.figure(figsize=(7, 6))
    plt.plot(x_ref, y_ref, "--", label="Reference trajectory")
    plt.plot(x, y, label="Actual trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("XY Plane Rectangle Trajectory Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_xy_trajectory.png", dpi=300)
    plt.close()

    # ============================================================
    # 5. 图 2：x/y/z 位置跟踪
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, x_ref, "--", label="x_ref")
    plt.plot(t, x, label="x")
    plt.plot(t, y_ref, "--", label="y_ref")
    plt.plot(t, y, label="y")
    plt.plot(t, z_ref, "--", label="z_ref")
    plt.plot(t, z, label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_xyz_tracking.png", dpi=300)
    plt.close()

    # ============================================================
    # 6. 图 3：x/y/z 分方向位置误差
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, ex, label="x error")
    plt.plot(t, ey, label="y error")
    plt.plot(t, ez, label="z error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.title("Position Tracking Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_position_error.png", dpi=300)
    plt.close()

    # ============================================================
    # 7. 图 4：姿态跟踪
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.rad2deg(roll_ref), "--", label="roll_ref")
    plt.plot(t, np.rad2deg(roll), label="roll")
    plt.plot(t, np.rad2deg(pitch_ref), "--", label="pitch_ref")
    plt.plot(t, np.rad2deg(pitch), label="pitch")
    plt.plot(t, np.rad2deg(yaw_ref), "--", label="yaw_ref")
    plt.plot(t, np.rad2deg(yaw), label="yaw")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Attitude Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_attitude_tracking.png", dpi=300)
    plt.close()

    # ============================================================
    # 8. 图 5：四电机差分推力
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, df1, label="motor 1 delta")
    plt.plot(t, df2, label="motor 2 delta")
    plt.plot(t, df3, label="motor 3 delta")
    plt.plot(t, df4, label="motor 4 delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Differential motor thrust (N)")
    plt.title("Motor Differential Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_motor_commands.png", dpi=300)
    plt.close()

    # ============================================================
    # 9. 图 6：外环期望加速度指令
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, acc_x_cmd, label="acc_x_cmd")
    plt.plot(t, acc_y_cmd, label="acc_y_cmd")
    plt.plot(t, acc_z_cmd, label="acc_z_cmd")
    plt.xlabel("Time (s)")
    plt.ylabel("Commanded acceleration (m/s²)")
    plt.title("Outer-loop Acceleration Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_acceleration_commands.png", dpi=300)
    plt.close()

    # ============================================================
    # 10. 图 7：姿态内环力矩指令
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, tau_x, label="tau_x")
    plt.plot(t, tau_y, label="tau_y")
    plt.plot(t, tau_z, label="tau_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque command (N·m)")
    plt.title("Attitude-loop Torque Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_torque_commands.png", dpi=300)
    plt.close()

    # ============================================================
    # 11. 图 8：总推力指令
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(t, thrust_total, label="thrust_total")
    plt.xlabel("Time (s)")
    plt.ylabel("Total thrust command (N)")
    plt.title("Total Thrust Command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "fig_total_thrust.png", dpi=300)
    plt.close()

    print("8 张图已生成并保存到 result 文件夹：")
    print("- fig_xy_trajectory.png")
    print("- fig_xyz_tracking.png")
    print("- fig_position_error.png")
    print("- fig_attitude_tracking.png")
    print("- fig_motor_commands.png")
    print("- fig_acceleration_commands.png")
    print("- fig_torque_commands.png")
    print("- fig_total_thrust.png")


if __name__ == "__main__":
    main()