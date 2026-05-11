"""
四电机 MuJoCo Crazyflie 基础飞行控制器。

1. 读取无人机当前位置、速度、姿态四元数、角速度；
2. 生成位置轨迹：起飞 -> 悬停 -> 按照矩形轨迹飞行，并在到达四个点时分别悬停2秒 -> 降落；
3. 位置外环：根据 x/y/z 位置误差计算期望加速度 acc_cmd;
4. 将 acc_cmd 转换为总推力 thrust_total、roll_ref、pitch_ref;
5. yaw_ref 默认保持初始航向 initial_yaw;
6. 姿态内环：根据 roll/pitch/yaw 误差计算三轴力矩 tau_x、tau_y、tau_z;
7. mixer:用控制分配矩阵的逆，将 [T, tau_x, tau_y, tau_z] 转换为四个电机推力；
8. 将四个电机推力写入 MuJoCo 的 data.ctrl。

"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 1. 参数定义
# ============================================================

@dataclass
class QuadParams:
    """
    无人机物理参数和控制器参数。
    """
    # -------------------------
    # 物理参数
    # -------------------------
    mass: float = 0.027          # 无人机质量，单位 kg
    gravity: float = 9.81        # 重力加速度，单位 m/s^2

    # 电机到机体中心的 x/y 投影距离。
    # 这个值会影响 roll/pitch 力矩计算。
    arm_xy: float = 0.0325       # 单位 m

    # 偏航反扭矩系数。
    # 用于描述电机推力带来的 yaw 方向反作用力矩。
    yaw_coeff: float = 0.006

    # -------------------------
    # 电机推力限制
    # -------------------------
    # 这里必须和 XML 文件里的 ctrlrange="0 0.16" 对应。
    motor_min: float = 0.0
    motor_max: float = 0.16

    # -------------------------
    # 位置外环 PID 参数
    # -------------------------
    # 外环作用：
    # 位置误差 + 速度误差 -> 期望加速度 acc_cmd
    kp_x: float = 0.95
    kd_x: float = 4.0

    kp_y: float = 0.95
    kd_y: float = 4.5

    kp_z: float = 5.2
    kd_z: float = 4.0

    # -------------------------
    # 姿态内环 PID 参数
    # -------------------------
    # 内环作用：
    # 姿态误差 + 角速度阻尼 -> 三轴力矩 tau_x/tau_y/tau_z
    kp_roll: float = 0.0045
    kd_roll: float = 0.00035

    kp_pitch: float = 0.0045
    kd_pitch: float = 0.00035

    kp_yaw: float = 0.0010
    kd_yaw: float = 0.00008

    # -------------------------
    # 安全限制参数
    # -------------------------
    # 最大允许倾斜角，单位 rad。
    # 0.20 rad 大约等于 11.5 度。
    max_tilt: float = 0.16

    # x/y 方向最大期望加速度，避免水平运动过激。
    max_acc_xy: float = 1.4

    # z 方向最大期望加速度，避免起飞/降落太猛。
    max_acc_z: float = 3.0

    # roll/pitch 最大力矩限制。
    max_tau_xy: float = 0.0025

    # yaw 最大力矩限制。
    max_tau_z: float = 0.0008

    # -------------------------
    # 任务轨迹参数
    # -------------------------
    # 相对于初始高度上升多少米。
    hover_height: float = 0.30

    # 矩形轨迹的 x 方向长度，单位 m。
    rectangle_x_length: float = 0.40

    # 矩形轨迹的 y 方向宽度，单位 m。
    rectangle_y_width: float = 0.25

# ============================================================
# 2. 状态和参考量的数据结构
# ============================================================

@dataclass
class DroneState:
    """
    当前无人机状态。

    pos:世界坐标系下的位置 [x, y, z]

    vel:世界坐标系下的速度 [vx, vy, vz]

    roll, pitch, yaw:由姿态四元数转换得到的欧拉角，单位 rad

    gyro:陀螺仪测得的机体角速度 [wx, wy, wz]
    """

    pos: np.ndarray
    vel: np.ndarray
    roll: float
    pitch: float
    yaw: float
    gyro: np.ndarray


@dataclass
class PositionReference:
    """
    轨迹生成器输出的位置参考。不是目标位置！

    pos:当前时刻参考位置 [x_ref, y_ref, z_ref]

    vel:当前时刻参考速度 [vx_ref, vy_ref, vz_ref]

    yaw:目标偏航角 yaw_ref。(当前版本默认保持初始 yaw,不主动规划转向。)
    """

    pos: np.ndarray
    vel: np.ndarray
    yaw: float


# ============================================================
# 3. 四电机控制器主体
# ============================================================

class FourMotorController:

    # 控制器初始化，只在开头控制器创建时运行一次
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: QuadParams):
        self.model = model
        self.data = data
        self.params = params

        # MuJoCo 仿真步长。
        self.dt = float(model.opt.timestep)

        # MuJoCo freejoint 的 qpos 通常为：
        # qpos = [x, y, z, qw, qx, qy, qz]
        self.initial_pos = np.array(data.qpos[0:3], dtype=float)
        #self.initial_z = float(self.initial_pos[2])

        # 让无人机一直保持初始航向。从初始四元数得出初始yaw，这样后面轨迹中yaw_ref = self.initial_yaw。
        initial_quat = np.array(data.qpos[3:7], dtype=float)
        _,_,self.initial_yaw = self.quat_to_euler_wxyz(initial_quat)

        # 建立传感器索引，方便按名字读取 sensor 数据。
        self.sensor_slices = {
            "body_quat": self._sensor_slice("body_quat"),
            "body_gyro": self._sensor_slice("body_gyro"),
        }

        # 建立控制分配矩阵 A 以及它的逆矩阵 A^-1。（[T, tau_x, tau_y, tau_z]^T = A [f1, f2, f3, f4]^T）
        # 后续 mixer 会使用 A^-1 将 [T, tau_x, tau_y, tau_z] 转成 [f1, f2, f3, f4]。
        self.allocation_matrix = self._build_allocation_matrix()
        self.allocation_matrix_inv = np.linalg.inv(self.allocation_matrix)

        # 用于保存最近一次控制器计算结果，方便打印和记录 CSV。
        self.last_info: dict[str, object] = {}

    # ------------------------------------------------------------
    # 3.1 传感器读取相关函数（MuJoCo 所有 sensor 数据都存在 data.sensordata 这个长数组里）
    # ------------------------------------------------------------

    # 根据 sensor 名字找到该 sensor 在 sensordata 里的位置。
    def _sensor_slice(self, sensor_name: str) -> slice:
        sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )

        if sensor_id < 0:
            raise ValueError(f"Sensor '{sensor_name}' not found in model.")

        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]

        return slice(adr, adr + dim)
    
    # 根据位置读取指定 sensor 的当前数值。 
    def _read_sensor(self, sensor_name: str) -> np.ndarray:
        s = self.sensor_slices[sensor_name]
        return np.array(self.data.sensordata[s], dtype=float)

    # ------------------------------------------------------------
    # 3.2 姿态数学工具函数
    # ------------------------------------------------------------

    @staticmethod
    def wrap_pi(angle: float) -> float:
        """
        将角度限制在 [-pi, pi] 范围内。（防止 yaw 角误差从 +179° 到 -179° 时被错误地认为差了 358°。)
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def quat_to_euler_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
        """
        将四元数 [w, x, y, z] 转换为欧拉角 roll, pitch, yaw。
        MuJoCo 里的自由关节姿态通常使用四元数表示。
        四元数不容易直接用于简单 PID 控制，因此这里转换成欧拉角。
        """

        w, x, y, z = quat

        # roll：绕 x 轴旋转
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch：绕 y 轴旋转
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        # yaw：绕 z 轴旋转
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # ------------------------------------------------------------
    # 3.3 获取当前无人机状态（每一次控制循环，都从 MuJoCo 当前仿真状态中读取无人机的位置、速度、姿态和角速度，然后打包成一个 DroneState 对象，供后面的外环和内环控制器使用。）
    # ------------------------------------------------------------

    def get_state(self) -> DroneState:
        """
        从 MuJoCo 中读取当前无人机状态。（注:qpos和qvel均为MujoCo内部状态)
        """

        # 从 sensor 读取姿态四元数和角速度。
        quat = self._read_sensor("body_quat")
        gyro = self._read_sensor("body_gyro")

        # 四元数转换为欧拉角。
        roll, pitch, yaw = self.quat_to_euler_wxyz(quat)

        # freejoint 的 qpos[0:3] 是世界坐标系下的位置。（取 qpos 中索引 0、1、2 的元素）
        pos = np.array(self.data.qpos[0:3], dtype=float)

        # qvel[0:3] 作为平动速度使用，后三个是角速度
        # qvel[0] = vx
        # qvel[1] = vy
        # qvel[2] = vz
        # qvel[3] = wx
        # qvel[4] = wy
        # qvel[5] = wz  
        vel = np.array(self.data.qvel[0:3], dtype=float)

        return DroneState(
            pos=pos,
            vel=vel,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gyro=gyro,
        )

    # ------------------------------------------------------------
    # 3.4 轨迹生成器
    # ------------------------------------------------------------

    @staticmethod
    def _smooth_segment(
        t: float,
        t0: float,
        t1: float,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        在 t0 到 t1 之间，让目标位置从 p0 平滑移动到 p1。

        使用 smoothstep公式:
            s = 3u^2 - 2u^3
        目标不会突然跳变，控制器更稳定。
        """
        #时间边界判断
        if t <= t0:
            return p0.copy(), np.zeros(3)

        if t >= t1:
            return p1.copy(), np.zeros(3)
        
        #标准化时间u
        duration = t1 - t0
        u = (t - t0) / duration

        # smoothstep公式（s是从P0到P1的时间完成比例）
        s = 3.0 * u * u - 2.0 * u * u * u

        # s 对时间的导数，用来计算参考速度
        ds_dt = (6.0 * u - 6.0 * u * u) / duration

        pos_ref = p0 + s * (p1 - p0)
        vel_ref = ds_dt * (p1 - p0)

        return pos_ref, vel_ref
                 
    def trajectory(self, t: float) -> PositionReference:
        """
        生成长方形轨迹。

        0-1 s:
            保持初始位置；

        1-5 s:
            起飞到悬停高度；

        5-7 s:
            在 A 点悬停；

        7-13 s:
            A -> B,沿 +x 方向飞行；

        13-15 s:
            在 B 点悬停；

        15-21 s:
            B -> C,沿 +y 方向飞行；

        21-23 s:
            在 C 点悬停；

        23-29 s:
            C -> D,沿 -x 方向飞行；

        29-31 s:
            在 D 点悬停；

        31-37 s:
            D -> A,沿 -y 方向飞行，回到矩形起点上方；

        37-40 s:
            在 A 点悬停；

        40-45 s:
            降落；

        45 s 以后:
            保持降落位置。
        """

        x0, y0, z0 = self.initial_pos
        hover_z = z0 + self.params.hover_height

        # 矩形尺寸
        rect_x = self.params.rectangle_x_length
        rect_y = self.params.rectangle_y_width

        # 初始地面/起飞位置
        p_initial = np.array([x0, y0, z0], dtype=float)

        # A 点：初始点正上方悬停位置
        p_A = np.array([x0, y0, hover_z], dtype=float)

        # B 点：从 A 沿 +x 方向移动
        p_B = np.array([x0 + rect_x, y0, hover_z], dtype=float)

        # C 点：从 B 沿 +y 方向移动
        p_C = np.array([x0 + rect_x, y0 + rect_y, hover_z], dtype=float)

        # D 点：从 C 沿 -x 方向移动
        p_D = np.array([x0, y0 + rect_y, hover_z], dtype=float)

        # 回到 A 点下方降落
        p_land = np.array([x0, y0, z0], dtype=float)

        # 当前版本保持初始 yaw，不主动转向。
        yaw_ref = self.initial_yaw

        # 0-1 s：保持初始位置
        if t < 1.0:
            pos_ref = p_initial
            vel_ref = np.zeros(3)

        # 1-5 s：起飞到 A 点
        elif t < 5.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=1.0,
                t1=5.0,
                p0=p_initial,
                p1=p_A,
            )

        # 5-7 s：A 点悬停
        elif t < 7.0:
            pos_ref = p_A
            vel_ref = np.zeros(3)

        # 7-13 s：A -> B，沿 +x 方向
        elif t < 13.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=7.0,
                t1=13.0,
                p0=p_A,
                p1=p_B,
            )

        # 13-15 s：B 点悬停
        elif t < 15.0:
            pos_ref = p_B
            vel_ref = np.zeros(3)

        # 15-21 s：B -> C，沿 +y 方向
        elif t < 21.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=15.0,
                t1=21.0,
                p0=p_B,
                p1=p_C,
            )

        # 21-23 s：C 点悬停
        elif t < 23.0:
            pos_ref = p_C
            vel_ref = np.zeros(3)

        # 23-29 s：C -> D，沿 -x 方向
        elif t < 29.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=23.0,
                t1=29.0,
                p0=p_C,
                p1=p_D,
            )

        # 29-31 s：D 点悬停
        elif t < 31.0:
            pos_ref = p_D
            vel_ref = np.zeros(3)

        # 31-37 s：D -> A，沿 -y 方向，回到起点上方
        elif t < 37.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=31.0,
                t1=37.0,
                p0=p_D,
                p1=p_A,
            )

        # 37-40 s：A 点悬停
        elif t < 40.0:
            pos_ref = p_A
            vel_ref = np.zeros(3)

        # 40-45 s：降落
        elif t < 45.0:
            pos_ref, vel_ref = self._smooth_segment(
                t=t,
                t0=40.0,
                t1=45.0,
                p0=p_A,
                p1=p_land,
            )

        # 45 s 以后：保持降落位置
        else:
            pos_ref = p_land
            vel_ref = np.zeros(3)

        return PositionReference(
            pos=pos_ref,
            vel=vel_ref,
            yaw=yaw_ref,
        )

    # ------------------------------------------------------------
    # 3.5 位置外环控制器
    # ------------------------------------------------------------

    def outer_position_controller(
        self,
        state: DroneState,
        ref: PositionReference,
    ) -> tuple[float, float, float, float, np.ndarray]:
        """
        位置外环控制器。

        输入：
            当前状态 state
            目标轨迹 ref

        输出：
            thrust_total:
                总推力，用于控制 z 方向运动；

            roll_ref:
                期望 roll,用于产生 y 方向水平加速度；

            pitch_ref:
                期望 pitch,用于产生 x 方向水平加速度；

            yaw_ref:
                期望 yaw,用于保持或控制机头朝向；

            acc_cmd:
                期望加速度指令，方便记录和调试。

        关键思想：
            外环不直接控制电机，而是先判断无人机应该往哪个方向加速。
        """

        p = self.params

        # 位置误差：目标位置 - 当前实际位置
        pos_err = ref.pos - state.pos

        # 速度误差：目标速度 - 当前实际速度
        vel_err = ref.vel - state.vel

        # 根据位置误差和速度误差计算期望加速度。
        # acc_cmd 不是实际加速度，而是控制器希望无人机产生的加速度。
        acc_cmd = np.array(
            [
                p.kp_x * pos_err[0] + p.kd_x * vel_err[0],
                p.kp_y * pos_err[1] + p.kd_y * vel_err[1],
                p.kp_z * pos_err[2] + p.kd_z * vel_err[2],
            ],
            dtype=float,
        )

        # 限制期望加速度，避免控制器太激进。
        acc_cmd[0] = float(np.clip(acc_cmd[0], -p.max_acc_xy, p.max_acc_xy))
        acc_cmd[1] = float(np.clip(acc_cmd[1], -p.max_acc_xy, p.max_acc_xy))
        acc_cmd[2] = float(np.clip(acc_cmd[2], -p.max_acc_z, p.max_acc_z))

        # yaw_ref 来自轨迹规划。
        # 当前版本默认保持初始 yaw。
        yaw_ref = ref.yaw

        # 将世界坐标系下的 x/y 方向期望加速度转换成 roll_ref 和 pitch_ref。
        # 当 yaw_ref = 0 时，近似关系为：
        #     pitch_ref ≈ acc_x / g
        #     roll_ref  ≈ -acc_y / g
        # 可以根据实际仿真效果反转 roll_ref 或 pitch_ref 的符号。
        roll_ref = (
            acc_cmd[0] * math.sin(yaw_ref)
            - acc_cmd[1] * math.cos(yaw_ref)
        ) / p.gravity

        pitch_ref = (
            acc_cmd[0] * math.cos(yaw_ref)
            + acc_cmd[1] * math.sin(yaw_ref)
        ) / p.gravity

        # 限制最大倾角，避免因为位置误差大而突然大角度倾斜。
        roll_ref = float(np.clip(roll_ref, -p.max_tilt, p.max_tilt))
        pitch_ref = float(np.clip(pitch_ref, -p.max_tilt, p.max_tilt))

        # 计算总推力。
        #
        # 如果无人机水平悬停：
        #     thrust_total = mass * gravity
        #
        # 如果需要向上加速：
        #     thrust_total = mass * (gravity + acc_z)
        #
        # 如果机身倾斜，需要除以 cos(roll)*cos(pitch) 做竖直分量补偿。
        tilt_comp = math.cos(state.roll) * math.cos(state.pitch)

        # 避免倾角过大时除以很小的数。
        tilt_comp = max(0.30, tilt_comp)

        thrust_total = p.mass * (p.gravity + acc_cmd[2]) / tilt_comp

        # 限制总推力不能超过四个电机的总最大推力。
        thrust_total = float(np.clip(thrust_total, 0.0, 4.0 * p.motor_max))

        return thrust_total, roll_ref, pitch_ref, yaw_ref, acc_cmd

    # ------------------------------------------------------------
    # 3.6 姿态内环控制器
    # ------------------------------------------------------------

    def inner_attitude_controller(
        self,
        state: DroneState,
        roll_ref: float,
        pitch_ref: float,
        yaw_ref: float,
    ) -> tuple[float, float, float]:
        """
        姿态内环控制器。

        输入：
            roll_ref, pitch_ref, yaw_ref

        输出：
            tau_x, tau_y, tau_z

        作用：
            让无人机实际姿态跟踪外环给出的期望姿态。
        """

        p = self.params

        wx, wy, wz = state.gyro

        # 姿态角误差。
        roll_err = self.wrap_pi(roll_ref - state.roll)
        pitch_err = self.wrap_pi(pitch_ref - state.pitch)
        yaw_err = self.wrap_pi(yaw_ref - state.yaw)

        # PD 控制。
        # 比例项：根据角度误差产生恢复力矩；
        # 微分项：根据角速度产生阻尼，抑制振荡。
        tau_x = p.kp_roll * roll_err - p.kd_roll * wx
        tau_y = p.kp_pitch * pitch_err - p.kd_pitch * wy
        tau_z = p.kp_yaw * yaw_err - p.kd_yaw * wz

        # 限制力矩，避免 mixer 产生过大的电机差分推力。
        tau_x = float(np.clip(tau_x, -p.max_tau_xy, p.max_tau_xy))
        tau_y = float(np.clip(tau_y, -p.max_tau_xy, p.max_tau_xy))
        tau_z = float(np.clip(tau_z, -p.max_tau_z, p.max_tau_z))

        return tau_x, tau_y, tau_z

    # ------------------------------------------------------------
    # 3.7 控制分配矩阵和 mixer
    # ------------------------------------------------------------

    def _build_allocation_matrix(self) -> np.ndarray:
        """
        建立控制分配矩阵 A。

        电机编号假设：

            motor1 = (+a, +a)
            motor2 = (+a, -a)
            motor3 = (-a, -a)
            motor4 = (-a, +a)

        对每个电机施加向上的推力 f1/f2/f3/f4。

        总推力：
            T = f1 + f2 + f3 + f4

        roll 力矩：
            tau_x = a( f1 - f2 - f3 + f4 )

        pitch 力矩：
            tau_y = a(-f1 - f2 + f3 + f4 )

        yaw 力矩：
            tau_z = mu(-f1 + f2 - f3 + f4 )

        写成矩阵形式：

            [T, tau_x, tau_y, tau_z]^T = A [f1, f2, f3, f4]^T

        mixer 中会使用 A 的逆：

            [f1, f2, f3, f4]^T = A^-1 [T, tau_x, tau_y, tau_z]^T
        """

        a = self.params.arm_xy
        mu = self.params.yaw_coeff

        allocation_matrix = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [a, -a, -a, a],
                [-a, -a, a, a],
                [-mu, mu, -mu, mu],
            ],
            dtype=float,
        )

        return allocation_matrix

    def mixer(
        self,
        thrust_total: float,
        tau_x: float,
        tau_y: float,
        tau_z: float,
    ) -> np.ndarray:
        """
        控制分配器。

        输入：
            thrust_total:
                总推力 T

            tau_x, tau_y, tau_z:
                三轴力矩

        输出：
            motor_cmd:
                四个电机推力 [f1, f2, f3, f4]
        """

        desired = np.array(
            [thrust_total, tau_x, tau_y, tau_z],
            dtype=float,
        )

        # 通过 A^-1 计算四个电机推力。[f1, f2, f3, f4]^T = A^-1 [T, tau_x, tau_y, tau_z]^T，@ 是 Python/NumPy 的矩阵乘法符号。
        motor_cmd = self.allocation_matrix_inv @ desired

        # 限制电机推力范围，必须和 XML 中 ctrlrange 保持一致。
        motor_cmd = np.clip(
            motor_cmd,
            self.params.motor_min,
            self.params.motor_max,
        )

        return motor_cmd

    # ------------------------------------------------------------
    # 3.8 主控制更新函数
    # ------------------------------------------------------------

    def update(self, t: float) -> np.ndarray:
        """
        每个仿真步调用一次的控制更新函数。

        执行顺序：
        1. 读取当前状态；
        2. 生成目标轨迹；
        3. 位置外环计算 thrust_total, roll_ref, pitch_ref, yaw_ref；
        4. 姿态内环计算 tau_x, tau_y, tau_z；
        5. mixer 计算四个电机推力；
        6. 写入 data.ctrl。
        """

        # 当前状态
        state = self.get_state()

        # 当前时刻的目标轨迹
        ref = self.trajectory(t)

        # 位置外环
        thrust_total, roll_ref, pitch_ref, yaw_ref, acc_cmd = self.outer_position_controller(
            state=state,
            ref=ref,
        )

        # 姿态内环
        tau_x, tau_y, tau_z = self.inner_attitude_controller(
            state=state,
            roll_ref=roll_ref,
            pitch_ref=pitch_ref,
            yaw_ref=yaw_ref,
        )

        # 控制分配
        motor_cmd = self.mixer(
            thrust_total=thrust_total,
            tau_x=tau_x,
            tau_y=tau_y,
            tau_z=tau_z,
        )

        # 写入 MuJoCo actuator 控制量。
        self.data.ctrl[:] = motor_cmd

        # 保存当前控制器信息，方便打印和记录。
        self.last_info = {
            "t": t,
            "pos": state.pos.copy(),
            "vel": state.vel.copy(),
            "ref_pos": ref.pos.copy(),
            "ref_vel": ref.vel.copy(),
            "rpy": np.array([state.roll, state.pitch, state.yaw], dtype=float),
            "rpy_ref": np.array([roll_ref, pitch_ref, yaw_ref], dtype=float),
            "acc_cmd": acc_cmd.copy(),
            "thrust_total": thrust_total,
            "tau": np.array([tau_x, tau_y, tau_z], dtype=float),
            "motor_cmd": motor_cmd.copy(),
        }

        return motor_cmd


# ============================================================
# 4. 仿真运行函数
# ============================================================

def run(
    xml_path: Path,
    duration: float,
    realtime: bool,
    log_csv: Path | None = None,
) -> None:
    """
    加载 MuJoCo XML,运行仿真，并可选择保存 CSV 日志。
    """

    # 加载模型
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # 前向计算一次，初始化 sensor 等数据。
    mujoco.mj_forward(model, data)

    # 创建控制器
    controller = FourMotorController(
        model=model,
        data=data,
        params=QuadParams(),
    )

    print(f"Loaded model: {xml_path}")
    print(f"Initial position: {controller.initial_pos}")
    print(f"Initial yaw: {controller.initial_yaw:.4f} rad")

    hover_thrust_per_motor = controller.params.mass * controller.params.gravity / 4.0
    print(f"Nominal hover thrust per motor: {hover_thrust_per_motor:.5f} N")

    # CSV 日志文件
    log_file = None
    writer = None

    if log_csv is not None:
        log_file = open(log_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)

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

        print(f"Logging CSV to: {log_csv}")

    try:
        # 启动 MuJoCo 可视化窗口
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_wall = time.time()
            next_print_time = 0.0

            while viewer.is_running() and data.time < duration:
                step_start = time.time()

                # 控制器更新
                motor_cmd = controller.update(float(data.time))

                # MuJoCo 前进一步
                mujoco.mj_step(model, data)

                # 同步可视化窗口
                viewer.sync()

                info = controller.last_info

                # 写入 CSV
                if writer is not None and info:
                    pos = info["pos"]
                    ref_pos = info["ref_pos"]
                    vel = info["vel"]
                    rpy = info["rpy"]
                    rpy_ref = info["rpy_ref"]
                    acc_cmd = info["acc_cmd"]
                    tau = info["tau"]

                    writer.writerow(
                        [
                            info["t"],
                            pos[0], pos[1], pos[2],
                            ref_pos[0], ref_pos[1], ref_pos[2],
                            vel[0], vel[1], vel[2],
                            rpy[0], rpy[1], rpy[2],
                            rpy_ref[0], rpy_ref[1], rpy_ref[2],
                            acc_cmd[0], acc_cmd[1], acc_cmd[2],
                            info["thrust_total"],
                            tau[0], tau[1], tau[2],
                            motor_cmd[0], motor_cmd[1], motor_cmd[2], motor_cmd[3],
                        ]
                    )

                # 每 1 秒打印一次关键信息
                if data.time >= next_print_time and info:
                    pos = info["pos"]
                    ref_pos = info["ref_pos"]
                    vel = info["vel"]
                    acc_cmd = info["acc_cmd"]
                    rpy = info["rpy"]
                    rpy_ref = info["rpy_ref"]

                    print(
                        f"t={data.time:6.2f} s | "
                        f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | "
                        f"ref=({ref_pos[0]:+.3f}, {ref_pos[1]:+.3f}, {ref_pos[2]:+.3f}) | "
                        f"vel=({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) | "
                        f"acc=({acc_cmd[0]:+.3f}, {acc_cmd[1]:+.3f}, {acc_cmd[2]:+.3f}) | "
                        f"ctrl=[{motor_cmd[0]:.7f}, {motor_cmd[1]:.7f}, "
                        f"{motor_cmd[2]:.7f}, {motor_cmd[3]:.7f}]"
                    )

                    next_print_time += 1.0

                # 是否按真实时间运行
                if realtime:
                    elapsed = time.time() - step_start
                    sleep_time = model.opt.timestep - elapsed

                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

            total_wall = time.time() - start_wall

            print(
                f"Simulation finished. "
                f"Sim time = {data.time:.2f} s, "
                f"wall time = {total_wall:.2f} s"
            )

    finally:
        if log_file is not None:
            log_file.close()


# ============================================================
# 5. 主函数和命令行参数
# ============================================================

def main() -> None:
    """
    命令行入口。

    示例：

    1. 默认运行：
        python cf2_4motor_control.py

    2. 指定仿真时间：
        python cf2_4motor_control.py --duration 30

    3. 保存 CSV;
        python cf2_4motor_control.py --duration 30 --log-csv flight_log.csv

    4. 不按真实时间运行，尽可能快速仿真：
        python cf2_4motor_control.py --duration 30 --no-realtime --log-csv flight_log.csv

    5. 指定 XML 文件：
        python cf2_4motor_control.py --xml scene_4motor.xml
    """

    parser = argparse.ArgumentParser(
        description="Run cascaded PID controller for four-motor MuJoCo Crazyflie."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="MuJoCo scene XML 文件路径。默认使用当前脚本目录下的 scene_4motor.xml。",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="仿真持续时间，单位秒。",
    )

    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不按真实时间运行，而是尽可能快速运行仿真。",
    )

    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="可选：保存控制器日志的 CSV 文件路径。",
    )

    args = parser.parse_args()

    # 默认 XML 路径：当前 Python 文件同目录下的 scene_4motor.xml
    script_dir = Path(__file__).resolve().parent
    xml_path = args.xml if args.xml is not None else (script_dir / "scene_4motor.xml")

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    run(
        xml_path=xml_path,
        duration=args.duration,
        realtime=not args.no_realtime,
        log_csv=args.log_csv,
    )


if __name__ == "__main__":
    main()
