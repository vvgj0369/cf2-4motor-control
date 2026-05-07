"""Four-motor MuJoCo Crazyflie controller (direct force at four motor sites).

Control logic:
1. Read altitude, attitude quaternion, and body angular velocity.
2. Generate a simple mission: idle -> takeoff -> ........-> land.
3. According to the error,compute total thrust from altitude PD control.
4. According to the error,compute body torques from roll/pitch/yaw PD control.
5. Mix total thrust + torques into four motor thrust commands.
6. Write the four motor commands to data.ctrl.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
#参数
class QuadParams:
    mass: float = 0.027 #无人机质量
    gravity: float = 9.81 #重力加速度

    # X/Y projection of each motor position from body center
    arm_xy: float = 0.0325

    # Yaw reaction torque coefficient(单位推力输入会带来多大的偏航反扭矩)
    yaw_coeff: float = 0.006

    # Single motor thrust limits
    motor_min: float = 0.0
    motor_max: float = 0.16

    # Vertical PD gains
    kp_z: float = 3.5
    kd_z: float = 2.2

    # Attitude PD gains
    kp_roll: float = 0.0045
    kd_roll: float = 0.00035

    kp_pitch: float = 0.0045
    kd_pitch: float = 0.00035

    kp_yaw: float = 0.0010
    kd_yaw: float = 0.00008


class FourMotorController:
    #保存模型、状态、参数，方便后续使用
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: QuadParams):
        self.model = model
        self.data = data
        self.params = params

        self.dt = float(model.opt.timestep)
        #初始高度
        self.initial_z = float(data.qpos[2])
        self.last_z = self.initial_z #保存上一步高度，给后面算垂直速度
        self.vz_est = 0.0 #当前估计的垂直速度，初始设 0

        self.sensor_slices = {
            "body_quat": self._sensor_slice("body_quat"),
            "body_gyro": self._sensor_slice("body_gyro"),
        }
    #为了后面能按名字读取传感器数据
    def _sensor_slice(self, sensor_name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            raise ValueError(f"Sensor '{sensor_name}' not found in model.")
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return slice(adr, adr + dim)
    #读取传感器数据（当前姿态四元数和角速度）(通过名字找到传感器,把这个传感器当前数值读出来)
    def _read_sensor(self, sensor_name: str) -> np.ndarray:
        s = self.sensor_slices[sensor_name]
        return np.array(self.data.sensordata[s], dtype=float)

    @staticmethod
    #将姿态四元数转换成欧拉角roll，pitch，yaw
    def quat_to_euler_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
        #(w,x,y,z)=(cos(θ/2)​,Vx*​sin(θ/2)​,Vy*​sin(θ/2)​,Vz*​sin(θ/2)​),  θ：总共转了多少角度; Vx,Vy,Vz:旋转轴方向，是单位向量
        #“绕某一根轴，转了某个角度”
        w, x, y, z = quat 

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    #任务生成，在不同时间，高度和角度应该是多少
    def reference(self, t: float) -> tuple[float, float]:
        z0 = self.initial_z#初始高度，来自XML
        takeoff_z = z0 + 0.3
        land_z = z0
        if t < 1.0:
           return z0, 0.0, 0.0, 0.0
        elif t < 3.0:
           return 0.3, 0.0, 0.0, 0.0
        elif t < 4.0:
           return 0.3, 0.0, 0.1, 0.0
        elif t < 5.5:
           return 0.3, 0.0, -0.1, 0.0
        elif t < 9.0:
           return 0.3, 0.0, 0.0, 0.0
        else:
          return land_z, 0.0, 0.0, 0.0
         
    #把整机需要的总推力和三轴力矩，分配成四个电机推力
    def mixer(self, thrust_total: float, tau_x: float, tau_y: float, tau_z: float) -> np.ndarray:
        """
        Motor numbering(a=0.0325):
            motor1 = (+a, +a)
            motor2 = (+a, -a)
            motor3 = (-a, -a)
            motor4 = (-a, +a)

        r=(x,y,z),F=(Fx,Fy,Fz),τ=r*F=(yFz-zFy,zFx-xFz,xFy-yFx)=(yFz,-xFz,0)

        So with direct upward force at each motor site:
            tau_x = a( f1 - f2 - f3 + f4 )
            tau_y = a(-f1 - f2 + f3 + f4 )
            tau_z = mu(-f1 + f2 - f3 + f4):1,3推力大,逆时针偏航;2,4推力大,顺时针偏航
            T     = f1 + f2 + f3 + f4
        """
        a = self.params.arm_xy
        mu = self.params.yaw_coeff

        f1 = thrust_total / 4.0 + tau_x / (4.0 * a) - tau_y / (4.0 * a) - tau_z / (4.0 * mu)
        f2 = thrust_total / 4.0 - tau_x / (4.0 * a) - tau_y / (4.0 * a) + tau_z / (4.0 * mu)
        f3 = thrust_total / 4.0 - tau_x / (4.0 * a) + tau_y / (4.0 * a) - tau_z / (4.0 * mu)
        f4 = thrust_total / 4.0 + tau_x / (4.0 * a) + tau_y / (4.0 * a) + tau_z / (4.0 * mu)

        motor_cmd = np.array([f1, f2, f3, f4], dtype=float)
        motor_cmd = np.clip(motor_cmd, self.params.motor_min, self.params.motor_max)
        return motor_cmd
    
    #仿真里的控制循环
    def update(self, t: float) -> np.ndarray:
        #读取传感器：当前姿态四元数和角速度
        quat = self._read_sensor("body_quat")
        gyro = self._read_sensor("body_gyro")
        #四元数转欧拉角
        roll, pitch, yaw = self.quat_to_euler_wxyz(quat)
        wx, wy, wz = gyro
        #读取当前高度
        z = float(self.data.qpos[2])
        self.vz_est = (z - self.last_z) / self.dt
        self.last_z = z
        #reference返回“高度，偏航角，roll角，pitch角“
        z_ref, roll_ref, pitch_ref, yaw_ref = self.reference(t)

        # Altitude 高度 PD control
        z_err = z_ref - z
        vz_err = 0.0 - self.vz_est
        #T=mg+kpz(zref-z)+kdz(0-vz)目标高度比现在高时，加大推力上升，快速下降时，增加推力
        thrust_total = self.params.mass * self.params.gravity
        thrust_total += self.params.kp_z * z_err + self.params.kd_z * vz_err
        thrust_total = float(np.clip(thrust_total, 0.0, 4.0 * self.params.motor_max))#总推力T不超过四个电机最大推力总和0.16*4

        # Attitude 姿态 PD control
        roll_err = roll_ref - roll
        pitch_err = pitch_ref - pitch
        yaw_err = yaw_ref - yaw
        #生成期望力矩
        tau_x = self.params.kp_roll * roll_err - self.params.kd_roll * wx
        tau_y = self.params.kp_pitch * pitch_err - self.params.kd_pitch * wy
        tau_z = self.params.kp_yaw * yaw_err - self.params.kd_yaw * wz
        
        #把数据给mixer,分配四个电机推力并写给Mujoco
        motor_cmd = self.mixer(thrust_total, tau_x, tau_y, tau_z)#把整机需求变成四个推力
        self.data.ctrl[:] = motor_cmd
        return motor_cmd

#加载模型，从xml读模型，建立仿真数据，初始化状态
def run(xml_path: Path, duration: float, realtime: bool) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    #创建控制器，把模型，数据，参数交给控制器
    controller = FourMotorController(model, data, QuadParams())

    print(f"Loaded model: {xml_path}")
    print(f"Initial z: {controller.initial_z:.4f} m")
    print(f"Nominal hover thrust per motor: {controller.params.mass * controller.params.gravity / 4.0:.5f} N")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.time()
        #仿真时间还没到duration就一直跑
        while viewer.is_running() and data.time < duration:
            step_start = time.time()

            motor_cmd = controller.update(float(data.time))
            mujoco.mj_step(model, data)
            viewer.sync()

            if int(data.time * 100) % 100 == 0:
                print(
                    f"t={data.time:6.2f} s | z={data.qpos[2]:.3f} m | "
                    f"ctrl=[{motor_cmd[0]:.3f}, {motor_cmd[1]:.3f}, {motor_cmd[2]:.3f}, {motor_cmd[3]:.3f}]"
                )

            if realtime:
                elapsed = time.time() - step_start
                sleep_time = model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        total_wall = time.time() - start_wall
        print(f"Simulation finished. Sim time = {data.time:.2f} s, wall time = {total_wall:.2f} s")

#默认加载同目录下的scene_4motor.xml,调用run
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-motor MuJoCo Crazyflie controller.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to the MuJoCo scene XML.",
    )
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of approximately real-time.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    xml_path = args.xml if args.xml is not None else (script_dir / "scene_4motor.xml")

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    run(xml_path=xml_path, duration=args.duration, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
