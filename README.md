# cf2-4motor-control
# 仿真流程：
### 1. 读取当前状态；
### 2. 生成目标轨迹；
### 3. 位置外环计算 thrust_total, roll_ref, pitch_ref, yaw_ref;
### 4. 姿态内环计算 tau_x, tau_y, tau_z;
### 5. mixer 计算四个电机推力；
### 6. 写入 data.ctrl。

# 要熟悉的公式：
### 1.四元数转换欧拉角 quat to euler;
### 2.标准化时间 u；
### 3.smoothstep公式；
### 4.用s和u求pos_ref和vel_ref公式；
### 5.求期望加速度acc_cmd公式；
### 6.加速度限幅公式；
### 7.acc_cmd转换roll_ref和pitch_ref公式（符号正负如何决定和判断）；
### 8.roll_ref和pitch_ref的限幅，限制最大倾角，避免因为位置误差大而突然大角度倾斜；
### 9.总推力thrust_total公式（thrust_total = p.mass * (p.gravity + acc_cmd[2]) / tilt_comp）；
### 10.计算tau_x/tau_y/tau_z的PD公式；
### 11.限制力矩公式，避免 mixer 产生过大的电机差分推力；
### 12.控制分配矩阵和mixer，计算推力公式；

# 需要注意的参数：
### 1.偏航反扭矩系数：yaw_coeff: float = 0.006
### 2.最大推力：motor_max: float = 0.16
### 3.外环/内环的PD参数
### 4.所有用于限幅的边界值
