# 系统介绍

## 数据流

`D435 彩色+深度 → 像素 (u,v) → 相机 XYZ → 外参变换 → RM65 Base XYZ → 蓝点/尖端补偿 → 受锁机械臂运动`

压力链路为：

`左右 CH343 → 每侧 32×uint16 → 手动零点 → 动态基线/std → 中位与时间滤波 → 低/高 mask → 松开门控 → 显示滤波 → 双侧 X/Y 夹取成功判断`

## 模块

- `src/vtla_grip/web/`：仅本机访问的 FastAPI、相机服务、RM65 网关和日志。
- `src/vtla_grip/hardware/infineon.py`：双串口采集、左右合成、缓存和夹取判断。
- `src/vtla_grip/hardware/signal_processing.py`：按侧信号处理链与夹取成功判据。
- `src/vtla_grip/sensor_settings.py`：97 个设置字段的范围验证、左右覆盖和原子持久化。
- `config/sensor_processing.json`：首次迁移自 `arm_gripping/config/gui_runtime.jsonc` 的现场当前值。
- `frontend/`：React 控制台；设置弹窗可切换左右侧和原设置分组。

## 安全状态

真机运动仍要求 `allow_motion=true` 且 RM65 已连接/使能，不再要求当前会话显式解锁。自动搬运在每段运动后检查实际 TCP 位姿连续到位，之后才进入夹爪等下一动作；应用不再额外限制单段移动距离。结构化搬运状态机使用双侧触觉成功判据控制夹爪闭合与空夹恢复。

## 运行边界

Windows 当前可满足 D435、串口和 RM SDK 联调。未来迁移 Linux 时，核心 Python/React 逻辑可复用，主要替换串口名、硬件驱动、RM SDK 与启动脚本。**推断/待验证**
