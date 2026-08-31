# 架构

```text
D435 frames
  -> color/depth alignment
  -> target pixel (manual click first)
  -> robust local depth
  -> camera XYZ
  -> T_base_camera (next milestone)
  -> RM65 Base XYZ + fixed orientation
  -> safety gate
  -> localhost RM65 JSON service

React local console
  -> loopback FastAPI (127.0.0.1:8000)
  -> D435 camera service / RM65 gateway / DH5 Modbus RTU
```

## 模块边界

- `camera_app.py`：D435 生命周期、对齐取流和交互显示。
- `depth.py`：不依赖硬件的深度过滤算法。
- `transforms.py`：坐标变换数学函数。
- `robot_client.py`：RM65 本机服务协议和运动请求。
- `calibration_app.py`：蓝点人工选择、RM65 状态同步和追加式原始数据记录。
- `config/default.json`：相机、服务和安全默认值。
- `web/app.py`：统一的本机硬件 API、日志与标定会话。
- `web/camera_service.py`：后台 D435 取流和 MJPEG 输出。
- `web/robot_gateway.py`：现有 RM65 服务的生命周期和配置运动开关。
- `hardware/dh5.py`：DH5 Modbus RTU 协议、串口事务和范围校验。
- `hardware/infineon.py`：左右32通道压力帧解析、后台采集、滤波、置零和64通道合成。
- `calibration_solver.py`：Kabsch/RANSAC 外参求解、留出验证与误差报告。
- `frontend/`：React/shadcn 风格的一体化操作界面。

视觉模块不直接导入现有 RM SDK。RM65 的网络、使能、反馈、单位转换和逆解继续由
现有 `robot_service.py` 与 `robot_drivers.py` 负责，避免复制硬件控制逻辑。
