from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

Field = tuple[str, str, str, float, float, str]

# Mirrors arm_gripping/gui/profile_override_schema.py and ConfigDialog additions.
FIELDS: list[Field] = [
    ("密集行数", "3D显示", "point_cloud.dense_rows", 16, 120, "int"),
    ("密集列数", "3D显示", "point_cloud.dense_cols", 8, 80, "int"),
    ("高度缩放", "3D显示", "point_cloud.height_scale", 0, 5, "float"),
    ("点大小", "3D显示", "point_cloud.dot_size", 1, 14, "float"),
    ("行向影响半径", "3D显示", "point_cloud.influence_sigma_row", 0.1, 5, "float"),
    ("列向影响半径", "3D显示", "point_cloud.influence_sigma_col", 0.1, 5, "float"),
    ("上升平滑", "3D显示", "point_cloud.response_alpha", 0.01, 1, "float"),
    ("回落平滑", "3D显示", "point_cloud.fall_alpha", 0.01, 1, "float"),
    ("左点阵 X", "3D显示", "point_cloud.left_offset.0", -5, 5, "float"),
    ("右点阵 X", "3D显示", "point_cloud.right_offset.0", -5, 5, "float"),
    (
        "Mountain active threshold",
        "3D显示",
        "point_cloud.mountain_active_threshold",
        0,
        100000,
        "float",
    ),
    ("Mountain peak gain", "3D显示", "point_cloud.mountain_peak_gain", 0, 5, "float"),
    ("Fall duration (s)", "3D显示", "point_cloud.fall_duration_s", 0, 5, "float"),
    ("显示量程", "显示", "heatmap_display.display.threshold", 0, 100000, "float"),
    ("Gamma", "显示", "heatmap_display.display.gamma", 0.05, 5, "float"),
    ("对数强度", "显示", "heatmap_display.display.log_strength", 0, 200, "float"),
    ("Smoothing", "显示", "heatmap_display.display.smoothing", 0, 31, "int"),
    ("Sharpening", "显示", "heatmap_display.display.sharpening", 0, 3, "float"),
    ("低位 mask", "预处理/基线", "heatmap_display.preprocessing.signal_mask", 0, 100000, "float"),
    (
        "高位 mask",
        "预处理/基线",
        "heatmap_display.preprocessing.signal_high_mask",
        0,
        100000,
        "float",
    ),
    (
        "最终低值显示滤波",
        "预处理/基线",
        "heatmap_display.preprocessing.final_signal_mask",
        0,
        100000,
        "float",
    ),
    (
        "中位值滤波",
        "预处理/基线",
        "heatmap_display.preprocessing.median_filter.enabled",
        0,
        1,
        "bool",
    ),
    (
        "中位值窗口",
        "预处理/基线",
        "heatmap_display.preprocessing.median_filter.window",
        1,
        31,
        "int",
    ),
    ("松开门控", "预处理/基线", "heatmap_display.preprocessing.release_gate.enabled", 0, 1, "bool"),
    (
        "松开下降比例",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.release_slope",
        0,
        1,
        "float",
    ),
    (
        "最小释放值",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.release_level",
        0,
        100000,
        "float",
    ),
    (
        "重新按下比例",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.rearm_slope",
        0,
        1,
        "float",
    ),
    (
        "重新放行阈值",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.rearm_level",
        0,
        100000,
        "float",
    ),
    (
        "松开确认帧数",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.release_frames",
        1,
        30,
        "int",
    ),
    (
        "重新放行帧数",
        "预处理/基线",
        "heatmap_display.preprocessing.release_gate.rearm_frames",
        1,
        30,
        "int",
    ),
    ("滑动平均滤波", "预处理/基线", "heatmap_display.temporal_filter.enabled", 0, 1, "bool"),
    ("滑动平均窗口", "预处理/基线", "heatmap_display.temporal_filter.window", 2, 60, "int"),
    ("动态基线", "预处理/基线", "heatmap_display.baseline.enabled", 0, 1, "bool"),
    (
        "初始化显示清零",
        "预处理/基线",
        "heatmap_display.baseline.suppress_during_initializing",
        0,
        1,
        "bool",
    ),
    ("基线窗口", "预处理/基线", "heatmap_display.baseline.tracker.window_size", 6, 300, "int"),
    (
        "突变倍数",
        "预处理/基线",
        "heatmap_display.baseline.tracker.spike_multiplier",
        0.1,
        100,
        "float",
    ),
    (
        "受压倍数",
        "预处理/基线",
        "heatmap_display.baseline.tracker.contact_level_multiplier",
        0.1,
        100,
        "float",
    ),
    (
        "平稳倍数",
        "预处理/基线",
        "heatmap_display.baseline.tracker.calm_diff_multiplier",
        0.1,
        100,
        "float",
    ),
    (
        "恢复倍数",
        "预处理/基线",
        "heatmap_display.baseline.tracker.recovery_level_multiplier",
        0.1,
        100,
        "float",
    ),
    ("最小噪声", "预处理/基线", "heatmap_display.baseline.tracker.min_noise", 0, 100, "float"),
    (
        "冻结噪声",
        "预处理/基线",
        "heatmap_display.baseline.tracker.freeze_noise_after_ready",
        0,
        1,
        "bool",
    ),
    ("基线日志", "预处理/基线", "heatmap_display.baseline.tracker.verbose", 0, 1, "bool"),
    ("std 扣除", "预处理/基线", "heatmap_display.std_filter.enabled", 0, 1, "bool"),
    ("std 倍数", "预处理/基线", "heatmap_display.std_filter.subtract_multiplier", 0, 100, "float"),
    ("std 更新间隔", "预处理/基线", "heatmap_display.std_filter.update_interval", 0, 60, "float"),
    ("动态量程", "预处理/基线", "heatmap_display.dynamic_threshold.enabled", 0, 1, "bool"),
    (
        "动态量程下限",
        "预处理/基线",
        "heatmap_display.dynamic_threshold.min_value",
        0,
        100000,
        "float",
    ),
    ("动态量程平滑", "预处理/基线", "heatmap_display.dynamic_threshold.smooth", 0.01, 1, "float"),
    (
        "自动夹取处理后信号阈值",
        "触发判断",
        "state_transitions.auto_grasp_trigger.point_threshold",
        0,
        100000,
        "float",
    ),
    (
        "自动夹取连续确认帧",
        "触发判断",
        "state_transitions.auto_grasp_trigger.confirm_frames",
        1,
        30,
        "int",
    ),
    (
        "处理后信号停止阈值 X",
        "触发判断",
        "state_transitions.contact_stop_trigger.point_threshold",
        0,
        100000,
        "float",
    ),
    (
        "停止连续帧数阈值 Y",
        "触发判断",
        "state_transitions.contact_stop_trigger.contact_confirm_frames",
        0,
        60,
        "int",
    ),
    (
        "空闲稳定时间",
        "触发判断",
        "state_transitions.idle_std_refresh.stable_duration",
        0,
        60,
        "float",
    ),
]

_RELEASE: list[Field] = [
    (
        "释放停止位置",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_stop_position",
        0,
        2000,
        "int",
    ),
    (
        "释放步长（越小越慢）",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_step_pos",
        1,
        2000,
        "int",
    ),
    (
        "释放步进间隔（秒）",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_interval",
        0,
        5,
        "float",
    ),
    (
        "脱离接触单点阈值",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_clear_point_threshold",
        0,
        100000,
        "float",
    ),
    (
        "脱离接触连续确认帧",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_clear_confirm_frames",
        1,
        60,
        "int",
    ),
    (
        "释放中调零采样帧数",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_zero_frames",
        1,
        200,
        "int",
    ),
    (
        "到位后最长补采时间（秒）",
        "释放动作与下一轮准备",
        "adaptive_grasper.release_zero_timeout",
        0,
        10,
        "float",
    ),
    (
        "释放后额外保护时间（秒）",
        "释放动作与下一轮准备",
        "state_transitions.auto_release_trigger.post_release_guard_seconds",
        0,
        10,
        "float",
    ),
    (
        "夹取成功后稳定等待（秒）",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.delay_seconds",
        0,
        60,
        "float",
    ),
    (
        "释放基准采样帧数",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.baseline_frames",
        1,
        120,
        "int",
    ),
    (
        "自动释放固定增加阈值",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.point_change_threshold",
        0,
        100000,
        "float",
    ),
    (
        "自动释放增加比例",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.point_change_ratio",
        0,
        100,
        "float",
    ),
    (
        "自动释放连续确认帧",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.confirm_frames",
        1,
        60,
        "int",
    ),
    (
        "从 0 快速上升时间窗（秒）",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.rapid_rise_window_seconds",
        0.01,
        10,
        "float",
    ),
    (
        "从 0 快速上升释放阈值",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.rapid_rise_threshold",
        0,
        100000,
        "float",
    ),
    (
        "快速上升连续确认帧",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.rapid_rise_confirm_frames",
        1,
        30,
        "int",
    ),
    (
        "释放热力图回落时间（秒）",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.release_heatmap_fall_seconds",
        0,
        10,
        "float",
    ),
    (
        "释放基准慢跟随时间常数（秒）",
        "夹住后的自动释放判断",
        "state_transitions.auto_release_trigger.baseline_follow_time_constant_seconds",
        0,
        600,
        "float",
    ),
]

_GRASP_NAMES = [
    ("靠近步长", "approach_step_pos", 1, 1000, "int"),
    ("单侧接触慢速步长", "approach_slow_step_pos", 1, 1000, "int"),
    ("夹取开始慢速屏蔽时间 N", "approach_slow_guard_seconds", 0, 60, "float"),
    ("靠近间隔", "approach_interval", 0, 5, "float"),
    ("接触帧数", "contact_frames", 1, 200, "int"),
    ("最小力", "min_force", 0, 1000, "int"),
    ("增力步长", "force_step", 0, 1000, "int"),
    ("小增力步长", "force_step_small", 0, 1000, "int"),
    ("循环间隔", "loop_interval", 0, 5, "float"),
    ("扩展最小增量", "spread_min_increase", 0, 10, "float"),
    ("方差高阈值倍数", "var_high_factor", 0, 100, "float"),
    ("CoP 滑移阈值", "slip_cop_disp", 0, 100, "float"),
    ("滑移参考帧", "slip_ref_frames", 1, 200, "int"),
    ("滑移速度窗口", "slip_vel_window", 1, 200, "int"),
    ("滑移速度阈值", "slip_vel_disp", 0, 100, "float"),
    ("滑移最小 R2", "slip_vel_min_r2", 0, 1, "float"),
    ("累计滑移阈值", "slip_cumul_disp", 0, 100, "float"),
    ("CoP 最少触点", "slip_min_cop_nz", 0, 64, "int"),
    ("接触丢失比例", "slip_loss_ratio", 0, 10, "float"),
    ("滑移确认帧", "slip_confirm_frames", 1, 200, "int"),
    ("保持间隔", "hold_interval", 0, 5, "float"),
    ("最大位置", "max_position", 0, 2000, "int"),
    ("释放位置", "open_position", 0, 2000, "int"),
    ("释放最小力", "min_release_force", 0, 1000, "int"),
    ("最小抓取位置", "min_grasp_position", 0, 2000, "int"),
]
FIELDS += [
    (label, "抓取参数", f"adaptive_grasper.{key}", lo, hi, kind)
    for label, key, lo, hi, kind in _GRASP_NAMES
]
FIELDS += _RELEASE


def _get_nested(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if isinstance(value, list) and part.isdigit():
            index = int(part)
            if index >= len(value):
                return None
            value = value[index]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    current: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        if isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})
    final = parts[-1]
    if isinstance(current, list) and final.isdigit():
        current[int(final)] = value
    else:
        current[final] = value


class SensorSettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = (
            path or Path(__file__).resolve().parents[2] / "config" / "sensor_processing.json"
        )
        self._lock = threading.RLock()
        self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def side(self, side: str) -> dict[str, Any]:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        with self._lock:
            return deepcopy(self._data["sensor_type_overrides"][side])

    def response(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(
                    self._data.get("signal_processing", {}).get("override_profile", True)
                ),
                "schema": [
                    {
                        "label": label,
                        "group": group,
                        "path": path,
                        "min": lo,
                        "max": hi,
                        "kind": kind,
                    }
                    for label, group, path, lo, hi, kind in FIELDS
                ],
                "values": {
                    side: {
                        **deepcopy(self._data["sensor_type_overrides"][side]),
                        "point_cloud": deepcopy(self._data.get("point_cloud", {})),
                    }
                    for side in ("left", "right")
                },
                "point_cloud": deepcopy(self._data.get("point_cloud", {})),
            }

    def update(self, side: str, values: dict[str, Any]) -> dict[str, Any]:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        schema = {item[2]: item for item in FIELDS}
        with self._lock:
            target = self._data["sensor_type_overrides"][side]
            shared_paths: list[tuple[str, Any]] = []
            for path, raw in values.items():
                if path not in schema:
                    raise ValueError(f"unknown sensor setting: {path}")
                _, _, _, minimum, maximum, kind = schema[path]
                value = bool(raw) if kind == "bool" else int(raw) if kind == "int" else float(raw)
                if kind != "bool" and not minimum <= value <= maximum:
                    raise ValueError(f"{path} must be between {minimum} and {maximum}")
                _set_nested(self._data if path.startswith("point_cloud.") else target, path, value)
                if (
                    path.startswith("adaptive_grasper.")
                    or path in {item[2] for item in _RELEASE}
                    or path == "state_transitions.auto_grasp_trigger.confirm_frames"
                ):
                    shared_paths.append((path, value))
            for path, value in shared_paths:
                _set_nested(
                    self._data["sensor_type_overrides"]["right" if side == "left" else "left"],
                    path,
                    value,
                )
            self._persist()
            return self.response()

    def _persist(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)


get_nested = _get_nested
