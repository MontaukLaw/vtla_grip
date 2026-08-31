from __future__ import annotations

import copy
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .config import TransferConfig
from .tactile_recognition import TactileRecognitionWorkspace


class TactileCaptureManager:
    """One safe manual gripper capture with an explicit save step."""

    def __init__(
        self,
        gripper: Any,
        sensors: Any,
        workspace: TactileRecognitionWorkspace,
        config: TransferConfig,
        log_callback=None,
    ) -> None:
        self.gripper = gripper
        self.sensors = sensors
        self.workspace = workspace
        self.config = config
        self.log_callback = log_callback
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: dict[str, Any] | None = None
        self._state: dict[str, Any] = self._initial_state()

    @staticmethod
    def _initial_state() -> dict[str, Any]:
        return {
            "status": "idle",
            "running": False,
            "current_step": None,
            "error": None,
            "vision_class": None,
            "property": None,
            "frame_count": 0,
            "duration_seconds": 0.0,
            "final_gripper_position": None,
            "curves": None,
            "started_at": None,
            "finished_at": None,
            "saved_sample": None,
        }

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**copy.deepcopy(self._state), "running": self.running}

    def start(self, vision_class: str, property_label: str) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("触觉采集正在进行")
            if property_label not in self.workspace.properties(vision_class):
                raise ValueError("请选择有效的视觉类别和性质")
            self._stop.clear()
            self._pending = None
            self._state = {
                **self._initial_state(),
                "status": "queued",
                "running": True,
                "current_step": "等待采集",
                "vision_class": str(vision_class),
                "property": str(property_label),
                "started_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(str(vision_class), str(property_label)),
                daemon=True,
                name="tactile-data-capture",
            )
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        try:
            self.gripper.set_position(self.config.gripper_open_position)
        except Exception as exc:  # noqa: BLE001 - emergency release is best effort.
            self._log("warning", "停止采集时夹爪松开失败", error=str(exc))
        with self._lock:
            if self.running:
                self._state.update(status="stopping", current_step="正在停止")
            return self.status()

    def save(self, vision_class: str, property_label: str) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("采集尚未完成")
            if self._pending is None:
                raise RuntimeError("没有等待保存的触觉采集")
            sample = self.workspace.save_sample(
                vision_class=vision_class,
                property_label=property_label,
                frames=self._pending["frames"],
                timestamps=self._pending["timestamps"],
                final_gripper_position=self._pending["final_gripper_position"],
            )
            self._pending = None
            self._state.update(
                status="saved",
                current_step="样本已保存",
                vision_class=vision_class,
                property=property_label,
                saved_sample=sample,
            )
            return self.status()

    def discard(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("采集尚未完成")
            self._pending = None
            self._state = self._initial_state()
            return self.status()

    def _run(self, vision_class: str, property_label: str) -> None:
        frames: list[list[float]] = []
        timestamps: list[float] = []
        try:
            self._update(status="running", current_step="张开夹爪")
            self.gripper.set_force(self.config.gripper_force)
            self.gripper.set_speed(self.config.gripper_speed)
            self.gripper.set_position(self.config.gripper_open_position)
            self._wait(self.config.release_wait_s)

            self._update(current_step="触觉调零")
            self.sensors.zero()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self._check_stop()
                features = self.sensors.data().get("features") or {}
                if all(bool((features.get(side) or {}).get("baseline_ready")) for side in ("left", "right")):
                    break
                self._wait(0.03)
            else:
                raise RuntimeError("触觉传感器基线未能在 3 秒内稳定")

            self._update(current_step="逐步闭合并等待双侧接触")
            contact_data: dict[str, Any] | None = None
            start = self.config.gripper_open_position
            stop = self.config.gripper_min_position
            step = max(1, self.config.gripper_close_step)
            for position in range(start - step, stop - 1, -step):
                self._check_stop()
                command_position = max(stop, position)
                self.gripper.set_position(command_position)
                self._wait(self.config.grasp_poll_interval_s)
                data = self.sensors.data()
                if (data.get("grasp_success") or {}).get("success"):
                    contact_data = data
                    break
                if command_position == stop:
                    break
            if contact_data is None:
                raise RuntimeError("夹爪到达最小位置仍未检测到双侧接触")

            self._update(current_step="保持夹爪并采集触觉序列")
            capture_started = time.monotonic()
            duration = 1.0
            while time.monotonic() - capture_started < duration:
                self._check_stop()
                data = contact_data if not frames else self.sensors.data()
                processed = data.get("processed")
                if isinstance(processed, list) and len(processed) >= 64:
                    frames.append([float(value) for value in processed[:64]])
                    timestamps.append(time.monotonic() - capture_started)
                self._update(frame_count=len(frames), duration_seconds=time.monotonic() - capture_started)
                self._wait(0.012)
            if len(frames) < 2:
                raise RuntimeError("采集到的有效触觉帧不足")
            final_position = None
            try:
                raw_position = self.gripper.feedback().position
                if isinstance(raw_position, int) and 0 <= raw_position <= 1000:
                    final_position = raw_position
            except Exception as exc:  # noqa: BLE001 - sample may retain missing position.
                self._log("warning", "触觉采集读取夹爪位置失败", error=str(exc))

            self._update(current_step="采集完成，正在松开夹爪")
            self.gripper.set_position(self.config.gripper_open_position)
            self._wait(self.config.release_wait_s)
            curves = self._curves(frames, timestamps)
            with self._lock:
                self._pending = {
                    "vision_class": vision_class,
                    "property": property_label,
                    "frames": frames,
                    "timestamps": timestamps,
                    "final_gripper_position": final_position,
                }
            self._update(
                status="ready",
                current_step="采集完成，请检查曲线并保存",
                frame_count=len(frames),
                duration_seconds=timestamps[-1],
                final_gripper_position=final_position,
                curves=curves,
                finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            )
            self._log("info", "触觉样本采集完成，等待保存", vision_class=vision_class, property=property_label, frame_count=len(frames))
        except InterruptedError:
            self._update(status="stopped", current_step="采集已停止", error=None, finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"))
        except Exception as exc:  # noqa: BLE001 - background task boundary.
            try:
                self.gripper.set_position(self.config.gripper_open_position)
            except Exception as release_exc:  # noqa: BLE001
                self._log("warning", "采集失败后夹爪松开失败", error=str(release_exc))
            self._update(status="failed", current_step="采集失败", error=str(exc), finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"))
            self._log("error", "触觉采集失败", error=str(exc))
        finally:
            self._update(running=False)

    @staticmethod
    def _curves(frames: list[list[float]], timestamps: list[float]) -> dict[str, list[float]]:
        data = np.asarray(frames, dtype=np.float32)
        return {
            "time": [float(value) for value in timestamps],
            "left_sum": data[:, :32].sum(axis=1).astype(float).tolist(),
            "right_sum": data[:, 32:].sum(axis=1).astype(float).tolist(),
            "left_peak": data[:, :32].max(axis=1).astype(float).tolist(),
            "right_peak": data[:, 32:].max(axis=1).astype(float).tolist(),
        }

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise InterruptedError

    def _wait(self, seconds: float) -> None:
        if self._stop.wait(max(0.0, seconds)):
            self._check_stop()

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _log(self, level: str, message: str, **data: Any) -> None:
        if self.log_callback is not None:
            self.log_callback(level, "tactile_recognition", message, **data)


class TactileTrainingManager:
    def __init__(self, workspace: TactileRecognitionWorkspace, log_callback=None) -> None:
        self.workspace = workspace
        self.log_callback = log_callback
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "idle", "running": False, "vision_class": None,
            "error": None, "result": None, "started_at": None, "finished_at": None,
        }

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**copy.deepcopy(self._state), "running": self.running}

    def start(self, vision_class: str) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("已有触觉模型正在训练")
            self._state = {
                "status": "running", "running": True, "vision_class": str(vision_class),
                "error": None, "result": None,
                "started_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "finished_at": None,
            }
            self._thread = threading.Thread(target=self._run, args=(str(vision_class),), daemon=True, name="tactile-model-training")
            self._thread.start()
            return self.status()

    def _run(self, vision_class: str) -> None:
        try:
            result = self.workspace.train(vision_class)
            self._update(status="completed", result=result, finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"))
            self._log("info", "触觉性质模型训练完成", vision_class=vision_class, version=result["version"], sample_count=result["sample_count"])
        except Exception as exc:  # noqa: BLE001
            self._update(status="failed", error=str(exc), finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"))
            self._log("error", "触觉性质模型训练失败", vision_class=vision_class, error=str(exc))
        finally:
            self._update(running=False)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _log(self, level: str, message: str, **data: Any) -> None:
        if self.log_callback is not None:
            self.log_callback(level, "tactile_recognition", message, **data)
