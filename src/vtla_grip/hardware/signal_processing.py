from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..sensor_settings import get_nested


class SensorSignalProcessor:
    """Per-side processing chain compatible with arm_gripping's configured stages."""

    def __init__(self, config: dict[str, Any], channels: int = 32) -> None:
        self.channels = channels
        self.configure(config)

    def configure(self, config: dict[str, Any]) -> None:
        self.config = config
        heatmap = config.get("heatmap_display", {})
        preprocessing = heatmap.get("preprocessing", {})
        baseline = heatmap.get("baseline", {})
        tracker = baseline.get("tracker", {})
        std_filter = heatmap.get("std_filter", {})
        temporal = heatmap.get("temporal_filter", {})
        dynamic = heatmap.get("dynamic_threshold", {})
        display = heatmap.get("display", {})
        release = preprocessing.get("release_gate", {})

        self.baseline_enabled = bool(baseline.get("enabled", True))
        self.suppress_initial = bool(baseline.get("suppress_during_initializing", True))
        self.baseline_window = max(6, int(tracker.get("window_size", 15)))
        self.spike_multiplier = float(tracker.get("spike_multiplier", 5.0))
        self.contact_multiplier = float(tracker.get("contact_level_multiplier", 10.0))
        self.calm_multiplier = float(tracker.get("calm_diff_multiplier", 2.0))
        self.recovery_multiplier = float(tracker.get("recovery_level_multiplier", 8.0))
        self.min_noise = max(0.0, float(tracker.get("min_noise", 1e-4)))
        self.freeze_noise = bool(tracker.get("freeze_noise_after_ready", False))
        self.std_enabled = bool(std_filter.get("enabled", True))
        self.std_multiplier = float(std_filter.get("subtract_multiplier", 1.0))
        self.median_enabled = bool(get_nested(preprocessing, "median_filter.enabled"))
        self.median_window = max(1, int(get_nested(preprocessing, "median_filter.window") or 3))
        self.temporal_enabled = bool(temporal.get("enabled", False))
        self.temporal_window = max(2, int(temporal.get("window", 3)))
        self.signal_mask = max(0.0, float(preprocessing.get("signal_mask", 0.0)))
        self.signal_high_mask = max(0.0, float(preprocessing.get("signal_high_mask", 0.0)))
        self.final_signal_mask = max(0.0, float(preprocessing.get("final_signal_mask", 0.0)))
        self.release_enabled = bool(release.get("enabled", False))
        self.release_slope = float(release.get("release_slope", 0.5))
        self.release_level = float(release.get("release_level", 0.0))
        self.rearm_slope = float(release.get("rearm_slope", 0.5))
        self.rearm_level = float(release.get("rearm_level", 0.0))
        self.release_frames = max(1, int(release.get("release_frames", 1)))
        self.rearm_frames = max(1, int(release.get("rearm_frames", 1)))
        self.dynamic_enabled = bool(dynamic.get("enabled", False))
        self.dynamic_min = float(dynamic.get("min_value", 5.0))
        self.dynamic_smooth = float(dynamic.get("smooth", 1.0))
        self.display_threshold = float(display.get("threshold", 100.0))
        self.reset()

    def reset(self) -> None:
        self._baseline_history: deque[np.ndarray] = deque(maxlen=self.baseline_window)
        self._median_history: deque[np.ndarray] = deque(maxlen=self.median_window)
        self._temporal_history: deque[np.ndarray] = deque(maxlen=self.temporal_window)
        self._baseline = np.zeros(self.channels, dtype=np.float64)
        self._noise = np.full(self.channels, self.min_noise, dtype=np.float64)
        self._last_raw: np.ndarray | None = None
        self._suppressed = np.zeros(self.channels, dtype=bool)
        self._release_counts = np.zeros(self.channels, dtype=np.int32)
        self._rearm_counts = np.zeros(self.channels, dtype=np.int32)
        self._previous_processed: np.ndarray | None = None

    @property
    def baseline_ready(self) -> bool:
        return not self.baseline_enabled or len(self._baseline_history) >= self.baseline_window

    def process(self, values: list[float]) -> tuple[list[float], list[float], dict[str, Any]]:
        raw = np.asarray(values, dtype=np.float64)
        if raw.size != self.channels:
            raise ValueError(f"expected {self.channels} sensor channels")
        self._update_baseline(raw)
        if self.baseline_enabled and self.suppress_initial and not self.baseline_ready:
            zeros = np.zeros(self.channels, dtype=np.float64)
            return zeros.tolist(), zeros.tolist(), self.debug_state()

        processed = raw - self._baseline if self.baseline_enabled else raw.copy()
        if self.std_enabled:
            processed -= self._noise * self.std_multiplier
        processed = np.maximum(processed, 0.0)
        if self.median_enabled and self.median_window > 1:
            self._median_history.append(processed.copy())
            if len(self._median_history) >= 2:
                processed = np.median(np.stack(self._median_history), axis=0)
        if self.temporal_enabled and self.temporal_window > 1:
            self._temporal_history.append(processed.copy())
            if len(self._temporal_history) >= 2:
                processed = np.mean(np.stack(self._temporal_history), axis=0)
        if self.signal_mask > 0:
            processed = np.where(processed < self.signal_mask, 0.0, processed)
        if self.signal_high_mask > 0:
            processed = np.where(processed > self.signal_high_mask, 0.0, processed)
        processed = self._apply_release_gate(processed)
        if self.dynamic_enabled:
            target = max(float(processed.max(initial=0.0)), self.dynamic_min)
            alpha = min(1.0, max(0.01, self.dynamic_smooth))
            self.display_threshold = (1 - alpha) * self.display_threshold + alpha * target
        display = processed.copy()
        if self.final_signal_mask > 0:
            display = np.where(display < self.final_signal_mask, 0.0, display)
        return processed.tolist(), display.tolist(), self.debug_state()

    def _update_baseline(self, raw: np.ndarray) -> None:
        if not self.baseline_enabled:
            return
        if not self.baseline_ready:
            self._baseline_history.append(raw.copy())
            stack = np.stack(self._baseline_history)
            self._baseline = np.mean(stack, axis=0)
            self._noise = np.maximum(
                np.std(stack, axis=0, ddof=1) if len(stack) > 1 else 0.0, self.min_noise
            )
            self._last_raw = raw.copy()
            return
        previous = self._last_raw if self._last_raw is not None else raw
        delta = raw - previous
        calm = np.abs(delta) < self._noise * self.calm_multiplier
        recovered = raw < self._baseline + self._noise * self.recovery_multiplier
        pressure = (delta > self._noise * self.spike_multiplier) | (
            (raw > self._baseline + self._noise * self.contact_multiplier) & calm
        )
        update_mask = recovered & calm & ~pressure
        if np.any(update_mask):
            candidate = self._baseline.copy()
            candidate[update_mask] = raw[update_mask]
            self._baseline_history.append(candidate)
            stack = np.stack(self._baseline_history)
            self._baseline = np.mean(stack, axis=0)
            if not self.freeze_noise:
                self._noise = np.maximum(np.std(stack, axis=0, ddof=1), self.min_noise)
        self._last_raw = raw.copy()

    def _apply_release_gate(self, values: np.ndarray) -> np.ndarray:
        if not self.release_enabled:
            self._previous_processed = values.copy()
            return values
        previous = self._previous_processed
        if previous is None:
            self._previous_processed = values.copy()
            return values
        result = values.copy()
        for index, (old, new) in enumerate(zip(previous, values, strict=True)):
            if self._suppressed[index]:
                level_ok = self.rearm_level <= 0 or new >= self.rearm_level
                ratio = (new - old) / max(abs(old), 1e-6)
                self._rearm_counts[index] = (
                    self._rearm_counts[index] + 1 if level_ok and ratio >= self.rearm_slope else 0
                )
                if self._rearm_counts[index] >= self.rearm_frames:
                    self._suppressed[index] = False
                    self._rearm_counts[index] = 0
                else:
                    result[index] = 0.0
            else:
                ratio = (old - new) / max(abs(old), 1e-6)
                level_ok = self.release_level > 0 and new < self.release_level
                drop_ok = ratio >= self.release_slope
                self._release_counts[index] = (
                    self._release_counts[index] + 1 if level_ok and drop_ok else 0
                )
                if self._release_counts[index] >= self.release_frames:
                    self._suppressed[index] = True
                    self._release_counts[index] = 0
                    result[index] = 0.0
        self._previous_processed = values.copy()
        return result

    def debug_state(self) -> dict[str, Any]:
        return {
            "baseline_ready": self.baseline_ready,
            "baseline_enabled": self.baseline_enabled,
            "display_threshold": self.display_threshold,
            "baseline_mean": float(self._baseline.mean()),
            "noise_mean": float(self._noise.mean()),
        }


class GraspSuccessJudge:
    """The original contact-stop rule: both sides exceed X for strictly more than Y frames."""

    def __init__(self) -> None:
        self.confirm_count = 0
        self.success = False

    def reset(self) -> None:
        self.confirm_count = 0
        self.success = False

    def update(
        self,
        left: np.ndarray,
        right: np.ndarray,
        left_config: dict[str, Any],
        right_config: dict[str, Any],
    ) -> dict[str, Any]:
        left_threshold = float(
            get_nested(left_config, "state_transitions.contact_stop_trigger.point_threshold") or 0.0
        )
        right_threshold = float(
            get_nested(right_config, "state_transitions.contact_stop_trigger.point_threshold")
            or 0.0
        )
        required = max(
            int(
                get_nested(
                    left_config, "state_transitions.contact_stop_trigger.contact_confirm_frames"
                )
                or 0
            ),
            int(
                get_nested(
                    right_config, "state_transitions.contact_stop_trigger.contact_confirm_frames"
                )
                or 0
            ),
        )
        left_peak = float(left.max(initial=0.0))
        right_peak = float(right.max(initial=0.0))
        both = left_peak > left_threshold and right_peak > right_threshold
        self.confirm_count = self.confirm_count + 1 if both else 0
        self.success = self.confirm_count > required
        return {
            "success": self.success,
            "both_sides_over_threshold": both,
            "confirm_count": self.confirm_count,
            "required_frames": required + 1,
            "left_peak": left_peak,
            "right_peak": right_peak,
            "left_threshold": left_threshold,
            "right_threshold": right_threshold,
        }


def bilateral_contact_ready(data: dict[str, Any]) -> bool:
    """Accept only fresh, baseline-ready bilateral contact for capture and inference."""
    features = data.get("features") or {}
    grasp = data.get("grasp_success") or {}
    return bool(
        data.get("frame_ready")
        and all((features.get(side) or {}).get("baseline_ready") for side in ("left", "right"))
        and grasp.get("both_sides_over_threshold")
        and grasp.get("success")
    )
