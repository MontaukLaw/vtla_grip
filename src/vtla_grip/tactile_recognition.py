from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_VERSION = 3
CLASSIFIER_MODE = "vision_conditioned_property"
POSITION_MATCH_MARGIN = 10.0
DEFAULT_PROPERTIES = ("软", "硬")
ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_property(value: str) -> str:
    label = " ".join(str(value).strip().split())
    if not 1 <= len(label) <= 32:
        raise ValueError("性质名称长度必须为 1–32 个字符")
    if any(character in '/\\:*?"<>|' or ord(character) < 32 for character in label):
        raise ValueError("性质名称包含不允许的路径或控制字符")
    return label


def _class_key(vision_class: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_-]+", "_", str(vision_class)).strip("_") or "class"
    digest = hashlib.sha256(str(vision_class).encode()).hexdigest()[:10]
    return f"{name[:48]}-{digest}"


@dataclass(frozen=True)
class RecognitionResult:
    vision_class: str
    property: str
    confidence: float


class TactileRecognitionWorkspace:
    """Editable properties, tactile samples and one model per visual class."""

    def __init__(self, data_root: Path | str | None = None, model_root: Path | str | None = None) -> None:
        self.data_root = Path(data_root) if data_root else _project_root() / "records" / "tactile_recognition"
        self.model_root = Path(model_root) if model_root else _project_root() / "models" / "tactile_recognition" / "by_vision_class"
        self.labels_path = self.data_root / "properties.json"
        self.samples_root = self.data_root / "samples"
        self.trash_root = self.data_root / ".trash"
        self._lock = threading.RLock()
        self._models: dict[str, tuple[int, Any, dict[str, Any]]] = {}
        self.capture_duration_seconds = 1.0

    def status(self, vision_classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
        self.sync_vision_classes(vision_classes)
        classes = []
        for item in vision_classes:
            name = str(item["name"])
            classes.append({
                **dict(item),
                "properties": self.properties(name),
                "sample_count": self.list_samples(vision_class=name, limit=1)["total"],
                "distribution": self.sample_distribution(name),
                "model": self.model_status(name),
            })
        return {"root": str(self.data_root.resolve()), "classes": classes, "total_samples": self.list_samples(limit=1)["total"]}

    def sync_vision_classes(self, vision_classes: Sequence[dict[str, Any]]) -> None:
        with self._lock:
            document = self._labels_document()
            changed = False
            for item in vision_classes:
                name = str(item.get("name", "")).strip()
                if name and name not in document["classes"]:
                    document["classes"][name] = list(DEFAULT_PROPERTIES)
                    changed = True
            if changed:
                self._write_json(self.labels_path, document)

    def properties(self, vision_class: str) -> list[str]:
        with self._lock:
            return [normalize_property(value) for value in self._labels_document()["classes"].get(str(vision_class), [])]

    def add_property(self, vision_class: str, value: str) -> list[str]:
        label = normalize_property(value)
        with self._lock:
            document = self._labels_document()
            current = list(document["classes"].setdefault(str(vision_class), []))
            if label.casefold() not in {str(item).casefold() for item in current}:
                current.append(label)
                document["classes"][str(vision_class)] = current
                self._write_json(self.labels_path, document)
            return [str(item) for item in current]

    def rename_property(self, vision_class: str, old: str, new: str) -> list[str]:
        old_label, new_label = normalize_property(old), normalize_property(new)
        with self._lock:
            document = self._labels_document()
            current = [str(item) for item in document["classes"].get(str(vision_class), [])]
            matches = [index for index, item in enumerate(current) if item.casefold() == old_label.casefold()]
            if not matches:
                raise ValueError(f"性质不存在：{old_label}")
            if new_label.casefold() != old_label.casefold() and new_label.casefold() in {item.casefold() for item in current}:
                raise ValueError(f"性质已经存在：{new_label}")
            current[matches[0]] = new_label
            document["classes"][str(vision_class)] = current
            for path in self._sample_files():
                raw = self._read_sample(path)
                if str(raw.get("vision_class")) == str(vision_class) and str(raw.get("property", "")).casefold() == old_label.casefold():
                    raw["property"] = new_label
                    self._write_json(path, raw)
            self._write_json(self.labels_path, document)
            self._mark_model_stale(vision_class, "性质标签已重命名，请重新训练")
            return current

    def delete_property(self, vision_class: str, value: str, *, confirm: bool) -> dict[str, Any]:
        if not confirm:
            raise ValueError("删除性质需要明确确认")
        label = normalize_property(value)
        with self._lock:
            document = self._labels_document()
            current = [str(item) for item in document["classes"].get(str(vision_class), [])]
            remaining = [item for item in current if item.casefold() != label.casefold()]
            if len(remaining) == len(current):
                raise ValueError(f"性质不存在：{label}")
            trashed = 0
            for path in list(self._sample_files()):
                raw = self._read_sample(path)
                if str(raw.get("vision_class")) == str(vision_class) and str(raw.get("property", "")).casefold() == label.casefold():
                    self._trash_sample_dir(path.parent)
                    trashed += 1
            document["classes"][str(vision_class)] = remaining
            self._write_json(self.labels_path, document)
            self._mark_model_stale(vision_class, "性质标签已删除，请重新训练")
            return {"properties": remaining, "trashed_sample_count": trashed}

    def save_sample(self, *, vision_class: str, property_label: str, frames: Sequence[Sequence[float]], timestamps: Sequence[float], final_gripper_position: int | None) -> dict[str, Any]:
        label = normalize_property(property_label)
        if label.casefold() not in {item.casefold() for item in self.properties(vision_class)}:
            raise ValueError("请先添加并选择有效的性质标签")
        data = np.asarray(frames, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 64:
            raise ValueError("触觉样本至少需要两帧 64 通道数据")
        if len(timestamps) != data.shape[0]:
            raise ValueError("时间戳数量与帧数不一致")
        if final_gripper_position is not None and not 0 <= int(final_gripper_position) <= 1000:
            raise ValueError("夹爪最终闭合值必须在 0–1000 之间")
        with self._lock:
            captured_at = datetime.now(UTC)
            while True:
                sample_id = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
                sample_dir = self.samples_root / sample_id[:8] / sample_id
                if not sample_dir.exists():
                    break
                captured_at += timedelta(microseconds=1)
            document = {
                "schema_version": 1,
                "sample_id": sample_id,
                "captured_at": captured_at.isoformat(timespec="milliseconds"),
                "vision_class": str(vision_class),
                "property": label,
                "duration_seconds": float(timestamps[-1]) if timestamps else 0.0,
                "final_gripper_position": final_gripper_position,
                "timestamps": [float(value) for value in timestamps],
                "frames": data[:, :64].astype(float).tolist(),
            }
            sample_dir.mkdir(parents=True, exist_ok=False)
            self._write_json(sample_dir / "sample.json", document)
            self._mark_model_stale(vision_class, "存在尚未训练的新样本")
        return self._sample_summary(document)

    def list_samples(self, *, vision_class: str | None = None, property_label: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        summaries = []
        with self._lock:
            for path in self._sample_files():
                try:
                    raw = self._read_sample(path)
                except (OSError, TypeError, json.JSONDecodeError):
                    continue
                if vision_class is not None and str(raw.get("vision_class")) != str(vision_class):
                    continue
                if property_label is not None and str(raw.get("property")) != str(property_label):
                    continue
                summaries.append(self._sample_summary(raw))
        start = max(0, int(offset))
        stop = start + max(1, min(int(limit), 500))
        return {"total": len(summaries), "offset": start, "samples": summaries[start:stop]}

    def sample(self, sample_id: str) -> dict[str, Any]:
        raw = self._read_sample(self._sample_path(sample_id))
        return {**raw, "frame_count": len(raw.get("frames") or [])}

    def trash_sample(self, sample_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._sample_path(sample_id)
            raw = self._read_sample(path)
            destination = self._trash_sample_dir(path.parent)
            self._mark_model_stale(str(raw["vision_class"]), "训练样本已删除，请重新训练")
            return {"sample_id": sample_id, "trashed": True, "recoverable_from": str(destination.resolve())}

    def sample_distribution(self, vision_class: str) -> dict[str, int]:
        return dict(Counter(str(item["property"]) for item in self.list_samples(vision_class=vision_class, limit=500)["samples"]))

    def train(self, vision_class: str) -> dict[str, Any]:
        samples = [self.sample(item["sample_id"]) for item in self.list_samples(vision_class=vision_class, limit=500)["samples"]]
        distribution = Counter(str(item["property"]) for item in samples)
        labels = sorted(distribution)
        if len(labels) < 2:
            raise ValueError("每个视觉类别至少需要两个不同性质才能训练")
        if len(samples) < 6 or any(distribution[label] < 2 for label in labels):
            raise ValueError("训练至少需要 6 条样本，且每个性质至少 2 条")
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("缺少 xgboost，无法训练触觉模型") from exc
        features = np.vstack([extract_features(item["frames"], final_gripper_position=item.get("final_gripper_position")) for item in samples])
        label_index = {label: index for index, label in enumerate(labels)}
        target = np.asarray([label_index[str(item["property"])] for item in samples])
        ranges = self._position_ranges(samples)
        params: dict[str, Any] = {
            "max_depth": 4, "eta": 0.06, "subsample": 0.9, "colsample_bytree": 0.9,
            "seed": 42, "nthread": 2,
            "eval_metric": "mlogloss" if len(labels) > 2 else "logloss",
        }
        params.update(objective="multi:softprob", num_class=len(labels)) if len(labels) > 2 else params.update(objective="binary:logistic")
        evaluation = self._evaluate(xgb, features, target, labels, params)
        model = xgb.train(params, xgb.DMatrix(features, label=target), num_boost_round=120)
        now = datetime.now(UTC)
        version = now.strftime("%Y%m%dT%H%M%S.%fZ")
        class_root = self._class_model_root(vision_class)
        version_root = class_root / "versions" / version
        version_root.mkdir(parents=True, exist_ok=False)
        model_path = version_root / "model.json"
        model.save_model(model_path)
        metadata = {
            "schema_version": 1, "classifier_mode": CLASSIFIER_MODE,
            "feature_version": FEATURE_VERSION, "feature_count": int(features.shape[1]),
            "vision_class": str(vision_class), "properties": labels,
            "property_position_ranges": {label: list(ranges[label]) if label in ranges else None for label in labels},
            "trained_at": now.isoformat(timespec="milliseconds"), "version": version,
            "sample_count": len(samples), "sample_distribution": dict(distribution),
            "capture_duration_seconds": float(np.median([max(0.1, float(item.get("duration_seconds", 1.0))) for item in samples])),
            "evaluation": evaluation, "stale": False, "stale_reason": None,
        }
        self._write_json(version_root / "metadata.json", metadata)
        active_root = class_root / "active"
        active_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, active_root / "model.json")
        self._write_json(active_root / "metadata.json", metadata)
        self._models.pop(str(vision_class), None)
        return metadata

    def model_status(self, vision_class: str) -> dict[str, Any]:
        metadata_path = self._class_model_root(vision_class) / "active" / "metadata.json"
        versions = self.model_versions(vision_class)
        if not metadata_path.is_file():
            return {"available": False, "loaded": False, "stale": False, "versions": versions}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {"available": True, "loaded": str(vision_class) in self._models, "versions": versions, **metadata}
        except (OSError, json.JSONDecodeError):
            return {"available": False, "loaded": False, "stale": False, "error": "模型元数据损坏", "versions": versions}

    def model_versions(self, vision_class: str) -> list[dict[str, Any]]:
        root = self._class_model_root(vision_class) / "versions"
        if not root.is_dir():
            return []
        result = []
        for path in sorted(root.glob("*/metadata.json"), reverse=True):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                result.append({
                    "version": metadata.get("version", path.parent.name), "trained_at": metadata.get("trained_at"),
                    "sample_count": metadata.get("sample_count"), "properties": metadata.get("properties", []),
                    "accuracy": (metadata.get("evaluation") or {}).get("accuracy"),
                })
            except (OSError, json.JSONDecodeError):
                continue
        return result

    def restore_model(self, vision_class: str, version: str) -> dict[str, Any]:
        if not ID_PATTERN.fullmatch(version):
            raise ValueError("模型版本格式无效")
        class_root = self._class_model_root(vision_class)
        source = class_root / "versions" / version
        if not (source / "model.json").is_file() or not (source / "metadata.json").is_file():
            raise FileNotFoundError(f"模型版本不存在：{version}")
        active = class_root / "active"
        active.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "model.json", active / "model.json")
        shutil.copy2(source / "metadata.json", active / "metadata.json")
        self._models.pop(str(vision_class), None)
        return self.model_status(vision_class)

    def predict(self, vision_class: str, frames: Sequence[Sequence[float]], final_gripper_position: int | None) -> RecognitionResult:
        model, metadata = self._load_model(vision_class)
        import xgboost as xgb

        features = extract_features(frames, final_gripper_position=final_gripper_position).reshape(1, -1)
        labels = [str(item) for item in metadata["properties"]]
        raw = np.asarray(model.predict(xgb.DMatrix(features)))
        probabilities = np.asarray([1.0 - float(raw.reshape(-1)[0]), float(raw.reshape(-1)[0])]) if len(labels) == 2 else raw.reshape(1, len(labels))[0]
        selected = self._select_property(labels, probabilities, metadata.get("property_position_ranges") or {}, final_gripper_position)
        return RecognitionResult(str(vision_class), labels[selected], float(probabilities[selected]))

    def _load_model(self, vision_class: str) -> tuple[Any, dict[str, Any]]:
        active = self._class_model_root(vision_class) / "active"
        model_path, metadata_path = active / "model.json", active / "metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"视觉类别 {vision_class} 尚未训练触觉性质模型")
        mtime = max(model_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
        cached = self._models.get(str(vision_class))
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("classifier_mode") != CLASSIFIER_MODE or int(metadata.get("feature_version", -1)) != FEATURE_VERSION:
            raise ValueError("触觉性质模型与当前程序不兼容")
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("缺少 xgboost，无法运行触觉模型") from exc
        model = xgb.Booster()
        model.load_model(model_path)
        self._models[str(vision_class)] = (mtime, model, metadata)
        self.capture_duration_seconds = max(0.1, float(metadata.get("capture_duration_seconds", 1.0)))
        return model, metadata

    @staticmethod
    def _select_property(labels: Sequence[str], probabilities: np.ndarray, ranges: dict[str, Any], position: int | None) -> int:
        ranked = [int(index) for index in np.argsort(probabilities)[::-1]]
        if position is None:
            return ranked[0]
        for index in ranked:
            expected = ranges.get(labels[index])
            if isinstance(expected, list) and len(expected) == 2:
                low, high = map(float, expected)
                if low - POSITION_MATCH_MARGIN <= position <= high + POSITION_MATCH_MARGIN:
                    return index
        raise RuntimeError(f"夹爪闭合值 {position} 与该视觉类别的所有性质均不匹配")

    @staticmethod
    def _position_ranges(samples: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        values: dict[str, list[float]] = {}
        for sample in samples:
            position = sample.get("final_gripper_position")
            if isinstance(position, (int, float)) and not isinstance(position, bool) and 0 <= float(position) <= 1000:
                values.setdefault(str(sample["property"]), []).append(float(position))
        return {label: (min(items), max(items)) for label, items in values.items()}

    @staticmethod
    def _evaluate(xgb: Any, features: np.ndarray, target: np.ndarray, labels: Sequence[str], params: dict[str, Any]) -> dict[str, Any]:
        rng = np.random.default_rng(42)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for class_index in range(len(labels)):
            indices = np.flatnonzero(target == class_index)
            rng.shuffle(indices)
            count = min(len(indices) - 1, max(1, round(len(indices) * 0.2))) if len(indices) >= 2 else 0
            validation_indices.extend(int(value) for value in indices[:count])
            train_indices.extend(int(value) for value in indices[count:])
        model = xgb.train(params, xgb.DMatrix(features[train_indices], label=target[train_indices]), num_boost_round=120)
        raw = np.asarray(model.predict(xgb.DMatrix(features[validation_indices])))
        predicted = (raw.reshape(-1) >= 0.5).astype(np.int64) if len(labels) == 2 else np.argmax(raw.reshape(-1, len(labels)), axis=1)
        actual = target[validation_indices].astype(np.int64)
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for expected, found in zip(actual, predicted, strict=True):
            matrix[int(expected), int(found)] += 1
        return {
            "method": "stratified_holdout", "validation_fraction": 0.2,
            "sample_count": len(validation_indices), "training_sample_count": len(train_indices),
            "accuracy": float(np.mean(predicted == actual)), "labels": list(labels),
            "confusion_matrix": matrix.tolist(),
            "note": "按性质分层留出约 20% 样本；最终活动模型使用全部样本训练",
        }

    def _labels_document(self) -> dict[str, Any]:
        if self.labels_path.is_file():
            try:
                raw = json.loads(self.labels_path.read_text(encoding="utf-8"))
                if isinstance(raw.get("classes"), dict):
                    return {"schema_version": 1, "classes": raw["classes"]}
            except (OSError, json.JSONDecodeError):
                pass
        return {"schema_version": 1, "classes": {}}

    def _sample_files(self) -> list[Path]:
        if not self.samples_root.is_dir():
            return []
        return sorted(self.samples_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/*/sample.json"), key=lambda path: path.stat().st_mtime, reverse=True)

    def _sample_path(self, sample_id: str) -> Path:
        if not ID_PATTERN.fullmatch(sample_id):
            raise ValueError("触觉样本 ID 格式无效")
        path = self.samples_root / sample_id[:8] / sample_id / "sample.json"
        if not path.is_file():
            raise FileNotFoundError(f"触觉样本不存在：{sample_id}")
        return path

    @staticmethod
    def _read_sample(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _sample_summary(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": raw.get("sample_id"), "captured_at": raw.get("captured_at"),
            "vision_class": raw.get("vision_class"), "property": raw.get("property"),
            "duration_seconds": raw.get("duration_seconds"),
            "final_gripper_position": raw.get("final_gripper_position"),
            "frame_count": len(raw.get("frames") or []),
        }

    def _trash_sample_dir(self, sample_dir: Path) -> Path:
        self.trash_root.mkdir(parents=True, exist_ok=True)
        destination = self.trash_root / sample_dir.name
        if destination.exists():
            destination = self.trash_root / f"{sample_dir.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        sample_dir.replace(destination)
        return destination

    def _class_model_root(self, vision_class: str) -> Path:
        return self.model_root / _class_key(vision_class)

    def _mark_model_stale(self, vision_class: str, reason: str) -> None:
        path = self._class_model_root(vision_class) / "active" / "metadata.json"
        if not path.is_file():
            return
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.update(stale=True, stale_reason=reason)
            self._write_json(path, metadata)
            self._models.pop(str(vision_class), None)
        except (OSError, json.JSONDecodeError):
            return

    @staticmethod
    def _write_json(path: Path, document: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


TactileObjectRecognizer = TactileRecognitionWorkspace


def extract_features(frames: Sequence[Sequence[float]], temporal_steps: int = 16, *, final_gripper_position: float | None = None) -> np.ndarray:
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 64:
        raise ValueError("识别数据必须至少包含一帧 64 通道数据")
    data = np.maximum(data[:, :64], 0.0)
    left, right = data[:, :32], data[:, 32:64]
    summaries = np.column_stack((
        left.sum(axis=1), right.sum(axis=1), left.max(axis=1), right.max(axis=1),
        np.count_nonzero(left > 0.0, axis=1), np.count_nonzero(right > 0.0, axis=1),
        left.std(axis=1), right.std(axis=1),
    ))
    source_x, target_x = np.linspace(0.0, 1.0, summaries.shape[0]), np.linspace(0.0, 1.0, temporal_steps)
    temporal = np.column_stack([np.interp(target_x, source_x, summaries[:, index]) for index in range(8)]).reshape(-1)
    channel_stats = np.concatenate((data.mean(axis=0), data.max(axis=0), data[-1]))
    texture = _extract_texture_features(data)
    if final_gripper_position is None:
        gripper = np.asarray([np.nan, 0.0], dtype=np.float32)
    else:
        position = float(final_gripper_position)
        if not np.isfinite(position) or not 0 <= position <= 1000:
            raise ValueError("夹爪最终闭合值必须在 0 到 1000 之间")
        gripper = np.asarray([position / 1000.0, 1.0], dtype=np.float32)
    return np.concatenate((temporal, channel_stats, texture, gripper)).astype(np.float32)


def _extract_texture_features(data: np.ndarray) -> np.ndarray:
    features: list[float] = []
    epsilon = np.float32(1e-6)
    for side in (data[:, :32], data[:, 32:64]):
        totals = side.sum(axis=1, keepdims=True)
        normalized = np.divide(side, totals, out=np.zeros_like(side, dtype=np.float32), where=totals > epsilon)
        for flat_map in (normalized.mean(axis=0), normalized[-1]):
            grid = flat_map.reshape(8, 4)
            horizontal, vertical = np.abs(np.diff(grid, axis=1)).reshape(-1), np.abs(np.diff(grid, axis=0)).reshape(-1)
            features.extend(grid.reshape(-1).astype(float))
            features.extend(grid.sum(axis=1).astype(float))
            features.extend(grid.sum(axis=0).astype(float))
            for gradient in (horizontal, vertical):
                features.extend((float(gradient.mean()), float(gradient.std()), float(gradient.max()), float(np.percentile(gradient, 75))))
            ratio = np.log((float(horizontal.mean()) + float(epsilon)) / (float(vertical.mean()) + float(epsilon)))
            features.append(float(np.clip(ratio, -10.0, 10.0)))
    return np.asarray(features, dtype=np.float32)
