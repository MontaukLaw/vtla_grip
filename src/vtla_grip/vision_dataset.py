from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_LABELS = ("绿色圆柱体", "灰色正方体", "绿色托盘", "空桌面")
DEFAULT_TRAINING_CLASSES = (
    {"class_id": 0, "name": "green_cylinder", "label": "绿色圆柱体"},
    {"class_id": 1, "name": "gray_cube", "label": "灰色正方体"},
    {"class_id": 2, "name": "green_tray", "label": "绿色托盘"},
)
INVALID_LABEL_CHARACTERS = set('/\\:*?"<>|')
SAMPLE_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "records" / "vision_dataset"


def default_labels_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "vision_capture_labels.json"


def normalize_label(value: str) -> str:
    label = " ".join(str(value).strip().split())
    if not 1 <= len(label) <= 32:
        raise ValueError("标签长度必须为 1–32 个字符")
    if any(character in INVALID_LABEL_CHARACTERS or ord(character) < 32 for character in label):
        raise ValueError("标签包含不允许的路径或控制字符")
    return label


class VisionDatasetStore:
    def __init__(
        self,
        root: str | Path | None = None,
        labels_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root else default_dataset_root()
        self.labels_path = Path(labels_path) if labels_path else default_labels_path()
        self.classes_path = self.root / "classes.json"
        self._lock = threading.RLock()

    def training_classes(self) -> list[dict[str, Any]]:
        with self._lock:
            custom: list[dict[str, Any]] = []
            if self.classes_path.is_file():
                try:
                    raw = json.loads(self.classes_path.read_text(encoding="utf-8"))
                    custom = list(raw.get("classes", []))
                except (OSError, TypeError, json.JSONDecodeError):
                    custom = []
            by_id = {int(item["class_id"]): dict(item) for item in DEFAULT_TRAINING_CLASSES}
            for item in custom:
                try:
                    class_id = int(item["class_id"])
                    if class_id >= len(DEFAULT_TRAINING_CLASSES):
                        by_id[class_id] = {
                            "class_id": class_id,
                            "name": normalize_label(str(item["name"])),
                            "label": normalize_label(str(item.get("label", item["name"]))),
                        }
                except (KeyError, TypeError, ValueError):
                    continue
            return [by_id[class_id] for class_id in sorted(by_id)]

    def add_training_class(self, value: str) -> list[dict[str, Any]]:
        label = normalize_label(value)
        with self._lock:
            classes = self.training_classes()
            if label.casefold() in {str(item["label"]).casefold() for item in classes}:
                return classes
            new_class = {"class_id": len(classes), "name": label, "label": label}
            classes.append(new_class)
            self._write_training_classes(classes)
            return classes

    def labels(self) -> list[str]:
        with self._lock:
            custom: list[str] = []
            if self.labels_path.exists():
                try:
                    raw = json.loads(self.labels_path.read_text(encoding="utf-8"))
                    custom = [normalize_label(item) for item in raw.get("custom_labels", [])]
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    custom = []
            return self._deduplicate([*DEFAULT_LABELS, *custom])

    def add_label(self, value: str) -> list[str]:
        label = normalize_label(value)
        with self._lock:
            current = self.labels()
            if label.casefold() in {item.casefold() for item in current}:
                return current
            custom = [item for item in current if item not in DEFAULT_LABELS]
            custom.append(label)
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.labels_path.with_suffix(self.labels_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {"schema_version": 1, "custom_labels": custom}, ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.labels_path)
            return self._deduplicate([*DEFAULT_LABELS, *custom])

    def capture(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        *,
        tags: list[str],
        metadata: dict[str, Any],
        auto_annotation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("彩色帧格式无效")
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise ValueError("深度帧必须是 16 位单通道原始数据")
        if color_bgr.shape[:2] != depth_raw.shape:
            raise ValueError("彩色帧和深度帧没有对齐")
        if auto_annotation is not None and not auto_annotation.get("valid"):
            raise ValueError("不能保存未通过校验的自动标注")
        normalized_tags = self._deduplicate([normalize_label(tag) for tag in tags])
        if not normalized_tags:
            raise ValueError("至少选择一个场景标签")
        known = {label.casefold() for label in self.labels()}
        unknown = [tag for tag in normalized_tags if tag.casefold() not in known]
        if unknown:
            raise ValueError(f"存在未添加的标签：{', '.join(unknown)}")

        captured_at = datetime.now(UTC)
        batch = captured_at.strftime("%Y%m%d")
        sample_id = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
        sample_dir = self.root / batch / sample_id
        with self._lock:
            sample_dir.mkdir(parents=True, exist_ok=False)
            color_ok, color_png = cv2.imencode(".png", color_bgr)
            depth_ok, depth_png = cv2.imencode(".png", depth_raw)
            if not color_ok or not depth_ok:
                raise RuntimeError("PNG 编码失败")
            (sample_dir / "color.png").write_bytes(color_png.tobytes())
            (sample_dir / "depth.png").write_bytes(depth_png.tobytes())
            files = {"color": "color.png", "depth": "depth.png"}
            if auto_annotation is not None:
                lines = [
                    " ".join(
                        [
                            str(item["class_id"]),
                            *(f"{float(value):.8f}" for value in item["yolo_xywh"]),
                        ]
                    )
                    for item in auto_annotation.get("annotations", [])
                ]
                (sample_dir / "color.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                files["labels"] = "color.txt"
                self._write_training_classes(self.training_classes())
            document = {
                "schema_version": 1,
                "sample_id": sample_id,
                "captured_at": captured_at.isoformat(timespec="milliseconds"),
                "tags": normalized_tags,
                "files": files,
                "color": {"format": "BGR8", "shape": list(color_bgr.shape)},
                "depth": {"format": "Z16", "shape": list(depth_raw.shape)},
                "auto_annotation": auto_annotation,
                **metadata,
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return {
            "sample_id": sample_id,
            "captured_at": document["captured_at"],
            "tags": normalized_tags,
            "directory": str(sample_dir.resolve()),
            "files": document["files"],
        }

    def status(self, recent_limit: int = 5) -> dict[str, Any]:
        with self._lock:
            metadata_files = self._metadata_files()
            return {
                "root": str(self.root.resolve()),
                "sample_count": len(metadata_files),
                "labels": self.labels(),
                "recent": self.list_samples(limit=recent_limit)["samples"],
            }

    def list_samples(
        self, *, tag: str | None = None, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        normalized_tag = normalize_label(tag) if tag else None
        with self._lock:
            summaries: list[dict[str, Any]] = []
            for path in self._metadata_files():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    tags = [str(item) for item in raw.get("tags", [])]
                    if normalized_tag and normalized_tag.casefold() not in {
                        item.casefold() for item in tags
                    }:
                        continue
                    summaries.append(self._summary(raw, path.parent))
                except (OSError, TypeError, json.JSONDecodeError):
                    continue
            total = len(summaries)
            start = max(0, int(offset))
            stop = start + max(1, min(int(limit), 200))
            return {"total": total, "offset": start, "samples": summaries[start:stop]}

    def sample(self, sample_id: str) -> dict[str, Any]:
        sample_dir = self._sample_dir(sample_id)
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"视觉样本不存在：{sample_id}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {**raw, "directory": str(sample_dir.resolve())}

    def update_tags(self, sample_id: str, tags: list[str]) -> dict[str, Any]:
        normalized = self._deduplicate([normalize_label(tag) for tag in tags])
        if not normalized:
            raise ValueError("至少保留一个场景标签")
        known = {label.casefold() for label in self.labels()}
        unknown = [tag for tag in normalized if tag.casefold() not in known]
        if unknown:
            raise ValueError(f"存在未添加的标签：{', '.join(unknown)}")
        with self._lock:
            sample_dir = self._sample_dir(sample_id)
            metadata_path = sample_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"视觉样本不存在：{sample_id}")
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw["tags"] = normalized
            temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(metadata_path)
            return self._summary(raw, sample_dir)

    def trash_sample(self, sample_id: str) -> dict[str, Any]:
        with self._lock:
            sample_dir = self._sample_dir(sample_id)
            if not sample_dir.is_dir():
                raise FileNotFoundError(f"视觉样本不存在：{sample_id}")
            trash_root = self.root / ".trash"
            trash_root.mkdir(parents=True, exist_ok=True)
            destination = trash_root / sample_id
            if destination.exists():
                suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                destination = trash_root / f"{sample_id}-{suffix}"
            sample_dir.replace(destination)
            return {
                "sample_id": sample_id,
                "trashed": True,
                "recoverable_from": str(destination.resolve()),
            }

    def image_path(self, sample_id: str, kind: str) -> Path:
        filename = {"color": "color.png", "depth": "depth.png"}.get(kind)
        if filename is None:
            raise ValueError("不支持的样本图像类型")
        path = self._sample_dir(sample_id) / filename
        if not path.is_file():
            raise FileNotFoundError(f"样本图像不存在：{sample_id}/{filename}")
        return path

    def depth_preview_png(self, sample_id: str) -> bytes:
        depth = cv2.imread(str(self.image_path(sample_id, "depth")), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise ValueError("样本深度图格式无效")
        valid = depth[depth > 0]
        if valid.size:
            lower, upper = np.percentile(valid, [2, 98])
            span = max(1.0, float(upper - lower))
            normalized = np.clip((depth.astype(np.float32) - lower) / span * 255.0, 0, 255)
            image = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
            image[depth == 0] = 0
        else:
            image = np.zeros((*depth.shape, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("深度预览编码失败")
        return encoded.tobytes()

    def _metadata_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        files: list[Path] = []
        for batch in self.root.iterdir():
            if not batch.is_dir() or not re.fullmatch(r"\d{8}", batch.name):
                continue
            files.extend(batch.glob("*/metadata.json"))
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)

    def _write_training_classes(self, classes: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "classes": classes}
        temporary = self.classes_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.classes_path)
        (self.root / "classes.txt").write_text(
            "\n".join(str(item["name"]) for item in classes) + "\n", encoding="utf-8"
        )

    def _sample_dir(self, sample_id: str) -> Path:
        if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise ValueError("视觉样本 ID 格式无效")
        root = self.root.resolve()
        candidate = (root / sample_id[:8] / sample_id).resolve()
        if len(candidate.parents) < 2 or candidate.parents[1] != root:
            raise ValueError("视觉样本路径超出数据集目录")
        return candidate

    @staticmethod
    def _summary(raw: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
        return {
            "sample_id": raw.get("sample_id"),
            "captured_at": raw.get("captured_at"),
            "tags": raw.get("tags", []),
            "directory": str(sample_dir.resolve()),
            "workspace_calibrated": raw.get("workspace") is not None,
            "auto_annotation": raw.get("auto_annotation"),
            "color_url": f"/api/vision/dataset/samples/{raw.get('sample_id')}/color",
            "depth_preview_url": (
                f"/api/vision/dataset/samples/{raw.get('sample_id')}/depth-preview"
            ),
        }

    @staticmethod
    def _deduplicate(values: list[str] | tuple[str, ...]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result
