from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vtla_grip.vision_dataset import VisionDatasetStore, normalize_label


def test_capture_saves_aligned_color_depth_and_metadata(tmp_path) -> None:
    store = VisionDatasetStore(tmp_path / "dataset", tmp_path / "labels.json")
    color = np.zeros((24, 32, 3), dtype=np.uint8)
    color[:, :, 1] = 180
    depth = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)

    result = store.capture(
        color,
        depth,
        tags=["绿色圆柱体", "绿色托盘"],
        metadata={"camera": {"depth_scale": 0.001}, "workspace": {"schema_version": 1}},
    )

    sample_dir = Path(result["directory"])
    assert np.array_equal(cv2.imread(str(sample_dir / "color.png")), color)
    assert np.array_equal(cv2.imread(str(sample_dir / "depth.png"), cv2.IMREAD_UNCHANGED), depth)
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tags"] == ["绿色圆柱体", "绿色托盘"]
    assert metadata["depth"]["format"] == "Z16"
    assert metadata["camera"]["depth_scale"] == 0.001
    assert store.status()["sample_count"] == 1


def test_capture_writes_yolo_labels_and_class_map(tmp_path) -> None:
    store = VisionDatasetStore(tmp_path / "dataset", tmp_path / "labels.json")
    color = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 700, dtype=np.uint16)
    annotation = {
        "valid": True,
        "target_class": "gray_cube",
        "image_size": [32, 24],
        "annotations": [
            {
                "class_id": 1,
                "class_name": "gray_cube",
                "bbox_xywh": [8, 6, 12, 10],
                "yolo_xywh": [0.4375, 0.45833333, 0.375, 0.41666667],
                "confidence": 0.9,
            }
        ],
        "reasons": [],
    }

    result = store.capture(
        color,
        depth,
        tags=["灰色正方体"],
        metadata={},
        auto_annotation=annotation,
    )

    sample_dir = Path(result["directory"])
    assert (sample_dir / "color.txt").read_text(encoding="utf-8") == (
        "1 0.43750000 0.45833333 0.37500000 0.41666667\n"
    )
    assert (tmp_path / "dataset" / "classes.txt").read_text(encoding="utf-8") == (
        "green_cylinder\ngray_cube\ngreen_tray\n"
    )
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["auto_annotation"]["annotations"][0]["class_id"] == 1


def test_custom_label_is_persisted_and_can_be_used_for_capture(tmp_path) -> None:
    labels_path = tmp_path / "labels.json"
    store = VisionDatasetStore(tmp_path / "dataset", labels_path)

    labels = store.add_label("  强反光场景  ")

    assert "强反光场景" in labels
    assert "强反光场景" in VisionDatasetStore(tmp_path / "dataset", labels_path).labels()


def test_training_class_ids_are_stable_and_custom_classes_append(tmp_path) -> None:
    root = tmp_path / "dataset"
    store = VisionDatasetStore(root, tmp_path / "labels.json")

    classes = store.add_training_class("蓝色海绵")

    assert classes[-1] == {"class_id": 3, "name": "蓝色海绵", "label": "蓝色海绵"}
    reloaded = VisionDatasetStore(root, tmp_path / "labels.json").training_classes()
    assert reloaded == classes
    assert store.add_training_class("蓝色海绵") == classes


def test_sample_gallery_tag_update_depth_preview_and_recoverable_delete(tmp_path) -> None:
    store = VisionDatasetStore(tmp_path / "dataset", tmp_path / "labels.json")
    store.add_label("强反光场景")
    color = np.full((20, 30, 3), 80, dtype=np.uint8)
    depth = np.linspace(400, 900, 20 * 30, dtype=np.uint16).reshape(20, 30)
    captured = store.capture(color, depth, tags=["绿色圆柱体"], metadata={})
    sample_id = captured["sample_id"]

    listing = store.list_samples()
    assert listing["total"] == 1
    assert listing["samples"][0]["color_url"].endswith(f"/{sample_id}/color")
    assert store.list_samples(tag="灰色正方体")["total"] == 0

    updated = store.update_tags(sample_id, ["绿色圆柱体", "强反光场景"])
    assert updated["tags"] == ["绿色圆柱体", "强反光场景"]
    assert store.depth_preview_png(sample_id).startswith(b"\x89PNG")

    deleted = store.trash_sample(sample_id)
    assert deleted["trashed"] is True
    assert Path(deleted["recoverable_from"]).is_dir()
    assert store.status()["sample_count"] == 0


def test_sample_id_cannot_escape_dataset_root(tmp_path) -> None:
    store = VisionDatasetStore(tmp_path / "dataset", tmp_path / "labels.json")
    with pytest.raises(ValueError, match="ID 格式无效"):
        store.sample("../../metadata")


@pytest.mark.parametrize("label", ["", "bad/name", "x" * 33])
def test_rejects_invalid_custom_labels(label: str) -> None:
    with pytest.raises(ValueError):
        normalize_label(label)
