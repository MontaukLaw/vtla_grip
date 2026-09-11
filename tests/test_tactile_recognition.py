from __future__ import annotations

import numpy as np

from vtla_grip.tactile_recognition import TactileRecognitionWorkspace, extract_features


def _frames(value: float = 1.0) -> list[list[float]]:
    return [[value] * 64, [value + 1.0] * 64, [value + 2.0] * 64]


def test_feature_vector_matches_property_model_version() -> None:
    features = extract_features(_frames(), final_gripper_position=420)

    assert features.shape == (534,)
    assert features[-2] == np.float32(0.42)
    assert features[-1] == 1.0


def test_visual_classes_receive_editable_default_properties(tmp_path) -> None:
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    classes = [{"class_id": 0, "name": "cylinder", "label": "圆柱体"}]

    status = workspace.status(classes)
    workspace.add_property("cylinder", "中等")

    assert status["classes"][0]["properties"] == ["软", "硬"]
    assert workspace.properties("cylinder") == ["软", "硬", "中等"]


def test_property_rename_updates_saved_samples(tmp_path) -> None:
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    workspace.sync_vision_classes([{"name": "cube"}])
    sample = workspace.save_sample(
        vision_class="cube",
        property_label="软",
        frames=_frames(),
        timestamps=[0.0, 0.5, 1.0],
        final_gripper_position=420,
    )

    workspace.rename_property("cube", "软", "柔软")

    assert workspace.sample(sample["sample_id"])["property"] == "柔软"
    assert workspace.sample_distribution("cube") == {"柔软": 1}


def test_property_delete_moves_related_samples_to_trash(tmp_path) -> None:
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    workspace.sync_vision_classes([{"name": "cube"}])
    workspace.save_sample(
        vision_class="cube",
        property_label="软",
        frames=_frames(),
        timestamps=[0.0, 0.5, 1.0],
        final_gripper_position=420,
    )

    result = workspace.delete_property("cube", "软", confirm=True)

    assert result["trashed_sample_count"] == 1
    assert workspace.list_samples(vision_class="cube")["total"] == 0
    assert any((tmp_path / "data" / ".trash").iterdir())


def test_training_creates_independent_model_for_visual_class(tmp_path) -> None:
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    workspace.sync_vision_classes([{"name": "cube"}, {"name": "cylinder"}])
    for index in range(3):
        workspace.save_sample(
            vision_class="cube",
            property_label="软",
            frames=_frames(1.0 + index),
            timestamps=[0.0, 0.5, 1.0],
            final_gripper_position=400 + index,
        )
        workspace.save_sample(
            vision_class="cube",
            property_label="硬",
            frames=_frames(20.0 + index),
            timestamps=[0.0, 0.5, 1.0],
            final_gripper_position=600 + index,
        )

    metadata = workspace.train("cube")
    result = workspace.predict("cube", _frames(2.0), 401)

    assert metadata["classifier_mode"] == "vision_conditioned_property"
    assert metadata["sample_count"] == 6
    assert result.vision_class == "cube"
    assert result.property == "软"
    assert workspace.model_status("cylinder")["available"] is False


def test_sample_summary_exposes_each_sides_peak_for_existing_samples(tmp_path):
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    workspace.sync_vision_classes([{"name": "cube"}])
    workspace.save_sample(
        vision_class="cube", property_label="软",
        frames=[[1.0] * 32 + [8.0] * 32, [5.0] * 32 + [2.0] * 32],
        timestamps=[0.0, 1.0], final_gripper_position=420,
    )
    summary = workspace.list_samples()["samples"][0]
    assert summary["left_peak"] == 5.0
    assert summary["right_peak"] == 8.0
