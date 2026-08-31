from vtla_grip.place_target import PlaceTargetStore


def test_place_target_persists_single_overwritable_point(tmp_path) -> None:
    path = tmp_path / "place_target.json"
    store = PlaceTargetStore(path)
    assert store.status()["configured"] is False

    store.save(
        (100, 200),
        (0.1, 0.2, 0.9),
        (10.0, 20.0, 30.0),
        "workspace-1",
        "calibration-1",
    )
    store.save(
        (300, 250),
        (0.3, 0.25, 0.8),
        (40.0, 50.0, 60.0),
        "workspace-1",
        "calibration-1",
    )

    restored = PlaceTargetStore(path).status()
    assert restored["configured"] is True
    assert restored["pixel_uv"] == (300, 250)
    assert restored["base_xyz_mm"] == (40.0, 50.0, 60.0)
