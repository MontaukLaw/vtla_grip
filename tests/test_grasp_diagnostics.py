from vtla_grip.grasp_diagnostics import closure_snapshot


def test_actual_position_is_not_replaced_by_command():
    class Gripper:
        def read_position(self):
            return 350

    snapshot = closure_snapshot(Gripper(), {
        "frame_ready": True,
        "features": {"left": {"baseline_ready": True}, "right": {"baseline_ready": True}},
        "grasp_success": {"left_peak": 30, "right_peak": 0, "success": False},
    }, 100, 100)
    assert snapshot["actual_position"] == 350
    assert snapshot["position_command"] == 100
    assert snapshot["position_error"] == 250
    assert snapshot["right_peak"] == 0
    assert not snapshot["contact_accepted"]


def test_position_read_failure_is_recorded_without_interrupting_control():
    class Gripper:
        def read_position(self):
            raise RuntimeError("serial timeout")

    snapshot = closure_snapshot(Gripper(), {}, 100, 100)
    assert snapshot["actual_position"] is None
    assert snapshot["position_error"] is None
    assert snapshot["position_read_error"] == "serial timeout"
