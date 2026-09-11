from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ..calibration_app import CalibrationRecorder, default_output_path
from ..calibration_solver import latest_calibration_record, save_report, solve_calibration
from ..camera_app import ClickResult
from ..config import AppConfig, TargetingConfig, TransferConfig, default_config_path, load_config
from ..hardware import (
    DH5Gripper,
    DH5ProtocolError,
    DualInfineonSensors,
    SensorProtocolError,
    list_serial_ports,
)
from ..place_target import PlaceTargetStore
from ..robot_client import RobotServiceError
from ..table_workspace import TableWorkspaceStore, WorkspacePoint
from ..tactile_capture import TactileCaptureManager, TactileTrainingManager
from ..tactile_recognition import TactileRecognitionWorkspace
from ..transfer import (
    SUPPORTED_TRANSFER_TARGETS,
    TransferError,
    TransferTaskManager,
    build_transfer_plan,
)
from ..transforms import transform_point
from ..vision import TabletopVisionDetector, depth_object_proposals
from ..vision_dataset import VisionDatasetStore
from .camera_service import CameraService
from .log_store import LogStore
from .robot_gateway import RobotGateway


class PixelRequest(BaseModel):
    u: int = Field(ge=0)
    v: int = Field(ge=0)


class RobotMoveRequest(BaseModel):
    pose: list[float] = Field(min_length=6, max_length=6)
    mode: str = "MoveJ_P"
    speed: int = Field(default=5, ge=1, le=100)
    acc: int = Field(default=10, ge=1, le=100)


class GripperConnectRequest(BaseModel):
    port: str
    baud_rate: int = Field(default=115200, gt=0)
    modbus_id: int = Field(default=1, ge=1, le=247)
    stop_bits: int = 1
    parity: str = "N"


class GripperCommandRequest(BaseModel):
    position: int | None = Field(default=None, ge=0, le=1000)
    speed: int | None = Field(default=None, ge=1, le=100)
    force: int | None = Field(default=None, ge=20, le=100)


class SensorConnectRequest(BaseModel):
    left_port: str
    right_port: str


class CalibrationSolveRequest(BaseModel):
    input_path: str | None = None


class CalibrationResumeRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=160)


class TargetingSettingsRequest(BaseModel):
    marker_to_tip_x_mm: float = Field(ge=-500, le=500)
    marker_to_tip_y_mm: float = Field(ge=-500, le=500)
    marker_to_tip_z_mm: float = Field(ge=0, le=800)


class TargetMoveRequest(BaseModel):
    confirm: bool
    speed: int = Field(default=3, ge=1, le=100)


class SensorSettingsUpdateRequest(BaseModel):
    side: str
    values: dict[str, Any]


class VisionDetectRequest(BaseModel):
    target_class: str | None = None


class TransferPreviewRequest(BaseModel):
    action: str = "place_into_tray"
    target_class: str
    target_property: str | None = None


class TransferExecuteRequest(BaseModel):
    plan_id: str = Field(min_length=8, max_length=64)
    confirm: bool


class TransferSettingsRequest(BaseModel):
    transit_speed: int = Field(ge=1, le=100)
    approach_speed: int = Field(ge=1, le=100)
    grasp_clearance_mm: float = Field(ge=-100, le=100)
    pick_lift_mm: float = Field(ge=20, le=300)
    place_approach_mm: float = Field(ge=20, le=300)
    place_clearance_mm: float = Field(ge=0, le=150)


class VisionCaptureRequest(BaseModel):
    class_id: int | None = Field(default=None, ge=0)
    empty_table: bool = False
    boxes: list[list[int]] = Field(default_factory=list, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)


class VisionTrainingClassRequest(BaseModel):
    label: str = Field(min_length=1, max_length=32)


class VisionLabelRequest(BaseModel):
    label: str = Field(min_length=1, max_length=32)


class VisionSampleTagsRequest(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=20)


class VisionSampleDeleteRequest(BaseModel):
    confirm: bool


class TactilePropertyRequest(BaseModel):
    vision_class: str = Field(min_length=1, max_length=64)
    property: str = Field(min_length=1, max_length=32)


class TactilePropertyRenameRequest(BaseModel):
    vision_class: str = Field(min_length=1, max_length=64)
    old_property: str = Field(min_length=1, max_length=32)
    new_property: str = Field(min_length=1, max_length=32)


class TactilePropertyDeleteRequest(TactilePropertyRequest):
    confirm: bool


class TactileSampleDeleteRequest(BaseModel):
    confirm: bool


class TactileTrainRequest(BaseModel):
    vision_class: str = Field(min_length=1, max_length=64)


class TactileRestoreRequest(TactileTrainRequest):
    version: str = Field(min_length=8, max_length=64)


def _preferred_calibration_record(config: AppConfig) -> Path | None:
    solution_path = Path(config.calibration.output_path)
    if solution_path.is_file():
        try:
            source = json.loads(solution_path.read_text(encoding="utf-8")).get("source_record")
            if source and Path(source).is_file():
                return Path(source)
        except (AttributeError, OSError, json.JSONDecodeError):
            pass
    try:
        return latest_calibration_record()
    except FileNotFoundError:
        return None


class CalibrationState:
    def __init__(self, preferred_record: Path | None = None) -> None:
        self.recorder: CalibrationRecorder | None = None
        self.output: Path | None = None
        self.sample_count = 0
        self.last_sample: dict[str, Any] | None = None
        try:
            latest = CalibrationRecorder.resume(preferred_record or latest_calibration_record())
        except (OSError, TypeError, ValueError):
            pass
        else:
            self.output = latest.path
            self.sample_count = latest.sample_count
            self.last_sample = latest.last_sample

    def attach(self, recorder: CalibrationRecorder) -> None:
        self.recorder = recorder
        self.output = recorder.path
        self.sample_count = recorder.sample_count
        self.last_sample = recorder.last_sample

    def stop(self) -> None:
        if self.recorder is not None:
            self.output = self.recorder.path
            self.sample_count = self.recorder.sample_count
            self.last_sample = self.recorder.last_sample
        self.recorder = None

    def status(self) -> dict[str, Any]:
        return {
            "active": self.recorder is not None,
            "sample_count": (self.recorder.sample_count if self.recorder else self.sample_count),
            "output": str(self.output.resolve()) if self.output else None,
            "last_sample": self.last_sample,
        }


class ControlState:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logs = LogStore()
        self.camera = CameraService(config.camera)
        self.robot = RobotGateway(config.robot)
        self.gripper = DH5Gripper(config.gripper)
        self.sensors = DualInfineonSensors(config.sensors)
        self.vision = TabletopVisionDetector(config.vision)
        self.vision_dataset = VisionDatasetStore()
        self.workspace = TableWorkspaceStore()
        self.place_target = PlaceTargetStore()
        self.tactile_recognizer = TactileRecognitionWorkspace()
        self.tactile_capture = TactileCaptureManager(
            self.gripper,
            self.sensors,
            self.tactile_recognizer,
            config.transfer,
            self.logs.add,
        )
        self.tactile_training = TactileTrainingManager(
            self.tactile_recognizer, self.logs.add
        )
        self.calibration = CalibrationState(_preferred_calibration_record(config))
        self.selected_point: ClickResult | None = None
        self.transfer = TransferTaskManager(
            self.robot,
            self.gripper,
            self.sensors,
            config.transfer,
            self.logs.add,
            self.tactile_recognizer,
        )


def create_app(config: AppConfig | None = None) -> FastAPI:
    state = ControlState(config or load_config())
    app = FastAPI(title="VTLA Grip Local Control API", version="0.2.0")
    app.state.control = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "vtla-grip", "version": app.version}

    @app.get("/api/devices")
    def devices() -> dict[str, Any]:
        return {
            "camera": state.camera.status(),
            "robot": state.robot.status(include_pose=True),
            "gripper": state.gripper.status(),
            "sensors": state.sensors.status(),
            "serial_ports": list_serial_ports(),
        }

    @app.post("/api/camera/start")
    def camera_start() -> dict[str, Any]:
        def execute() -> dict[str, Any]:
            result = state.camera.start()
            state.selected_point = None
            return result

        return _run(state, "camera", "D435 started", execute)

    @app.post("/api/camera/stop")
    def camera_stop() -> dict[str, Any]:
        def execute() -> dict[str, Any]:
            result = state.camera.stop()
            state.selected_point = None
            return result

        return _run(state, "camera", "D435 stopped", execute)

    @app.get("/api/camera/status")
    def camera_status() -> dict[str, Any]:
        return state.camera.status()

    @app.get("/api/camera/selected-point")
    def camera_selected_point() -> dict[str, Any] | None:
        return _click_dict(state.selected_point) if state.selected_point else None

    @app.get("/api/camera/stream")
    def camera_stream() -> StreamingResponse:
        if not state.camera.status()["running"]:

            def execute() -> dict[str, Any]:
                result = state.camera.start()
                state.selected_point = None
                return result

            _run(state, "camera", "D435 started", execute)
        return StreamingResponse(
            state.camera.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/camera/deproject")
    def camera_deproject(request: PixelRequest) -> dict[str, Any]:
        try:
            point = state.camera.deproject(request.u, request.v)
            state.selected_point = point
            result = _click_dict(point)
            state.logs.add("info", "camera", "Selected camera point", **result)
            return result
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/vision/detect")
    def vision_detect(request: VisionDetectRequest | None = None) -> dict[str, Any]:
        target_class = request.target_class if request else None
        if target_class not in {None, "green_cylinder", "gray_cube"}:
            raise HTTPException(status_code=400, detail="unsupported visual target class")
        try:
            enriched = _detect_vision(state, target_class)
            selected = enriched.get("recommended_object")
            if selected and selected.get("camera_xyz_m"):
                state.selected_point = ClickResult(
                    tuple(selected["center_uv"]),
                    float(selected["depth_m"]),
                    tuple(selected["camera_xyz_m"]),
                )
            state.logs.add(
                "info",
                "vision",
                "Tabletop visual detection completed",
                detection_count=len(enriched["detections"]),
                recommended_object_id=enriched.get("recommended_object_id"),
                destination_id=enriched.get("destination_id"),
            )
            return enriched
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/transfer/status")
    def transfer_status() -> dict[str, Any]:
        return state.transfer.status()

    @app.get("/api/transfer/settings")
    def transfer_settings() -> dict[str, Any]:
        return _transfer_settings_dict(state.config.transfer)

    @app.post("/api/transfer/settings")
    def transfer_settings_update(request: TransferSettingsRequest) -> dict[str, Any]:
        if (
            max(request.transit_speed, request.approach_speed)
            > state.config.targeting.maximum_speed
        ):
            raise HTTPException(status_code=403, detail="搬运速度超过系统运动速度上限")
        updated = replace(
            state.config.transfer,
            transit_speed=request.transit_speed,
            approach_speed=request.approach_speed,
            grasp_clearance_mm=request.grasp_clearance_mm,
            pick_lift_mm=request.pick_lift_mm,
            place_approach_mm=request.place_approach_mm,
            place_clearance_mm=request.place_clearance_mm,
        )
        try:
            state.transfer.update_config(updated)
            state.tactile_capture.config = updated
            _persist_transfer_settings(updated)
            state.config = replace(state.config, transfer=updated)
        except (OSError, TransferError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state.logs.add(
            "warning",
            "transfer",
            "自动搬运速度与高度设置已保存；旧预览已失效",
            **_transfer_settings_dict(updated),
        )
        return _transfer_settings_dict(updated)

    @app.post("/api/transfer/preview")
    def transfer_preview(request: TransferPreviewRequest) -> dict[str, Any]:
        if request.action not in {"place_into_tray", "place_at_position"}:
            raise HTTPException(status_code=400, detail="不支持的放置目标")
        if request.target_class not in SUPPORTED_TRANSFER_TARGETS:
            raise HTTPException(status_code=400, detail="第一版只支持绿色圆柱体和灰色正方体")
        if state.workspace.workspace is None:
            raise HTTPException(status_code=409, detail="必须先完成四点桌面工作区标定")
        solution = _calibration_solution(state.config)
        if not solution or not solution.get("quality", {}).get("hover_ready"):
            raise HTTPException(status_code=409, detail="外参尚未通过自动搬运质量门槛")
        try:
            vision = _detect_vision(state, request.target_class)
            destination_override = None
            if request.action == "place_at_position":
                target = state.place_target.target
                if target is None:
                    raise TransferError("请先用鼠标标定指定放置点")
                workspace = state.workspace.workspace
                if (
                    workspace is None
                    or target.workspace_created_at != workspace.created_at
                    or target.calibration_created_at != solution.get("created_at")
                ):
                    raise TransferError("桌面或外参已改变，请重新标定指定放置点")
                u, v = target.pixel_uv
                destination_override = {
                    "detection_id": "specified-position",
                    "class_name": "specified_position",
                    "confidence": 1.0,
                    "bbox_xywh": [u - 8, v - 8, 16, 16],
                    "center_uv": [u, v],
                    "depth_m": target.camera_xyz_m[2],
                    "base_xyz_mm": list(target.base_xyz_mm),
                }
                vision["warnings"] = [
                    warning
                    for warning in vision.get("warnings", [])
                    if "绿色实心托盘" not in warning
                ]
            plan = build_transfer_plan(
                vision,
                request.target_class,
                state.config.targeting,
                state.config.transfer,
                solution,
                destination_override=destination_override,
                target_property=request.target_property,
            )
            status = state.transfer.set_preview(plan)
            state.logs.add(
                "warning",
                "transfer",
                "结构化搬运预览已生成，等待人工确认",
                plan_id=plan["plan_id"],
                target_class=request.target_class,
                destination=request.action,
            )
            return {"vision": vision, "transfer": status}
        except (RuntimeError, ValueError, TransferError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/transfer/execute")
    def transfer_execute(request: TransferExecuteRequest) -> dict[str, Any]:
        if request.confirm is not True:
            raise HTTPException(status_code=400, detail="执行真实搬运必须明确确认")
        try:
            _validate_transfer_readiness(state)
            status = state.transfer.status()
            plan = status.get("plan")
            if not isinstance(plan, dict) or plan.get("plan_id") != request.plan_id:
                raise TransferError("搬运预览不存在或已被替换")
            return state.transfer.start(request.plan_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (OSError, RobotServiceError, RuntimeError, ValueError, TransferError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/transfer/stop")
    def transfer_stop() -> dict[str, Any]:
        return state.transfer.request_stop()

    @app.get("/api/workspace/status")
    def workspace_status() -> dict[str, Any]:
        return state.workspace.status()

    def current_place_target_status() -> dict[str, Any]:
        status = state.place_target.status()
        target = state.place_target.target
        workspace = state.workspace.workspace
        solution = _calibration_solution(state.config)
        status["valid"] = bool(
            target
            and workspace
            and solution
            and target.workspace_created_at == workspace.created_at
            and target.calibration_created_at == solution.get("created_at")
        )
        return status

    @app.get("/api/place-target/status")
    def place_target_status() -> dict[str, Any]:
        return current_place_target_status()

    @app.post("/api/place-target/point")
    def place_target_point(request: PixelRequest) -> dict[str, Any]:
        workspace = state.workspace.workspace
        if workspace is None:
            raise HTTPException(status_code=409, detail="必须先完成四点桌面工作区标定")
        solution = _calibration_solution(state.config)
        if not solution or not solution.get("quality", {}).get("hover_ready"):
            raise HTTPException(status_code=409, detail="外参尚未通过自动搬运质量门槛")
        try:
            clicked = state.camera.deproject(request.u, request.v)
            camera_xyz = workspace.intersect_camera_ray_with_plane(clicked.camera_xyz_m)
            base_xyz = transform_point(
                np.asarray(solution["transform_4x4"]),
                np.asarray(camera_xyz, dtype=np.float64) * 1000.0,
            )
            state.transfer.clear_preview()
            state.place_target.save(
                (request.u, request.v),
                camera_xyz,
                tuple(map(float, base_xyz)),
                workspace.created_at,
                str(solution["created_at"]),
            )
            state.logs.add(
                "warning",
                "transfer",
                "Mouse-specified place target saved; old preview invalidated",
                pixel_uv=[request.u, request.v],
                base_xyz_mm=list(map(float, base_xyz)),
            )
            return current_place_target_status()
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/workspace/start")
    def workspace_start() -> dict[str, Any]:
        status = state.workspace.start()
        state.logs.add("info", "workspace", "Table workspace calibration started")
        return status

    @app.post("/api/workspace/point")
    def workspace_point(request: PixelRequest) -> dict[str, Any]:
        solution = _calibration_solution(state.config)
        if not solution or not solution.get("quality", {}).get("hover_ready"):
            raise HTTPException(
                status_code=409,
                detail="相机到 Base 外参尚未通过质量门槛，不能保存桌面工作区",
            )
        try:
            point = state.camera.deproject(request.u, request.v)
            camera_mm = np.asarray(point.camera_xyz_m, dtype=np.float64) * 1000.0
            base_xyz = transform_point(np.asarray(solution["transform_4x4"]), camera_mm)
            status = state.workspace.add_point(
                WorkspacePoint(
                    pixel_uv=point.pixel,
                    depth_m=point.depth_m,
                    camera_xyz_m=point.camera_xyz_m,
                    base_xyz_mm=tuple(map(float, base_xyz)),
                )
            )
            state.logs.add(
                "info",
                "workspace",
                "Table workspace point captured",
                pixel_uv=list(point.pixel),
                point_count=status["point_count"],
                calibrated=status["calibrated"],
            )
            return status
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/vision/dataset/status")
    def vision_dataset_status() -> dict[str, Any]:
        return state.vision_dataset.status()

    @app.get("/api/vision/labels")
    def vision_labels() -> dict[str, Any]:
        return {"labels": state.vision_dataset.labels()}

    @app.post("/api/vision/labels")
    def vision_label_add(request: VisionLabelRequest) -> dict[str, Any]:
        try:
            labels = state.vision_dataset.add_label(request.label)
            state.logs.add(
                "info", "vision_dataset", "Vision capture label added", label=request.label
            )
            return {"labels": labels}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/vision/classes")
    def vision_training_classes() -> dict[str, Any]:
        return {"classes": state.vision_dataset.training_classes()}

    @app.post("/api/vision/classes")
    def vision_training_class_add(request: VisionTrainingClassRequest) -> dict[str, Any]:
        try:
            classes = state.vision_dataset.add_training_class(request.label)
            state.logs.add(
                "info", "vision_dataset", "Vision training class added", label=request.label
            )
            return {"classes": classes}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/vision/capture")
    def vision_capture(request: VisionCaptureRequest) -> dict[str, Any]:
        try:
            color, depth, _ = state.camera.snapshot()
            camera_context = state.camera.capture_context()
            solution = _calibration_solution(state.config)
            workspace = state.workspace.workspace
            if workspace is None:
                raise ValueError("请先完成四点桌面工作区标定，再采集自动标注数据")
            classes = state.vision_dataset.training_classes()
            selected_class = next(
                (item for item in classes if item["class_id"] == request.class_id), None
            )
            if request.empty_table:
                if request.boxes:
                    raise ValueError("空桌面负样本不能包含标注框")
                class_label = "空桌面"
                auto_annotation = _annotation_from_boxes(
                    [], None, None, color.shape[1], color.shape[0], workspace.polygon_uv
                )
            else:
                if selected_class is None:
                    raise ValueError("请选择有效的训练类别")
                if not request.boxes:
                    raise ValueError("普通物体样本至少需要保留一个标注框")
                class_label = str(selected_class["label"])
                auto_annotation = _annotation_from_boxes(
                    request.boxes,
                    int(selected_class["class_id"]),
                    str(selected_class["name"]),
                    color.shape[1],
                    color.shape[0],
                    workspace.polygon_uv,
                )
            result = state.vision_dataset.capture(
                color,
                depth,
                tags=list(dict.fromkeys([class_label, *request.tags])),
                auto_annotation=auto_annotation,
                metadata={
                    "camera": {**state.camera.metadata(), **camera_context},
                    "workspace": workspace.to_dict() if workspace else None,
                    "calibration": (
                        {
                            "transform_name": solution.get("transform_name"),
                            "transform_4x4": solution.get("transform_4x4"),
                            "quality": solution.get("quality"),
                            "source_record": solution.get("source_record"),
                            "created_at": solution.get("created_at"),
                        }
                        if solution
                        else None
                    ),
                    "vision_config": asdict(state.config.vision),
                },
            )
            state.logs.add(
                "info",
                "vision_dataset",
                "Aligned visual training sample captured",
                sample_id=result["sample_id"],
                tags=result["tags"],
                directory=result["directory"],
            )
            return {**result, "dataset": state.vision_dataset.status()}
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/vision/auto-annotations")
    def vision_auto_annotations() -> dict[str, Any]:
        try:
            workspace = state.workspace.workspace
            if workspace is None:
                raise ValueError("请先完成四点桌面工作区标定")
            _, depth, depth_scale = state.camera.snapshot()
            context = state.camera.capture_context()
            return depth_object_proposals(
                depth,
                depth_scale,
                context["intrinsics"],
                workspace,
                state.config.vision,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/vision/dataset/samples")
    def vision_dataset_samples(
        tag: str | None = None,
        limit: int = Query(default=100, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return state.vision_dataset.list_samples(tag=tag, limit=limit, offset=offset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/vision/dataset/samples/{sample_id}")
    def vision_dataset_sample(sample_id: str) -> dict[str, Any]:
        try:
            return state.vision_dataset.sample(sample_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/vision/dataset/samples/{sample_id}/color")
    def vision_dataset_color(sample_id: str) -> FileResponse:
        try:
            return FileResponse(
                state.vision_dataset.image_path(sample_id, "color"), media_type="image/png"
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/vision/dataset/samples/{sample_id}/depth-preview")
    def vision_dataset_depth_preview(sample_id: str) -> Response:
        try:
            return Response(
                content=state.vision_dataset.depth_preview_png(sample_id), media_type="image/png"
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/vision/dataset/samples/{sample_id}/tags")
    def vision_dataset_tags_update(
        sample_id: str, request: VisionSampleTagsRequest
    ) -> dict[str, Any]:
        try:
            result = state.vision_dataset.update_tags(sample_id, request.tags)
            state.logs.add(
                "info",
                "vision_dataset",
                "Vision sample tags updated",
                sample_id=sample_id,
                tags=result["tags"],
            )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/vision/dataset/samples/{sample_id}")
    def vision_dataset_sample_delete(
        sample_id: str, request: VisionSampleDeleteRequest
    ) -> dict[str, Any]:
        if request.confirm is not True:
            raise HTTPException(status_code=400, detail="删除视觉样本需要明确确认")
        try:
            result = state.vision_dataset.trash_sample(sample_id)
            state.logs.add(
                "warning",
                "vision_dataset",
                "Vision sample moved to dataset trash",
                **result,
            )
            return {**result, "dataset": state.vision_dataset.status()}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def require_tactile_vision_class(vision_class: str) -> dict[str, Any]:
        selected = next(
            (
                item
                for item in state.vision_dataset.training_classes()
                if str(item["name"]) == str(vision_class)
            ),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=400, detail=f"视觉类别不存在：{vision_class}")
        return selected

    @app.get("/api/tactile/status")
    def tactile_status() -> dict[str, Any]:
        return {
            **state.tactile_recognizer.status(state.vision_dataset.training_classes()),
            "capture": state.tactile_capture.status(),
            "training": state.tactile_training.status(),
        }

    @app.post("/api/tactile/properties")
    def tactile_property_add(request: TactilePropertyRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        try:
            properties = state.tactile_recognizer.add_property(
                request.vision_class, request.property
            )
            state.logs.add(
                "info",
                "tactile_recognition",
                "Tactile property added",
                vision_class=request.vision_class,
                property=request.property,
            )
            return {"properties": properties}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/tactile/properties/rename")
    def tactile_property_rename(request: TactilePropertyRenameRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        try:
            properties = state.tactile_recognizer.rename_property(
                request.vision_class,
                request.old_property,
                request.new_property,
            )
            state.logs.add(
                "warning",
                "tactile_recognition",
                "Tactile property renamed; model marked stale",
                vision_class=request.vision_class,
                old_property=request.old_property,
                new_property=request.new_property,
            )
            return {"properties": properties}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/tactile/properties")
    def tactile_property_delete(request: TactilePropertyDeleteRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        try:
            result = state.tactile_recognizer.delete_property(
                request.vision_class, request.property, confirm=request.confirm
            )
            state.logs.add(
                "warning",
                "tactile_recognition",
                "Tactile property and related samples moved to trash",
                vision_class=request.vision_class,
                property=request.property,
                **result,
            )
            return result
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/tactile/samples")
    def tactile_samples(
        vision_class: str | None = None,
        property: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return state.tactile_recognizer.list_samples(
            vision_class=vision_class,
            property_label=property,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/tactile/samples/{sample_id}")
    def tactile_sample(sample_id: str) -> dict[str, Any]:
        try:
            return state.tactile_recognizer.sample(sample_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/tactile/samples/{sample_id}")
    def tactile_sample_delete(
        sample_id: str, request: TactileSampleDeleteRequest
    ) -> dict[str, Any]:
        if request.confirm is not True:
            raise HTTPException(status_code=400, detail="删除触觉样本需要明确确认")
        try:
            result = state.tactile_recognizer.trash_sample(sample_id)
            state.logs.add(
                "warning", "tactile_recognition", "Tactile sample moved to trash", **result
            )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/tactile/capture/status")
    def tactile_capture_status() -> dict[str, Any]:
        return state.tactile_capture.status()

    @app.post("/api/tactile/capture/start")
    def tactile_capture_start(request: TactilePropertyRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        if state.transfer.running:
            raise HTTPException(status_code=409, detail="自动搬运正在执行，不能采集触觉数据")
        gripper = state.gripper.status()
        sensors = state.sensors.status()
        if not gripper.get("connected"):
            raise HTTPException(status_code=409, detail="请先连接 DH5 夹爪")
        if not sensors.get("connected") or not sensors.get("frame_ready"):
            raise HTTPException(status_code=409, detail="请先连接双侧触觉传感器并等待数据就绪")
        try:
            return state.tactile_capture.start(request.vision_class, request.property)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tactile/capture/stop")
    def tactile_capture_stop() -> dict[str, Any]:
        return state.tactile_capture.stop()

    @app.post("/api/tactile/capture/save")
    def tactile_capture_save(request: TactilePropertyRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        try:
            result = state.tactile_capture.save(request.vision_class, request.property)
            state.logs.add(
                "info",
                "tactile_recognition",
                "Tactile sample saved",
                vision_class=request.vision_class,
                property=request.property,
            )
            return result
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tactile/capture/discard")
    def tactile_capture_discard() -> dict[str, Any]:
        try:
            return state.tactile_capture.discard()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/tactile/training/status")
    def tactile_training_status() -> dict[str, Any]:
        return state.tactile_training.status()

    @app.post("/api/tactile/train")
    def tactile_train(request: TactileTrainRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        try:
            return state.tactile_training.start(request.vision_class)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tactile/models/restore")
    def tactile_model_restore(request: TactileRestoreRequest) -> dict[str, Any]:
        require_tactile_vision_class(request.vision_class)
        if state.tactile_training.running:
            raise HTTPException(status_code=409, detail="触觉模型训练中，不能恢复历史版本")
        try:
            result = state.tactile_recognizer.restore_model(
                request.vision_class, request.version
            )
            state.logs.add(
                "warning",
                "tactile_recognition",
                "Tactile model version restored",
                vision_class=request.vision_class,
                version=request.version,
            )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/robot/service/start")
    def robot_service_start(simulate: bool = False) -> dict[str, Any]:
        return _run(
            state,
            "robot",
            "RM65 service started",
            lambda: state.robot.start_service(simulate=simulate),
        )

    @app.get("/api/robot/status")
    def robot_status() -> dict[str, Any]:
        return state.robot.status()

    @app.post("/api/robot/connect")
    def robot_connect() -> dict[str, Any]:
        return _run(state, "robot", "RM65 connected", state.robot.connect)

    @app.post("/api/robot/disconnect")
    def robot_disconnect() -> dict[str, Any]:
        return _run(state, "robot", "RM65 disconnected", state.robot.disconnect)

    @app.post("/api/robot/enable")
    def robot_enable() -> dict[str, Any]:
        return _run(state, "robot", "RM65 enabled", state.robot.enable)

    @app.post("/api/robot/disable")
    def robot_disable() -> dict[str, Any]:
        return _run(state, "robot", "RM65 disabled", state.robot.disable)

    @app.post("/api/robot/slow-stop")
    def robot_slow_stop() -> dict[str, Any]:
        if state.transfer.running:
            return state.transfer.request_stop()
        return _run(state, "safety", "RM65 slow stop requested", state.robot.slow_stop)

    @app.get("/api/robot/pose")
    def robot_pose() -> dict[str, Any]:
        return _run(state, "robot", "RM65 pose read", state.robot.pose, log_success=False)

    @app.post("/api/robot/move")
    def robot_move(request: RobotMoveRequest) -> dict[str, Any]:
        _validate_motion_target(state, request.pose, request.speed)
        return _run(
            state,
            "robot",
            "RM65 motion command completed",
            lambda: state.robot.move(request.pose, request.mode, request.speed, request.acc),
        )

    @app.get("/api/targeting/settings")
    def targeting_settings() -> dict[str, Any]:
        return _targeting_settings_dict(state.config)

    @app.post("/api/targeting/settings")
    def targeting_settings_update(request: TargetingSettingsRequest) -> dict[str, Any]:
        updated = replace(
            state.config.targeting,
            marker_to_tip_xyz_mm=(
                request.marker_to_tip_x_mm,
                request.marker_to_tip_y_mm,
                request.marker_to_tip_z_mm,
            ),
        )
        state.config = replace(state.config, targeting=updated)
        _persist_targeting_settings(updated)
        state.logs.add(
            "warning",
            "safety",
            "Targeting offsets updated",
            marker_to_tip_x_mm=request.marker_to_tip_x_mm,
            marker_to_tip_y_mm=request.marker_to_tip_y_mm,
            marker_to_tip_z_mm=request.marker_to_tip_z_mm,
        )
        return _targeting_settings_dict(state.config)

    @app.get("/api/targeting/preview")
    def targeting_preview() -> dict[str, Any]:
        return _target_preview(state)

    @app.post("/api/targeting/move-target")
    def targeting_move_target(request: TargetMoveRequest) -> dict[str, Any]:
        if request.confirm is not True:
            raise HTTPException(
                status_code=400, detail="explicit target move confirmation is required"
            )
        preview = _target_preview(state)
        pose = preview["target_pose_mm_deg"]
        _validate_motion_target(state, pose, request.speed)
        return _run(
            state,
            "robot",
            "Validated visual target motion completed",
            lambda: state.robot.move(pose, "MoveJ_P", request.speed, 10),
        )

    @app.get("/api/gripper/ports")
    def gripper_ports() -> list[dict[str, Any]]:
        return list_serial_ports()

    @app.get("/api/gripper/status")
    def gripper_status(read_feedback: bool = False) -> dict[str, Any]:
        return _run(
            state,
            "gripper",
            "DH5 status read",
            lambda: state.gripper.status(read_feedback=read_feedback),
            log_success=False,
        )

    @app.post("/api/gripper/connect")
    def gripper_connect(request: GripperConnectRequest) -> dict[str, Any]:
        config_value = replace(
            state.config.gripper,
            port=request.port,
            baud_rate=request.baud_rate,
            modbus_id=request.modbus_id,
            stop_bits=request.stop_bits,
            parity=request.parity.upper(),
        )
        return _run(
            state,
            "gripper",
            f"DH5 connected on {request.port}",
            lambda: state.gripper.connect(config_value),
        )

    @app.post("/api/gripper/disconnect")
    def gripper_disconnect() -> dict[str, Any]:
        return _run(state, "gripper", "DH5 disconnected", state.gripper.disconnect)

    @app.post("/api/gripper/initialize")
    def gripper_initialize() -> dict[str, Any]:
        return _run(state, "gripper", "DH5 initialized", state.gripper.initialize)

    @app.post("/api/gripper/command")
    def gripper_command(request: GripperCommandRequest) -> dict[str, Any]:
        def execute() -> dict[str, Any]:
            result: dict[str, Any] = {}
            if request.force is not None:
                result.update(state.gripper.set_force(request.force))
            if request.speed is not None:
                result.update(state.gripper.set_speed(request.speed))
            if request.position is not None:
                result.update(state.gripper.set_position(request.position))
            if not result:
                raise ValueError("at least one gripper command value is required")
            return result

        return _run(state, "gripper", "DH5 command sent", execute)

    @app.get("/api/sensors/status")
    def sensors_status() -> dict[str, Any]:
        return state.sensors.status()

    @app.get("/api/sensors/data")
    def sensors_data() -> dict[str, Any]:
        return state.sensors.data()

    @app.get("/api/sensors/settings")
    def sensors_settings() -> dict[str, Any]:
        return state.sensors.settings.response()

    @app.post("/api/sensors/settings")
    def sensors_settings_update(request: SensorSettingsUpdateRequest) -> dict[str, Any]:
        try:
            result = state.sensors.settings.update(request.side, request.values)
            state.sensors.apply_settings()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state.logs.add("info", "sensors", f"{request.side} sensor processing settings saved")
        return result

    @app.post("/api/sensors/connect")
    def sensors_connect(request: SensorConnectRequest) -> dict[str, Any]:
        return _run(
            state,
            "sensors",
            f"Pressure sensors connected: {request.left_port}, {request.right_port}",
            lambda: state.sensors.connect(request.left_port, request.right_port),
        )

    @app.post("/api/sensors/disconnect")
    def sensors_disconnect() -> dict[str, Any]:
        return _run(state, "sensors", "Pressure sensors disconnected", state.sensors.disconnect)

    @app.post("/api/sensors/zero")
    def sensors_zero() -> dict[str, Any]:
        return _run(state, "sensors", "Pressure sensor baseline captured", state.sensors.zero)

    @app.post("/api/sensors/clear-zero")
    def sensors_clear_zero() -> dict[str, Any]:
        return _run(state, "sensors", "Pressure sensor baseline cleared", state.sensors.clear_zero)

    def require_calibration_frame() -> None:
        if not state.camera.status()["frame_ready"]:
            raise HTTPException(status_code=409, detail="D435 frame is not ready")

    def new_calibration_recorder() -> CalibrationRecorder:
        metadata = {
            "camera": state.camera.metadata(),
            "robot": {
                "name": "RM65",
                "initial_state": None,
                "note": "RM65 pose is required and captured for every saved sample.",
            },
            "units": {
                "camera_xyz": "m",
                "robot_tcp_position": "mm",
                "robot_tcp_orientation": "degree",
                "robot_joint_values": "RM SDK native",
            },
            "notes": "Blue-dot raw observations; marker-to-TCP offset is unresolved.",
        }
        return CalibrationRecorder(default_output_path(), metadata)

    def calibration_record_path(filename: str) -> Path:
        if Path(filename).name != filename or not filename.startswith("blue-dot-"):
            raise ValueError("invalid calibration record filename")
        directory = Path("records/calibration").resolve()
        source = (directory / filename).resolve()
        if source.parent != directory or not source.is_file():
            raise FileNotFoundError(f"calibration record does not exist: {filename}")
        return source

    @app.get("/api/calibration/records")
    def calibration_records() -> list[dict[str, Any]]:
        solution = _calibration_solution(state.config)
        solved_source = (
            Path(solution["source_record"]).resolve()
            if solution and solution.get("source_record")
            else None
        )
        selected = state.calibration.output.resolve() if state.calibration.output else None
        records: list[dict[str, Any]] = []
        paths = sorted(
            Path("records/calibration").glob("blue-dot-*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                recorder = CalibrationRecorder.resume(path)
            except (OSError, TypeError, ValueError):
                continue
            resolved = path.resolve()
            records.append(
                {
                    "filename": path.name,
                    "sample_count": recorder.sample_count,
                    "selected": resolved == selected,
                    "solved_source": resolved == solved_source,
                }
            )
        return records

    @app.post("/api/calibration/resume")
    def calibration_resume(request: CalibrationResumeRequest) -> dict[str, Any]:
        require_calibration_frame()
        try:
            recorder = CalibrationRecorder.resume(calibration_record_path(request.filename))
            state.calibration.attach(recorder)
            state.logs.add(
                "info",
                "calibration",
                "Selected calibration record resumed",
                output=str(recorder.path.resolve()),
                sample_count=recorder.sample_count,
            )
            return state.calibration.status()
        except (OSError, TypeError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/start")
    def calibration_start() -> dict[str, Any]:
        if state.calibration.recorder is not None:
            return state.calibration.status()
        require_calibration_frame()
        try:
            source = state.calibration.output
            if source is None or not source.is_file():
                try:
                    source = latest_calibration_record()
                except FileNotFoundError:
                    source = None
            recorder = (
                CalibrationRecorder.resume(source)
                if source is not None
                else new_calibration_recorder()
            )
            state.calibration.attach(recorder)
            state.logs.add(
                "info",
                "calibration",
                "Calibration record resumed"
                if recorder.sample_count
                else "Calibration session started",
                output=str(recorder.path.resolve()),
                sample_count=recorder.sample_count,
            )
            return state.calibration.status()
        except (OSError, RobotServiceError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/new")
    def calibration_new() -> dict[str, Any]:
        if state.calibration.recorder is not None:
            raise HTTPException(status_code=409, detail="stop the active calibration first")
        require_calibration_frame()
        try:
            recorder = new_calibration_recorder()
            state.calibration.attach(recorder)
            state.logs.add(
                "info",
                "calibration",
                "New calibration session started",
                output=str(recorder.path.resolve()),
            )
            return state.calibration.status()
        except (OSError, RobotServiceError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/sample")
    def calibration_sample(request: PixelRequest | None = None) -> dict[str, Any]:
        recorder = state.calibration.recorder
        if recorder is None:
            raise HTTPException(status_code=409, detail="calibration session is not active")
        try:
            robot_status = state.robot.status()
            if not robot_status.get("service_online"):
                raise ValueError("RM65 本机服务未启动，请先在机械臂面板启动服务")
            if not robot_status.get("connected"):
                raise ValueError("RM65 尚未连接，请先在机械臂面板连接机械臂")
            point = (
                state.camera.deproject(request.u, request.v)
                if request is not None
                else state.selected_point
            )
            if point is None:
                raise ValueError("select the blue dot before saving a sample")
            sample = recorder.append_sample(point, state.robot.pose())
            state.calibration.last_sample = sample
            state.logs.add(
                "info",
                "calibration",
                f"Calibration sample {sample['sample_index']} saved",
            )
            return state.calibration.status()
        except (OSError, RobotServiceError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/stop")
    def calibration_stop() -> dict[str, Any]:
        status = state.calibration.status()
        state.calibration.stop()
        state.logs.add("info", "calibration", "Calibration session stopped", **status)
        return {**status, "active": False}

    @app.get("/api/calibration/status")
    def calibration_status() -> dict[str, Any]:
        return {**state.calibration.status(), "solution": _calibration_solution(state.config)}

    @app.get("/api/calibration/solution")
    def calibration_solution() -> dict[str, Any] | None:
        return _calibration_solution(state.config)

    @app.post("/api/calibration/solve")
    def calibration_solve(request: CalibrationSolveRequest | None = None) -> dict[str, Any]:
        def execute() -> dict[str, Any]:
            source = (
                Path(request.input_path)
                if request and request.input_path
                else latest_calibration_record()
            )
            report = solve_calibration(
                source,
                threshold_mm=state.config.calibration.ransac_threshold_mm,
                validation_fraction=state.config.calibration.validation_fraction,
            )
            output = save_report(report, state.config.calibration.output_path)
            return {**report, "output_path": str(output.resolve())}

        return _run(state, "calibration", "Camera-to-base transform solved", execute)

    @app.get("/api/logs")
    def logs(after: int = Query(default=0, ge=0)) -> list[dict[str, Any]]:
        return state.logs.list(after)

    return app


def _run(
    state: ControlState,
    source: str,
    message: str,
    action,
    *,
    log_success: bool = True,
) -> Any:
    try:
        result = action()
        if log_success:
            state.logs.add("info", source, message)
        return result
    except PermissionError as exc:
        state.logs.add("warning", source, str(exc))
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (
        DH5ProtocolError,
        SensorProtocolError,
        FileNotFoundError,
        OSError,
        RobotServiceError,
        RuntimeError,
        ValueError,
    ) as exc:
        state.logs.add("error", source, str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _click_dict(point: ClickResult) -> dict[str, Any]:
    return {
        "pixel_uv": list(point.pixel),
        "depth_m": point.depth_m,
        "camera_xyz_m": list(point.camera_xyz_m),
    }


def _detect_vision(state: ControlState, target_class: str | None) -> dict[str, Any]:
    color, depth, depth_scale = state.camera.snapshot()
    workspace = state.workspace.workspace
    result = state.vision.detect(
        color,
        depth,
        depth_scale,
        roi_polygon_uv=workspace.polygon_uv if workspace else None,
    )
    return _enrich_vision_result(state, result, target_class)


def _validate_transfer_readiness(state: ControlState) -> None:
    robot = state.robot.status()
    if not robot.get("motion_configured"):
        raise PermissionError("config/default.json 尚未允许机械臂运动")
    if not robot.get("connected") or not robot.get("enabled"):
        raise TransferError("RM65 必须已连接并使能")
    if not state.gripper.status().get("connected"):
        raise TransferError("DH5 夹爪尚未连接")
    sensors = state.sensors.status()
    if not sensors.get("connected") or not sensors.get("frame_ready"):
        raise TransferError("双侧触觉传感器尚未连接或数据未就绪")


def _enrich_vision_result(
    state: ControlState, result: dict[str, Any], target_class: str | None
) -> dict[str, Any]:
    solution = _calibration_solution(state.config)
    transform = None
    if solution and solution.get("quality", {}).get("hover_ready"):
        transform = np.asarray(solution["transform_4x4"], dtype=np.float64)
    workspace = state.workspace.workspace

    detections: list[dict[str, Any]] = []
    for raw in result["detections"]:
        item = dict(raw)
        item["camera_xyz_m"] = None
        item["base_xyz_mm"] = None
        item["height_above_table_mm"] = None
        if item.get("depth_m") is not None:
            u, v = map(int, item["center_uv"])
            point = state.camera.deproject_depth(u, v, float(item["depth_m"]))
            item["camera_xyz_m"] = list(point.camera_xyz_m)
            if transform is not None:
                camera_mm = np.asarray(point.camera_xyz_m, dtype=np.float64) * 1000.0
                item["base_xyz_mm"] = transform_point(transform, camera_mm).tolist()
            if workspace is not None and not workspace.contains_camera_point(point.camera_xyz_m):
                item["graspable"] = False
                item["rejection_reason"] = "目标三维坐标位于桌面工作区外"
            elif workspace is not None:
                item["height_above_table_mm"] = (
                    workspace.height_above_plane_m(point.camera_xyz_m) * 1000.0
                )
        detections.append(item)

    object_candidates = [
        item
        for item in detections
        if item["kind"] == "object"
        and item["graspable"]
        and item["base_xyz_mm"] is not None
        and (target_class is None or item["class_name"] == target_class)
    ]
    recommended = max(
        object_candidates,
        key=lambda item: (item["confidence"], item["depth_valid_fraction"]),
        default=None,
    )
    destinations = [
        item
        for item in detections
        if item["kind"] == "destination"
        and item["rejection_reason"] is None
        and item["base_xyz_mm"] is not None
    ]
    destination = max(
        destinations,
        key=lambda item: (item["confidence"], item["area_px"]),
        default=None,
    )
    warnings = list(result.get("warnings", []))
    if transform is None:
        warnings.append("外参尚未通过质量门槛，仅输出相机坐标，不推荐机械臂目标")
    elif not recommended:
        warnings.append(
            f"没有找到可抓取的目标类别：{target_class}"
            if target_class
            else "桌面工作区内没有可推荐的抓取目标"
        )
    if workspace is None:
        warnings.append("尚未完成四点桌面工作区标定，当前使用矩形视觉 ROI")
    if not destination:
        warnings.append("桌面工作区内没有可用的绿色实心托盘")
    return {
        **result,
        "detections": detections,
        "target_class": target_class,
        "recommended_object_id": recommended["detection_id"] if recommended else None,
        "destination_id": destination["detection_id"] if destination else None,
        "recommended_object": recommended,
        "destination": destination,
        "workspace_calibrated": workspace is not None,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _annotation_from_boxes(
    boxes: list[list[int]],
    class_id: int | None,
    class_name: str | None,
    image_width: int,
    image_height: int,
    roi_polygon_uv: list[list[int]],
) -> dict[str, Any]:
    polygon = np.asarray(roi_polygon_uv, dtype=np.int32)
    annotations: list[dict[str, Any]] = []
    for index, raw in enumerate(boxes):
        if len(raw) != 4:
            raise ValueError(f"第 {index + 1} 个标注框格式无效")
        x, y, width, height = (int(value) for value in raw)
        if width < 4 or height < 4:
            raise ValueError(f"第 {index + 1} 个标注框过小")
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise ValueError(f"第 {index + 1} 个标注框超出图像范围")
        center = (x + width / 2.0, y + height / 2.0)
        if cv2.pointPolygonTest(polygon, center, False) < 0:
            raise ValueError(f"第 {index + 1} 个标注框中心位于工作区外")
        if class_id is None or class_name is None:
            raise ValueError("非空标注缺少训练类别")
        annotations.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xywh": [x, y, width, height],
                "yolo_xywh": [
                    (x + width / 2.0) / image_width,
                    (y + height / 2.0) / image_height,
                    width / image_width,
                    height / image_height,
                ],
                "confidence": 1.0,
                "source": "depth_proposal_or_user_edit",
            }
        )
    return {
        "valid": True,
        "target_class": class_name if class_name is not None else "empty_table",
        "image_size": [image_width, image_height],
        "annotations": annotations,
        "reasons": [],
        "roi_polygon_uv": roi_polygon_uv,
        "annotation_source": "class_agnostic_depth_with_user_confirmation",
    }


def _calibration_solution(config: AppConfig) -> dict[str, Any] | None:
    path = Path(config.calibration.output_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _targeting_settings_dict(config: AppConfig) -> dict[str, Any]:
    settings = config.targeting
    return {
        "marker_to_tip_xyz_mm": list(settings.marker_to_tip_xyz_mm),
        "maximum_speed": settings.maximum_speed,
    }


def _persist_targeting_settings(settings: TargetingConfig) -> None:
    path = default_config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["targeting"] = {
        "marker_to_tip_xyz_mm": list(settings.marker_to_tip_xyz_mm),
        "maximum_speed": settings.maximum_speed,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _transfer_settings_dict(settings: TransferConfig) -> dict[str, Any]:
    return {
        "transit_speed": settings.transit_speed,
        "approach_speed": settings.approach_speed,
        "grasp_clearance_mm": settings.grasp_clearance_mm,
        "pick_lift_mm": settings.pick_lift_mm,
        "place_approach_mm": settings.place_approach_mm,
        "place_clearance_mm": settings.place_clearance_mm,
    }


def _persist_transfer_settings(settings: TransferConfig) -> None:
    path = default_config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["transfer"] = asdict(settings)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _target_preview(state: ControlState) -> dict[str, Any]:
    point = state.selected_point
    if point is None:
        raise HTTPException(
            status_code=409, detail="select an object point in the camera view first"
        )
    solution = _calibration_solution(state.config)
    if not solution or not solution.get("quality", {}).get("hover_ready"):
        raise HTTPException(
            status_code=409, detail="calibration is not approved for motion validation"
        )
    orientation = solution.get("fixed_tool_orientation_deg")
    orientation_range = solution.get("tool_orientation_range_deg")
    if not isinstance(orientation, list) or len(orientation) != 3:
        raise HTTPException(
            status_code=409, detail="re-solve calibration to store fixed tool orientation"
        )
    if not isinstance(orientation_range, list) or max(map(abs, orientation_range)) > 2.0:
        raise HTTPException(
            status_code=409, detail="calibration tool orientation was not sufficiently fixed"
        )
    camera_mm = np.asarray(point.camera_xyz_m, dtype=np.float64) * 1000.0
    fitted = transform_point(np.asarray(solution["transform_4x4"]), camera_mm)
    marker_offset = np.asarray(state.config.targeting.marker_to_tip_xyz_mm)
    target_xyz = fitted + marker_offset
    return {
        "pixel_uv": list(point.pixel),
        "camera_xyz_m": list(point.camera_xyz_m),
        "fitted_reference_xyz_mm": fitted.tolist(),
        "marker_to_tip_xyz_mm": marker_offset.tolist(),
        "target_pose_mm_deg": [*target_xyz.tolist(), *map(float, orientation)],
    }


def _validate_motion_target(state: ControlState, pose: list[float], speed: int) -> None:
    if speed > state.config.targeting.maximum_speed:
        raise HTTPException(status_code=403, detail="speed exceeds the configured motion limit")
    if len(pose) != 6 or not np.all(np.isfinite(np.asarray(pose, dtype=np.float64))):
        raise HTTPException(status_code=400, detail="target pose is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description="VTLA Grip local-only hardware API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("VTLA API only permits loopback host addresses")
    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port)


app = create_app()


if __name__ == "__main__":
    main()
