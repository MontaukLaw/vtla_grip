export type Panel =
  | "camera"
  | "calibration"
  | "robot"
  | "gripper"
  | "sensors"
  | "tactile-data"
  | "vision-data"
  | "devices"
  | "logs"
  | "settings";

export type SensorSideState = {
  connected: boolean;
  port: string;
  baud_rate: number;
  frame_ready: boolean;
  frame_age_s?: number | null;
  fps: number;
  last_error?: string | null;
};
export type SensorsState = {
  connected: boolean;
  frame_ready: boolean;
  left: SensorSideState;
  right: SensorSideState;
};
export type SensorData = SensorsState & {
  raw: number[] | null;
  zeroed_at?: number | null;
  zero_offset?: number[];
  frame_ids?: number[];
  processed: number[] | null;
  display: number[] | null;
  features: { left: SensorFeatures; right: SensorFeatures } | null;
  grasp_success: GraspSuccess | null;
};
export type SensorFeatures = {
  sum: number;
  max: number;
  nonzero: number;
  baseline_ready: boolean;
  baseline_enabled?: boolean;
  display_threshold: number;
  baseline_mean: number;
  noise_mean: number;
};
export type GraspSuccess = {
  success: boolean;
  both_sides_over_threshold: boolean;
  confirm_count: number;
  required_frames: number;
  left_peak: number;
  right_peak: number;
  left_threshold: number;
  right_threshold: number;
};
export type SensorSettingField = {
  label: string;
  group: string;
  path: string;
  min: number;
  max: number;
  kind: "bool" | "int" | "float";
};
export type SensorSettings = {
  enabled: boolean;
  schema: SensorSettingField[];
  values: Record<"left" | "right", Record<string, unknown>>;
  point_cloud: Record<string, unknown>;
};
export type TargetingSettings = {
  marker_to_tip_xyz_mm: [number, number, number];
  maximum_speed: number;
};
export type TargetPreview = {
  pixel_uv: [number, number];
  camera_xyz_m: [number, number, number];
  fitted_reference_xyz_mm: [number, number, number];
  marker_to_tip_xyz_mm: [number, number, number];
  target_pose_mm_deg: [number, number, number, number, number, number];
};

export type DeviceState = {
  camera: {
    running: boolean;
    frame_ready: boolean;
    error?: string | null;
    frame_age_s?: number | null;
  };
  robot: Record<string, unknown> & {
    service_online?: boolean;
    connected?: boolean;
    enabled?: boolean;
    motion_configured?: boolean;
  };
  gripper: {
    connected: boolean;
    port: string;
    baud_rate: number;
    last_error?: string | null;
  };
  sensors: SensorsState;
  serial_ports: Array<{ device: string; description: string }>;
};

export type SelectedPoint = {
  pixel_uv: [number, number];
  depth_m: number;
  camera_xyz_m: [number, number, number];
};
export type VisionClass = "green_cylinder" | "gray_cube" | "green_tray";
export type VisionDetection = {
  detection_id: string;
  class_name: VisionClass;
  kind: "object" | "destination";
  confidence: number;
  bbox_xywh: [number, number, number, number];
  center_uv: [number, number];
  area_px: number;
  depth_m: number | null;
  depth_spread_m: number | null;
  depth_valid_fraction: number;
  graspable: boolean;
  rejection_reason: string | null;
  camera_xyz_m: [number, number, number] | null;
  base_xyz_mm: [number, number, number] | null;
  height_above_table_mm: number | null;
};
export type VisionResult = {
  roi_xyxy: [number, number, number, number];
  roi_polygon_uv: Array<[number, number]> | null;
  detections: VisionDetection[];
  target_class: VisionClass | null;
  recommended_object_id: string | null;
  destination_id: string | null;
  recommended_object: VisionDetection | null;
  destination: VisionDetection | null;
  workspace_calibrated: boolean;
  warnings: string[];
};
export type TransferPoseName = "pick_above" | "pick" | "place_above" | "place";
export type TransferPlan = {
  plan_id: string;
  created_at: string;
  expires_in_s: number;
  command: {
    action: "place_into_tray" | "place_at_position";
    target_class: "green_cylinder" | "gray_cube";
    target_property?: "软" | "硬";
    destination_class: "green_tray" | "specified_position";
  };
  source: Pick<
    VisionDetection,
    | "detection_id"
    | "class_name"
    | "confidence"
    | "bbox_xywh"
    | "center_uv"
    | "depth_m"
    | "base_xyz_mm"
    | "height_above_table_mm"
  >;
  destination: {
    detection_id: string;
    class_name: "green_tray" | "specified_position";
    confidence: number;
    bbox_xywh: [number, number, number, number];
    center_uv: [number, number];
    depth_m: number | null;
    base_xyz_mm: [number, number, number] | null;
    height_above_table_mm?: number | null;
  };
  poses: Record<
    TransferPoseName,
    [number, number, number, number, number, number]
  >;
  motion: {
    transit_speed: number;
    approach_speed: number;
    acceleration: number;
    grasp_clearance_mm: number;
    pick_lift_mm: number;
    place_approach_mm: number;
    place_clearance_mm: number;
    source_height_above_table_mm: number;
  };
  candidates?: Pick<TransferPlan, "source" | "poses" | "motion">[];
  steps: string[];
};
export type TransferStatus = {
  status:
    | "idle"
    | "preview_ready"
    | "queued"
    | "running"
    | "stopping"
    | "completed"
    | "failed"
    | "stopped";
  running: boolean;
  current_step: string | null;
  completed_steps: number;
  error: string | null;
  holding_object: boolean;
  started_at: string | null;
  finished_at: string | null;
  plan: TransferPlan | null;
  candidate_index?: number;
  candidate_count?: number;
  attempts?: {
    source: TransferPlan["source"];
    recognition_result: { status: string; property?: string; error?: string };
    matched: boolean;
  }[];
  grasp_result?: Record<string, unknown>;
  recognition_result?: {
    status: "completed" | "failed" | "unavailable";
    vision_class?: string;
    property?: string;
    confidence?: number;
    final_gripper_position?: number | null;
    frame_count?: number;
    capture_duration_s?: number;
    error?: string;
  } | null;
};
export type TactileModelVersion = {
  version: string;
  trained_at: string;
  sample_count: number;
  properties: string[];
  accuracy: number | null;
};
export type TactileModelStatus = {
  available: boolean;
  loaded: boolean;
  stale: boolean;
  stale_reason?: string | null;
  version?: string;
  trained_at?: string;
  sample_count?: number;
  properties?: string[];
  capture_duration_seconds?: number;
  evaluation?: {
    accuracy: number;
    labels: string[];
    confusion_matrix: number[][];
    sample_count: number;
    note: string;
  };
  versions: TactileModelVersion[];
};
export type TactileClassStatus = TrainingClass & {
  properties: string[];
  sample_count: number;
  distribution: Record<string, number>;
  model: TactileModelStatus;
};
export type TactileCaptureStatus = {
  status: "idle" | "queued" | "running" | "stopping" | "ready" | "saved" | "failed" | "stopped";
  running: boolean;
  current_step: string | null;
  error: string | null;
  vision_class: string | null;
  property: string | null;
  frame_count: number;
  duration_seconds: number;
  final_gripper_position: number | null;
  curves: Record<"time" | "left_sum" | "right_sum" | "left_peak" | "right_peak", number[]> | null;
  saved_sample?: TactileSample | null;
};
export type TactileTrainingStatus = {
  status: "idle" | "running" | "completed" | "failed";
  running: boolean;
  vision_class: string | null;
  error: string | null;
  result: TactileModelStatus | null;
};
export type TactileStatus = {
  root: string;
  classes: TactileClassStatus[];
  total_samples: number;
  capture: TactileCaptureStatus;
  training: TactileTrainingStatus;
};
export type TactileSample = {
  left_peak: number;
  right_peak: number;
  sample_id: string;
  captured_at: string;
  vision_class: string;
  property: string;
  duration_seconds: number;
  final_gripper_position: number | null;
  frame_count: number;
};
export type TactileSampleList = {
  total: number;
  offset: number;
  samples: TactileSample[];
};
export type TactileSampleDetail = TactileSample & {
  timestamps: number[];
  frames: number[][];
};
export type TransferPreviewResult = {
  vision: VisionResult;
  transfer: TransferStatus;
};
export type TransferSettings = {
  transit_speed: number;
  approach_speed: number;
  grasp_clearance_mm: number;
  pick_lift_mm: number;
  place_approach_mm: number;
  place_clearance_mm: number;
};
export type WorkspacePoint = {
  pixel_uv: [number, number];
  depth_m: number;
  camera_xyz_m: [number, number, number];
  base_xyz_mm: [number, number, number];
};
export type WorkspaceStatus = {
  calibrated: boolean;
  active: boolean;
  point_count: number;
  next_label: "左上" | "右上" | "右下" | "左下" | null;
  draft_points: WorkspacePoint[];
  polygon_uv: Array<[number, number]> | null;
  max_plane_residual_mm: number | null;
  created_at: string | null;
  path: string;
};
export type PlaceTargetStatus = {
  configured: boolean;
  valid?: boolean;
  pixel_uv?: [number, number];
  camera_xyz_m?: [number, number, number];
  base_xyz_mm?: [number, number, number];
  created_at?: string;
  path: string;
};
export type AutoAnnotation = {
  valid: boolean;
  target_class: string;
  image_size: [number, number];
  annotations: Array<{
    class_id: number;
    class_name: string;
    bbox_xywh: [number, number, number, number];
    yolo_xywh: [number, number, number, number];
    confidence: number;
  }>;
  reasons: string[];
  roi_xyxy?: [number, number, number, number] | null;
  roi_polygon_uv: Array<[number, number]> | null;
};
export type TrainingClass = { class_id: number; name: string; label: string };
export type DepthProposalResult = {
  image_size: [number, number];
  boxes: Array<[number, number, number, number]>;
  roi_polygon_uv: Array<[number, number]>;
  foreground_fraction: number;
  warnings: string[];
  thresholds: Record<string, number>;
};
export type VisionDatasetSample = {
  sample_id: string;
  captured_at: string;
  tags: string[];
  directory: string;
  workspace_calibrated: boolean;
  auto_annotation?: AutoAnnotation | null;
  color_url: string;
  depth_preview_url: string;
};
export type VisionDatasetList = {
  total: number;
  offset: number;
  samples: VisionDatasetSample[];
};
export type VisionDatasetStatus = {
  root: string;
  sample_count: number;
  labels: string[];
  recent: VisionDatasetSample[];
};
export type VisionCaptureResult = {
  sample_id: string;
  captured_at: string;
  tags: string[];
  directory: string;
  files: { color: string; depth: string; labels?: string };
  dataset: VisionDatasetStatus;
};
export type CalibrationSolution = {
  output_path?: string;
  sample_count: number;
  inlier_count: number;
  outlier_sample_indices: number[];
  transform_4x4: number[][];
  validation: {
    count: number;
    rms_mm: number;
    max_mm: number;
    mean_mm: number;
    median_mm: number;
  };
  final_all_samples: { rms_mm: number; max_mm: number };
  quality: { hover_ready: boolean };
  warning: string;
};
export type CalibrationState = {
  active: boolean;
  sample_count: number;
  output?: string | null;
  last_sample?: Record<string, unknown> | null;
  solution?: CalibrationSolution | null;
};
export type LogEvent = {
  id: number;
  timestamp: string;
  level: string;
  source: string;
  message: string;
};

export type PanelContext = {
  devices: DeviceState;
  point: SelectedPoint | null;
  calibration: CalibrationState;
  workspace: WorkspaceStatus;
  logs: LogEvent[];
  busy: string | null;
  action: (
    name: string,
    work: () => Promise<unknown>,
    success: string,
  ) => Promise<void>;
};
