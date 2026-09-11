"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bot,
  Camera,
  Crosshair,
  Database,
  Hand,
  BrainCircuit,
  Images,
  ListTree,
  LoaderCircle,
  PackageCheck,
  RadioTower,
  ScanSearch,
  Settings,
  ShieldAlert,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { api, API_BASE, post } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  CalibrationState,
  DeviceState,
  LogEvent,
  Panel,
  PlaceTargetStatus,
  SelectedPoint,
  SensorData,
  TransferPreviewResult,
  TransferStatus,
  VisionClass,
  VisionResult,
  WorkspaceStatus,
} from "@/lib/types";
import { renderPanel } from "@/components/panels";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { StatusDot } from "@/components/ui/status-dot";

const emptyDevices: DeviceState = {
  camera: { running: false, frame_ready: false },
  robot: {
    service_online: false,
    connected: false,
    enabled: false,
    motion_configured: false,
  },
  gripper: { connected: false, port: "COM11", baud_rate: 115200 },
  sensors: {
    connected: false,
    frame_ready: false,
    left: {
      connected: false,
      port: "COM78",
      baud_rate: 1000000,
      frame_ready: false,
      fps: 0,
    },
    right: {
      connected: false,
      port: "COM79",
      baud_rate: 1000000,
      frame_ready: false,
      fps: 0,
    },
  },
  serial_ports: [],
};

const emptyCalibration: CalibrationState = { active: false, sample_count: 0 };
const emptyWorkspace: WorkspaceStatus = {
  calibrated: false,
  active: false,
  point_count: 0,
  next_label: null,
  draft_points: [],
  polygon_uv: null,
  max_plane_residual_mm: null,
  created_at: null,
  path: "",
};
const emptyPlaceTarget: PlaceTargetStatus = { configured: false, path: "" };

const visionLabels: Record<VisionClass, string> = {
  green_cylinder: "绿色圆柱体",
  gray_cube: "灰色正方体",
  green_tray: "绿色托盘",
};

const navigation: Array<{ id: Panel; label: string; icon: typeof Camera }> = [
  { id: "camera", label: "相机", icon: Camera },
  { id: "calibration", label: "标定采集", icon: Target },
  { id: "robot", label: "机械臂", icon: Bot },
  { id: "gripper", label: "夹爪", icon: Hand },
  { id: "sensors", label: "压力传感", icon: RadioTower },
  { id: "tactile-data", label: "触觉识别", icon: BrainCircuit },
  { id: "vision-data", label: "视觉数据", icon: Images },
  { id: "devices", label: "设备状态", icon: Database },
  { id: "logs", label: "日志", icon: ListTree },
  { id: "settings", label: "设置", icon: Settings },
];

type PressurePoint = {
  time: number;
  leftSum: number;
  rightSum: number;
  leftPeak: number;
  rightPeak: number;
};

export function ControlDashboard() {
  const [panel, setPanel] = useState<Panel | null>(null);
  const [devices, setDevices] = useState<DeviceState>(emptyDevices);
  const [calibration, setCalibration] =
    useState<CalibrationState>(emptyCalibration);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [point, setPoint] = useState<SelectedPoint | null>(null);
  const [vision, setVision] = useState<VisionResult | null>(null);
  const [pressureHistory, setPressureHistory] = useState<PressurePoint[]>([]);
  const [workspace, setWorkspace] = useState<WorkspaceStatus>(emptyWorkspace);
  const [placeTarget, setPlaceTarget] =
    useState<PlaceTargetStatus>(emptyPlaceTarget);
  const [placeTargetSelecting, setPlaceTargetSelecting] = useState(false);
  const [transferTarget, setTransferTarget] = useState<
    "green_cylinder" | "gray_cube"
  >("green_cylinder");
  const [transferProperty, setTransferProperty] = useState<"" | "软" | "硬">("");
  const [transferDestination, setTransferDestination] = useState<
    "green_tray" | "specified_position"
  >("green_tray");
  const [transfer, setTransfer] = useState<TransferStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{
    kind: "ok" | "error";
    text: string;
  } | null>(null);

  const refresh = useCallback(async () => {
    const [
      nextDevices,
      nextCalibration,
      nextSelectedPoint,
      nextLogs,
      nextWorkspace,
      nextPlaceTarget,
      nextTransfer,
    ] = await Promise.all([
      api<DeviceState>("/api/devices"),
      api<CalibrationState>("/api/calibration/status"),
      api<SelectedPoint | null>("/api/camera/selected-point"),
      api<LogEvent[]>("/api/logs"),
      api<WorkspaceStatus>("/api/workspace/status"),
      api<PlaceTargetStatus>("/api/place-target/status"),
      api<TransferStatus>("/api/transfer/status"),
    ]);
    setDevices(nextDevices);
    setCalibration(nextCalibration);
    setPoint(nextSelectedPoint);
    setLogs(nextLogs);
    setWorkspace(nextWorkspace);
    setPlaceTarget(nextPlaceTarget);
    setTransfer(nextTransfer);
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      void refresh().catch((error: Error) =>
        setNotice({ kind: "error", text: `后端未连接：${error.message}` }),
      );
    }, 0);
    const timer = window.setInterval(
      () => void refresh().catch(() => undefined),
      1500,
    );
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => {
    let active = true;
    const update = () =>
      void api<SensorData>("/api/sensors/data").then((data) => {
        if (!active || !data.features) return;
        const left = data.features.left;
        const right = data.features.right;
        setPressureHistory((current) => [
          ...current,
          {
            time: Date.now(),
            leftSum: left.sum,
            rightSum: right.sum,
            leftPeak: left.max,
            rightPeak: right.max,
          },
        ].slice(-100));
      }).catch(() => undefined);
    update();
    const timer = window.setInterval(update, 250);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const action = useCallback(
    async (name: string, work: () => Promise<unknown>, success: string) => {
      setBusy(name);
      try {
        await work();
        setNotice({ kind: "ok", text: success });
        await refresh();
      } catch (error) {
        setNotice({
          kind: "error",
          text: error instanceof Error ? error.message : "操作失败",
        });
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const selectPixel = async (event: React.MouseEvent<HTMLDivElement>) => {
    if (!devices.camera.frame_ready) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const u = Math.max(
      0,
      Math.min(
        639,
        Math.round(((event.clientX - rect.left) / rect.width) * 640),
      ),
    );
    const v = Math.max(
      0,
      Math.min(
        479,
        Math.round(((event.clientY - rect.top) / rect.height) * 480),
      ),
    );
    if (workspace.active) {
      const label = workspace.next_label ?? "桌面角点";
      await action(
        "workspace-point",
        async () =>
          setWorkspace(
            await post<WorkspaceStatus>("/api/workspace/point", { u, v }),
          ),
        `已采集${label} (${u}, ${v})`,
      );
      return;
    }
    if (placeTargetSelecting) {
      await action(
        "place-target-point",
        async () => {
          setPlaceTarget(
            await post<PlaceTargetStatus>("/api/place-target/point", { u, v }),
          );
          setPlaceTargetSelecting(false);
          setTransfer(null);
        },
        `指定放置点已保存 (${u}, ${v})`,
      );
      return;
    }
    await action(
      "deproject",
      async () => {
        const selected = await post<SelectedPoint>("/api/camera/deproject", {
          u,
          v,
        });
        setPoint(selected);
      },
      `已读取像素 (${u}, ${v}) 的深度`,
    );
  };

  const startWorkspaceCalibration = async () => {
    await action(
      "workspace-start",
      async () => {
        setWorkspace(await post<WorkspaceStatus>("/api/workspace/start"));
        setVision(null);
      },
      "请依次点击左上、右上、右下、左下四个桌面角点",
    );
  };

  const detectObjects = async () => {
    await action(
      "vision-detect",
      async () => {
        const result = await post<VisionResult>("/api/vision/detect", {});
        setVision(result);
        const selected = result.recommended_object;
        if (selected?.camera_xyz_m && selected.depth_m != null) {
          setPoint({
            pixel_uv: selected.center_uv,
            depth_m: selected.depth_m,
            camera_xyz_m: selected.camera_xyz_m,
          });
        }
      },
      "视觉识别完成",
    );
  };

  const previewTransfer = async () => {
    await action(
      "transfer-preview",
      async () => {
        const result = await post<TransferPreviewResult>(
          "/api/transfer/preview",
          {
            action:
              transferDestination === "green_tray"
                ? "place_into_tray"
                : "place_at_position",
            target_class: transferTarget,
            target_property: transferProperty || null,
          },
        );
        setVision(result.vision);
        setTransfer(result.transfer);
        const selected = result.vision.recommended_object;
        if (selected?.camera_xyz_m && selected.depth_m != null)
          setPoint({
            pixel_uv: selected.center_uv,
            depth_m: selected.depth_m,
            camera_xyz_m: selected.camera_xyz_m,
          });
      },
      "搬运预览已生成，请检查后确认执行",
    );
  };

  const executeTransfer = async () => {
    if (!transfer?.plan) return;
    await action(
      "transfer-execute",
      async () =>
        setTransfer(
          await post<TransferStatus>("/api/transfer/execute", {
            plan_id: transfer.plan!.plan_id,
            confirm: true,
          }),
        ),
      "搬运任务已开始",
    );
  };

  const stopTransfer = async () => {
    await action(
      "transfer-stop",
      async () => setTransfer(await post<TransferStatus>("/api/transfer/stop")),
      "已请求停止搬运并慢停机械臂",
    );
  };

  const robotOnline = Boolean(devices.robot.service_online);
  const overallReady = devices.camera.frame_ready && robotOnline;
  const flow = useMemo(
    () => [
      [
        "01",
        "目标像素",
        point ? `${point.pixel_uv[0]}, ${point.pixel_uv[1]}` : "等待点击",
      ],
      [
        "02",
        "深度读取",
        point ? `${(point.depth_m * 1000).toFixed(1)} mm` : "—",
      ],
      ["03", "相机坐标", point ? "已计算" : "—"],
      [
        "04",
        "手眼转换",
        calibration.solution
          ? calibration.solution.quality.hover_ready
            ? "已验证"
            : "精度不足"
          : "待求解",
      ],
      [
        "05",
        "运动控制",
        devices.robot.connected && devices.robot.enabled
          ? "已连接使能"
          : "未就绪",
      ],
    ],
    [
      calibration.solution,
      devices.robot.connected,
      devices.robot.enabled,
      point,
    ],
  );

  return (
    <main className="flex min-h-screen bg-[#071016] text-[#e8f1f5]">
      <aside className="fixed inset-y-0 left-0 z-30 flex w-[92px] flex-col border-r border-[#203842] bg-[#09151c]">
        <div className="flex h-[76px] items-center justify-center border-b border-[#203842]">
          <div className="grid size-10 place-items-center border border-[#2d7769] bg-[#102a2a] text-[#42e7bd]">
            <Crosshair className="size-5" />
          </div>
        </div>
        <nav className="flex flex-1 flex-col py-3">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setPanel(id)}
              className={cn(
                "relative flex h-[70px] flex-col items-center justify-center gap-1.5 text-[11px] transition-colors",
                panel === id
                  ? "bg-[#10262b] text-[#42e7bd]"
                  : "text-[#718791] hover:bg-[#0e2028] hover:text-white",
              )}
            >
              {panel === id && (
                <span className="absolute inset-y-3 left-0 w-0.5 bg-[#42e7bd]" />
              )}
              <Icon className="size-[19px]" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="border-t border-[#203842] py-4 text-center font-mono text-[9px] tracking-wider text-[#49606a]">
          VTLA / 0.2
        </div>
      </aside>

      <section className="ml-[92px] min-w-0 flex-1">
        <header className="flex h-[76px] items-center justify-between border-b border-[#203842] bg-[#0a171e]/95 px-6">
          <div>
            <h1 className="text-lg font-semibold tracking-wide">
              VTLA_物体抓取任务
            </h1>
            <p className="mt-0.5 text-xs text-[#637b85]">
              RealSense D435 · RM65 · DH5
            </p>
          </div>
          <div className="flex items-center gap-3">
            <HeaderStatus label="D435" ok={devices.camera.frame_ready} />
            <HeaderStatus label="RM65" ok={Boolean(devices.robot.connected)} />
            <HeaderStatus label="DH5" ok={devices.gripper.connected} />
            <Button
              variant="danger"
              className="ml-2 h-11 px-5"
              disabled={busy === "slow-stop"}
              onClick={() =>
                void action(
                  "slow-stop",
                  () => post("/api/robot/slow-stop"),
                  "已发送机械臂慢停命令",
                )
              }
            >
              <ShieldAlert className="size-4" />
              紧急慢停
            </Button>
          </div>
        </header>

        <div className="grid min-h-[calc(100vh-76px)] grid-cols-[minmax(620px,1fr)_330px] items-start gap-4 p-4">
          <section className="flex min-h-0 flex-col border border-[#203b46] bg-[#0a161d]">
            <div className="flex h-12 items-center justify-between border-b border-[#203b46] px-4">
              <div className="flex items-center gap-2 text-sm">
                <Camera className="size-4 text-[#42e7bd]" />
                D435 彩色 / 对齐深度
              </div>
              <div className="flex items-center gap-2 text-xs text-[#718791]">
                <StatusDot ok={devices.camera.frame_ready} />
                {devices.camera.frame_ready ? "实时画面" : "等待相机"}
                <span className="font-mono">640×480</span>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={
                    !devices.camera.frame_ready ||
                    workspace.active ||
                    busy === "workspace-start"
                  }
                  onClick={() => void startWorkspaceCalibration()}
                >
                  <Target className="size-3.5" />
                  {workspace.calibrated ? "重新标定" : "标定工作区"}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={
                    !devices.camera.frame_ready ||
                    workspace.active ||
                    busy === "vision-detect"
                  }
                  onClick={() => void detectObjects()}
                >
                  <ScanSearch className="size-3.5" />
                  识别物体
                </Button>
              </div>
            </div>
            <div className="technical-grid flex min-h-[480px] flex-1 items-center justify-center overflow-hidden p-5">
              <div
                className="relative aspect-[4/3] max-h-[calc(100vh-190px)] w-full max-w-[960px] overflow-hidden border border-[#294854] bg-[#04090c]"
                role={devices.camera.running ? "button" : undefined}
                tabIndex={devices.camera.running ? 0 : undefined}
                onClick={
                  devices.camera.running
                    ? (event) => void selectPixel(event)
                    : undefined
                }
                onKeyDown={
                  devices.camera.running
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          void action(
                            "deproject",
                            async () =>
                              setPoint(
                                await post<SelectedPoint>(
                                  "/api/camera/deproject",
                                  { u: 320, v: 240 },
                                ),
                              ),
                            "已读取画面中心的深度",
                          );
                        }
                      }
                    : undefined
                }
                aria-label={
                  devices.camera.running
                    ? "点击画面选择目标像素；按回车选择画面中心"
                    : undefined
                }
              >
                {devices.camera.running ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG cannot use the optimizer. */}
                    <img
                      src={`${API_BASE}/api/camera/stream`}
                      alt="RealSense D435 实时画面"
                      className="h-full w-full cursor-crosshair object-fill"
                    />
                  </>
                ) : (
                  <div className="grid h-full place-items-center text-center">
                    <div>
                      <Camera className="mx-auto size-12 text-[#29434e]" />
                      <p className="mt-4 text-sm text-[#718791]">
                        D435 尚未启动
                      </p>
                      <Button
                        className="mt-4"
                        onClick={() =>
                          void action(
                            "camera-start",
                            () => post("/api/camera/start"),
                            "D435 已启动",
                          )
                        }
                      >
                        启动相机
                      </Button>
                    </div>
                  </div>
                )}
                {vision && !vision.roi_polygon_uv && (
                  <div
                    className="pointer-events-none absolute border border-dashed border-[#42e7bd]/50"
                    style={{
                      left: `${(vision.roi_xyxy[0] / 640) * 100}%`,
                      top: `${(vision.roi_xyxy[1] / 480) * 100}%`,
                      width: `${((vision.roi_xyxy[2] - vision.roi_xyxy[0]) / 640) * 100}%`,
                      height: `${((vision.roi_xyxy[3] - vision.roi_xyxy[1]) / 480) * 100}%`,
                    }}
                  />
                )}
                {(workspace.polygon_uv ||
                  workspace.draft_points.length > 0) && (
                  <svg
                    className="pointer-events-none absolute inset-0 h-full w-full"
                    viewBox="0 0 640 480"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    {workspace.polygon_uv && (
                      <polygon
                        points={workspace.polygon_uv
                          .map((item) => item.join(","))
                          .join(" ")}
                        fill="rgba(255,76,88,0.04)"
                        stroke="#ff4c58"
                        strokeWidth="2"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                    {workspace.active && workspace.draft_points.length > 0 && (
                      <polyline
                        points={workspace.draft_points
                          .map((item) => item.pixel_uv.join(","))
                          .join(" ")}
                        fill="none"
                        stroke="#f6b94a"
                        strokeWidth="2"
                        strokeDasharray="6 4"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                    {workspace.draft_points.map((item, index) => (
                      <g key={`${item.pixel_uv.join("-")}-${index}`}>
                        <circle
                          cx={item.pixel_uv[0]}
                          cy={item.pixel_uv[1]}
                          r="6"
                          fill="#f6b94a"
                          stroke="#071016"
                          strokeWidth="2"
                          vectorEffect="non-scaling-stroke"
                        />
                        <text
                          x={item.pixel_uv[0] + 9}
                          y={item.pixel_uv[1] - 9}
                          fill="#f6b94a"
                          fontSize="13"
                          fontWeight="700"
                        >
                          {index + 1}
                        </text>
                      </g>
                    ))}
                  </svg>
                )}
                {placeTarget.configured && placeTarget.pixel_uv && (
                  <div
                    className={cn(
                      "pointer-events-none absolute size-10 -translate-x-1/2 -translate-y-1/2 rounded-full border-2",
                      placeTarget.valid
                        ? "border-[#f6b94a] shadow-[0_0_18px_rgba(246,185,74,0.65)]"
                        : "border-[#ff666d]",
                    )}
                    style={{
                      left: `${(placeTarget.pixel_uv[0] / 640) * 100}%`,
                      top: `${(placeTarget.pixel_uv[1] / 480) * 100}%`,
                    }}
                  >
                    <span
                      className={cn(
                        "absolute left-1/2 top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full",
                        placeTarget.valid ? "bg-[#f6b94a]" : "bg-[#ff666d]",
                      )}
                    />
                    <span
                      className={cn(
                        "absolute left-9 top-0 whitespace-nowrap px-1.5 py-1 text-[10px] font-bold text-[#071016]",
                        placeTarget.valid ? "bg-[#f6b94a]" : "bg-[#ff666d]",
                      )}
                    >
                      {placeTarget.valid ? "指定放置点" : "放置点已失效"}
                    </span>
                  </div>
                )}
                {vision?.detections.map((item) => {
                  const [x, y, width, height] = item.bbox_xywh;
                  const recommended =
                    item.detection_id === vision.recommended_object_id ||
                    item.detection_id === vision.destination_id;
                  return (
                    <div
                      key={item.detection_id}
                      className={cn(
                        "pointer-events-none absolute border-2",
                        item.rejection_reason
                          ? "border-[#ff666d]"
                          : item.kind === "destination"
                            ? "border-[#f6b94a]"
                            : "border-[#42e7bd]",
                        recommended &&
                          "shadow-[0_0_16px_rgba(66,231,189,0.35)]",
                      )}
                      style={{
                        left: `${(x / 640) * 100}%`,
                        top: `${(y / 480) * 100}%`,
                        width: `${(width / 640) * 100}%`,
                        height: `${(height / 480) * 100}%`,
                      }}
                    >
                      <span
                        className={cn(
                          "absolute -top-6 left-[-2px] whitespace-nowrap px-1.5 py-1 font-mono text-[10px] text-[#061014]",
                          item.rejection_reason
                            ? "bg-[#ff666d]"
                            : item.kind === "destination"
                              ? "bg-[#f6b94a]"
                              : "bg-[#42e7bd]",
                        )}
                      >
                        {visionLabels[item.class_name]}{" "}
                        {(item.confidence * 100).toFixed(0)}%
                        {recommended ? " · 推荐" : ""}
                      </span>
                    </div>
                  );
                })}
                {point && (
                  <div
                    className="pointer-events-none absolute size-8 -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${(point.pixel_uv[0] / 640) * 100}%`,
                      top: `${(point.pixel_uv[1] / 480) * 100}%`,
                    }}
                  >
                    <span className="absolute left-1/2 top-0 h-full w-px bg-[#ff4c58]" />
                    <span className="absolute left-0 top-1/2 h-px w-full bg-[#ff4c58]" />
                  </div>
                )}
              </div>
            </div>
            <div className="flex h-11 items-center justify-between border-t border-[#203b46] px-4 text-xs text-[#718791]">
              <span>
                {workspace.active
                  ? `工作区标定：请点击${workspace.next_label ?? "下一点"}（${workspace.point_count + 1}/4）`
                  : placeTargetSelecting
                    ? "指定位置标定：请点击桌面上的放置中心"
                    : "点击目标中心读取像素、深度与相机坐标"}
              </span>
              <span
                className={cn(
                  "font-mono",
                  workspace.active ? "text-[#f6b94a]" : "text-[#42e7bd]",
                )}
              >
                {workspace.active
                  ? "按 左上 → 右上 → 右下 → 左下"
                  : placeTargetSelecting
                    ? "等待点击放置点"
                    : point
                      ? `UV (${point.pixel_uv.join(", ")})`
                      : "UV (—, —)"}
              </span>
            </div>
          </section>

          <aside className="space-y-4">
            <PressureCharts history={pressureHistory} />
            <InfoCard title="目标坐标" icon={Crosshair}>
              <div className="grid grid-cols-3 divide-x divide-[#203b46] border border-[#203b46] bg-[#08131a]">
                {(["X", "Y", "Z"] as const).map((axis, index) => (
                  <div key={axis} className="px-3 py-4">
                    <div className="text-[10px] text-[#637b85]">{axis} / m</div>
                    <div className="mt-2 font-mono text-sm text-[#42e7bd]">
                      {point ? point.camera_xyz_m[index].toFixed(4) : "—"}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 flex justify-between text-xs text-[#718791]">
                <span>深度</span>
                <span className="font-mono text-white">
                  {point ? `${point.depth_m.toFixed(4)} m` : "—"}
                </span>
              </div>
            </InfoCard>

            <InfoCard title="视觉识别" icon={ScanSearch}>
              <div className="mb-3 flex items-center justify-between border-b border-[#1c333d] pb-2 text-xs">
                <span className="text-[#718791]">桌面工作区</span>
                <span
                  className={
                    workspace.calibrated ? "text-[#42e7bd]" : "text-[#f6b94a]"
                  }
                >
                  {workspace.calibrated
                    ? `已标定 · 残差 ${(workspace.max_plane_residual_mm ?? 0).toFixed(1)} mm`
                    : workspace.active
                      ? `采集中 ${workspace.point_count}/4`
                      : "未标定"}
                </span>
              </div>
              {vision ? (
                <div className="space-y-2 text-xs">
                  {vision.detections.map((item) => (
                    <div
                      key={item.detection_id}
                      className="border-b border-[#1c333d] pb-2 last:border-0"
                    >
                      <div className="flex justify-between">
                        <span
                          className={
                            item.rejection_reason
                              ? "text-[#ff8e93]"
                              : "text-[#c7d2d6]"
                          }
                        >
                          {visionLabels[item.class_name]}
                        </span>
                        <span className="font-mono text-[#42e7bd]">
                          {(item.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-[#637b85]">
                        UV {item.center_uv.join(", ")} ·{" "}
                        {item.depth_m == null
                          ? "深度无效"
                          : `${(item.depth_m * 1000).toFixed(1)} mm`}
                      </div>
                      {item.rejection_reason && (
                        <div className="mt-1 text-[10px] text-[#ff8e93]">
                          {item.rejection_reason}
                        </div>
                      )}
                    </div>
                  ))}
                  {vision.warnings.map((warning) => (
                    <div
                      key={warning}
                      className="text-[10px] leading-4 text-[#f6b94a]"
                    >
                      {warning}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs leading-5 text-[#718791]">
                  先标定桌面工作区，再点击“识别物体”检测绿色圆柱体、灰色正方体和绿色实心托盘。
                </p>
              )}
            </InfoCard>

            <InfoCard title="结构化搬运指令" icon={PackageCheck}>
              <label className="block text-xs text-[#718791]">
                目标物体
                <select
                  value={transferTarget}
                  disabled={Boolean(transfer?.running)}
                  onChange={(event) => {
                    setTransferTarget(
                      event.target.value as "green_cylinder" | "gray_cube",
                    );
                    setTransfer(null);
                  }}
                  className="mt-2 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white outline-none focus:border-[#26d0a8]"
                >
                  <option value="green_cylinder">绿色圆柱体</option>
                  <option value="gray_cube">灰色正方体</option>
                </select>
              </label>
              <label className="mt-3 block text-xs text-[#718791]">
                软硬要求
                <select
                  value={transferProperty}
                  disabled={Boolean(transfer?.running)}
                  onChange={(event) => {
                    setTransferProperty(event.target.value as "" | "软" | "硬");
                    setTransfer(null);
                  }}
                  className="mt-2 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white outline-none focus:border-[#26d0a8]"
                >
                  <option value="">不限</option>
                  <option value="软">软</option>
                  <option value="硬">硬</option>
                </select>
              </label>
              {transferProperty && (
                <p className="mt-2 text-xs leading-5 text-[#718791]">
                  逐个夹持判断软硬，不符合则原位松开再抓下一个；全部不符合时提示未找到匹配物体。
                </p>
              )}
              <label className="mt-3 block text-xs text-[#718791]">
                放置目标
                <select
                  value={transferDestination}
                  disabled={Boolean(transfer?.running)}
                  onChange={(event) => {
                    setTransferDestination(
                      event.target.value as
                        | "green_tray"
                        | "specified_position",
                    );
                    setTransfer(null);
                    setPlaceTargetSelecting(false);
                  }}
                  className="mt-2 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white outline-none focus:border-[#26d0a8]"
                >
                  <option value="green_tray">绿色托盘</option>
                  <option value="specified_position">鼠标指定位置</option>
                </select>
              </label>
              {transferDestination === "specified_position" && (
                <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-[#294550] bg-[#071116] px-3 py-2 text-xs">
                  <span
                    className={
                      placeTarget.valid
                        ? "text-[#42e7bd]"
                        : "text-[#f6b94a]"
                    }
                  >
                    {placeTarget.valid && placeTarget.pixel_uv
                      ? `已标定 UV (${placeTarget.pixel_uv.join(", ")})`
                      : placeTarget.configured
                        ? "桌面或外参已改变，请重新标定"
                        : "尚未标定放置点"}
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={
                      !devices.camera.frame_ready ||
                      !workspace.calibrated ||
                      workspace.active ||
                      Boolean(transfer?.running)
                    }
                    onClick={() => {
                      setPlaceTargetSelecting((current) => !current);
                      setTransfer(null);
                    }}
                  >
                    <Crosshair className="size-3.5" />
                    {placeTargetSelecting
                      ? "取消标定"
                      : placeTarget.configured
                        ? "重新标定"
                        : "标定放置点"}
                  </Button>
                </div>
              )}
              <div className="mt-3 rounded-md border border-[#294550] bg-[#071116] px-3 py-2 text-xs text-[#a9bbc2]">
                把{transferProperty ? `${transferProperty}的` : ""}{visionLabels[transferTarget]}
                {transferDestination === "green_tray"
                  ? "放入绿色托盘里面"
                  : "放到鼠标指定位置"}
              </div>
              <div className="mt-3 flex gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={
                    !devices.camera.frame_ready ||
                    !workspace.calibrated ||
                    (transferDestination === "specified_position" &&
                      !placeTarget.valid) ||
                    Boolean(transfer?.running) ||
                    busy === "transfer-preview"
                  }
                  onClick={() => void previewTransfer()}
                >
                  生成搬运预览
                </Button>
                <Button
                  size="sm"
                  disabled={
                    !transfer?.plan ||
                    transfer.status !== "preview_ready" ||
                    !devices.robot.connected ||
                    !devices.robot.enabled ||
                    busy === "transfer-execute"
                  }
                  onClick={() => void executeTransfer()}
                >
                  确认执行
                </Button>
                {transfer?.running && (
                  <Button
                    size="sm"
                    variant="danger"
                    disabled={busy === "transfer-stop"}
                    onClick={() => void stopTransfer()}
                  >
                    停止
                  </Button>
                )}
              </div>
              {transfer?.plan && (
                <div className="mt-3 space-y-2 border-t border-[#1c333d] pt-3 text-[10px] leading-4 text-[#718791]">
                  <div className="flex justify-between">
                    <span>任务状态</span>
                    <span
                      className={
                        transfer.status === "failed"
                          ? "text-[#ff8e93]"
                          : transfer.status === "completed"
                            ? "text-[#42e7bd]"
                            : "text-[#f6b94a]"
                      }
                    >
                      {transfer.current_step ?? transfer.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>搬运 / 升降速度</span>
                    <span>
                      {transfer.plan.motion.transit_speed} /{" "}
                      {transfer.plan.motion.approach_speed}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>抓取高度微调</span>
                    <span>
                      {transfer.plan.motion.grasp_clearance_mm.toFixed(0)} mm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>物体桌面高度</span>
                    <span>
                      {transfer.plan.motion.source_height_above_table_mm.toFixed(
                        1,
                      )}{" "}
                      mm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>抓取后抬升</span>
                    <span>
                      {transfer.plan.motion.pick_lift_mm.toFixed(0)} mm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>
                      {transfer.plan.command.destination_class === "green_tray"
                        ? "托盘上方距离"
                        : "指定点上方距离"}
                    </span>
                    <span>
                      {transfer.plan.motion.place_approach_mm.toFixed(0)} mm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>放置间隙</span>
                    <span>
                      {transfer.plan.motion.place_clearance_mm.toFixed(0)} mm
                    </span>
                  </div>
                  {transfer.plan.command.target_property && (
                    <div className="rounded-md border border-[#294550] px-3 py-2">
                      <div>要求：{transfer.plan.command.target_property} · 候选 {transfer.plan.candidates?.length ?? 1} 个</div>
                      {transfer.candidate_index && <div>当前检查第 {transfer.candidate_index} 个</div>}
                      {transfer.attempts?.map((attempt, index) => (
                        <div key={attempt.source.detection_id} className="mt-1 text-[#91a4ac]">
                          第 {index + 1} 个：{attempt.recognition_result.property ?? "识别失败"}
                          {attempt.matched ? "，符合要求" : "，未匹配"}
                        </div>
                      ))}
                    </div>
                  )}
                  {transfer.recognition_result && (
                    <div className="rounded-md border border-[#294550] bg-[#071116] px-3 py-2">
                      <div className="flex justify-between gap-3">
                        <span>触觉识别</span>
                        <span
                          className={
                            transfer.recognition_result.status === "completed"
                              ? "text-[#42e7bd]"
                              : "text-[#f6b94a]"
                          }
                        >
                          {transfer.recognition_result.status === "completed"
                            ? transfer.recognition_result.property
                            : transfer.recognition_result.status === "failed"
                              ? "识别失败"
                              : "不可用"}
                        </span>
                      </div>
                      {transfer.recognition_result.status === "completed" && (
                        <div className="mt-1 flex justify-between text-[#91a4ac]">
                          <span>
                            视觉种类 {transfer.recognition_result.vision_class} · 性质{" "}
                            {transfer.recognition_result.property}
                          </span>
                          <span>
                            {Math.round(
                              (transfer.recognition_result.confidence ?? 0) * 100,
                            )}
                            %
                          </span>
                        </div>
                      )}
                      {transfer.recognition_result.error && (
                        <div className="mt-1 text-[#f6b94a]">
                          {transfer.recognition_result.error}
                        </div>
                      )}
                    </div>
                  )}
                  <details>
                    <summary className="cursor-pointer text-[#91a4ac]">
                      查看完整动作步骤
                    </summary>
                    <ol className="mt-2 list-decimal space-y-1 pl-4">
                      {transfer.plan.steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ol>
                  </details>
                  {transfer.error && (
                    <div className="text-[#ff8e93]">{transfer.error}</div>
                  )}
                  {transfer.holding_object && (
                    <div className="text-[#f6b94a]">
                      夹爪可能仍持有物体，请勿直接断电或松开。
                    </div>
                  )}
                </div>
              )}
              {(!devices.robot.connected || !devices.robot.enabled) && (
                <p className="mt-3 text-[10px] leading-4 text-[#f6b94a]">
                  执行前需要在“机械臂”面板连接并使能。
                </p>
              )}
            </InfoCard>

            <InfoCard title="抓取流程" icon={SlidersHorizontal}>
              <div className="space-y-0">
                {flow.map(([number, label, value], index) => (
                  <div
                    key={number}
                    className="grid grid-cols-[30px_1fr_auto] items-center border-b border-[#1c333d] py-3 last:border-0"
                  >
                    <span className="font-mono text-[10px] text-[#3e5964]">
                      {number}
                    </span>
                    <span className="text-xs text-[#a9b9bf]">{label}</span>
                    <span
                      className={cn(
                        "font-mono text-[11px]",
                        index < 3 && point
                          ? "text-[#42e7bd]"
                          : "text-[#718791]",
                      )}
                    >
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </InfoCard>

            <InfoCard title="系统状态" icon={Activity}>
              <Metric
                label="总体就绪"
                value={overallReady ? "就绪" : "未就绪"}
                ok={overallReady}
              />
              <Metric
                label="标定外参"
                value={
                  calibration.solution
                    ? `${calibration.solution.validation.rms_mm.toFixed(1)} mm RMS`
                    : "未求解"
                }
                ok={Boolean(calibration.solution?.quality.hover_ready)}
              />
              <Metric
                label="压力传感器"
                value={devices.sensors.frame_ready ? "双侧数据正常" : "未就绪"}
                ok={devices.sensors.frame_ready}
              />
              <Metric
                label="运动配置"
                value={devices.robot.motion_configured ? "允许" : "禁止"}
                ok={Boolean(devices.robot.motion_configured)}
              />
              <Metric label="API" value="127.0.0.1:8000" ok />
            </InfoCard>
          </aside>
        </div>
      </section>

      <Dialog
        open={panel !== null}
        onOpenChange={(open) => !open && setPanel(null)}
      >
        <DialogContent
          className={
            panel === "vision-data" || panel === "tactile-data"
              ? "w-[min(1180px,calc(100vw-48px))]"
              : undefined
          }
        >
          {renderPanel(panel, {
            devices,
            point,
            calibration,
            workspace,
            logs,
            busy,
            action,
          })}
        </DialogContent>
      </Dialog>

      {notice && (
        <div
          className={cn(
            "fixed bottom-5 right-5 z-[80] flex max-w-md items-center gap-3 border px-4 py-3 text-sm shadow-2xl",
            notice.kind === "ok"
              ? "border-[#287060] bg-[#102b28] text-[#bcebdd]"
              : "border-[#824046] bg-[#32171a] text-[#ffc4c7]",
          )}
        >
          {busy && <LoaderCircle className="size-4 animate-spin" />}
          {notice.text}
        </div>
      )}
    </main>
  );
}

function HeaderStatus({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex h-9 items-center gap-2 border border-[#26414c] bg-[#0b1b23] px-3 text-xs text-[#a3b4ba]">
      <StatusDot ok={ok} />
      {label}
    </div>
  );
}

function PressureCharts({ history }: { history: PressurePoint[] }) {
  return (
    <InfoCard title="夹爪压力变化" icon={RadioTower}>
      <div className="space-y-3">
        <PressureChart
          title="左夹爪"
          color="#42e7bd"
          sum={history.map((point) => point.leftSum)}
          peak={history.map((point) => point.leftPeak)}
        />
        <PressureChart
          title="右夹爪"
          color="#6ea8ff"
          sum={history.map((point) => point.rightSum)}
          peak={history.map((point) => point.rightPeak)}
        />
      </div>
    </InfoCard>
  );
}

function PressureChart({
  title,
  color,
  sum,
  peak,
}: {
  title: string;
  color: string;
  sum: number[];
  peak: number[];
}) {
  const scale = Math.max(1, ...sum, ...peak);
  const points = (values: number[]) =>
    values
      .map(
        (value, index) =>
          `${(index / Math.max(1, values.length - 1)) * 100},${46 - (value / scale) * 42}`,
      )
      .join(" ");
  return (
    <div className="rounded-md border border-[#294550] bg-[#071116] p-2">
      <div className="mb-1 flex items-center justify-between text-[10px]">
        <span style={{ color }}>{title}</span>
        <span className="text-[#91a4ac]">量程 0–{scale.toFixed(1)}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex h-20 w-9 shrink-0 flex-col justify-between text-right font-mono text-[9px] text-[#637b85]">
          <span>{scale.toFixed(0)}</span><span>{(scale / 2).toFixed(0)}</span><span>0</span>
        </div>
        <svg viewBox="0 0 100 48" className="h-20 min-w-0 flex-1" preserveAspectRatio="none" aria-label={`${title}总压力和峰值压力曲线`}>
          {[4, 25, 46].map((y) => <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="#203740" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />)}
          <polyline points={points(sum)} fill="none" stroke={color} strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
          <polyline points={points(peak)} fill="none" stroke="#ffffff" strokeWidth="0.9" strokeDasharray="2 1" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-[#718791]"><span>总压力和</span><span className="text-white">峰值压力</span></div>
      <div className="mt-1 flex justify-between font-mono text-[9px] text-[#91a4ac]"><span>当前总和 {sum.at(-1)?.toFixed(1) ?? "—"}</span><span>当前峰值 {peak.at(-1)?.toFixed(1) ?? "—"}</span></div>
    </div>
  );
}

function InfoCard({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: typeof Activity;
  children: React.ReactNode;
}) {
  if (["目标坐标", "视觉识别", "抓取流程", "系统状态"].includes(title)) return null;
  return (
    <section className="border border-[#203b46] bg-[#0b1820]">
      <div className="flex h-11 items-center gap-2 border-b border-[#203b46] px-4 text-sm">
        <Icon className="size-4 text-[#42e7bd]" />
        {title}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}
function Metric({
  label,
  value,
  ok,
}: {
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div className="flex items-center justify-between border-b border-[#1c333d] py-2.5 text-xs last:border-0">
      <span className="text-[#718791]">{label}</span>
      <span className="flex items-center gap-2 text-[#c7d2d6]">
        <StatusDot ok={ok} />
        {value}
      </span>
    </div>
  );
}
