"use client";

/* eslint-disable @next/next/no-img-element -- MJPEG and dataset endpoints are not compatible with the Next.js image optimizer. */

import { useEffect, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Bot,
  Camera,
  Check,
  Hand,
  ImagePlus,
  Play,
  Plus,
  Power,
  RadioTower,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  Square,
  Trash2,
  Unplug,
} from "lucide-react";
import { API_BASE, api, post } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  DepthProposalResult,
  Panel,
  PanelContext,
  SensorData,
  SensorSettings,
  TactileSample,
  TactileSampleDetail,
  TactileSampleList,
  TactileStatus,
  TargetingSettings,
  TargetPreview,
  TrainingClass,
  TransferSettings,
  VisionCaptureResult,
  VisionDatasetList,
  VisionDatasetSample,
  VisionDatasetStatus,
} from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function renderPanel(panel: Panel | null, context: PanelContext) {
  if (!panel) return null;
  if (panel === "camera") return <CameraPanel {...context} />;
  if (panel === "calibration") return <CalibrationPanel {...context} />;
  if (panel === "robot") return <RobotPanel {...context} />;
  if (panel === "gripper") return <GripperPanel {...context} />;
  if (panel === "sensors") return <SensorsPanel {...context} />;
  if (panel === "tactile-data") return <TactileRecognitionPanel {...context} />;
  if (panel === "vision-data") return <VisionDataPanel {...context} />;
  if (panel === "devices") return <DevicesPanel {...context} />;
  if (panel === "logs") return <LogsPanel {...context} />;
  return <SettingsPanel {...context} />;
}

function CameraPanel({ devices, action }: PanelContext) {
  return (
    <>
      <PanelHeader
        title="相机控制"
        description="管理 D435 取流并检查彩色与深度对齐状态。"
      />
      <div className="grid grid-cols-2 gap-3">
        <StateBlock
          label="取流状态"
          value={devices.camera.running ? "运行中" : "已停止"}
          ok={devices.camera.frame_ready}
        />
        <StateBlock
          label="最近帧"
          value={
            devices.camera.frame_age_s == null
              ? "—"
              : `${devices.camera.frame_age_s.toFixed(2)} s`
          }
          ok={devices.camera.frame_ready}
        />
      </div>
      {devices.camera.error && <Warning>{devices.camera.error}</Warning>}
      <div className="mt-5 flex gap-2">
        <Button
          onClick={() =>
            void action(
              "camera-start",
              () => post("/api/camera/start"),
              "D435 已启动",
            )
          }
        >
          <Play className="size-4" />
          启动
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            void action(
              "camera-stop",
              () => post("/api/camera/stop"),
              "D435 已停止",
            )
          }
        >
          <Square className="size-4" />
          停止
        </Button>
      </div>
    </>
  );
}

function CalibrationPanel({ point, calibration, action }: PanelContext) {
  const solution = calibration.solution;
  const [records, setRecords] = useState<
    Array<{
      filename: string;
      sample_count: number;
      selected: boolean;
      solved_source: boolean;
    }>
  >([]);
  const [selectedRecord, setSelectedRecord] = useState("");
  useEffect(() => {
    void api<
      Array<{
        filename: string;
        sample_count: number;
        selected: boolean;
        solved_source: boolean;
      }>
    >("/api/calibration/records").then((items) => {
      setRecords(items);
      const serverSelected = items.find((item) => item.selected);
      const preferred = calibration.active
        ? serverSelected
        : (items.find((item) => item.solved_source) ?? serverSelected);
      if (preferred) setSelectedRecord(preferred.filename);
    });
  }, [calibration.active, calibration.output, calibration.sample_count]);
  const selectedInfo = records.find(
    (record) => record.filename === selectedRecord,
  );
  const displayedSampleCount = calibration.active
    ? calibration.sample_count
    : (selectedInfo?.sample_count ?? calibration.sample_count);
  return (
    <>
      <PanelHeader
        title="蓝点标定与外参"
        description="采集相机点与 RM65 状态，并求解相机到 Base 的刚体变换。"
      />
      <div className="grid grid-cols-3 gap-3">
        <StateBlock
          label="会话"
          value={calibration.active ? "采集中" : "未开始"}
          ok={calibration.active}
        />
        <StateBlock
          label="当前记录样本"
          value={String(displayedSampleCount)}
          ok={displayedSampleCount >= 10}
        />
        <StateBlock
          label="外参状态"
          value={
            solution
              ? solution.quality.hover_ready
                ? "通过运动门槛"
                : "精度不足"
              : "未求解"
          }
          ok={Boolean(solution?.quality.hover_ready)}
        />
      </div>
      <div className="mt-4 rounded-lg border border-[#294550] bg-[#09151c] p-4 font-mono text-xs leading-6 text-[#a9bbc2]">
        {point ? (
          <>
            camera_xyz_m = [
            {point.camera_xyz_m.map((value) => value.toFixed(5)).join(", ")}]
            <br />
            depth_m = {point.depth_m.toFixed(5)}
          </>
        ) : (
          "请先关闭弹窗，在主画面点击蓝点中心。"
        )}
      </div>
      {records.length > 0 && (
        <div className="mt-4">
          <label
            htmlFor="calibration-record"
            className="mb-2 block text-xs text-[#8fa6af]"
          >
            要继续追加的标定记录
          </label>
          <select
            id="calibration-record"
            value={selectedRecord}
            onChange={(event) => setSelectedRecord(event.target.value)}
            className="h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
          >
            {records.map((record) => (
              <option key={record.filename} value={record.filename}>
                {record.sample_count} 点
                {record.solved_source ? " · 当前外参来源" : ""} · {record.filename}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="mt-5 flex flex-wrap gap-2">
        <Button
          onClick={() =>
            void action(
              "cal-start",
              () =>
                selectedRecord
                  ? post("/api/calibration/resume", {
                      filename: selectedRecord,
                    })
                  : post("/api/calibration/start"),
              selectedInfo
                ? `已继续 ${selectedInfo.sample_count} 点的标定记录`
                : "标定会话已开始",
            )
          }
        >
          {selectedInfo ? `继续采集（${selectedInfo.sample_count} 点）` : "开始采集"}
        </Button>
        <Button
          variant="outline"
          disabled={calibration.active}
          onClick={() => {
            if (
              window.confirm(
                "确定新建空标定记录吗？已有记录不会删除，但新记录会从 0 点开始。",
              )
            ) {
              void action(
                "cal-new",
                () => post("/api/calibration/new"),
                "新的标定记录已创建",
              );
            }
          }}
        >
          <Plus className="size-4" />
          新建记录
        </Button>
        <Button
          variant="secondary"
          disabled={!calibration.active || !point}
          onClick={() =>
            void action(
              "cal-save",
              () =>
                post("/api/calibration/sample", {
                  u: point!.pixel_uv[0],
                  v: point!.pixel_uv[1],
                }),
              "标定样本已保存",
            )
          }
        >
          保存当前样本
        </Button>
        <Button
          variant="outline"
          disabled={!calibration.active}
          onClick={() =>
            void action(
              "cal-stop",
              () => post("/api/calibration/stop"),
              "标定会话已结束",
            )
          }
        >
          结束会话
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            void action(
              "cal-solve",
              () => post("/api/calibration/solve"),
              "外参求解完成",
            )
          }
        >
          求解最新记录
        </Button>
      </div>
      {solution && (
        <div className="mt-5 border-t border-[#294550] pt-5">
          <div className="grid grid-cols-3 gap-3">
            <StateBlock
              label="有效样本"
              value={`${solution.inlier_count} / ${solution.sample_count}`}
              ok={solution.inlier_count >= 6}
            />
            <StateBlock
              label="验证 RMS"
              value={`${solution.validation.rms_mm.toFixed(2)} mm`}
              ok={solution.validation.rms_mm <= 10}
            />
            <StateBlock
              label="验证最大误差"
              value={`${solution.validation.max_mm.toFixed(2)} mm`}
              ok={solution.validation.max_mm <= 15}
            />
          </div>
          <div className="mt-4 overflow-auto rounded-lg border border-[#294550] bg-[#071116] p-3 font-mono text-[11px] leading-5 text-[#9bb0b8]">
            {solution.transform_4x4.map((row) => (
              <div key={row.join(",")}>
                [{row.map((value) => value.toFixed(6).padStart(12)).join(", ")}]
              </div>
            ))}
          </div>
          {!solution.quality.hover_ready && (
            <Warning>
              当前验证误差没有达到运动门槛，禁止据此开启自动运动。建议检查异常样本并补采。
            </Warning>
          )}
          <p className="mt-3 text-xs leading-5 text-[#8297a1]">
            异常样本：{solution.outlier_sample_indices.join(", ") || "无"}
            。蓝点到 TCP 偏移仍未独立求解。
          </p>
        </div>
      )}
      {(calibration.active ? calibration.output : selectedRecord) && (
        <p className="mt-4 break-all text-xs text-[#6f858e]">
          记录文件：{calibration.active ? calibration.output : selectedRecord}
        </p>
      )}
    </>
  );
}

function RobotPanel({ devices, point, busy, action }: PanelContext) {
  const [pose, setPose] = useState([0, 0, 300, 180, 0, 0]);
  const [speed, setSpeed] = useState(5);
  const [preview, setPreview] = useState<TargetPreview | null>(null);
  const robot = devices.robot;
  const currentPose = Array.isArray(robot.pose)
    ? (robot.pose as number[])
    : null;
  const setValue = (index: number, value: string) =>
    setPose((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? Number(value) : item,
      ),
    );
  return (
    <>
      <PanelHeader
        title="RM65 机械臂"
        description="连接、实时位姿、安全使能和受控位姿运动。连接并使能后无需额外解锁。"
      />
      <div className="grid grid-cols-3 gap-3">
        <StateBlock
          label="后台服务"
          value={robot.service_online ? "在线" : "离线"}
          ok={Boolean(robot.service_online)}
        />
        <StateBlock
          label="控制连接"
          value={robot.connected ? "已连接" : "未连接"}
          ok={Boolean(robot.connected)}
        />
        <StateBlock
          label="关节使能"
          value={robot.enabled ? "已使能" : "未使能"}
          ok={Boolean(robot.enabled)}
        />
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          onClick={() =>
            void action(
              "robot-service",
              () => post("/api/robot/service/start"),
              "RM65 服务已启动",
            )
          }
        >
          启动服务
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            void action(
              "robot-connect",
              () => post("/api/robot/connect"),
              "RM65 已连接",
            )
          }
        >
          连接
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            void action(
              "robot-disconnect",
              () => post("/api/robot/disconnect"),
              "RM65 已断开",
            )
          }
        >
          断开
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            void action(
              "robot-enable",
              () => post("/api/robot/enable"),
              "RM65 已使能",
            )
          }
        >
          <Power className="size-4" />
          使能
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            void action(
              "robot-disable",
              () => post("/api/robot/disable"),
              "RM65 已失能",
            )
          }
        >
          <Unplug className="size-4" />
          失能
        </Button>
      </div>
      <div className="mt-4 rounded-lg border border-[#294550] bg-[#08141a] p-3">
        <div className="mb-2 text-xs text-[#708790]">
          当前 TCP 位姿 [mm, deg] · 约 1.5 秒刷新
        </div>
        <div className="grid grid-cols-6 gap-2">
          {["X", "Y", "Z", "RX", "RY", "RZ"].map((label, index) => (
            <div key={label}>
              <div className="text-[10px] text-[#607680]">{label}</div>
              <div className="mt-1 font-mono text-xs text-[#42e7bd]">
                {currentPose && typeof currentPose[index] === "number"
                  ? currentPose[index].toFixed(2)
                  : "—"}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-5 border-t border-[#294550] pt-5">
        <h3 className="mb-3 text-sm font-medium">视觉目标</h3>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            disabled={!point}
            onClick={() =>
              void action(
                "target-preview",
                async () =>
                  setPreview(
                    await api<TargetPreview>("/api/targeting/preview"),
                  ),
                "视觉目标已计算",
              )
            }
          >
            计算预览
          </Button>
          <Button
            disabled={
              !preview ||
              !robot.connected ||
              !robot.enabled ||
              busy === "target-move"
            }
            onClick={() =>
              void action(
                "target-move",
                () =>
                  post("/api/targeting/move-target", { confirm: true, speed }),
                "机械臂已到达视觉目标位置",
              )
            }
          >
            确认并移动到目标点
          </Button>
        </div>
        {preview && (
          <div className="mt-3 rounded-lg border border-[#294550] bg-[#071116] p-3 font-mono text-xs leading-5 text-[#a9bbc2]">
            target = [
            {preview.target_pose_mm_deg
              .map((value) => value.toFixed(2))
              .join(", ")}
            ]<br />
            蓝点→尖端补偿 XYZ = [
            {preview.marker_to_tip_xyz_mm
              .map((value) => value.toFixed(1))
              .join(", ")}
            ] mm
          </div>
        )}
        <div className="mt-4">
          <Range
            label="运动速度"
            value={speed}
            min={1}
            max={100}
            onChange={setSpeed}
          />
        </div>
        {robot.motion_configured ? (
          <Warning>
            运动配置已允许，连接并使能后无需额外解锁。高速运动风险显著增加，请从低速逐步验证。
          </Warning>
        ) : (
          <Warning>配置文件仍禁止真机运动。</Warning>
        )}
      </div>
      <details className="mt-5 border-t border-[#294550] pt-4">
        <summary className="cursor-pointer text-xs text-[#708790]">
          高级：手动目标位姿
        </summary>
        <div className="mt-3 grid grid-cols-6 gap-2">
          {["X", "Y", "Z", "RX", "RY", "RZ"].map((label, index) => (
            <label key={label} className="text-[11px] text-[#8297a1]">
              {label}
              <input
                type="number"
                value={pose[index]}
                onChange={(event) => setValue(index, event.target.value)}
                className="mt-1 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-2 py-2 font-mono text-xs text-white outline-none focus:border-[#26d0a8]"
              />
            </label>
          ))}
        </div>
        <Button
          className="mt-3"
          variant="outline"
          disabled={!robot.connected || !robot.enabled || busy === "robot-move"}
          onClick={() =>
            void action(
              "robot-move",
              () =>
                post("/api/robot/move", {
                  pose,
                  mode: "MoveJ_P",
                  speed,
                  acc: 10,
                }),
              "手动目标运动完成",
            )
          }
        >
          移动到手动目标
        </Button>
      </details>
    </>
  );
}

function GripperPanel({ devices, action }: PanelContext) {
  const recommendedPort = devices.serial_ports.find((item) =>
    item.description.toLowerCase().includes("usb serial port"),
  )?.device;
  const [port, setPort] = useState(
    devices.gripper.connected
      ? devices.gripper.port
      : recommendedPort || devices.gripper.port || "COM11",
  );
  const [position, setPosition] = useState(1000);
  const [speed, setSpeed] = useState(20);
  const [force, setForce] = useState(20);
  return (
    <>
      <PanelHeader
        title="DH5 串口夹爪"
        description="连接时会验证 DH5 Modbus 回包；仅打开串口不再视为连接成功。"
      />
      <div className="grid grid-cols-[1fr_auto] gap-2">
        <select
          value={port}
          onChange={(event) => setPort(event.target.value)}
          className="h-10 rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
        >
          <option value={port}>{port}</option>
          {devices.serial_ports
            .filter((item) => item.device !== port)
            .map((item) => (
              <option key={item.device} value={item.device}>
                {item.device} · {item.description}
              </option>
            ))}
        </select>
        <Button
          onClick={() =>
            void action(
              "gripper-connect",
              () =>
                post("/api/gripper/connect", {
                  port,
                  baud_rate: 115200,
                  modbus_id: 1,
                  stop_bits: 1,
                  parity: "N",
                }),
              "DH5 握手验证成功",
            )
          }
        >
          连接并验证
        </Button>
      </div>
      {recommendedPort && (
        <p className="mt-2 text-xs text-[#42e7bd]">
          检测到夹爪候选串口：{recommendedPort} · USB Serial Port
        </p>
      )}
      <div className="mt-3 flex gap-2">
        <Button
          variant="secondary"
          disabled={!devices.gripper.connected}
          onClick={() =>
            void action(
              "gripper-init",
              () => post("/api/gripper/initialize"),
              "DH5 初始化完成",
            )
          }
        >
          初始化
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            void action(
              "gripper-disconnect",
              () => post("/api/gripper/disconnect"),
              "DH5 已断开",
            )
          }
        >
          断开
        </Button>
      </div>
      <div className="mt-5 space-y-4 border-t border-[#294550] pt-5">
        <Range
          label="位置"
          value={position}
          min={0}
          max={1000}
          onChange={setPosition}
        />
        <Range
          label="速度"
          value={speed}
          min={1}
          max={100}
          onChange={setSpeed}
        />
        <Range
          label="力"
          value={force}
          min={20}
          max={100}
          onChange={setForce}
        />
        <Button
          disabled={!devices.gripper.connected}
          onClick={() =>
            void action(
              "gripper-command",
              () => post("/api/gripper/command", { position, speed, force }),
              "DH5 控制参数已下发",
            )
          }
        >
          下发夹爪命令
        </Button>
      </div>
      <Warning>
        确认选择 USB Serial Port，不要选择 Windows 蓝牙 COM
        口；执行前确保夹爪周围无人手、线缆和障碍物。
      </Warning>
    </>
  );
}

function SensorsPanel({ devices, action }: PanelContext) {
  const [leftPort, setLeftPort] = useState(
    devices.sensors.left.port || "COM78",
  );
  const [rightPort, setRightPort] = useState(
    devices.sensors.right.port || "COM79",
  );
  const [data, setData] = useState<SensorData | null>(null);
  useEffect(() => {
    let active = true;
    const update = () =>
      void api<SensorData>("/api/sensors/data")
        .then((value) => active && setData(value))
        .catch(() => undefined);
    update();
    const timer = window.setInterval(update, 250);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);
  const display = data?.display;
  const grasp = data?.grasp_success;
  return (
    <>
      <PanelHeader
        title="左右压力传感器"
        description="两个独立 Infineon 串口，每侧 32 通道；采用 arm_gripping 的信号处理参数和双侧夹取成功判断。"
      />
      <div className="grid grid-cols-2 gap-3">
        <PortSelect
          label="左传感器"
          value={leftPort}
          onChange={setLeftPort}
          ports={devices.serial_ports}
          fallback="COM78"
        />
        <PortSelect
          label="右传感器"
          value={rightPort}
          onChange={setRightPort}
          ports={devices.serial_ports}
          fallback="COM79"
        />
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          onClick={() =>
            void action(
              "sensors-connect",
              () =>
                post("/api/sensors/connect", {
                  left_port: leftPort,
                  right_port: rightPort,
                }),
              "左右压力传感器已连接",
            )
          }
        >
          连接双侧
        </Button>
        <Button
          variant="secondary"
          disabled={!data?.frame_ready}
          onClick={() =>
            void action(
              "sensors-zero",
              () => post("/api/sensors/zero"),
              "当前空载值已设为基线",
            )
          }
        >
          空载置零
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            void action(
              "sensors-clear",
              () => post("/api/sensors/clear-zero"),
              "传感器基线已清除",
            )
          }
        >
          清除基线
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            void action(
              "sensors-disconnect",
              () => post("/api/sensors/disconnect"),
              "左右压力传感器已断开",
            )
          }
        >
          断开
        </Button>
      </div>
      {grasp && (
        <div
          className={cn(
            "mt-4 rounded-lg border p-3 text-sm",
            grasp.success
              ? "border-[#26715e] bg-[#102b24] text-[#66efc7]"
              : "border-[#294550] bg-[#09151c] text-[#91a4ac]",
          )}
        >
          <div className="flex items-center justify-between">
            <span>夹取成功判断</span>
            <span className="font-mono">
              {grasp.success
                ? "成功"
                : `${grasp.confirm_count}/${grasp.required_frames}`}
            </span>
          </div>
          <div className="mt-1 font-mono text-[10px] text-[#708790]">
            L {grasp.left_peak.toFixed(1)} &gt;{" "}
            {grasp.left_threshold.toFixed(1)} · R {grasp.right_peak.toFixed(1)}{" "}
            &gt; {grasp.right_threshold.toFixed(1)}
          </div>
        </div>
      )}
      <div className="mt-3 text-xs leading-5 text-[#91a4ac]">
        <div>最近空载置零：{data?.zeroed_at ? new Date(data.zeroed_at * 1000).toLocaleString() : "本次连接尚无手动置零记录"}</div>
        <div>原始读数包含零点偏置；接触和采集使用扣除基线后的处理值。空载残余明显时，请在完全张开且无接触时置零。</div>
      </div>
      <div className="mt-5 grid grid-cols-1 gap-5 border-t border-[#294550] pt-5">
        <SensorHeatmap
          title="左侧 8×4"
          values={display?.slice(0, 32) ?? null}
          fps={data?.left.fps ?? 0}
          features={data?.features?.left}
          raw={data?.raw?.slice(0, 32)}
          processed={data?.processed?.slice(0, 32)}
        />
        <SensorHeatmap
          title="右侧 8×4"
          values={display?.slice(32, 64) ?? null}
          fps={data?.right.fps ?? 0}
          features={data?.features?.right}
          raw={data?.raw?.slice(32, 64)}
          processed={data?.processed?.slice(32, 64)}
        />
      </div>
      {!data?.frame_ready && (
        <Warning>
          双侧数据尚未同时就绪。确认两个 CH343 串口没有被 arm_gripping
          或其他程序占用。
        </Warning>
      )}
    </>
  );
}

function TactileRecognitionPanel({ devices, busy, action }: PanelContext) {
  const [status, setStatus] = useState<TactileStatus | null>(null);
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedProperty, setSelectedProperty] = useState("");
  const [newProperty, setNewProperty] = useState("");
  const [samples, setSamples] = useState<TactileSample[]>([]);
  const [activeSample, setActiveSample] = useState<TactileSampleDetail | null>(null);

  const refreshStatus = async () => {
    const next = await api<TactileStatus>("/api/tactile/status");
    setStatus(next);
    const className = selectedClass || next.classes[0]?.name || "";
    if (!selectedClass) setSelectedClass(className);
    const classInfo = next.classes.find((item) => item.name === className);
    setSelectedProperty((current) =>
      classInfo?.properties.includes(current)
        ? current
        : classInfo?.properties[0] ?? "",
    );
    return next;
  };
  const refreshSamples = async (visionClass = selectedClass) => {
    if (!visionClass) {
      setSamples([]);
      return [];
    }
    const result = await api<TactileSampleList>(
      "/api/tactile/samples?limit=500&vision_class=" +
        encodeURIComponent(visionClass),
    );
    setSamples(result.samples);
    return result.samples;
  };

  useEffect(() => {
    let active = true;
    void api<TactileStatus>("/api/tactile/status")
      .then((next) => {
        if (!active) return;
        setStatus(next);
        const initialClass = next.classes[0];
        setSelectedClass(initialClass?.name ?? "");
        setSelectedProperty(initialClass?.properties[0] ?? "");
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedClass) return;
    void api<TactileSampleList>(
      "/api/tactile/samples?limit=500&vision_class=" +
        encodeURIComponent(selectedClass),
    )
      .then((result) => setSamples(result.samples))
      .catch(() => undefined);
  }, [selectedClass]);

  useEffect(() => {
    if (!status?.capture.running && !status?.training.running) return;
    const timer = window.setInterval(() => {
      void refreshStatus().then((next) => {
        if (!next.capture.running && !next.training.running)
          void refreshSamples(selectedClass);
      });
    }, 400);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.capture.running, status?.training.running, selectedClass]);

  const currentClass = status?.classes.find((item) => item.name === selectedClass) ?? null;
  const capture = status?.capture;
  const model = currentClass?.model;
  const canCapture = Boolean(
    selectedClass &&
      selectedProperty &&
      devices.gripper.connected &&
      devices.sensors.frame_ready &&
      !capture?.running,
  );

  const addProperty = () => {
    if (!newProperty.trim()) return;
    void action(
      "tactile-property-add",
      async () => {
        await post("/api/tactile/properties", {
          vision_class: selectedClass,
          property: newProperty,
        });
        setNewProperty("");
        await refreshStatus();
      },
      "性质标签已添加",
    );
  };
  const renameProperty = () => {
    if (!selectedProperty) return;
    const next = window.prompt("新的性质名称", selectedProperty)?.trim();
    if (!next || next === selectedProperty) return;
    void action(
      "tactile-property-rename",
      async () => {
        await post("/api/tactile/properties/rename", {
          vision_class: selectedClass,
          old_property: selectedProperty,
          new_property: next,
        });
        setSelectedProperty(next);
        await refreshStatus();
        await refreshSamples();
      },
      "性质及关联样本已重命名；旧模型已标记为需要重训",
    );
  };
  const deleteProperty = () => {
    if (
      !selectedProperty ||
      !window.confirm(
        `删除性质“${selectedProperty}”？\n关联样本将移入 .trash，活动模型会标记为需要重训。`,
      )
    )
      return;
    void action(
      "tactile-property-delete",
      async () => {
        await api("/api/tactile/properties", {
          method: "DELETE",
          body: JSON.stringify({
            vision_class: selectedClass,
            property: selectedProperty,
            confirm: true,
          }),
        });
        setSelectedProperty("");
        await refreshStatus();
        await refreshSamples();
      },
      "性质及关联样本已移入 .trash",
    );
  };
  const startCapture = () =>
    action(
      "tactile-capture",
      async () => {
        const result = await post<TactileStatus["capture"]>(
          "/api/tactile/capture/start",
          { vision_class: selectedClass, property: selectedProperty },
        );
        setStatus((current) => (current ? { ...current, capture: result } : current));
      },
      "触觉采集已开始，请勿触碰夹爪或移动物体",
    );
  const saveCapture = () =>
    action(
      "tactile-save",
      async () => {
        await post("/api/tactile/capture/save", {
          vision_class: selectedClass,
          property: selectedProperty,
        });
        await refreshStatus();
        await refreshSamples();
      },
      "触觉样本已保存",
    );
  const discardCapture = () =>
    action(
      "tactile-discard",
      async () => {
        await post("/api/tactile/capture/discard");
        await refreshStatus();
      },
      "本次未保存的采集已丢弃",
    );
  const train = () =>
    action(
      "tactile-train",
      async () => {
        await post("/api/tactile/train", { vision_class: selectedClass });
        await refreshStatus();
      },
      "训练任务已启动",
    );
  const openSample = async (sample: TactileSample) =>
    setActiveSample(
      await api<TactileSampleDetail>(
        "/api/tactile/samples/" + encodeURIComponent(sample.sample_id),
      ),
    );
  const deleteSample = (sample: TactileSample) => {
    if (!window.confirm(`删除触觉样本 ${sample.sample_id}？\n样本将移入 .trash。`)) return;
    void action(
      "tactile-sample-delete",
      async () => {
        await api("/api/tactile/samples/" + encodeURIComponent(sample.sample_id), {
          method: "DELETE",
          body: JSON.stringify({ confirm: true }),
        });
        setActiveSample(null);
        await refreshStatus();
        await refreshSamples();
      },
      "触觉样本已移入 .trash",
    );
  };

  return (
    <>
      <PanelHeader
        title="视觉类别条件下的触觉性质识别"
        description="视觉先确定具体种类；本页为每个视觉类别设计性质、采集触觉样本、训练并管理独立模型。采集不会移动机械臂。"
      />
      <div className="mt-4 max-h-[74vh] space-y-4 overflow-y-auto pr-1">
        <div className="grid gap-4 lg:grid-cols-[330px_minmax(0,1fr)]">
          <section className="space-y-4 rounded-lg border border-[#294550] bg-[#09151c] p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <BrainCircuit className="size-4 text-[#42e7bd]" />
              1 · 类别与性质标签
            </div>
            <label className="block text-xs text-[#8297a1]">
              视觉类别
              <select
                value={selectedClass}
                disabled={Boolean(capture?.running)}
                onChange={(event) => {
                  const nextClass = status?.classes.find(
                    (item) => item.name === event.target.value,
                  );
                  setSelectedClass(event.target.value);
                  setSelectedProperty(nextClass?.properties[0] ?? "");
                  setActiveSample(null);
                }}
                className="mt-1.5 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
              >
                {status?.classes.map((item) => (
                  <option key={item.name} value={item.name}>{item.label}</option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-[#8297a1]">
              性质
              <select
                value={selectedProperty}
                disabled={Boolean(capture?.running)}
                onChange={(event) => setSelectedProperty(event.target.value)}
                className="mt-1.5 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
              >
                {currentClass?.properties.map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
            <div className="flex gap-2">
              <input
                value={newProperty}
                onChange={(event) => setNewProperty(event.target.value)}
                placeholder="新增性质，如：中等"
                className="min-w-0 flex-1 rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-xs text-white"
              />
              <Button size="sm" onClick={addProperty} disabled={!newProperty.trim()}><Plus className="size-4" />添加</Button>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" disabled={!selectedProperty} onClick={renameProperty}>重命名</Button>
              <Button size="sm" variant="danger" disabled={!selectedProperty} onClick={deleteProperty}><Trash2 className="size-4" />删除</Button>
            </div>
            <div className="rounded-md border border-[#203740] bg-[#071116] p-3 text-[11px] leading-5 text-[#8297a1]">
              {currentClass
                ? Object.entries(currentClass.distribution).map(([label, count]) => <div key={label} className="flex justify-between"><span>{label}</span><span>{count} 条</span></div>)
                : "暂无类别"}
              {currentClass && Object.keys(currentClass.distribution).length === 0 && "尚未保存触觉样本"}
            </div>
          </section>

          <section className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium">2 · 单次夹取采集</h3>
                <p className="mt-1 text-xs text-[#708790]">把物体放入夹爪后开始；检测双侧接触后保持约 1 秒，随后自动松开。</p>
              </div>
              <span className={cn("text-xs", capture?.status === "ready" ? "text-[#42e7bd]" : capture?.status === "failed" ? "text-[#ff8e93]" : "text-[#f6b94a]")}>{capture?.current_step ?? "等待开始"}</span>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3">
              <StateBlock label="夹爪" value={devices.gripper.connected ? "已连接" : "未连接"} ok={devices.gripper.connected} />
              <StateBlock label="双侧触觉" value={devices.sensors.frame_ready ? "数据正常" : "未就绪"} ok={devices.sensors.frame_ready} />
              <StateBlock label="已采集帧" value={String(capture?.frame_count ?? 0)} ok={Boolean(capture?.frame_count)} />
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button disabled={!canCapture || busy === "tactile-capture"} onClick={() => void startCapture()}><Play className="size-4" />进行一次夹取并采集</Button>
              {capture?.running && <Button variant="danger" onClick={() => void action("tactile-stop", () => post("/api/tactile/capture/stop"), "已请求停止并松开夹爪")}><Square className="size-4" />停止</Button>}
              <Button disabled={capture?.status !== "ready"} onClick={() => void saveCapture()}><Save className="size-4" />保存标注样本</Button>
              <Button variant="outline" disabled={capture?.status !== "ready"} onClick={() => void discardCapture()}>丢弃</Button>
            </div>
            {capture?.curves && <TactileCurves curves={capture.curves} />}
            {capture?.error && <Warning>{capture.error}</Warning>}
          </section>
        </div>

        <section className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
          <div className="flex items-center justify-between gap-3">
            <div><h3 className="text-sm font-medium">3 · 数据管理</h3><p className="mt-1 text-xs text-[#708790]">{samples.length} 条当前类别样本；点击记录查看压力曲线。</p></div>
            <Button size="sm" variant="outline" onClick={() => void refreshSamples()}><RefreshCw className="size-4" />刷新</Button>
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
            <div className="max-h-64 overflow-auto rounded-md border border-[#203740]">
              <table className="w-full text-left text-[11px]">
                <thead className="sticky top-0 bg-[#0d1d24] text-[#708790]"><tr><th className="px-3 py-2">时间</th><th>性质</th><th>左压力峰值</th><th>右压力峰值</th><th /></tr></thead>
                <tbody>
                  {samples.map((sample) => (
                    <tr key={sample.sample_id} className="cursor-pointer border-t border-[#1c333d] text-[#a9bbc2] hover:bg-[#10232b]" onClick={() => void openSample(sample)}>
                      <td className="px-3 py-2 font-mono">{new Date(sample.captured_at).toLocaleString()}</td><td>{sample.property}</td><td>{sample.left_peak?.toFixed(1) ?? "—"}</td><td>{sample.right_peak?.toFixed(1) ?? "—"}</td>
                      <td><button type="button" className="p-2 text-[#ff8e93]" onClick={(event) => { event.stopPropagation(); deleteSample(sample); }}><Trash2 className="size-3.5" /></button></td>
                    </tr>
                  ))}
                  {!samples.length && <tr><td colSpan={5} className="px-3 py-8 text-center text-[#526a75]">暂无样本</td></tr>}
                </tbody>
              </table>
            </div>
            <div className="rounded-md border border-[#203740] bg-[#071116] p-3">
              {activeSample ? <><div className="text-xs text-[#91a4ac]">{activeSample.property} · {activeSample.sample_id}</div><TactileCurves curves={sampleCurves(activeSample)} /></> : <div className="py-12 text-center text-xs text-[#526a75]">选择一条记录查看曲线</div>}
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div><h3 className="text-sm font-medium">4 · 模型训练、评估与版本</h3><p className="mt-1 text-xs text-[#708790]">每个视觉类别独立训练性质分类器；至少两个性质、总计 6 条且每个性质至少 2 条。</p></div>
            <Button disabled={!selectedClass || status?.training.running} onClick={() => void train()}>{status?.training.running ? <RefreshCw className="size-4 animate-spin" /> : <BrainCircuit className="size-4" />}训练当前类别模型</Button>
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <div className="rounded-md border border-[#203740] bg-[#071116] p-4 text-xs leading-6 text-[#91a4ac]">
              <div className="flex justify-between"><span>活动模型</span><span className={model?.available ? "text-[#42e7bd]" : "text-[#f6b94a]"}>{model?.available ? model.version : "尚未训练"}</span></div>
              <div className="flex justify-between"><span>训练样本</span><span>{model?.sample_count ?? 0}</span></div>
              <div className="flex justify-between"><span>验证准确率</span><span>{model?.evaluation ? `${Math.round(model.evaluation.accuracy * 100)}%` : "—"}</span></div>
              {model?.stale && <div className="mt-2 text-[#f6b94a]">需要重训：{model.stale_reason}</div>}
              {status?.training.error && <div className="mt-2 text-[#ff8e93]">{status.training.error}</div>}
              {model?.evaluation && <ConfusionMatrix evaluation={model.evaluation} />}
            </div>
            <div className="max-h-64 overflow-auto rounded-md border border-[#203740] bg-[#071116] p-3">
              <div className="mb-2 text-xs text-[#708790]">历史版本</div>
              {model?.versions.map((version) => (
                <div key={version.version} className="flex items-center justify-between gap-3 border-t border-[#1c333d] py-2 text-[11px]">
                  <div><div className="font-mono text-[#a9bbc2]">{version.version}</div><div className="text-[#607680]">{version.sample_count} 条 · {version.accuracy == null ? "未评估" : `${Math.round(version.accuracy * 100)}%`}</div></div>
                  <Button size="sm" variant="outline" disabled={version.version === model.version} onClick={() => void action("tactile-restore", async () => { await post("/api/tactile/models/restore", { vision_class: selectedClass, version: version.version }); await refreshStatus(); }, "历史模型已恢复为活动版本")}><RotateCcw className="size-3.5" />恢复</Button>
                </div>
              ))}
              {!model?.versions.length && <div className="py-8 text-center text-xs text-[#526a75]">暂无模型版本</div>}
            </div>
          </div>
        </section>
      </div>
    </>
  );
}

function sampleCurves(sample: TactileSampleDetail) {
  return {
    time: sample.timestamps,
    left_sum: sample.frames.map((frame) => frame.slice(0, 32).reduce((sum, value) => sum + value, 0)),
    right_sum: sample.frames.map((frame) => frame.slice(32, 64).reduce((sum, value) => sum + value, 0)),
    left_peak: sample.frames.map((frame) => Math.max(...frame.slice(0, 32))),
    right_peak: sample.frames.map((frame) => Math.max(...frame.slice(32, 64))),
  };
}

function TactileCurves({ curves }: { curves: { time: number[]; left_sum: number[]; right_sum: number[]; left_peak: number[]; right_peak: number[] } }) {
  const maximum = Math.max(1, ...curves.left_peak, ...curves.right_peak);
  const duration = Math.max(0.001, ...curves.time);
  return (
    <div className="mt-3 space-y-3">
      {([
        ["左传感器", curves.left_peak, "#42e7bd"],
        ["右传感器", curves.right_peak, "#6ea8ff"],
      ] as const).map(([label, series, color]) => (
        <div key={label} className="rounded-md border border-[#203740] bg-[#050d11] p-3">
          <div className="mb-2 flex flex-wrap justify-between gap-2 text-xs">
            <span style={{ color }}>{label} · 单帧通道最大值</span>
            <span>峰值 {Math.max(0, ...series).toFixed(1)} · 显示量程 0–{maximum.toFixed(1)}（传感器读数）</span>
          </div>
          <div className="flex gap-2">
            <div className="flex w-14 shrink-0 flex-col justify-between text-right font-mono text-[10px] text-[#708790]">
              <span>{maximum.toFixed(1)}</span><span>{(maximum / 2).toFixed(1)}</span><span>0</span>
            </div>
            <svg viewBox="0 0 100 50" className="h-36 min-w-0 flex-1" preserveAspectRatio="none" role="img" aria-label={`${label}压力峰值曲线，量程0至${maximum.toFixed(1)}`}>
              {[0, 25, 50].map((y) => <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="#203740" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />)}
              <polyline points={series.map((value, index) => `${((curves.time[index] ?? 0) / duration) * 100},${50 - (value / maximum) * 50}`).join(" ")} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            </svg>
          </div>
          <div className="ml-16 mt-1 flex justify-between font-mono text-[10px] text-[#708790]"><span>0 s</span><span>{(duration / 2).toFixed(2)} s</span><span>{duration.toFixed(2)} s</span></div>
        </div>
      ))}
    </div>
  );
}

function ConfusionMatrix({ evaluation }: { evaluation: NonNullable<NonNullable<TactileStatus["classes"][number]["model"]["evaluation"]>> }) {
  return (
    <div className="mt-3 overflow-auto"><table className="w-full text-center text-[10px]"><thead><tr><th>真实＼预测</th>{evaluation.labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{evaluation.confusion_matrix.map((row, index) => <tr key={evaluation.labels[index]}><th>{evaluation.labels[index]}</th>{row.map((value, column) => <td key={column} className="border border-[#203740] p-1">{value}</td>)}</tr>)}</tbody></table><div className="mt-2 text-[#526a75]">{evaluation.note}</div></div>
  );
}

function VisionDataPanel({ devices, workspace, busy, action }: PanelContext) {
  const [dataset, setDataset] = useState<VisionDatasetStatus | null>(null);
  const [samples, setSamples] = useState<VisionDatasetSample[]>([]);
  const [trainingClasses, setTrainingClasses] = useState<TrainingClass[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [emptyTable, setEmptyTable] = useState(false);
  const [boxes, setBoxes] = useState<Array<[number, number, number, number]>>(
    [],
  );
  const [boxesDirty, setBoxesDirty] = useState(false);
  const [proposal, setProposal] = useState<DepthProposalResult | null>(null);
  const [proposalError, setProposalError] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState<[number, number] | null>(null);
  const [dragCurrent, setDragCurrent] = useState<[number, number] | null>(null);
  const [filter, setFilter] = useState<string | null>(null);
  const [sceneTags, setSceneTags] = useState<string[]>([]);
  const [newSceneLabel, setNewSceneLabel] = useState("");
  const [newClassLabel, setNewClassLabel] = useState("");
  const [activeSample, setActiveSample] = useState<VisionDatasetSample | null>(
    null,
  );
  const [sampleMetadata, setSampleMetadata] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [editTags, setEditTags] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    const query = filter
      ? "?tag=" + encodeURIComponent(filter) + "&limit=100"
      : "?limit=100";
    void Promise.all([
      api<VisionDatasetStatus>("/api/vision/dataset/status"),
      api<VisionDatasetList>("/api/vision/dataset/samples" + query),
      api<{ classes: TrainingClass[] }>("/api/vision/classes"),
    ])
      .then(([status, listing, classResult]) => {
        if (!active) return;
        setDataset(status);
        setSamples(listing.samples);
        setTrainingClasses(classResult.classes);
        setSelectedClassId(
          (current) => current ?? classResult.classes[0]?.class_id ?? null,
        );
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [filter]);

  useEffect(() => {
    let active = true;
    if (!devices.camera.frame_ready || !workspace.calibrated)
      return () => {
        active = false;
      };
    const update = () =>
      void api<DepthProposalResult>("/api/vision/auto-annotations")
        .then((result) => {
          if (!active) return;
          setProposal(result);
          setProposalError(null);
          if (!boxesDirty && !emptyTable) setBoxes(result.boxes);
        })
        .catch((error: unknown) => {
          if (!active) return;
          setProposalError(
            error instanceof Error ? error.message : "深度候选框检测失败",
          );
        });
    update();
    const timer = window.setInterval(update, 700);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [
    boxesDirty,
    devices.camera.frame_ready,
    emptyTable,
    workspace.calibrated,
  ]);

  const reload = async () => {
    const query = filter
      ? "?tag=" + encodeURIComponent(filter) + "&limit=100"
      : "?limit=100";
    const [status, listing] = await Promise.all([
      api<VisionDatasetStatus>("/api/vision/dataset/status"),
      api<VisionDatasetList>("/api/vision/dataset/samples" + query),
    ]);
    setDataset(status);
    setSamples(listing.samples);
    return listing.samples;
  };
  const selectedClass =
    trainingClasses.find((item) => item.class_id === selectedClassId) ?? null;
  const classLabels = trainingClasses.map((item) => item.label);
  const extraLabels =
    dataset?.labels.filter(
      (label) => !classLabels.includes(label) && label !== "空桌面",
    ) ?? [];
  const toggleSceneTag = (label: string) =>
    setSceneTags((current) =>
      current.includes(label)
        ? current.filter((item) => item !== label)
        : [...current, label],
    );
  const toggleEditTag = (label: string) =>
    setEditTags((current) =>
      current.includes(label)
        ? current.filter((item) => item !== label)
        : [...current, label],
    );
  const openSample = async (sample: VisionDatasetSample) => {
    setActiveSample(sample);
    setEditTags(sample.tags);
    setSampleMetadata(null);
    setSampleMetadata(
      await api<Record<string, unknown>>(
        "/api/vision/dataset/samples/" + encodeURIComponent(sample.sample_id),
      ),
    );
  };
  const addSceneLabel = () =>
    action(
      "vision-label-add",
      async () => {
        const result = await post<{ labels: string[] }>("/api/vision/labels", {
          label: newSceneLabel,
        });
        setDataset((current) =>
          current ? { ...current, labels: result.labels } : current,
        );
        const normalized = newSceneLabel.trim().replace(/\s+/g, " ");
        if (normalized)
          setSceneTags((current) =>
            current.includes(normalized) ? current : [...current, normalized],
          );
        setNewSceneLabel("");
      },
      "场景标签已添加",
    );
  const addTrainingClass = () =>
    action(
      "vision-class-add",
      async () => {
        const result = await post<{ classes: TrainingClass[] }>(
          "/api/vision/classes",
          { label: newClassLabel },
        );
        setTrainingClasses(result.classes);
        const normalized = newClassLabel.trim().replace(/\s+/g, " ");
        const added = result.classes.find((item) => item.label === normalized);
        if (added) {
          setSelectedClassId(added.class_id);
          setEmptyTable(false);
        }
        setNewClassLabel("");
      },
      "训练类别已添加，类别编号保持固定",
    );
  const capture = () =>
    action(
      "vision-capture",
      async () => {
        const result = await post<VisionCaptureResult>("/api/vision/capture", {
          class_id: emptyTable ? null : selectedClassId,
          empty_table: emptyTable,
          boxes: emptyTable ? [] : boxes,
          tags: sceneTags,
        });
        setDataset(result.dataset);
        setBoxesDirty(false);
        const updated = await reload();
        const captured = updated.find(
          (sample) => sample.sample_id === result.sample_id,
        );
        if (captured) await openSample(captured);
      },
      "彩色图、深度图、元数据和 YOLO 标注已保存",
    );
  const saveTags = () => {
    if (!activeSample) return Promise.resolve();
    return action(
      "vision-tags-save",
      async () => {
        const updated = await post<VisionDatasetSample>(
          "/api/vision/dataset/samples/" +
            encodeURIComponent(activeSample.sample_id) +
            "/tags",
          { tags: editTags },
        );
        setActiveSample(updated);
        setEditTags(updated.tags);
        await reload();
        setSampleMetadata(
          await api<Record<string, unknown>>(
            "/api/vision/dataset/samples/" +
              encodeURIComponent(updated.sample_id),
          ),
        );
      },
      "样本标签已更新",
    );
  };
  const deleteSample = () => {
    if (
      !activeSample ||
      !window.confirm(
        "确认删除样本 " +
          activeSample.sample_id +
          "？\n样本将移动到 .trash，可手动恢复。",
      )
    )
      return;
    const sampleId = activeSample.sample_id;
    void action(
      "vision-sample-delete",
      async () => {
        await api(
          "/api/vision/dataset/samples/" + encodeURIComponent(sampleId),
          { method: "DELETE", body: JSON.stringify({ confirm: true }) },
        );
        setActiveSample(null);
        setSampleMetadata(null);
        setEditTags([]);
        await reload();
      },
      "样本已移入 .trash",
    );
  };
  const imagePoint = (
    event: React.MouseEvent<SVGSVGElement>,
  ): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(
        0,
        Math.min(
          640,
          Math.round(((event.clientX - rect.left) / rect.width) * 640),
        ),
      ),
      Math.max(
        0,
        Math.min(
          480,
          Math.round(((event.clientY - rect.top) / rect.height) * 480),
        ),
      ),
    ];
  };
  const finishBox = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!dragStart) return;
    const end = imagePoint(event);
    const x = Math.min(dragStart[0], end[0]);
    const y = Math.min(dragStart[1], end[1]);
    const width = Math.abs(end[0] - dragStart[0]);
    const height = Math.abs(end[1] - dragStart[1]);
    if (width >= 8 && height >= 8) {
      setBoxes((current) => [...current, [x, y, width, height]]);
      setBoxesDirty(true);
    }
    setDragStart(null);
    setDragCurrent(null);
  };
  const resetBoxes = () => {
    setBoxes(proposal?.boxes ?? []);
    setBoxesDirty(false);
    setEmptyTable(false);
  };
  const annotationReady = emptyTable
    ? Boolean(proposal && proposal.boxes.length === 0)
    : Boolean(selectedClass && boxes.length > 0);
  const statusMessage = !workspace.calibrated
    ? "请先完成四点桌面工作区标定"
    : !devices.camera.frame_ready
      ? "请先启动 D435，等待彩色和对齐深度帧"
      : proposalError
        ? proposalError
        : emptyTable
          ? proposal?.boxes.length
            ? "检测到桌面上存在物体，不能保存空桌面负样本"
            : "空桌面校验通过，将保存无框负样本。"
          : boxes.length
            ? "当前保留 " + boxes.length + " 个框；点击框删除，拖动画面可补框。"
            : "没有自动候选框，可直接在画面拖拽补框。";

  return (
    <>
      <PanelHeader
        title="陌生物体训练数据采集"
        description="自动框只依据桌面平面与深度凸起，不要求系统事先认识物体；选择训练类别后可删除误框或拖拽补框。"
      />
      <div className="mt-4 max-h-[72vh] space-y-5 overflow-y-auto pr-1">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.45fr)_370px]">
          <section className="overflow-hidden rounded-lg border border-[#294550] bg-[#050d11]">
            <div className="flex items-center justify-between border-b border-[#203740] px-3 py-2">
              <span className="flex items-center gap-2 text-xs text-[#a9bbc2]">
                <Camera className="size-4 text-[#42e7bd]" />
                类别无关深度框
              </span>
              <span
                className={cn(
                  "text-[10px]",
                  annotationReady ? "text-[#42e7bd]" : "text-[#f6b94a]",
                )}
              >
                {annotationReady ? "可以保存" : "需要检查"}
              </span>
            </div>
            <div className="relative flex aspect-[4/3] items-center justify-center">
              {devices.camera.running ? (
                <>
                  <img
                    src={API_BASE + "/api/camera/stream"}
                    alt="D435 实时彩色画面"
                    className="h-full w-full object-fill"
                  />
                  <svg
                    className="absolute inset-0 h-full w-full cursor-crosshair"
                    viewBox="0 0 640 480"
                    preserveAspectRatio="none"
                    aria-label="可编辑自动标注框"
                    onMouseDown={(event) => {
                      const point = imagePoint(event);
                      setDragStart(point);
                      setDragCurrent(point);
                    }}
                    onMouseMove={(event) => {
                      if (dragStart) setDragCurrent(imagePoint(event));
                    }}
                    onMouseUp={finishBox}
                    onMouseLeave={(event) => {
                      if (dragStart) finishBox(event);
                    }}
                  >
                    {proposal?.roi_polygon_uv && (
                      <polygon
                        points={proposal.roi_polygon_uv
                          .map((point) => point.join(","))
                          .join(" ")}
                        fill="none"
                        stroke="#ff4c58"
                        strokeWidth="2"
                        strokeDasharray="7 5"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                    {boxes.map((box, index) => {
                      const [x, y, width, height] = box;
                      return (
                        <g
                          key={index}
                          className="cursor-pointer"
                          onMouseDown={(event) => event.stopPropagation()}
                          onClick={(event) => {
                            event.stopPropagation();
                            setBoxes((current) =>
                              current.filter(
                                (_, itemIndex) => itemIndex !== index,
                              ),
                            );
                            setBoxesDirty(true);
                          }}
                        >
                          <rect
                            x={x}
                            y={y}
                            width={width}
                            height={height}
                            fill="rgba(38,208,168,0.10)"
                            stroke="#26d0a8"
                            strokeWidth="2.5"
                            vectorEffect="non-scaling-stroke"
                          />
                          <rect
                            x={x}
                            y={Math.max(0, y - 20)}
                            width="100"
                            height="20"
                            fill="#126755"
                          />
                          <text
                            x={x + 4}
                            y={Math.max(14, y - 6)}
                            fill="white"
                            fontSize="12"
                          >
                            {selectedClass?.label ?? "未选类别"} {index + 1}
                          </text>
                        </g>
                      );
                    })}
                    {dragStart && dragCurrent && (
                      <rect
                        x={Math.min(dragStart[0], dragCurrent[0])}
                        y={Math.min(dragStart[1], dragCurrent[1])}
                        width={Math.abs(dragCurrent[0] - dragStart[0])}
                        height={Math.abs(dragCurrent[1] - dragStart[1])}
                        fill="rgba(246,185,74,0.10)"
                        stroke="#f6b94a"
                        strokeWidth="2"
                        strokeDasharray="6 4"
                      />
                    )}
                  </svg>
                </>
              ) : (
                <div className="text-center text-xs text-[#607680]">
                  <Camera className="mx-auto mb-3 size-8" />
                  <p>相机尚未启动</p>
                  <Button
                    className="mt-3"
                    size="sm"
                    onClick={() =>
                      void action(
                        "camera-start",
                        () => post("/api/camera/start"),
                        "D435 已启动",
                      )
                    }
                  >
                    <Play className="size-3.5" />
                    启动相机
                  </Button>
                </div>
              )}
            </div>
            <div className="flex items-center justify-between border-t border-[#203740] px-3 py-2 text-[10px] text-[#708790]">
              <span>点击已有框删除 · 在空白位置拖拽补框</span>
              <Button size="sm" variant="outline" onClick={resetBoxes}>
                <RefreshCw className="size-3" />
                恢复自动框
              </Button>
            </div>
          </section>
          <section className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
            <div className="grid grid-cols-3 gap-2">
              <StateBlock
                label="工作区"
                value={workspace.calibrated ? "已标定" : "未标定"}
                ok={workspace.calibrated}
              />
              <StateBlock
                label="当前框"
                value={String(boxes.length)}
                ok={annotationReady}
              />
              <StateBlock
                label="样本"
                value={String(dataset?.sample_count ?? 0)}
                ok={Boolean(dataset?.sample_count)}
              />
            </div>
            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-[#91a4ac]">训练类别（单选）</span>
              <button
                type="button"
                onClick={() => {
                  setEmptyTable(!emptyTable);
                  if (!emptyTable) {
                    setBoxes([]);
                    setBoxesDirty(true);
                  } else {
                    setBoxesDirty(false);
                  }
                }}
                className={cn(
                  "rounded border px-2 py-1 text-[10px]",
                  emptyTable
                    ? "border-[#2d7769] bg-[#163b34] text-[#66efc7]"
                    : "border-[#294550] text-[#8297a1]",
                )}
              >
                空桌面负样本
              </button>
            </div>
            <select
              aria-label="训练类别"
              disabled={emptyTable}
              value={selectedClassId ?? ""}
              onChange={(event) => {
                setSelectedClassId(Number(event.target.value));
                setEmptyTable(false);
              }}
              className="mt-2 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
            >
              {trainingClasses.map((item) => (
                <option key={item.class_id} value={item.class_id}>
                  {item.class_id} · {item.label}
                </option>
              ))}
            </select>
            <div className="mt-2 flex gap-2">
              <input
                aria-label="新增训练类别"
                value={newClassLabel}
                maxLength={32}
                onChange={(event) => setNewClassLabel(event.target.value)}
                placeholder="新增陌生物体类别"
                className="h-9 min-w-0 flex-1 rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-xs text-white"
              />
              <Button
                size="sm"
                variant="outline"
                disabled={!newClassLabel.trim() || busy === "vision-class-add"}
                onClick={() => void addTrainingClass()}
              >
                <Plus className="size-3.5" />
                添加类别
              </Button>
            </div>
            <div className="mt-4 text-xs text-[#91a4ac]">
              补充场景标签（可选）
            </div>
            <div className="mt-2 flex max-h-16 flex-wrap gap-2 overflow-y-auto">
              {extraLabels.length ? (
                extraLabels.map((label) => (
                  <TagButton
                    key={label}
                    label={label}
                    active={sceneTags.includes(label)}
                    onClick={() => toggleSceneTag(label)}
                  />
                ))
              ) : (
                <span className="text-[10px] text-[#607680]">
                  暂无自定义场景标签
                </span>
              )}
            </div>
            <div className="mt-2 flex gap-2">
              <input
                aria-label="新增场景标签"
                value={newSceneLabel}
                maxLength={32}
                onChange={(event) => setNewSceneLabel(event.target.value)}
                placeholder="例如：强反光"
                className="h-9 min-w-0 flex-1 rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-xs text-white"
              />
              <Button
                size="sm"
                variant="outline"
                disabled={!newSceneLabel.trim() || busy === "vision-label-add"}
                onClick={() => void addSceneLabel()}
              >
                <Plus className="size-3.5" />
                添加标签
              </Button>
            </div>
            <div
              className={cn(
                "mt-3 rounded-md border p-2.5 text-[11px] leading-4",
                annotationReady
                  ? "border-[#26715e] bg-[#102b24] text-[#66efc7]"
                  : "border-[#624a25] bg-[#251d10] text-[#e6c98c]",
              )}
            >
              {statusMessage}
              {proposal?.warnings.map((warning) => (
                <div key={warning} className="mt-1 text-[#f6b94a]">
                  {warning}
                </div>
              ))}
            </div>
            <Button
              className="mt-3 w-full"
              disabled={
                !devices.camera.frame_ready ||
                !workspace.calibrated ||
                !annotationReady ||
                busy === "vision-capture"
              }
              onClick={() => void capture()}
            >
              <ImagePlus className="size-4" />
              确认框选并保存
            </Button>
          </section>
        </div>
        <section className="border-t border-[#294550] pt-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-medium">历史样本库</h3>
              <p className="mt-1 text-[11px] text-[#607680]">
                点击缩略图查看彩色图、深度图、自动标注和完整元数据。
              </p>
            </div>
            <Button size="sm" variant="outline" onClick={() => void reload()}>
              <RefreshCw className="size-3.5" />
              刷新
            </Button>
          </div>
          <div className="mb-3 flex flex-wrap gap-2">
            <TagButton
              label="全部"
              active={filter === null}
              onClick={() => setFilter(null)}
            />
            {dataset?.labels.map((label) => (
              <TagButton
                key={label}
                label={label}
                active={filter === label}
                onClick={() => setFilter(label)}
              />
            ))}
          </div>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_390px]">
            <div className="grid max-h-[430px] grid-cols-2 gap-3 overflow-y-auto pr-1 sm:grid-cols-3">
              {samples.length ? (
                samples.map((sample) => (
                  <button
                    type="button"
                    key={sample.sample_id}
                    onClick={() => void openSample(sample)}
                    className={cn(
                      "overflow-hidden rounded-lg border bg-[#071116] text-left transition-colors",
                      activeSample?.sample_id === sample.sample_id
                        ? "border-[#34b99d] ring-1 ring-[#34b99d]"
                        : "border-[#294550] hover:border-[#47717f]",
                    )}
                  >
                    <div className="relative aspect-video bg-black">
                      <img
                        src={API_BASE + sample.color_url}
                        alt={"样本 " + sample.sample_id}
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                      {sample.auto_annotation && (
                        <span className="absolute right-1.5 top-1.5 rounded bg-[#126755]/90 px-1.5 py-0.5 text-[9px] text-white">
                          {sample.auto_annotation.annotations.length} 框
                        </span>
                      )}
                    </div>
                    <div className="p-2">
                      <div className="truncate font-mono text-[10px] text-[#42e7bd]">
                        {sample.sample_id}
                      </div>
                      <div className="mt-1 truncate text-[10px] text-[#708790]">
                        {sample.tags.join(" · ")}
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <div className="col-span-full rounded-lg border border-dashed border-[#294550] py-12 text-center text-xs text-[#607680]">
                  {filter ? "没有匹配此标签的样本" : "尚未采集数据"}
                </div>
              )}
            </div>
            <aside className="min-h-64 rounded-lg border border-[#294550] bg-[#09151c] p-3">
              {activeSample ? (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <PreviewImage
                      title="彩色图"
                      src={API_BASE + activeSample.color_url}
                    />
                    <PreviewImage
                      title="深度预览"
                      src={API_BASE + activeSample.depth_preview_url}
                    />
                  </div>
                  <div className="mt-3 flex items-center justify-between gap-2">
                    <span className="font-mono text-[11px] text-[#42e7bd]">
                      {activeSample.sample_id}
                    </span>
                    {activeSample.auto_annotation && (
                      <span className="rounded bg-[#163b34] px-2 py-1 text-[9px] text-[#66efc7]">
                        YOLO {activeSample.auto_annotation.annotations.length}{" "}
                        框
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[10px] text-[#607680]">
                    {new Date(activeSample.captured_at).toLocaleString()}
                  </div>
                  <div className="mt-4 text-xs text-[#91a4ac]">
                    编辑样本标签
                  </div>
                  <div className="mt-2 flex max-h-24 flex-wrap gap-2 overflow-y-auto">
                    {dataset?.labels.map((label) => (
                      <TagButton
                        key={label}
                        label={label}
                        active={editTags.includes(label)}
                        onClick={() => toggleEditTag(label)}
                      />
                    ))}
                  </div>
                  <div className="mt-3 flex gap-2">
                    <Button
                      size="sm"
                      disabled={
                        editTags.length === 0 || busy === "vision-tags-save"
                      }
                      onClick={() => void saveTags()}
                    >
                      <Save className="size-3.5" />
                      保存标签
                    </Button>
                    <Button
                      size="sm"
                      variant="danger"
                      disabled={busy === "vision-sample-delete"}
                      onClick={deleteSample}
                    >
                      <Trash2 className="size-3.5" />
                      删除
                    </Button>
                  </div>
                  <details className="mt-4 border-t border-[#294550] pt-3">
                    <summary className="cursor-pointer text-[11px] text-[#8297a1]">
                      查看完整元数据与自动标注
                    </summary>
                    <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-[#050d11] p-2 font-mono text-[9px] leading-4 text-[#708790]">
                      {sampleMetadata
                        ? JSON.stringify(sampleMetadata, null, 2)
                        : "正在读取…"}
                    </pre>
                  </details>
                </>
              ) : (
                <div className="flex h-full min-h-64 items-center justify-center text-center text-xs leading-5 text-[#607680]">
                  从左侧选择一个样本
                  <br />
                  即可查看和管理
                </div>
              )}
            </aside>
          </div>
          <p className="mt-3 break-all font-mono text-[9px] text-[#52666f]">
            数据目录：{dataset?.root ?? "—"}
          </p>
        </section>
      </div>
    </>
  );
}

function TagButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 rounded-md border px-2.5 py-1.5 text-[11px] transition-colors",
        active
          ? "border-[#2d7769] bg-[#163b34] text-[#66efc7]"
          : "border-[#294550] bg-[#071116] text-[#8297a1] hover:text-white",
      )}
    >
      {active && <Check className="size-3" />}
      {label}
    </button>
  );
}

function PreviewImage({ title, src }: { title: string; src: string }) {
  return (
    <div>
      <div className="mb-1 text-[10px] text-[#708790]">{title}</div>
      <div className="aspect-video overflow-hidden rounded border border-[#294550] bg-black">
        <img src={src} alt={title} className="h-full w-full object-contain" />
      </div>
    </div>
  );
}

function DevicesPanel({ devices }: PanelContext) {
  return (
    <>
      <PanelHeader
        title="设备状态"
        description="统一查看相机、机械臂、夹爪、压力传感器和可用串口。"
      />
      <div className="grid grid-cols-4 gap-3">
        <DeviceTile
          icon={Camera}
          title="RealSense D435"
          state={devices.camera.frame_ready ? "视频流正常" : "未就绪"}
          ok={devices.camera.frame_ready}
        />
        <DeviceTile
          icon={Bot}
          title="RM65"
          state={devices.robot.connected ? "控制连接正常" : "未连接"}
          ok={Boolean(devices.robot.connected)}
        />
        <DeviceTile
          icon={Hand}
          title="DH5"
          state={devices.gripper.connected ? devices.gripper.port : "未连接"}
          ok={devices.gripper.connected}
        />
        <DeviceTile
          icon={RadioTower}
          title="左右压力"
          state={devices.sensors.frame_ready ? "双侧数据正常" : "未就绪"}
          ok={devices.sensors.frame_ready}
        />
      </div>
      <h3 className="mb-2 mt-6 text-sm font-medium">检测到的串口</h3>
      <div className="max-h-52 overflow-auto rounded-lg border border-[#294550]">
        {devices.serial_ports.length ? (
          devices.serial_ports.map((item) => (
            <div
              key={item.device}
              className="flex justify-between border-b border-[#203740] px-4 py-3 text-sm last:border-0"
            >
              <span className="font-mono text-[#42e7bd]">{item.device}</span>
              <span className="text-[#8297a1]">{item.description}</span>
            </div>
          ))
        ) : (
          <div className="p-4 text-sm text-[#708790]">未检测到串口设备</div>
        )}
      </div>
    </>
  );
}

function LogsPanel({ logs }: PanelContext) {
  return (
    <>
      <PanelHeader
        title="运行日志"
        description="来自相机、机械臂、夹爪、标定和安全控制的最近事件。"
      />
      <div className="max-h-[60vh] overflow-auto rounded-lg border border-[#294550] bg-[#071116] font-mono text-xs">
        {logs.length ? (
          [...logs].reverse().map((event) => (
            <div
              key={event.id}
              className="grid grid-cols-[92px_80px_90px_1fr] gap-3 border-b border-[#172c35] px-3 py-2.5 last:border-0"
            >
              <span className="text-[#607680]">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
              <span
                className={cn(
                  event.level === "error"
                    ? "text-[#ff7777]"
                    : event.level === "warning"
                      ? "text-[#f6b94a]"
                      : "text-[#42e7bd]",
                )}
              >
                {event.level.toUpperCase()}
              </span>
              <span className="text-[#8297a1]">{event.source}</span>
              <span className="text-[#c6d3d8]">{event.message}</span>
            </div>
          ))
        ) : (
          <div className="p-6 text-center text-[#607680]">暂无日志</div>
        )}
      </div>
    </>
  );
}

function SettingsPanel({ devices, action }: PanelContext) {
  const [settings, setSettings] = useState<TargetingSettings | null>(null);
  const [transfer, setTransfer] = useState<TransferSettings | null>(null);
  const [markerX, setMarkerX] = useState(0);
  const [markerY, setMarkerY] = useState(0);
  const [markerZ, setMarkerZ] = useState(400);
  useEffect(() => {
    void Promise.all([
      api<TargetingSettings>("/api/targeting/settings"),
      api<TransferSettings>("/api/transfer/settings"),
    ])
      .then(([targeting, moving]) => {
        setSettings(targeting);
        setMarkerX(targeting.marker_to_tip_xyz_mm[0]);
        setMarkerY(targeting.marker_to_tip_xyz_mm[1]);
        setMarkerZ(targeting.marker_to_tip_xyz_mm[2]);
        setTransfer(moving);
      })
      .catch(() => undefined);
  }, []);
  const offsetFields = [
    ["X", markerX, setMarkerX, -500, 500],
    ["Y", markerY, setMarkerY, -500, 500],
    ["Z", markerZ, setMarkerZ, 0, 800],
  ] as const;
  const transferFields: Array<
    [keyof TransferSettings, string, number, number, string]
  > = [
    ["transit_speed", "水平搬运速度", 1, 100, "RM65 速度单位"],
    ["approach_speed", "垂直升降速度", 1, 100, "RM65 速度单位"],
    ["grasp_clearance_mm", "抓取高度微调", -100, 100, "mm；正值抬高，负值降低"],
    ["pick_lift_mm", "抓取成功后抬升", 20, 300, "mm"],
    ["place_approach_mm", "托盘上方距离", 20, 300, "mm"],
    ["place_clearance_mm", "松开放置间隙", 0, 150, "mm"],
  ];
  return (
    <>
      <PanelHeader
        title="系统与目标设置"
        description="目标补偿、安全限制和双侧传感器处理参数统一管理。"
      />
      <div className="space-y-4">
        <div className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
          <div className="mb-3 text-xs text-[#91a4ac]">
            固定物理距离：蓝点到夹爪尖端 / mm（Base坐标方向）
          </div>
          <div className="grid grid-cols-3 gap-3">
            {offsetFields.map(([axis, value, setter, minimum, maximum]) => (
              <label key={axis} className="text-xs text-[#708790]">
                Δ{axis}
                <input
                  aria-label={`蓝点到夹爪尖端${axis}`}
                  type="number"
                  min={minimum}
                  max={maximum}
                  value={value}
                  onChange={(event) => setter(Number(event.target.value))}
                  className="mt-1.5 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 py-2 font-mono text-sm text-white"
                />
              </label>
            ))}
          </div>
          <Button
            className="mt-4"
            onClick={() =>
              void action(
                "target-settings",
                async () =>
                  setSettings(
                    await post<TargetingSettings>("/api/targeting/settings", {
                      marker_to_tip_x_mm: markerX,
                      marker_to_tip_y_mm: markerY,
                      marker_to_tip_z_mm: markerZ,
                    }),
                  ),
                "目标补偿设置已保存",
              )
            }
          >
            保存目标设置
          </Button>
        </div>
        {transfer && (
          <div className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
            <div className="mb-1 text-xs text-[#91a4ac]">
              自动搬运速度与高度
            </div>
            <p className="mb-4 text-[11px] leading-5 text-[#708790]">
              抓取高度微调独立于蓝点到夹爪尖端的固定补偿，只沿 Base Z 生效。
            </p>
            <div className="grid grid-cols-2 gap-3">
              {transferFields.map(([key, label, minimum, maximum, unit]) => (
                <label key={key} className="text-xs text-[#708790]">
                  {label}
                  <input
                    aria-label={label}
                    type="number"
                    min={minimum}
                    max={maximum}
                    value={transfer[key]}
                    onChange={(event) =>
                      setTransfer({
                        ...transfer,
                        [key]: Number(event.target.value),
                      })
                    }
                    className="mt-1.5 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 py-2 font-mono text-sm text-white"
                  />
                  <span className="mt-1 block text-[10px] text-[#526a75]">
                    {unit}
                  </span>
                </label>
              ))}
            </div>
            <Button
              className="mt-4"
              onClick={() =>
                void action(
                  "transfer-settings",
                  async () =>
                    setTransfer(
                      await post<TransferSettings>(
                        "/api/transfer/settings",
                        transfer,
                      ),
                    ),
                  "自动搬运设置已保存；请重新生成预览",
                )
              }
            >
              保存自动搬运设置
            </Button>
          </div>
        )}
        <SettingRow title="后端监听" value="127.0.0.1:8000" />
        <SettingRow
          title="运动配置锁"
          value={
            devices.robot.motion_configured ? "允许真机运动" : "禁止真机运动"
          }
        />
        <SettingRow
          title="运动速度上限"
          value={String(settings?.maximum_speed ?? 100)}
        />
      </div>
      <SensorSettingsEditor action={action} />
      <Warning>
        应用不再限制单段移动距离，运动范围与碰撞限制由 RM65
        控制服务和机械臂自身负责。保存自动搬运设置会让旧预览失效。
      </Warning>
    </>
  );
}

function SensorSettingsEditor({ action }: Pick<PanelContext, "action">) {
  const [settings, setSettings] = useState<SensorSettings | null>(null);
  const [side, setSide] = useState<"left" | "right">("left");
  const [group, setGroup] = useState("显示");
  const [draft, setDraft] = useState<Record<string, number | boolean>>({});
  useEffect(() => {
    void api<SensorSettings>("/api/sensors/settings")
      .then((loaded) => {
        setSettings(loaded);
        setDraft(sensorDraft(loaded, "left"));
      })
      .catch(() => undefined);
  }, []);
  if (!settings)
    return (
      <div className="mt-5 text-xs text-[#708790]">正在读取传感器设置…</div>
    );
  const groups = Array.from(
    new Set(settings.schema.map((field) => field.group)),
  );
  const fields = settings.schema.filter((field) => field.group === group);
  const selectSide = (nextSide: "left" | "right") => {
    setSide(nextSide);
    setDraft(sensorDraft(settings, nextSide));
  };
  const save = () =>
    action(
      "sensor-settings",
      async () => {
        const updated = await post<SensorSettings>("/api/sensors/settings", {
          side,
          values: draft,
        });
        setSettings(updated);
        setDraft(sensorDraft(updated, side));
      },
      `${side === "left" ? "左" : "右"}传感器参数已保存并应用`,
    );
  return (
    <div className="mt-5 rounded-lg border border-[#294550] bg-[#09151c] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium">压力传感器与夹取判断参数</h3>
          <p className="mt-1 text-xs text-[#708790]">
            字段、范围和当前值来自 arm_gripping
            设置页；左右侧独立，夹爪动作参数自动保持一致。
          </p>
        </div>
        <div className="flex gap-2">
          {(["left", "right"] as const).map((item) => (
            <Button
              key={item}
              size="sm"
              variant={side === item ? "secondary" : "outline"}
              onClick={() => selectSide(item)}
            >
              {item === "left" ? "左传感器" : "右传感器"}
            </Button>
          ))}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2 border-b border-[#294550] pb-3">
        {groups.map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setGroup(item)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs",
              group === item
                ? "bg-[#1d6558] text-white"
                : "bg-[#0a171d] text-[#8297a1]",
            )}
          >
            {item}
          </button>
        ))}
      </div>
      <div className="mt-4 grid max-h-[42vh] grid-cols-2 gap-3 overflow-auto pr-1">
        {fields.map((field) => (
          <label
            key={field.path}
            className="rounded-md border border-[#203740] bg-[#071116] p-3 text-xs text-[#91a4ac]"
          >
            <span className="block min-h-8 leading-4">{field.label}</span>
            {field.kind === "bool" ? (
              <input
                type="checkbox"
                checked={Boolean(draft[field.path])}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    [field.path]: event.target.checked,
                  }))
                }
                className="mt-2 size-4 accent-[#26d0a8]"
              />
            ) : (
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.kind === "int" ? 1 : 0.1}
                value={Number(draft[field.path] ?? 0)}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    [field.path]: Number(event.target.value),
                  }))
                }
                className="mt-2 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-2 py-2 font-mono text-sm text-white"
              />
            )}
            <span className="mt-1 block font-mono text-[9px] text-[#52666f]">
              {field.path}
            </span>
          </label>
        ))}
      </div>
      <Button className="mt-4" onClick={() => void save()}>
        应用并保存
      </Button>
    </div>
  );
}

function sensorDraft(
  settings: SensorSettings,
  side: "left" | "right",
): Record<string, number | boolean> {
  const result: Record<string, number | boolean> = {};
  for (const field of settings.schema)
    result[field.path] = nestedValue(settings.values[side], field.path) as
      | number
      | boolean;
  return result;
}

function nestedValue(root: Record<string, unknown>, path: string): unknown {
  let value: unknown = root;
  for (const key of path.split("."))
    value =
      typeof value === "object" && value !== null
        ? (value as Record<string, unknown>)[key]
        : undefined;
  return value;
}

function PortSelect({
  label,
  value,
  onChange,
  ports,
  fallback,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  ports: Array<{ device: string; description: string }>;
  fallback: string;
}) {
  return (
    <label className="text-xs text-[#8297a1]">
      {label}
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1.5 h-10 w-full rounded-md border border-[#2a4652] bg-[#08141a] px-3 text-sm text-white"
      >
        <option value={value}>{value || fallback}</option>
        {ports
          .filter((item) => item.device !== value)
          .map((item) => (
            <option key={item.device} value={item.device}>
              {item.device} · {item.description}
            </option>
          ))}
      </select>
    </label>
  );
}

function SensorHeatmap({
  title,
  values,
  fps,
  features,
  raw,
  processed,
}: {
  title: string;
  values: number[] | null;
  fps: number;
  features?: NonNullable<SensorData["features"]>["left"];
  raw?: number[];
  processed?: number[];
}) {
  const scale = Math.max(1, features?.display_threshold ?? 100);
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm">
          <RadioTower className="size-4 text-[#42e7bd]" />
          {title}
        </span>
        <span className="font-mono text-[11px] text-[#708790]">
          {fps.toFixed(0)} FPS
        </span>
      </div>
      <div className="grid grid-cols-4 gap-1 rounded-lg border border-[#294550] bg-[#071116] p-2">
        {Array.from({ length: 32 }, (_, index) => {
          const value = values?.[index] ?? 0;
          const intensity = Math.min(1, Math.log1p(value) / Math.log1p(scale));
          return (
            <div
              key={index}
              title={`CH${index + 1} 原始 ${raw?.[index]?.toFixed(1) ?? "—"} · 处理 ${processed?.[index]?.toFixed(1) ?? "—"} · 显示 ${value.toFixed(1)}`}
              className="flex h-12 flex-col items-center justify-center rounded-sm border border-white/5 font-mono text-xs text-white"
              style={{
                backgroundColor: `rgba(${Math.round(38 + 217 * intensity)}, ${Math.round(85 + 88 * intensity)}, ${Math.round(110 - 70 * intensity)}, ${0.16 + intensity * 0.84})`,
              }}
            ><span className="text-[9px] text-[#91a4ac]">CH{index + 1}</span><span>{processed?.[index]?.toFixed(1) ?? "—"}</span></div>
          );
        })}
      </div>
      <div className="mt-2 text-xs leading-5 text-[#91a4ac]">
        <div>颜色量程 0–{scale.toFixed(1)} · 数字为处理后读数（未换算 N/kPa）</div>
        <div>原始峰值 {raw ? Math.max(...raw).toFixed(1) : "—"} · 处理后峰值 {features?.max.toFixed(1) ?? "—"}</div>
        <div>基线{features?.baseline_enabled === false ? "未启用" : features?.baseline_ready ? "就绪" : "初始化中"} · 基线均值 {features?.baseline_mean.toFixed(1) ?? "—"} · 噪声均值 {features?.noise_mean.toFixed(1) ?? "—"}</div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[10px] text-[#8297a1]">
        <span>Σ {features?.sum.toFixed(0) ?? "—"}</span>
        <span>MAX {features?.max.toFixed(0) ?? "—"}</span>
        <span>NZ {features?.nonzero ?? "—"}</span>
      </div>
    </div>
  );
}

function PanelHeader({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <DialogHeader>
      <DialogTitle>{title}</DialogTitle>
      <DialogDescription>{description}</DialogDescription>
    </DialogHeader>
  );
}
function StateBlock({
  label,
  value,
  ok,
}: {
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
      <div className="text-xs text-[#708790]">{label}</div>
      <div className="mt-2 flex items-center gap-2 text-sm">
        <span
          className={cn(
            "size-2 rounded-full",
            ok ? "bg-[#26d0a8]" : "bg-[#5b6e76]",
          )}
        />
        {value}
      </div>
    </div>
  );
}
function Warning({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-4 flex gap-2 rounded-lg border border-[#624a25] bg-[#251d10] p-3 text-xs leading-5 text-[#e6c98c]">
      <ShieldAlert className="mt-0.5 size-4 shrink-0 text-[#f6b94a]" />
      {children}
    </div>
  );
}
function Range({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-2 flex justify-between text-xs">
        <span className="text-[#91a4ac]">{label}</span>
        <span className="font-mono text-[#42e7bd]">{value}</span>
      </div>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-[#26d0a8]"
      />
    </label>
  );
}
function DeviceTile({
  icon: Icon,
  title,
  state,
  ok,
}: {
  icon: typeof Activity;
  title: string;
  state: string;
  ok: boolean;
}) {
  return (
    <div className="rounded-lg border border-[#294550] bg-[#09151c] p-4">
      <Icon
        className={cn("size-6", ok ? "text-[#42e7bd]" : "text-[#607680]")}
      />
      <div className="mt-3 text-sm font-medium">{title}</div>
      <div className="mt-1 text-xs text-[#708790]">{state}</div>
    </div>
  );
}
function SettingRow({ title, value }: { title: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-[#294550] bg-[#09151c] px-4 py-3">
      <span className="text-sm text-[#aabac0]">{title}</span>
      <span className="font-mono text-xs text-[#42e7bd]">{value}</span>
    </div>
  );
}
