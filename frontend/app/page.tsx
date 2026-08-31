import type { Metadata } from "next";
import { ControlDashboard } from "@/components/control-dashboard";

export const metadata: Metadata = {
  title: "VTLA_物体抓取任务",
  description: "RealSense D435、RM65 与 DH5 夹爪本地控制台",
};

export default function Home() {
  return <ControlDashboard />;
}
