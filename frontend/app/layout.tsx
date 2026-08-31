import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VTLA_物体抓取任务",
  description: "本地视觉抓取与设备控制工作台",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
