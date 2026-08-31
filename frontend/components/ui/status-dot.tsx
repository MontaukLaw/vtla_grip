import { cn } from "@/lib/utils";

export function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return <span className="inline-flex items-center gap-2 rounded-full border border-[#294550] bg-[#0d1c24] px-3 py-1.5 text-xs text-[#b9c8ce]"><span className={cn("size-2 rounded-full", ok ? "bg-[#26d0a8] shadow-[0_0_10px_#26d0a8]" : "bg-[#5d7078]")} />{label}</span>;
}
