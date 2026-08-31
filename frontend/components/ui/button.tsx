import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const variants = cva(
  "inline-flex items-center justify-center gap-2 rounded-md border text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#26d0a8]/60",
  {
    variants: {
      variant: {
        default: "border-[#32caa8] bg-[#26d0a8] text-[#04110e] hover:bg-[#42e7bd]",
        secondary: "border-[#2b4652] bg-[#142731] text-[#dce8ed] hover:bg-[#1a3440]",
        ghost: "border-transparent bg-transparent text-[#9fb2ba] hover:bg-[#162a34] hover:text-white",
        danger: "border-[#f26b6b] bg-[#d93f45] text-white hover:bg-[#ed5358]",
        outline: "border-[#35515d] bg-transparent text-[#dbe8ec] hover:border-[#4d727f] hover:bg-[#12232d]",
      },
      size: { default: "h-10 px-4", sm: "h-8 px-3 text-xs", icon: "size-10 p-0" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export function Button({ className, variant, size, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof variants>) {
  return <button className={cn(variants({ variant, size }), className)} {...props} />;
}
