import * as React from "react";
import { cn } from "@/lib/cn";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
};

export function Button({ className, variant = "primary", size = "md", ...props }: Props) {
  return (
    <button
      className={cn(
        "focus-ring inline-flex items-center justify-center gap-2 rounded-full font-semibold transition disabled:cursor-not-allowed disabled:opacity-45",
        variant === "primary" && "bg-ink text-white hover:bg-black/80",
        variant === "secondary" && "border border-black/10 bg-white hover:bg-black/5",
        variant === "ghost" && "hover:bg-black/5",
        variant === "danger" && "bg-red-50 text-red-700 hover:bg-red-100",
        size === "sm" && "h-9 px-4 text-sm",
        size === "md" && "h-11 px-5 text-sm",
        size === "lg" && "h-14 px-7 text-base",
        className,
      )}
      {...props}
    />
  );
}
