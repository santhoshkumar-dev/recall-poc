import { cn } from "@/lib/cn";

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "warn" | "bad" }) {
  return (
    <span className={cn(
      "inline-flex rounded-full px-2.5 py-1 text-xs font-semibold",
      tone === "neutral" && "bg-black/5 text-black/60",
      tone === "good" && "bg-emerald-100 text-emerald-800",
      tone === "warn" && "bg-amber-100 text-amber-800",
      tone === "bad" && "bg-red-100 text-red-800",
    )}>{children}</span>
  );
}
