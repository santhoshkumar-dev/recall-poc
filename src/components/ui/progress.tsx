import { cn } from "@/lib/cn";

export function Progress({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-2 overflow-hidden rounded-full bg-black/10", className)} role="progressbar" aria-valuenow={value}>
      <div className="h-full rounded-full bg-moss transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}
