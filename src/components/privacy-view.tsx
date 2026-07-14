"use client";

import { Check, Download, HardDrive, LockKeyhole, ShieldCheck, WifiOff } from "lucide-react";
import { recallApi } from "@/lib/tauri";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

export function PrivacyView({ onRefresh }: { onRefresh: () => Promise<void> }) {
  const { model, bootstrap } = useRecallStore();
  const install = async () => { await recallApi.installModels(); await onRefresh(); };
  const facts = [
    [ShieldCheck, "Processing location", "On this device"], [LockKeyhole, "Remote AI requests", "0"], [HardDrive, "Files uploaded", "0"], [WifiOff, "Internet needed for search", model?.offlineReady ? "No" : "After setup"],
  ] as const;
  return (
    <div><p className="eyebrow">Trust by architecture</p><h1 className="mt-2 text-4xl font-semibold tracking-tight">Privacy & models</h1><p className="mt-4 max-w-2xl text-base leading-7 text-black/50">Recall has no account, cloud backend, telemetry pipeline, or remote inference endpoint. It only accesses folders you explicitly approve.</p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2">{facts.map(([Icon, label, value]) => <div className="panel p-6" key={label}><Icon className="text-moss" /><p className="mt-5 text-sm text-black/45">{label}</p><p className="mt-1 text-2xl font-semibold">{value}</p></div>)}</div>
      <section className="panel mt-8 p-6"><div className="flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">Local model</p><h2 className="mt-2 text-xl font-semibold">{model?.embeddingModel ?? "all-MiniLM-L6-v2"}</h2><p className="mt-2 text-sm text-black/45">English OCR + 384-dimensional text embeddings</p></div><Badge tone={model?.offlineReady ? "good" : "warn"}>{model?.offlineReady ? "Offline ready" : model?.state ?? "Missing"}</Badge></div>{model?.state === "downloading" && <div className="mt-5"><Progress value={model.progress} /><p className="mt-2 text-xs text-black/40">{model.message}</p></div>}{!model?.offlineReady && <Button className="mt-5" onClick={() => void install()}><Download size={16} /> Install or retry models</Button>}</section>
      <section className="panel mt-8 p-6"><p className="eyebrow">Answer generation</p><div className="mt-5 space-y-3"><Option selected title="Disabled" body="Search results and exact citations only. No generative model required." /><Option title="Lightweight local answers" body="Planned capability — not included in this POC." disabled /><Option title="Custom local model" body="Future capability." disabled /></div></section>
      <div className="mt-7 flex items-center gap-2 text-sm text-black/45"><Check size={16} className="text-moss" /> {bootstrap?.indexedFiles ?? 0} indexed files remain in your private local database.</div>
    </div>
  );
}

function Option({ title, body, selected, disabled }: { title: string; body: string; selected?: boolean; disabled?: boolean }) {
  return <div className={`flex gap-4 rounded-2xl border p-4 ${selected ? "border-moss/30 bg-moss/5" : "border-black/10"} ${disabled ? "opacity-45" : ""}`}><div className={`mt-1 h-4 w-4 rounded-full border-2 ${selected ? "border-moss bg-moss shadow-[inset_0_0_0_3px_white]" : "border-black/25"}`} /><div><h3 className="text-sm font-semibold">{title}</h3><p className="mt-1 text-xs leading-5 text-black/45">{body}</p></div></div>;
}
