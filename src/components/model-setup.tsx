"use client";

import { useState } from "react";
import { ArrowRight, Check, Cpu, FolderOpen, LockKeyhole, Wifi } from "lucide-react";
import { recallApi, isTauri } from "@/lib/tauri";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";

export function ModelSetup({ onComplete }: { onComplete: () => Promise<void> }) {
  const { model, setModel, setFolders } = useRecallStore();
  const [error, setError] = useState<string>();
  const [choosing, setChoosing] = useState(false);

  const install = async () => {
    setError(undefined);
    if (!isTauri()) {
      setModel({ state: "ready", progress: 100, message: "Preview mode", embeddingModel: "all-MiniLM-L6-v2", offlineReady: true });
      return;
    }
    try { setModel(await recallApi.installModels()); await onComplete(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  };

  const choose = async () => {
    setChoosing(true); setError(undefined);
    try { const folders = await recallApi.chooseFolders(); setFolders(folders); await onComplete(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setChoosing(false); }
  };

  const ready = model?.state === "ready";
  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="mx-auto grid min-h-[calc(100vh-4rem)] max-w-6xl overflow-hidden rounded-[36px] border border-black/10 bg-white/65 shadow-soft backdrop-blur-xl lg:grid-cols-[1.15fr_.85fr]">
        <section className="flex flex-col justify-between p-8 md:p-14">
          <div>
            <div className="flex items-center gap-3"><img src="/recall-mark.svg" alt="" className="h-12 w-12 rounded-2xl" /><span className="text-xl font-semibold">Recall</span></div>
            <p className="eyebrow mt-20">Private local search</p>
            <h1 className="mt-5 max-w-2xl text-5xl font-semibold leading-[1.02] tracking-[-0.04em] md:text-7xl">Find what you saved. <span className="text-moss">Nothing leaves.</span></h1>
            <p className="mt-7 max-w-xl text-lg leading-8 text-black/55">Recall reads and understands files in folders you choose. OCR, embeddings, and retrieval run entirely on this Windows PC.</p>
          </div>
          <div className="mt-12 flex flex-wrap gap-4 text-sm text-black/55"><span className="flex items-center gap-2"><LockKeyhole size={16} /> No account</span><span className="flex items-center gap-2"><Cpu size={16} /> Local inference</span><span className="flex items-center gap-2"><Check size={16} /> Source citations</span></div>
        </section>
        <section className="bg-ink p-8 text-white md:p-12">
          <p className="eyebrow !text-white/35">Set up in two steps</p>
          <div className="mt-8 space-y-5">
            <div className={`rounded-3xl border p-6 ${ready ? "border-lime/40 bg-lime/10" : "border-white/10 bg-white/5"}`}>
              <div className="flex items-start gap-4"><div className="flex h-10 w-10 items-center justify-center rounded-full bg-lime font-bold text-ink">1</div><div className="flex-1"><h2 className="text-lg font-semibold">Install local intelligence</h2><p className="mt-2 text-sm leading-6 text-white/50">Downloads compact English OCR and embedding models once. Files and queries are never part of the download.</p></div></div>
              {model?.state === "downloading" && <div className="mt-5"><Progress value={model.progress} className="bg-white/10 [&>div]:bg-lime" /><p className="mt-2 text-xs text-white/45">{model.message}</p></div>}
              <Button className="mt-5 w-full bg-lime text-ink hover:bg-lime/80" disabled={model?.state === "downloading" || ready} onClick={() => void install()}>{ready ? <><Check size={17} /> Models ready</> : <><Wifi size={17} /> Download models</>}</Button>
            </div>
            <div className={`rounded-3xl border p-6 ${ready ? "border-white/15 bg-white/5" : "border-white/10 bg-white/5"}`}>
              <div className="flex items-start gap-4"><div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 font-bold">2</div><div><h2 className="text-lg font-semibold">Choose your folders</h2><p className="mt-2 text-sm leading-6 text-white/50">Recall indexes supported files in place. Originals are never changed or copied.</p></div></div>
              <Button className="mt-5 w-full" variant="secondary" disabled={choosing || !isTauri()} onClick={() => void choose()}><FolderOpen size={17} /> {choosing ? "Opening picker…" : ready ? "Choose folders" : "Continue with keyword search"}<ArrowRight size={17} /></Button>
            </div>
          </div>
          {error && <p className="mt-4 rounded-2xl bg-red-500/15 p-4 text-sm text-red-200">{error}</p>}
          <p className="mt-7 text-xs leading-5 text-white/35">Search works offline after model setup. Without models, text files remain available through keyword search; image OCR waits for setup.</p>
        </section>
      </div>
    </div>
  );
}
