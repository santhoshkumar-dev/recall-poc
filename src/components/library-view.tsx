"use client";

import { useState } from "react";
import { AlertTriangle, FolderOpen, Pause, Play, RefreshCw, Trash2 } from "lucide-react";
import { recallApi } from "@/lib/tauri";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

export function LibraryView({ onRefresh }: { onRefresh: () => Promise<void> }) {
  const { folders, indexing, assets, setFolders } = useRecallStore();
  const [forceDeleting, setForceDeleting] = useState(false);
  const background = (indexing?.backgroundPending ?? 0) + (indexing?.backgroundProcessing ?? 0);
  const active = (indexing?.pending ?? 0) + (indexing?.processing ?? 0) + background;
  const total = (indexing?.pending ?? 0) + (indexing?.processing ?? 0) + (indexing?.indexed ?? 0) + (indexing?.failed ?? 0) + (indexing?.skipped ?? 0);
  const done = (indexing?.indexed ?? 0) + (indexing?.failed ?? 0) + (indexing?.skipped ?? 0);
  const rawProgress = total ? (done / total) * 100 : 0;
  const progress = active > 0 ? Math.min(99, Math.floor(rawProgress)) : Math.round(rawProgress);
  const stageLabel = indexing?.currentStage
    ? ({
        index: "Indexing file",
        visual: "Embedding image",
        visual_regions: "Embedding image regions",
        text_embedding: "Embedding text",
        analysis: "Analyzing text",
        visual_tagging: "Tagging image",
        visual_region_tagging: "Tagging image regions",
      } as Record<string, string>)[indexing.currentStage] ?? "Processing"
    : undefined;
  const statusText = indexing?.currentFile
    ? `${stageLabel ?? (background > 0 ? "Processing visual evidence" : "Processing")} for ${indexing.currentFile}`
    : indexing?.paused
      ? "Queue is paused"
      : background > 0
        ? "Finalizing background visual and text evidence"
        : "Queue is up to date";

  const choose = async () => { setFolders(await recallApi.chooseFolders()); await onRefresh(); };
  const queueToggle = async () => { indexing?.paused ? await recallApi.resume() : await recallApi.pause(); await onRefresh(); };
  const forceDelete = async () => {
    const answer = window.prompt("This deletes all folders, queues, indexed files, thumbnails, and Recall database state. Downloaded models are kept. Type DELETE to continue.");
    if (answer !== "DELETE") return;
    setForceDeleting(true);
    try {
      await recallApi.forceDeleteLibrary();
      await onRefresh();
    } finally {
      setForceDeleting(false);
    }
  };

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Local library</p><h1 className="mt-2 text-4xl font-semibold tracking-tight">Folders & indexing</h1></div><Button onClick={() => void choose()}><FolderOpen size={17} /> Add folders</Button></div>
      <section className="panel mt-8 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="text-lg font-semibold">Indexing queue</h2><p className="mt-1 text-sm text-black/45">{statusText}</p></div><Button variant="secondary" size="sm" onClick={() => void queueToggle()}>{indexing?.paused ? <><Play size={15} /> Resume</> : <><Pause size={15} /> Pause</>}</Button></div>
        <Progress value={progress} className="mt-5" />
        <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-6">{[["Pending", indexing?.pending], ["Processing", indexing?.processing], ["Background tasks", background], ["Indexed", indexing?.indexed], ["Skipped", indexing?.skipped], ["Failed", indexing?.failed]].map(([label, value]) => <div key={String(label)} className="rounded-2xl bg-black/[.035] p-4"><div className="text-2xl font-semibold">{value ?? 0}</div><div className="mt-1 text-xs text-black/45">{label}</div></div>)}</div>
        {background > 0 && <p className="mt-3 text-xs leading-5 text-black/45">Background tasks are derived OCR/text/visual stages. They can be higher than the file count because one image may need embeddings, tags, metadata, and regional crops.</p>}
      </section>
      <section className="mt-9"><div className="flex items-center justify-between"><h2 className="text-xl font-semibold">Watched folders</h2><span className="text-sm text-black/40">{folders.length} approved</span></div><div className="mt-4 grid gap-4 xl:grid-cols-2">{folders.map((folder) => <div className="panel p-5" key={folder.id}><div className="flex items-start gap-4"><div className="rounded-2xl bg-lime/50 p-3"><FolderOpen size={20} /></div><div className="min-w-0 flex-1"><h3 className="truncate font-semibold">{folder.path.split(/[\\/]/).pop()}</h3><p className="mt-1 truncate text-xs text-black/40">{folder.path}</p><div className="mt-3 flex gap-2"><Badge>{folder.availableFiles} files</Badge><Badge tone="good">{folder.indexedFiles} indexed</Badge></div></div></div><div className="mt-4 flex gap-2 border-t border-black/10 pt-4"><Button size="sm" variant="secondary" onClick={async () => { await recallApi.rescanFolder(folder.id); await onRefresh(); }}><RefreshCw size={14} /> Rescan</Button><Button size="sm" variant="ghost" className="ml-auto text-red-700" onClick={async () => { await recallApi.removeFolder(folder.id); await onRefresh(); }}><Trash2 size={14} /> Remove</Button></div></div>)}{folders.length === 0 && <div className="rounded-3xl border border-dashed border-black/15 p-8 text-center text-sm text-black/45">No folders selected yet.</div>}</div></section>
      <section className="mt-9"><h2 className="text-xl font-semibold">Recent files</h2><div className="panel mt-4 divide-y divide-black/10 overflow-hidden">{assets.map((asset) => <div key={asset.id} className="flex items-center gap-4 p-4"><div className="min-w-0 flex-1"><p className="truncate text-sm font-medium">{asset.filename}</p><p className="mt-1 truncate text-xs text-black/35">{asset.sourcePath}</p></div><Badge tone={asset.status === "failed" ? "bad" : asset.status === "indexed" ? "good" : "neutral"}>{asset.status}</Badge>{asset.status === "failed" && <AlertTriangle size={16} className="text-red-600" />}</div>)}{assets.length === 0 && <p className="p-8 text-center text-sm text-black/40">Discovered files will appear here.</p>}</div></section>
      <section className="panel mt-9 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-red-700">Danger zone</h2>
            <p className="mt-1 text-sm text-black/45">Deletes all Recall database state, queues, watched folders, index records, and thumbnails. Downloaded model files are kept.</p>
          </div>
          <Button variant="danger" disabled={forceDeleting} onClick={() => void forceDelete()}><Trash2 size={15} /> {forceDeleting ? "Deleting..." : "Force delete library"}</Button>
        </div>
      </section>
    </div>
  );
}
