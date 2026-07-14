"use client";

import { useCallback, useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Library, LockKeyhole, Search, Sparkles } from "lucide-react";
import { recallApi, isTauri } from "@/lib/tauri";
import type { IndexingEvent, ModelProgressEvent } from "@/lib/types";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { ModelSetup } from "./model-setup";
import { SearchView } from "./search-view";
import { LibraryView } from "./library-view";
import { PrivacyView } from "./privacy-view";

export function AppShell() {
  const view = useRecallStore((state) => state.view);
  const bootstrap = useRecallStore((state) => state.bootstrap);
  const model = useRecallStore((state) => state.model);
  const folders = useRecallStore((state) => state.folders);
  const setView = useRecallStore((state) => state.setView);
  const setBootstrap = useRecallStore((state) => state.setBootstrap);
  const setModel = useRecallStore((state) => state.setModel);
  const setFolders = useRecallStore((state) => state.setFolders);
  const setIndexing = useRecallStore((state) => state.setIndexing);
  const setAssets = useRecallStore((state) => state.setAssets);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const refresh = useCallback(async () => {
    if (!isTauri()) {
      setBootstrap({ databaseReady: true, modelState: "missing", folders: 0, indexedFiles: 0, queuePaused: false });
      setModel({ state: "missing", progress: 0, message: "Models are not installed", embeddingModel: "all-MiniLM-L6-v2", offlineReady: false });
      setLoading(false);
      return;
    }
    try {
      const [bootstrapState, modelStatus, watchedFolders, indexing, assets] = await Promise.all([
        recallApi.bootstrap(),
        recallApi.modelStatus(),
        recallApi.folders(),
        recallApi.indexingStatus(),
        recallApi.recentAssets(),
      ]);
      setBootstrap(bootstrapState);
      setModel(modelStatus);
      setFolders(watchedFolders);
      setIndexing(indexing);
      setAssets(assets);
      setError(undefined);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, [setAssets, setBootstrap, setFolders, setIndexing, setModel]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!isTauri()) return;

    let disposed = false;
    let refreshTimer: ReturnType<typeof setTimeout> | undefined;
    const unlisteners: UnlistenFn[] = [];
    const scheduleRefresh = () => {
      if (refreshTimer) return;
      refreshTimer = setTimeout(() => {
        refreshTimer = undefined;
        if (!disposed) void refresh();
      }, 300);
    };

    void (async () => {
      const subscriptions = await Promise.all([
        listen<ModelProgressEvent>("models://progress", (event) => {
          const current = useRecallStore.getState().model;
          if (current) {
            setModel({
              ...current,
              ...event.payload,
              offlineReady: event.payload.state === "ready",
            });
          }
        }),
        ...[
          "indexing://file-progress",
          "indexing://file-completed",
          "indexing://file-failed",
          "indexing://folder-completed",
          "indexing://queue-state",
        ].map((name) => listen<IndexingEvent>(name, scheduleRefresh)),
      ]);

      if (disposed) subscriptions.forEach((unlisten) => unlisten());
      else unlisteners.push(...subscriptions);
    })();

    return () => {
      disposed = true;
      if (refreshTimer) clearTimeout(refreshTimer);
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, [refresh, setModel]);

  if (loading) return <Splash message="Opening your private index…" />;

  if (!bootstrap?.databaseReady || error) {
    return (
      <div className="flex min-h-screen items-center justify-center p-8">
        <div className="panel max-w-lg p-8">
          <p className="eyebrow">Native service unavailable</p>
          <h1 className="mt-3 text-3xl font-semibold">Recall could not open its local index.</h1>
          <p className="mt-4 text-black/60">{error ?? "The local database did not initialize."}</p>
          <Button className="mt-6" onClick={() => void refresh()}>Retry</Button>
        </div>
      </div>
    );
  }

  const needsOnboarding = model?.state !== "ready" && folders.length === 0;
  if (needsOnboarding) return <ModelSetup onComplete={refresh} />;

  const nav = [
    { id: "search" as const, label: "Search", icon: Search },
    { id: "library" as const, label: "Library", icon: Library },
    { id: "privacy" as const, label: "Privacy & models", icon: LockKeyhole },
  ];

  return (
    <div className="min-h-screen p-4 lg:p-6">
      <div className="mx-auto grid min-h-[calc(100vh-3rem)] max-w-[1500px] grid-cols-1 overflow-hidden rounded-[32px] border border-black/10 bg-white/55 shadow-soft backdrop-blur-xl lg:grid-cols-[250px_1fr]">
        <aside className="flex flex-col border-b border-black/10 bg-ink p-5 text-white lg:border-b-0 lg:border-r">
          <div className="flex items-center gap-3 px-2 py-3">
            <img src="/recall-mark.svg" alt="" className="h-10 w-10 rounded-xl border border-white/10" />
            <div><div className="text-lg font-semibold">Recall</div><div className="text-xs text-white/45">Local memory</div></div>
          </div>
          <nav className="mt-6 flex gap-2 overflow-x-auto lg:flex-col">
            {nav.map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setView(id)} className={`focus-ring flex min-w-fit items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm font-medium transition ${view === id ? "bg-lime text-ink" : "text-white/65 hover:bg-white/10 hover:text-white"}`}>
                <Icon size={18} />{label}
              </button>
            ))}
          </nav>
          <div className="mt-auto hidden rounded-2xl border border-white/10 bg-white/5 p-4 lg:block">
            <div className="flex items-center gap-2 text-sm"><Sparkles size={16} className="text-lime" /> On-device AI</div>
            <p className="mt-2 text-xs leading-5 text-white/45">Your files and queries stay on this computer.</p>
          </div>
        </aside>
        <main className="min-w-0 p-5 md:p-8 lg:p-10">
          {view === "search" && <SearchView />}
          {view === "library" && <LibraryView onRefresh={refresh} />}
          {view === "privacy" && <PrivacyView onRefresh={refresh} />}
        </main>
      </div>
    </div>
  );
}

function Splash({ message }: { message: string }) {
  return <div className="flex min-h-screen flex-col items-center justify-center gap-5"><img src="/recall-mark.svg" alt="Recall" className="h-20 w-20 rounded-3xl shadow-soft" /><p className="text-sm text-black/50">{message}</p></div>;
}
