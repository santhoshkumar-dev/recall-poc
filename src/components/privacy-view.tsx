"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Check,
  Download,
  HardDrive,
  LockKeyhole,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  WifiOff,
} from "lucide-react";
import { isTauri, recallApi } from "@/lib/tauri";
import type { ModelCatalog, VisualDiagnostics } from "@/lib/types";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

export function PrivacyView({ onRefresh }: { onRefresh: () => Promise<void> }) {
  const model = useRecallStore((state) => state.model);
  const bootstrap = useRecallStore((state) => state.bootstrap);
  const indexing = useRecallStore((state) => state.indexing);
  const [catalog, setCatalog] = useState<ModelCatalog>();
  const [ocrModelId, setOcrModelId] = useState("");
  const [embeddingModelId, setEmbeddingModelId] = useState("");
  const [visualModelId, setVisualModelId] = useState("");
  const [ocrMaxSide, setOcrMaxSide] = useState(1280);
  const [saving, setSaving] = useState(false);
  const [visualReindexing, setVisualReindexing] = useState(false);
  const [error, setError] = useState<string>();
  const [diagnostics, setDiagnostics] = useState<VisualDiagnostics>();

  const loadCatalog = useCallback(async () => {
    if (!isTauri()) return;
    const next = await recallApi.modelCatalog();
    setCatalog(next);
    setOcrModelId(next.activeOcrModelId);
    setEmbeddingModelId(next.activeEmbeddingModelId);
    setVisualModelId(next.activeVisualModelId);
    setOcrMaxSide(next.ocrMaxSide);
  }, []);

  const loadDiagnostics = useCallback(async () => {
    if (!isTauri()) return;
    try {
      setDiagnostics(await recallApi.visualDiagnostics());
    } catch {
      /* diagnostics are best-effort */
    }
  }, []);

  useEffect(() => {
    void loadDiagnostics();
  }, [loadDiagnostics]);

  useEffect(() => {
    void loadCatalog().catch((cause) => {
      setError(cause instanceof Error ? cause.message : String(cause));
    });
  }, [loadCatalog]);

  const install = async () => {
    setSaving(true);
    setError(undefined);
    try {
      await recallApi.installModels();
      await Promise.all([onRefresh(), loadCatalog()]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  };

  const applySelection = async () => {
    if (!ocrModelId || !embeddingModelId) return;
    setSaving(true);
    setError(undefined);
    try {
      await recallApi.updateModelSelection(ocrModelId, embeddingModelId, ocrMaxSide, visualModelId);
      await Promise.all([onRefresh(), loadCatalog(), loadDiagnostics()]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  };

  const reindexVisualLibrary = async () => {
    if (!window.confirm("Re-index all image embeddings? OCR and document-text embeddings will not be changed.")) return;
    setVisualReindexing(true);
    setError(undefined);
    try {
      await recallApi.reindexVisualLibrary();
      await Promise.all([onRefresh(), loadDiagnostics()]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setVisualReindexing(false);
    }
  };

  const selectionChanged =
    !!catalog &&
    (ocrModelId !== catalog.activeOcrModelId ||
      embeddingModelId !== catalog.activeEmbeddingModelId ||
      visualModelId !== catalog.activeVisualModelId ||
      ocrMaxSide !== catalog.ocrMaxSide);
  const busy = saving || model?.state === "downloading";
  const processing = (indexing?.processing ?? 0) > 0;

  const facts = [
    [ShieldCheck, "Processing location", "On this device"],
    [LockKeyhole, "Remote AI requests", "0"],
    [HardDrive, "Files uploaded", "0"],
    [WifiOff, "Internet needed for search", model?.offlineReady ? "No" : "After setup"],
  ] as const;

  return (
    <div>
      <p className="eyebrow">Trust by architecture</p>
      <h1 className="mt-2 text-4xl font-semibold tracking-tight">Privacy & models</h1>
      <p className="mt-4 max-w-2xl text-base leading-7 text-black/50">
        Recall has no account, cloud backend, telemetry pipeline, or remote inference endpoint. It
        only accesses folders you explicitly approve.
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {facts.map(([Icon, label, value]) => (
          <div className="panel p-6" key={label}>
            <Icon className="text-moss" />
            <p className="mt-5 text-sm text-black/45">{label}</p>
            <p className="mt-1 text-2xl font-semibold">{value}</p>
          </div>
        ))}
      </div>

      <section className="panel mt-8 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="eyebrow">Active local models</p>
            <h2 className="mt-2 text-xl font-semibold">
              {model?.ocrModel ?? "PP-OCRv6 Tiny"} +{" "}
              {model?.embeddingModel ?? "Multilingual E5 Small"}
            </h2>
            <p className="mt-2 text-sm text-black/45">
              OCR capped at {model?.ocrMaxSide ?? 1280}px; embeddings and search run locally.
            </p>
          </div>
          <Badge tone={model?.offlineReady ? "good" : "warn"}>
            {model?.offlineReady ? "Offline ready" : model?.state ?? "Missing"}
          </Badge>
        </div>

        {model?.state === "downloading" && (
          <div className="mt-5">
            <Progress value={model.progress} />
            <p className="mt-2 text-xs text-black/40">{model.message}</p>
          </div>
        )}

        {!model?.offlineReady && (
          <Button className="mt-5" disabled={busy} onClick={() => void install()}>
            <Download size={16} /> Install or retry selected models
          </Button>
        )}
      </section>

      <section className="panel mt-8 p-6">
        <div className="flex items-center gap-3">
          <SlidersHorizontal className="text-moss" size={20} />
          <div>
            <p className="eyebrow">Developer model lab</p>
            <h2 className="mt-1 text-xl font-semibold">Runtime model selector</h2>
          </div>
        </div>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-black/50">
          PP-OCRv6 Tiny and Multilingual E5 Small are the POC defaults. Use these controls for local
          benchmarking; a model change downloads the selected pack before it becomes active.
        </p>

        {catalog && (
          <div className="mt-6 grid gap-5 lg:grid-cols-3">
            <ModelSelect
              label="OCR model"
              value={ocrModelId}
              onChange={setOcrModelId}
              options={catalog.ocrModels}
            />
            <ModelSelect
              label="Search model"
              value={embeddingModelId}
              onChange={setEmbeddingModelId}
              options={catalog.embeddingModels}
            />
            <label className="block">
              <span className="text-sm font-semibold">OCR resolution</span>
              <select
                className="focus-ring mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-3 text-sm"
                value={ocrMaxSide}
                onChange={(event) => setOcrMaxSide(Number(event.target.value))}
              >
                <option value={1280}>1280 px - fastest</option>
                <option value={1600}>1600 px - balanced</option>
                <option value={2048}>2048 px - detailed</option>
                <option value={4096}>4096 px - maximum</option>
              </select>
              <p className="mt-2 text-xs leading-5 text-black/45">
                Images larger than this are reduced before OCR to prevent long indexing stalls.
              </p>
            </label>
          </div>
        )}

        {catalog && (
          <div className="mt-5">
            <ModelSelect
              label="Visual image search model"
              value={visualModelId}
              onChange={setVisualModelId}
              options={catalog.visualModels}
            />
            <p className="mt-2 text-xs leading-5 text-black/45">
              MobileCLIP2-S0 adds visual and cross-modal search over screenshots and photos, alongside
              OCR and text search. Its embeddings live in a separate vector space from the text model.
            </p>
          </div>
        )}

        <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-900">
          Changing OCR or resolution reindexes image files. Changing the search model reindexes all
          available files because embedding dimensions and vector spaces are model-specific. Changing
          the visual-search model regenerates image embeddings only — it does not affect OCR or
          document-text embeddings.
        </div>

        {processing && (
          <p className="mt-3 text-xs text-black/50">
            Wait for the currently processing file to finish, or pause indexing, before applying.
          </p>
        )}
        {error && <p className="mt-4 rounded-2xl bg-red-50 p-4 text-sm text-red-700">{error}</p>}

        <Button
          className="mt-5"
          disabled={!selectionChanged || busy || processing}
          onClick={() => void applySelection()}
        >
          <Download size={16} />
          {busy ? "Preparing models..." : "Download and apply selection"}
        </Button>
      </section>

      <VisualDiagnosticsPanel
        diagnostics={diagnostics}
        onRefresh={loadDiagnostics}
        onReindex={reindexVisualLibrary}
        reindexing={visualReindexing}
        reindexDisabled={busy || visualReindexing || processing || (indexing?.pending ?? 0) > 0}
      />

      <section className="panel mt-8 p-6">
        <p className="eyebrow">Answer generation</p>
        <div className="mt-5 space-y-3">
          <Option
            selected
            title="Disabled"
            body="Search results and exact citations only. No generative model required."
          />
          <Option
            title="Lightweight local answers"
            body="Planned capability - not included in this POC."
            disabled
          />
          <Option title="Custom local model" body="Future capability." disabled />
        </div>
      </section>

      <div className="mt-7 flex items-center gap-2 text-sm text-black/45">
        <Check size={16} className="text-moss" /> {bootstrap?.indexedFiles ?? 0} indexed files remain
        in your private local database.
      </div>
    </div>
  );
}

function VisualDiagnosticsPanel({
  diagnostics,
  onRefresh,
  onReindex,
  reindexing,
  reindexDisabled,
}: {
  diagnostics?: VisualDiagnostics;
  onRefresh: () => Promise<void>;
  onReindex: () => Promise<void>;
  reindexing: boolean;
  reindexDisabled: boolean;
}) {
  if (!diagnostics) return null;
  const d = diagnostics;
  const loadOk = d.runtimeLoaded;
  const coverage =
    d.imageAssets > 0 ? Math.round((d.imagesWithEmbeddings / d.imageAssets) * 100) : 0;

  const Row = ({ label, value, tone }: { label: string; value: string; tone?: "good" | "warn" | "bad" }) => (
    <div className="flex items-center justify-between border-b border-black/5 py-2 text-sm last:border-0">
      <span className="text-black/50">{label}</span>
      <span
        className={
          tone === "good" ? "font-medium text-moss"
          : tone === "bad" ? "font-medium text-red-600"
          : tone === "warn" ? "font-medium text-amber-600"
          : "font-medium tabular-nums"
        }
      >
        {value}
      </span>
    </div>
  );

  return (
    <section className="panel mt-8 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="eyebrow">Developer diagnostics</p>
          <h2 className="mt-1 text-xl font-semibold">Visual search status</h2>
        </div>
        <Button variant="secondary" size="sm" onClick={() => void onRefresh()}>Refresh</Button>
      </div>
      <div className="mt-4 grid gap-x-8 md:grid-cols-2">
        <div>
          <Row label="Selected model" value={d.visualModelId} />
          <Row label="Enabled" value={d.visualEnabled ? "Yes" : "No"} tone={d.visualEnabled ? "good" : "warn"} />
          <Row label="Files installed" value={d.filesInstalled ? "Yes" : "No"} tone={d.filesInstalled ? "good" : "bad"} />
          <Row
            label="Runtime loaded"
            value={loadOk ? "Yes" : "No"}
            tone={loadOk ? "good" : "bad"}
          />
          <Row label="Load status" value={d.loadStatus} tone={loadOk ? "good" : "bad"} />
          <Row label="Embedding dims" value={d.embeddingDims ? String(d.embeddingDims) : "-"} />
          <Row label="Prompt bank" value={d.promptBankLoaded ? "Loaded" : "Not built"} tone={d.promptBankLoaded ? "good" : "warn"} />
        </div>
        <div>
          <Row label="Image assets" value={String(d.imageAssets)} />
          <Row label="Images indexed" value={String(d.imagesIndexed)} />
          <Row
            label="Images with embeddings"
            value={`${d.imagesWithEmbeddings} (${coverage}%)`}
            tone={coverage >= 99 ? "good" : coverage > 0 ? "warn" : "bad"}
          />
          <Row label="Adaptive regions" value={`${d.regionEmbeddings} across ${d.imagesWithRegions} images`} />
          <Row label="Images classified" value={String(d.imagesClassified)} />
          <Row label="Images tagged" value={String(d.imagesTagged)} />
          <Row label="Visual tags stored" value={String(d.visualTags)} />
          <Row label="Pending jobs" value={String(d.pendingJobs)} tone={d.pendingJobs > 0 ? "warn" : "good"} />
          <Row label="Failed jobs" value={String(d.failedJobs)} tone={d.failedJobs > 0 ? "bad" : "good"} />
        </div>
      </div>
      {!loadOk && d.visualEnabled && (
        <p className="mt-4 rounded-2xl bg-red-50 p-3 text-xs leading-5 text-red-700">
          Visual search is enabled but the runtime is not loaded ({d.loadStatus}). Visual results
          will be empty. Check the dev terminal for a `[visual] runtime FAILED to load` line.
        </p>
      )}
      {loadOk && d.imagesWithEmbeddings < d.imageAssets && (
        <p className="mt-4 rounded-2xl bg-amber-50 p-3 text-xs leading-5 text-amber-800">
          {d.imageAssets - d.imagesWithEmbeddings} image(s) have no visual embedding yet. They will
          not appear in visual results until indexing finishes ({d.pendingJobs} job(s) pending).
        </p>
      )}
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-black/10 pt-5">
        <Button
          variant="secondary"
          disabled={!loadOk || reindexDisabled}
          onClick={() => void onReindex()}
        >
          <RefreshCw size={15} className={reindexing ? "animate-spin" : undefined} />
          {reindexing ? "Queuing visual re-index..." : "Re-index visual library"}
        </Button>
        <p className="max-w-2xl text-xs leading-5 text-black/45">
          Rebuilds MobileCLIP image embeddings, prompt-bank diagnostics, and model-produced visual tags.
          Normal query text changes reuse existing image evidence and do not require this action.
        </p>
      </div>
    </section>
  );
}

function ModelSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: ModelCatalog["ocrModels"];
}) {
  const selected = options.find((option) => option.id === value);
  return (
    <label className="block">
      <span className="text-sm font-semibold">{label}</span>
      <select
        className="focus-ring mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-3 text-sm"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
            {option.recommended ? " - recommended" : ""}
            {option.installed ? " - installed" : ""}
          </option>
        ))}
      </select>
      {selected && (
        <p className="mt-2 text-xs leading-5 text-black/45">
          {selected.description} Download: about {selected.downloadMb} MB.
        </p>
      )}
    </label>
  );
}

function Option({
  title,
  body,
  selected,
  disabled,
}: {
  title: string;
  body: string;
  selected?: boolean;
  disabled?: boolean;
}) {
  const containerClass =
    "flex gap-4 rounded-2xl border p-4 " +
    (selected ? "border-moss/30 bg-moss/5" : "border-black/10") +
    (disabled ? " opacity-45" : "");
  const dotClass =
    "mt-1 h-4 w-4 rounded-full border-2 " +
    (selected
      ? "border-moss bg-moss shadow-[inset_0_0_0_3px_white]"
      : "border-black/25");
  return (
    <div className={containerClass}>
      <div className={dotClass} />
      <div>
        <h3 className="text-sm font-semibold">{title}</h3>
        <p className="mt-1 text-xs leading-5 text-black/45">{body}</p>
      </div>
    </div>
  );
}
