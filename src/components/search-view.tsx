"use client";

import { useState } from "react";
import { Clipboard, ExternalLink, FileText, FolderSearch, Search, Sparkles, FlaskConical } from "lucide-react";
import { recallApi } from "@/lib/tauri";
import type { MatchReason, SearchDebugReport, SearchResult } from "@/lib/types";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";

const MATCH_REASON_LABEL: Record<MatchReason, string> = {
  exact_text: "Exact text",
  semantic_text: "Semantic text",
  visual_similarity: "Visual match",
  visual_tag: "Visual tag",
  visual_category: "Visual category",
  date: "Date",
  amount: "Amount",
  filename: "Filename",
  folder: "Folder",
  file_type: "File type",
  metadata: "Metadata",
  document_type: "Document type",
  entity: "Entity",
};

export function SearchView() {
  const { folders, model, bootstrap } = useRecallStore();
  const [query, setQuery] = useState("");
  const [folderId, setFolderId] = useState("");
  const [extension, setExtension] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string>();
  const [devMode, setDevMode] = useState(false);
  const [report, setReport] = useState<SearchDebugReport>();

  const search = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setError(undefined); setSearched(true); setReport(undefined);
    const filters = { folderId: folderId || undefined, extensions: extension ? [extension] : [] };
    try {
      if (devMode) {
        const debug = await recallApi.searchDebug(query.trim(), filters);
        setReport(debug);
        setResults(debug.results);
      } else {
        setResults(await recallApi.search(query.trim(), filters));
      }
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); setResults([]); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-5">
        <div><p className="eyebrow">Private retrieval</p><h1 className="mt-2 text-4xl font-semibold tracking-tight md:text-5xl">What are you looking for?</h1></div>
        <Badge tone={model?.offlineReady ? "good" : "warn"}>{model?.offlineReady ? "Offline ready" : "Keyword mode"}</Badge>
      </div>
      <form className="panel mt-9 p-3" onSubmit={search}>
        <div className="flex items-center gap-3 px-3"><Search className="text-black/35" /><input aria-label="Search your files" value={query} onChange={(e) => setQuery(e.target.value)} className="h-16 min-w-0 flex-1 bg-transparent text-xl outline-none placeholder:text-black/25" placeholder="Try “train ticket for Bengaluru”" /><Button size="lg" disabled={loading || !query.trim()}>{loading ? "Searching…" : "Search"}</Button></div>
        <div className="flex flex-wrap gap-2 border-t border-black/10 p-3">
          <select aria-label="Filter by folder" value={folderId} onChange={(e) => setFolderId(e.target.value)} className="h-9 rounded-full border border-black/10 bg-white px-4 text-sm"><option value="">All folders</option>{folders.map((folder) => <option key={folder.id} value={folder.id}>{folder.path}</option>)}</select>
          <select aria-label="Filter by file type" value={extension} onChange={(e) => setExtension(e.target.value)} className="h-9 rounded-full border border-black/10 bg-white px-4 text-sm"><option value="">All file types</option>{["txt", "md", "pdf", "png", "jpg", "jpeg", "webp"].map((ext) => <option key={ext} value={ext}>.{ext}</option>)}</select>
          <label className="ml-auto flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 text-sm text-black/60"><input type="checkbox" checked={devMode} onChange={(e) => setDevMode(e.target.checked)} /><FlaskConical size={14} /> Retrieval inspector</label>
        </div>
      </form>
      <div className="mt-5 flex items-center justify-between text-sm text-black/45"><span>{bootstrap?.indexedFiles ?? 0} locally indexed files</span>{searched && !loading && <span>{results.length} results</span>}</div>
      {error && <div className="mt-6 rounded-2xl bg-red-50 p-4 text-sm text-red-700">{error}</div>}
      {!searched && <Empty icon={Sparkles} title="Search by meaning, not filenames" body="Recall combines local semantic similarity with exact keyword matches and always shows the source." />}
      {searched && !loading && !error && results.length === 0 && <Empty icon={FolderSearch} title="No sufficiently relevant local files were found" body="Try broader wording, remove a filter, or check the Library for indexing failures." />}
      {report && <Inspector report={report} />}
      <div className="mt-7 space-y-4">
        {results.map((result) => <ResultCard key={result.assetId} result={result} />)}
      </div>
    </div>
  );
}

function Inspector({ report }: { report: SearchDebugReport }) {
  return (
    <section className="mt-7 rounded-2xl border border-black/10 bg-black/[0.02] p-5 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold">Retrieval inspector</span>
        <span className="text-black/40">· {report.totalLatencyMs} ms total</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-4">
        <div><span className="text-black/45">Intents:</span> {report.intents.length ? report.intents.map((i) => <Badge key={i}>{i}</Badge>) : <span className="text-black/40">none</span>}</div>
        <div><span className="text-black/45">Visual query:</span> {report.visualQuery ? "yes" : "no"}</div>
        {report.visualPrompts.length > 0 && <div><span className="text-black/45">Visual prompts:</span> {report.visualPrompts.join(" · ")}</div>}
        {report.expandedCategories.length > 0 && <div><span className="text-black/45">Categories:</span> {report.expandedCategories.map((c) => <Badge key={c} tone="neutral">{c}</Badge>)}</div>}
        {report.appliedFilters.length > 0 && <div><span className="text-black/45">Filters:</span> {report.appliedFilters.join("; ")}</div>}
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {report.channels.map((channel) => (
          <div key={channel.channel} className="rounded-xl border border-black/10 bg-white p-3">
            <div className="flex items-center justify-between font-medium"><span>{channel.channel}</span><span className="text-black/40">{channel.latencyMs} ms · {channel.candidateCount}</span></div>
            <ol className="mt-2 space-y-1 text-xs text-black/60">
              {channel.results.slice(0, 8).map((r) => (
                <li key={r.assetId} className="flex items-center justify-between gap-2"><span className="truncate">#{r.rank + 1} {r.filename}</span><span className="tabular-nums text-black/45" title="raw channel score (cosine / BM25 / category / match-count)">{r.rawScore.toFixed(3)}</span></li>
              ))}
              {channel.results.length === 0 && <li className="text-black/35">no candidates</li>}
            </ol>
          </div>
        ))}
      </div>
      {report.results.some((result) => result.visualScore > 0 || result.topVisualTags.length > 0) && (
        <div className="mt-4 overflow-x-auto rounded-xl border border-black/10 bg-white p-3">
          <p className="font-medium">Qualified visual evidence</p>
          <div className="mt-2 space-y-1 text-xs text-black/60">
            {report.results.slice(0, 8).map((result) => (
              <div key={result.assetId} className="grid min-w-[860px] grid-cols-[1fr_repeat(5,90px)_170px] gap-2">
                <span className="truncate">{result.filename}</span>
                <span>cos {result.visualScore.toFixed(3)}</span>
                <span>z {result.visualZScore.toFixed(2)}</span>
                <span>region {result.visualRegionId ?? "-"}</span>
                <span>cat+ {result.categoryPositiveScore.toFixed(3)}</span>
                <span>margin {result.categoryScore.toFixed(3)}</span>
                <span className="truncate">
                  tags {result.topVisualTags.slice(0, 3).map((tag) => tag.label).join(", ") || "-"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ResultCard({ result }: { result: SearchResult }) {
  const confidenceLabel = result.confidence === "strong" ? "Strong match" : "Moderate match";
  return (
    <article className="panel p-6 transition hover:-translate-y-0.5 hover:bg-white">
      <div className="flex items-start gap-4"><div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-lime/50"><FileText size={22} /></div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><h2 className="truncate text-lg font-semibold">{result.filename}</h2><Badge>{result.extension?.toUpperCase() ?? "FILE"}</Badge>{result.pageNumber && <Badge>Page {result.pageNumber}</Badge>}<Badge tone={result.confidence === "strong" ? "good" : "neutral"}>{confidenceLabel}</Badge></div><p className="mt-1 truncate text-xs text-black/40">{result.sourcePath}</p>{result.matchReasons?.length > 0 && <div className="mt-3 flex flex-wrap items-center gap-1.5"><span className="text-xs text-black/40">Matched because:</span>{result.matchReasons.map((reason) => <Badge key={reason} tone="neutral">{MATCH_REASON_LABEL[reason]}</Badge>)}</div>}{result.topVisualTags.length > 0 && <div className="mt-3 flex flex-wrap items-center gap-1.5"><span className="text-xs text-black/40">Visual tags:</span>{result.topVisualTags.slice(0, 5).map((tag) => <Badge key={`${tag.regionId}-${tag.label}`} tone="neutral">{tag.label}</Badge>)}</div>}<p className="mt-4 max-w-4xl text-sm leading-7 text-black/65">{result.snippet}</p><div className="mt-5 flex flex-wrap gap-2"><Button size="sm" onClick={() => void recallApi.open(result.assetId)}><ExternalLink size={15} /> Open</Button><Button size="sm" variant="secondary" onClick={() => void recallApi.reveal(result.assetId)}><FolderSearch size={15} /> Show in folder</Button><Button size="sm" variant="ghost" onClick={() => void recallApi.copyPath(result.assetId)}><Clipboard size={15} /> Copy path</Button></div></div></div>
    </article>
  );
}

function Empty({ icon: Icon, title, body }: { icon: typeof Sparkles; title: string; body: string }) {
  return <div className="mt-14 flex flex-col items-center text-center"><div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-black/5"><Icon className="text-black/40" /></div><h2 className="mt-4 text-lg font-semibold">{title}</h2><p className="mt-2 max-w-md text-sm leading-6 text-black/45">{body}</p></div>;
}
