"use client";

import { useState } from "react";
import { Clipboard, ExternalLink, FileText, FolderSearch, Search, Sparkles } from "lucide-react";
import { recallApi } from "@/lib/tauri";
import type { SearchResult } from "@/lib/types";
import { useRecallStore } from "@/store/recall-store";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";

export function SearchView() {
  const { folders, model, bootstrap } = useRecallStore();
  const [query, setQuery] = useState("");
  const [folderId, setFolderId] = useState("");
  const [extension, setExtension] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string>();

  const search = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setError(undefined); setSearched(true);
    try { setResults(await recallApi.search(query.trim(), { folderId: folderId || undefined, extensions: extension ? [extension] : [] })); }
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
        </div>
      </form>
      <div className="mt-5 flex items-center justify-between text-sm text-black/45"><span>{bootstrap?.indexedFiles ?? 0} locally indexed files</span>{searched && !loading && <span>{results.length} results</span>}</div>
      {error && <div className="mt-6 rounded-2xl bg-red-50 p-4 text-sm text-red-700">{error}</div>}
      {!searched && <Empty icon={Sparkles} title="Search by meaning, not filenames" body="Recall combines local semantic similarity with exact keyword matches and always shows the source." />}
      {searched && !loading && !error && results.length === 0 && <Empty icon={FolderSearch} title="No matching source found" body="Try broader wording, remove a filter, or check the Library for indexing failures." />}
      <div className="mt-7 space-y-4">
        {results.map((result) => <ResultCard key={result.assetId} result={result} />)}
      </div>
    </div>
  );
}

function ResultCard({ result }: { result: SearchResult }) {
  const pct = Math.round(result.combinedScore * 100);
  return (
    <article className="panel p-6 transition hover:-translate-y-0.5 hover:bg-white">
      <div className="flex items-start gap-4"><div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-lime/50"><FileText size={22} /></div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><h2 className="truncate text-lg font-semibold">{result.filename}</h2><Badge>{result.extension?.toUpperCase() ?? "FILE"}</Badge>{result.pageNumber && <Badge>Page {result.pageNumber}</Badge>}<Badge tone={pct >= 70 ? "good" : "neutral"}>{pct}% match</Badge></div><p className="mt-1 truncate text-xs text-black/40">{result.sourcePath}</p><p className="mt-4 max-w-4xl text-sm leading-7 text-black/65">{result.snippet}</p><div className="mt-5 flex flex-wrap gap-2"><Button size="sm" onClick={() => void recallApi.open(result.assetId)}><ExternalLink size={15} /> Open</Button><Button size="sm" variant="secondary" onClick={() => void recallApi.reveal(result.assetId)}><FolderSearch size={15} /> Show in folder</Button><Button size="sm" variant="ghost" onClick={() => void recallApi.copyPath(result.assetId)}><Clipboard size={15} /> Copy path</Button></div></div></div>
    </article>
  );
}

function Empty({ icon: Icon, title, body }: { icon: typeof Sparkles; title: string; body: string }) {
  return <div className="mt-14 flex flex-col items-center text-center"><div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-black/5"><Icon className="text-black/40" /></div><h2 className="mt-4 text-lg font-semibold">{title}</h2><p className="mt-2 max-w-md text-sm leading-6 text-black/45">{body}</p></div>;
}
