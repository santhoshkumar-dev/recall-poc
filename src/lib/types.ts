export type ModelState = "missing" | "downloading" | "ready" | "error";
export type JobState = "pending" | "processing" | "indexed" | "skipped" | "failed" | "cancelled";

export interface BootstrapState {
  databaseReady: boolean;
  modelState: ModelState;
  folders: number;
  indexedFiles: number;
  queuePaused: boolean;
}

export interface ModelStatus {
  state: ModelState;
  progress: number;
  message: string;
  embeddingModel: string;
  offlineReady: boolean;
}

export interface WatchedFolder {
  id: string;
  path: string;
  createdAt: string;
  availableFiles: number;
  indexedFiles: number;
}

export interface IndexingStatus {
  paused: boolean;
  pending: number;
  processing: number;
  indexed: number;
  skipped: number;
  failed: number;
  currentFile?: string;
}

export interface AssetSummary {
  id: string;
  filename: string;
  extension?: string;
  sourcePath: string;
  status: JobState;
  errorMessage?: string;
  indexedAt?: string;
}

export interface SearchFilters {
  extensions: string[];
  folderId?: string;
}

export interface SearchResult {
  assetId: string;
  filename: string;
  extension?: string;
  sourcePath: string;
  snippet: string;
  pageNumber?: number;
  semanticScore: number;
  keywordScore: number;
  combinedScore: number;
}

export interface ModelProgressEvent {
  progress: number;
  message: string;
  state: ModelState;
}

export interface IndexingEvent {
  assetId?: string;
  folderId?: string;
  filename?: string;
  completed?: number;
  total?: number;
  message?: string;
}
