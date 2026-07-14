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
  ocrModel: string;
  embeddingModel: string;
  visualModel: string;
  visualEnabled: boolean;
  ocrMaxSide: number;
  offlineReady: boolean;
}

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  downloadMb: number;
  recommended: boolean;
  installed: boolean;
}

export interface ModelCatalog {
  ocrModels: ModelOption[];
  embeddingModels: ModelOption[];
  visualModels: ModelOption[];
  activeOcrModelId: string;
  activeEmbeddingModelId: string;
  activeVisualModelId: string;
  ocrMaxSide: number;
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
  backgroundPending: number;
  backgroundProcessing: number;
  currentStage?: string;
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

export type QueryIntent =
  | "exact_identifier"
  | "filename"
  | "semantic_text"
  | "visual"
  | "category"
  | "date_filtered"
  | "amount_filtered"
  | "folder_filtered"
  | "file_type_filtered"
  | "mixed";

export type MatchReason =
  | "exact_text"
  | "semantic_text"
  | "visual_similarity"
  | "visual_tag"
  | "date"
  | "amount"
  | "filename"
  | "folder"
  | "file_type"
  | "metadata"
  | "document_type"
  | "entity";

export interface VisualCategory {
  label: string;
  score: number;
}

export interface VisualTag {
  regionId: number;
  namespace: string;
  label: string;
  confidence: number;
  rank: number;
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
  visualScore: number;
  visualZScore: number;
  visualRegionId?: number;
  categoryScore: number;
  categoryPositiveScore: number;
  categoryNegativeScore: number;
  matchReasons: MatchReason[];
  topCategories: VisualCategory[];
  topVisualTags: VisualTag[];
  confidence: "strong" | "moderate";
}

export interface ChannelResult {
  channel: string;
  assetId: string;
  filename: string;
  rank: number;
  rawScore: number;
  normalizedScore: number;
}

export interface ChannelDiagnostics {
  channel: string;
  latencyMs: number;
  candidateCount: number;
  results: ChannelResult[];
}

export interface SearchDebugReport {
  query: string;
  visualQuery: boolean;
  visualPrompts: string[];
  intents: QueryIntent[];
  expandedCategories: string[];
  appliedFilters: string[];
  channels: ChannelDiagnostics[];
  results: SearchResult[];
  totalLatencyMs: number;
}

export interface VisualDiagnostics {
  visualModelId: string;
  visualEnabled: boolean;
  filesInstalled: boolean;
  runtimeLoaded: boolean;
  taggerFilesInstalled: boolean;
  taggerRuntimeLoaded: boolean;
  embeddingDims?: number;
  promptBankLoaded: boolean;
  loadStatus: string;
  imageAssets: number;
  imagesIndexed: number;
  imagesWithEmbeddings: number;
  regionEmbeddings: number;
  imagesWithRegions: number;
  imagesClassified: number;
  imagesTagged: number;
  visualTags: number;
  pendingJobs: number;
  failedJobs: number;
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
