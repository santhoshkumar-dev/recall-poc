import { invoke } from "@tauri-apps/api/core";
import type {
  AssetSummary,
  AssetStageStatus,
  BootstrapState,
  IndexingStatus,
  ModelCatalog,
  ModelStatus,
  SearchDebugReport,
  SearchFilters,
  SearchResult,
  VisualDiagnostics,
  WatchedFolder,
} from "./types";

export const isTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function desktopInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) throw new Error("Recall native services are available in the desktop application.");
  return invoke<T>(command, args);
}

export const recallApi = {
  bootstrap: () => desktopInvoke<BootstrapState>("get_bootstrap_state"),
  modelStatus: () => desktopInvoke<ModelStatus>("get_model_status"),
  modelCatalog: () => desktopInvoke<ModelCatalog>("get_model_catalog"),
  installModels: () => desktopInvoke<ModelStatus>("install_models"),
  updateModelSelection: (
    ocrModelId: string,
    embeddingModelId: string,
    ocrMaxSide: number,
    visualModelId?: string,
  ) =>
    desktopInvoke<ModelStatus>("update_model_selection", {
      ocrModelId,
      embeddingModelId,
      ocrMaxSide,
      visualModelId,
    }),
  chooseFolders: () => desktopInvoke<WatchedFolder[]>("choose_folders"),
  folders: () => desktopInvoke<WatchedFolder[]>("list_watched_folders"),
  removeFolder: (folderId: string) => desktopInvoke<void>("remove_watched_folder", { folderId }),
  rescanFolder: (folderId: string) => desktopInvoke<void>("rescan_folder", { folderId }),
  pause: () => desktopInvoke<void>("pause_indexing"),
  resume: () => desktopInvoke<void>("resume_indexing"),
  forceDeleteLibrary: () => desktopInvoke<void>("force_delete_library"),
  retry: (jobId: string) => desktopInvoke<void>("retry_failed_job", { jobId }),
  indexingStatus: () => desktopInvoke<IndexingStatus>("get_indexing_status"),
  recentAssets: (limit = 20) => desktopInvoke<AssetSummary[]>("list_recent_assets", { limit }),
  search: (query: string, filters: SearchFilters) =>
    desktopInvoke<SearchResult[]>("search_files", { query, filters }),
  searchDebug: (query: string, filters: SearchFilters) =>
    desktopInvoke<SearchDebugReport>("search_files_debug", { query, filters }),
  visualDiagnostics: () => desktopInvoke<VisualDiagnostics>("get_visual_diagnostics"),
  reindexVisualLibrary: () => desktopInvoke<IndexingStatus>("reindex_visual_library"),
  thumbnail: (assetId: string) => desktopInvoke<number[]>("get_asset_thumbnail", { assetId }),
  assetPipelineStatus: (assetId: string) => desktopInvoke<AssetStageStatus[]>("get_asset_pipeline_status", { assetId }),
  reindexStatus: () => desktopInvoke<IndexingStatus>("get_reindex_status"),
  open: (assetId: string) => desktopInvoke<void>("open_source_file", { assetId }),
  reveal: (assetId: string) => desktopInvoke<void>("reveal_source_file", { assetId }),
  copyPath: (assetId: string) => desktopInvoke<void>("copy_source_path", { assetId }),
};
