import { create } from "zustand";
import type { AssetSummary, BootstrapState, IndexingStatus, ModelStatus, WatchedFolder } from "@/lib/types";

type View = "search" | "library" | "privacy";

interface RecallStore {
  view: View;
  bootstrap?: BootstrapState;
  model?: ModelStatus;
  folders: WatchedFolder[];
  indexing?: IndexingStatus;
  assets: AssetSummary[];
  setView: (view: View) => void;
  setBootstrap: (value: BootstrapState) => void;
  setModel: (value: ModelStatus) => void;
  setFolders: (value: WatchedFolder[]) => void;
  setIndexing: (value: IndexingStatus) => void;
  setAssets: (value: AssetSummary[]) => void;
}

export const useRecallStore = create<RecallStore>((set) => ({
  view: "search",
  folders: [],
  assets: [],
  setView: (view) => set({ view }),
  setBootstrap: (bootstrap) => set({ bootstrap }),
  setModel: (model) => set({ model }),
  setFolders: (folders) => set({ folders }),
  setIndexing: (indexing) => set({ indexing }),
  setAssets: (assets) => set({ assets }),
}));
