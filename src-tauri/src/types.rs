use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapState {
    pub database_ready: bool,
    pub model_state: String,
    pub folders: i64,
    pub indexed_files: i64,
    pub queue_paused: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub state: String,
    pub progress: u8,
    pub message: String,
    pub ocr_model: String,
    pub embedding_model: String,
    pub visual_model: String,
    pub visual_enabled: bool,
    pub ocr_max_side: u32,
    pub offline_ready: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOption {
    pub id: String,
    pub label: String,
    pub description: String,
    pub download_mb: u32,
    pub recommended: bool,
    pub installed: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalog {
    pub ocr_models: Vec<ModelOption>,
    pub embedding_models: Vec<ModelOption>,
    pub visual_models: Vec<ModelOption>,
    pub active_ocr_model_id: String,
    pub active_embedding_model_id: String,
    pub active_visual_model_id: String,
    pub ocr_max_side: u32,
}

/// Developer diagnostics for the visual (MobileCLIP) subsystem.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualDiagnostics {
    pub visual_model_id: String,
    pub visual_enabled: bool,
    /// Model files present on disk.
    pub files_installed: bool,
    /// Runtime successfully loaded into memory (search actually works).
    pub runtime_loaded: bool,
    pub embedding_dims: Option<usize>,
    pub prompt_bank_loaded: bool,
    /// Last load status message ("loaded" or an error).
    pub load_status: String,
    // Coverage counts.
    pub image_assets: i64,
    pub images_indexed: i64,
    pub images_with_embeddings: i64,
    pub images_classified: i64,
    pub pending_jobs: i64,
    pub failed_jobs: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProgressEvent {
    pub state: String,
    pub progress: u8,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WatchedFolder {
    pub id: String,
    pub path: String,
    pub created_at: String,
    pub available_files: i64,
    pub indexed_files: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexingStatus {
    pub paused: bool,
    pub pending: i64,
    pub processing: i64,
    pub indexed: i64,
    pub skipped: i64,
    pub failed: i64,
    pub current_file: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetSummary {
    pub id: String,
    pub filename: String,
    pub extension: Option<String>,
    pub source_path: String,
    pub status: String,
    pub error_message: Option<String>,
    pub indexed_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AssetRecord {
    pub id: String,
    pub folder_id: String,
    pub absolute_path: String,
    pub filename: String,
    pub extension: String,
}

/// Lightweight per-asset info for fusion (covers image-only assets that may
/// have no text chunks).
#[derive(Debug, Clone)]
pub struct AssetBrief {
    pub id: String,
    #[allow(dead_code)] // carried for folder-scoped features; not yet read in fusion.
    pub folder_id: String,
    pub filename: String,
    pub extension: Option<String>,
    pub source_path: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilters {
    #[serde(default)]
    pub extensions: Vec<String>,
    pub folder_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub asset_id: String,
    pub filename: String,
    pub extension: Option<String>,
    pub source_path: String,
    pub snippet: String,
    pub page_number: Option<i64>,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub combined_score: f32,
    #[serde(default)]
    pub visual_score: f32,
    #[serde(default)]
    pub category_score: f32,
    #[serde(default)]
    pub match_reasons: Vec<MatchReason>,
    #[serde(default)]
    pub top_categories: Vec<VisualCategory>,
}

impl SearchResult {
    /// Convenience constructor keeping the legacy text-only fields, with the
    /// multimodal fields defaulted (populated later by fusion).
    pub fn text_only(
        asset_id: String,
        filename: String,
        extension: Option<String>,
        source_path: String,
        snippet: String,
        page_number: Option<i64>,
        semantic_score: f32,
        keyword_score: f32,
        combined_score: f32,
    ) -> Self {
        Self {
            asset_id,
            filename,
            extension,
            source_path,
            snippet,
            page_number,
            semantic_score,
            keyword_score,
            combined_score,
            visual_score: 0.0,
            category_score: 0.0,
            match_reasons: Vec::new(),
            top_categories: Vec::new(),
        }
    }
}

/// Deterministically-detected search intents (no LLM). A query may carry several.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    ExactIdentifier,
    Filename,
    SemanticText,
    Visual,
    Category,
    DateFiltered,
    AmountFiltered,
    FolderFiltered,
    FileTypeFiltered,
    Mixed,
}

/// Why a result matched, surfaced to the user as human-readable reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchReason {
    ExactText,
    SemanticText,
    VisualSimilarity,
    VisualCategory,
    Date,
    Amount,
    Filename,
    Folder,
    FileType,
    Metadata,
}

/// One retrieval channel's ranking of a single asset (developer inspector).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelResult {
    pub channel: String,
    pub asset_id: String,
    pub filename: String,
    pub rank: usize,
    pub raw_score: f32,
    pub normalized_score: f32,
}

/// Per-channel timing and candidate counts for the inspector.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelDiagnostics {
    pub channel: String,
    pub latency_ms: u128,
    pub candidate_count: usize,
    pub results: Vec<ChannelResult>,
}

/// Full developer retrieval report for one query.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchDebugReport {
    pub query: String,
    pub intents: Vec<QueryIntent>,
    pub expanded_categories: Vec<String>,
    pub applied_filters: Vec<String>,
    pub channels: Vec<ChannelDiagnostics>,
    pub results: Vec<SearchResult>,
    pub total_latency_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexingEvent {
    pub asset_id: Option<String>,
    pub folder_id: Option<String>,
    pub filename: Option<String>,
    pub completed: Option<usize>,
    pub total: Option<usize>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Amount {
    pub raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
}

/// Generic, deterministically-extracted metadata for an asset (no LLM).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExtractedMetadata {
    pub dates: Vec<String>,
    pub times: Vec<String>,
    pub amounts: Vec<Amount>,
    pub urls: Vec<String>,
    pub emails: Vec<String>,
    pub phone_numbers: Vec<String>,
    pub identifiers: Vec<String>,
    pub hashtags: Vec<String>,
    pub mentions: Vec<String>,
    pub possible_locations: Vec<String>,
    pub possible_provider_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_qr_code: Option<bool>,
}

/// One visual category score for an asset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VisualCategory {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct PageText {
    pub page_number: Option<i64>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ChunkInput {
    pub index: i64,
    pub page_number: Option<i64>,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}
