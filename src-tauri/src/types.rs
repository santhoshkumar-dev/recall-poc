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
    pub embedding_model: String,
    pub offline_ready: bool,
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
