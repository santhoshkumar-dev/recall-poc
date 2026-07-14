mod ai;
mod commands;
mod db;
mod document_intent;
mod error;
mod extract;
mod fusion;
mod indexer;
mod metadata;
mod query_intent;
mod search;
mod types;
mod visual;

use std::{
    fs,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
};

use ai::AiRuntime;
use parking_lot::RwLock;
use tauri::Manager;

pub struct AppCore {
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
    pub thumbnail_dir: PathBuf,
    pub paused: AtomicBool,
    pub worker_running: AtomicBool,
    pub model_installing: AtomicBool,
    pub ai: RwLock<Option<Arc<AiRuntime>>>,
    /// Optional visual (MobileCLIP2) runtime — separate space from `ai`.
    pub visual: RwLock<Option<Arc<visual::VisualRuntime>>>,
    /// Optional ONNX visual tagger runtime for model-produced image tags.
    pub visual_tagger: RwLock<Option<Arc<visual::VisualTaggerRuntime>>>,
    /// Cached zero-shot prompt embeddings; rebuilt when the visual model or
    /// prompt bank changes.
    pub visual_prompts: RwLock<Option<Arc<visual::PromptBank>>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let model_dir = data_dir.join("models");
            let thumbnail_dir = data_dir.join("thumbnails");
            fs::create_dir_all(&model_dir)?;
            fs::create_dir_all(&thumbnail_dir)?;
            let db_path = data_dir.join("recall.db");
            db::migrate(&db_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            ai::ensure_defaults(&db_path, &model_dir)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let connection = db::connect(&db_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let paused = db::setting(&connection, "queue_paused", "false")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                == "true";

            let runtime = if ai::model_dir_is_complete(&db_path, &model_dir) {
                match ai::selection(&db_path)
                    .and_then(|selected| AiRuntime::load_for_selection(&model_dir, &selected))
                {
                    Ok(runtime) => Some(Arc::new(runtime)),
                    Err(error) => {
                        ai::set_error(
                            &db_path,
                            &format!("Installed models could not be loaded: {error}"),
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                        None
                    }
                }
            } else {
                None
            };

            // Optional visual runtime; failure here must NOT block text search.
            let visual_runtime = match ai::selection(&db_path)
                .and_then(|selected| ai::load_visual_for_selection(&model_dir, &selected))
            {
                Ok(Some(runtime)) => {
                    eprintln!("[visual] runtime loaded (dims={})", runtime.dims());
                    ai::set_visual_status(&db_path, "loaded");
                    Some(runtime)
                }
                Ok(None) => {
                    let selected = ai::selection(&db_path).ok();
                    let enabled = selected.map(|s| s.visual_enabled()).unwrap_or(false);
                    let msg = if enabled {
                        "enabled but model files are missing"
                    } else {
                        "disabled"
                    };
                    eprintln!("[visual] runtime not loaded: {msg}");
                    ai::set_visual_status(&db_path, msg);
                    None
                }
                Err(error) => {
                    eprintln!("[visual] runtime FAILED to load: {error}");
                    ai::set_visual_status(&db_path, &format!("load error: {error}"));
                    None
                }
            };
            let visual_tagger_runtime = match ai::selection(&db_path)
                .and_then(|selected| ai::load_visual_tagger_for_selection(&model_dir, &selected))
            {
                Ok(Some(runtime)) => {
                    eprintln!("[visual-tags] runtime loaded");
                    Some(runtime)
                }
                Ok(None) => None,
                Err(error) => {
                    eprintln!("[visual-tags] runtime FAILED to load: {error}");
                    None
                }
            };

            if db::setting(&connection, "model_state", "missing")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                == "downloading"
            {
                ai::set_error(
                    &db_path,
                    "The previous model download was interrupted. Select Download models to resume it.",
                )
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            }

            let core = Arc::new(AppCore {
                db_path,
                model_dir,
                thumbnail_dir,
                paused: AtomicBool::new(paused),
                worker_running: AtomicBool::new(false),
                model_installing: AtomicBool::new(false),
                ai: RwLock::new(runtime),
                visual: RwLock::new(visual_runtime),
                visual_tagger: RwLock::new(visual_tagger_runtime),
                visual_prompts: RwLock::new(None),
            });
            if core.visual.read().is_some() {
                let selected = ai::selection(&core.db_path)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let queued = db::queue_stale_visual_reindex(
                    &core.db_path,
                    &selected.visual_model_id,
                    ai::VISUAL_MODEL_VERSION,
                )
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                if queued > 0 {
                    eprintln!("[visual] queued {queued} stale image(s) for pipeline refresh");
                }
            }
            if core.visual_tagger.read().is_some() {
                let queued = db::queue_stale_visual_tagging_reindex(
                    &core.db_path,
                    ai::VISUAL_TAGGER_GENERAL,
                    ai::VISUAL_TAGGER_VERSION,
                )
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                if queued > 0 {
                    eprintln!("[visual-tags] queued {queued} stale image(s) for tag refresh");
                }
            }
            app.manage(core.clone());
            if !paused {
                indexer::start_worker(app.handle().clone(), core);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_bootstrap_state,
            commands::get_model_status,
            commands::get_model_catalog,
            commands::install_models,
            commands::update_model_selection,
            commands::choose_folders,
            commands::list_watched_folders,
            commands::remove_watched_folder,
            commands::rescan_folder,
            commands::pause_indexing,
            commands::resume_indexing,
            commands::force_delete_library,
            commands::retry_failed_job,
            commands::get_indexing_status,
            commands::list_recent_assets,
            commands::search_files,
            commands::search_files_debug,
            commands::get_visual_diagnostics,
            commands::reindex_visual_library,
            commands::open_source_file,
            commands::reveal_source_file,
            commands::copy_source_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Recall");
}
