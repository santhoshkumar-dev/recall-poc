mod ai;
mod commands;
mod db;
mod error;
mod extract;
mod indexer;
mod search;
mod types;

use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
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
            let connection = db::connect(&db_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let paused = db::setting(&connection, "queue_paused", "false")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                == "true";
            let runtime = if ai::model_dir_is_complete(&model_dir) {
                AiRuntime::load(&model_dir).ok().map(Arc::new)
            } else {
                None
            };
            if runtime.is_some() {
                ai::set_state(&db_path, "ready")
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            } else if db::setting(&connection, "model_state", "missing")
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
            });
            app.manage(core.clone());
            if !paused {
                indexer::start_worker(app.handle().clone(), core);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_bootstrap_state,
            commands::get_model_status,
            commands::install_models,
            commands::choose_folders,
            commands::list_watched_folders,
            commands::remove_watched_folder,
            commands::rescan_folder,
            commands::pause_indexing,
            commands::resume_indexing,
            commands::retry_failed_job,
            commands::get_indexing_status,
            commands::list_recent_assets,
            commands::search_files,
            commands::open_source_file,
            commands::reveal_source_file,
            commands::copy_source_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Recall");
}
