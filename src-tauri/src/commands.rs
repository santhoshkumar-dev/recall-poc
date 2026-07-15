use std::{
    fs,
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
    thread,
    time::{Duration, Instant},
};

use arboard::Clipboard;
use rusqlite::OptionalExtension;
use tauri::{AppHandle, State};

use crate::{
    ai, db,
    error::{RecallError, Result},
    fusion, indexer,
    types::{
        AssetSummary, BootstrapState, IndexingStatus, ModelCatalog, ModelStatus, SearchFilters,
        SearchResult, WatchedFolder,
    },
    AppCore,
};

#[tauri::command]
pub fn get_bootstrap_state(core: State<'_, Arc<AppCore>>) -> Result<BootstrapState> {
    db::bootstrap(&core.db_path)
}

#[tauri::command]
pub fn get_model_status(core: State<'_, Arc<AppCore>>) -> Result<ModelStatus> {
    let status = ai::status(&core.db_path)?;
    if status.state == "downloading" && !core.model_installing.load(Ordering::SeqCst) {
        ai::set_error(
            &core.db_path,
            "The previous model download was interrupted. Select Download models to resume it.",
        )?;
        return ai::status(&core.db_path);
    }
    Ok(status)
}

#[tauri::command]
pub fn get_model_catalog(core: State<'_, Arc<AppCore>>) -> Result<ModelCatalog> {
    ai::catalog(&core.db_path, &core.model_dir)
}

#[tauri::command(rename_all = "camelCase")]
pub async fn update_model_selection(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
    ocr_model_id: String,
    embedding_model_id: String,
    ocr_max_side: u32,
    visual_model_id: Option<String>,
) -> Result<ModelStatus> {
    let core = core.inner().clone();
    let previous = ai::selection(&core.db_path)?;
    let selected = ai::ModelSelection {
        ocr_model_id,
        embedding_model_id,
        ocr_max_side,
        visual_model_id: visual_model_id.unwrap_or_else(|| previous.visual_model_id.clone()),
    };
    if selected.ocr_model_id == previous.ocr_model_id
        && selected.embedding_model_id == previous.embedding_model_id
        && selected.ocr_max_side == previous.ocr_max_side
        && selected.visual_model_id == previous.visual_model_id
    {
        return ai::status(&core.db_path);
    }
    if core.model_installing.swap(true, Ordering::SeqCst) {
        return Err("Model installation is already running".into());
    }

    let was_paused = core.paused.swap(true, Ordering::SeqCst);
    let processing = match db::indexing_status(&core.db_path) {
        Ok(status) => status.processing,
        Err(error) => {
            core.paused.store(was_paused, Ordering::SeqCst);
            core.model_installing.store(false, Ordering::SeqCst);
            return Err(error);
        }
    };
    if processing > 0 {
        core.paused.store(was_paused, Ordering::SeqCst);
        core.model_installing.store(false, Ordering::SeqCst);
        return Err(
            "Wait for the current file to finish, or pause indexing, before changing models."
                .into(),
        );
    }

    let app_for_work = app.clone();
    let install_core = core.clone();
    let selected_for_work = selected.clone();
    let task = tauri::async_runtime::spawn_blocking(move || {
        ai::install_selection(
            &app_for_work,
            &install_core.db_path,
            &install_core.model_dir,
            &selected_for_work,
        )
    })
    .await;
    core.model_installing.store(false, Ordering::SeqCst);

    let result = match task {
        Ok(result) => result,
        Err(error) => {
            core.paused.store(was_paused, Ordering::SeqCst);
            return Err(RecallError::Message(error.to_string()));
        }
    };
    match result {
        Ok(runtime) => {
            let ocr_changed = selected.ocr_model_id != previous.ocr_model_id
                || selected.ocr_max_side != previous.ocr_max_side;
            let embedding_changed = selected.embedding_model_id != previous.embedding_model_id;
            let visual_changed = selected.visual_model_id != previous.visual_model_id;
            *core.ai.write() = Some(runtime);
            db::resume_waiting_model_jobs(&core.db_path)?;

            // Swap the visual runtime to match the new selection (load errors
            // are non-fatal: text search continues). Drop the cached prompt bank
            // so it rebuilds for the new model / prompt version.
            match ai::load_visual_for_selection(&core.model_dir, &selected) {
                Ok(Some(runtime)) => {
                    eprintln!("[visual] runtime loaded (dims={})", runtime.dims());
                    ai::set_visual_status(&core.db_path, "loaded");
                    *core.visual.write() = Some(runtime);
                }
                Ok(None) => {
                    *core.visual.write() = None;
                    ai::set_visual_status(&core.db_path, "disabled");
                }
                Err(error) => {
                    eprintln!("[visual] runtime FAILED to load: {error}");
                    ai::set_visual_status(&core.db_path, &format!("load error: {error}"));
                    *core.visual.write() = None;
                }
            }
            *core.visual_prompts.write() = None;
            match ai::load_visual_tagger_for_selection(&core.model_dir, &selected) {
                Ok(Some(runtime)) => {
                    eprintln!("[visual-tags] runtime loaded");
                    *core.visual_tagger.write() = Some(runtime);
                }
                Ok(None) => {
                    *core.visual_tagger.write() = None;
                }
                Err(error) => {
                    eprintln!("[visual-tags] runtime FAILED to load: {error}");
                    *core.visual_tagger.write() = None;
                }
            }

            if let Err(error) = db::queue_model_reindex(
                &core.db_path,
                ocr_changed,
                embedding_changed,
                &selected.embedding_model_id,
                crate::ai::CHUNKING_VERSION,
            ) {
                core.paused.store(was_paused, Ordering::SeqCst);
                return Err(error);
            }
            // If only the visual model changed, re-embed image assets only
            // (never text/E5). A full embedding reindex already covers images.
            if visual_changed && !embedding_changed && selected.visual_enabled() {
                if let Err(error) = db::queue_visual_stage_reindex(
                    &core.db_path,
                    &selected.visual_model_id,
                    crate::ai::VISUAL_MODEL_VERSION,
                ) {
                    core.paused.store(was_paused, Ordering::SeqCst);
                    return Err(error);
                }
            }
            if visual_changed && selected.visual_enabled() && core.visual_tagger.read().is_some() {
                if let Err(error) = db::queue_visual_tagging_stage_reindex(
                    &core.db_path,
                    crate::ai::VISUAL_TAGGER_GENERAL,
                    crate::ai::VISUAL_TAGGER_VERSION,
                ) {
                    core.paused.store(was_paused, Ordering::SeqCst);
                    return Err(error);
                }
            }
            core.paused.store(was_paused, Ordering::SeqCst);
            if !was_paused {
                indexer::start_worker(app, core.clone());
            }
            ai::status(&core.db_path)
        }
        Err(error) => {
            core.paused.store(was_paused, Ordering::SeqCst);
            if core.ai.read().is_some() {
                let _ = ai::set_state(&core.db_path, "ready");
                let _ = ai::set_last_error(&core.db_path, &error.to_string());
            } else {
                let _ = ai::set_error(&core.db_path, &error.to_string());
            }
            Err(error)
        }
    }
}

#[tauri::command]
pub async fn install_models(app: AppHandle, core: State<'_, Arc<AppCore>>) -> Result<ModelStatus> {
    let core = core.inner().clone();
    if core.model_installing.swap(true, Ordering::SeqCst) {
        return Err("Model installation is already running".into());
    }

    let app_for_work = app.clone();
    let install_core = core.clone();
    let task = tauri::async_runtime::spawn_blocking(move || {
        ai::install(
            &app_for_work,
            &install_core.db_path,
            &install_core.model_dir,
        )
    })
    .await;
    core.model_installing.store(false, Ordering::SeqCst);

    let result = task.map_err(|e| RecallError::Message(e.to_string()))?;
    match result {
        Ok(runtime) => {
            *core.ai.write() = Some(runtime);
            let selection = ai::selection(&core.db_path)?;
            match ai::load_visual_for_selection(&core.model_dir, &selection) {
                Ok(visual) => {
                    *core.visual.write() = visual;
                    if core.visual.read().is_some() {
                        ai::set_visual_status(&core.db_path, "loaded");
                        db::queue_stale_visual_reindex(
                            &core.db_path,
                            &selection.visual_model_id,
                            ai::VISUAL_MODEL_VERSION,
                        )?;
                    }
                }
                Err(error) => {
                    ai::set_visual_status(&core.db_path, &format!("load error: {error}"));
                    return Err(error);
                }
            }
            indexer::start_worker(app, core.clone());
            ai::status(&core.db_path)
        }
        Err(error) => {
            if core.ai.read().is_some() {
                let _ = ai::set_state(&core.db_path, "ready");
                let _ = ai::set_last_error(&core.db_path, &error.to_string());
            } else {
                let _ = ai::set_error(&core.db_path, &error.to_string());
            }
            Err(error)
        }
    }
}

#[tauri::command]
pub async fn choose_folders(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
) -> Result<Vec<WatchedFolder>> {
    let folders = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("Choose folders for Recall")
            .pick_folders()
    })
    .await
    .map_err(|e| RecallError::Message(e.to_string()))?
    .unwrap_or_default();
    let core = core.inner().clone();
    for folder in folders {
        let worker_core = core.clone();
        let worker_app = app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            indexer::add_and_scan(&worker_app, &worker_core, &folder)
        })
        .await
        .map_err(|e| RecallError::Message(e.to_string()))??;
    }
    indexer::list_folders(&core)
}

#[tauri::command]
pub fn list_watched_folders(core: State<'_, Arc<AppCore>>) -> Result<Vec<WatchedFolder>> {
    indexer::list_folders(&core)
}

#[tauri::command(rename_all = "camelCase")]
pub fn remove_watched_folder(core: State<'_, Arc<AppCore>>, folder_id: String) -> Result<()> {
    indexer::remove_folder(&core, &folder_id)
}

#[tauri::command(rename_all = "camelCase")]
pub async fn rescan_folder(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
    folder_id: String,
) -> Result<()> {
    let core = core.inner().clone();
    tauri::async_runtime::spawn_blocking(move || indexer::scan_folder(&app, &core, &folder_id))
        .await
        .map_err(|e| RecallError::Message(e.to_string()))?
}

#[tauri::command]
pub fn pause_indexing(core: State<'_, Arc<AppCore>>) -> Result<()> {
    core.paused.store(true, Ordering::SeqCst);
    let connection = db::connect(&core.db_path)?;
    db::set_setting(&connection, "queue_paused", "true")
}

#[tauri::command]
pub fn resume_indexing(app: AppHandle, core: State<'_, Arc<AppCore>>) -> Result<()> {
    core.paused.store(false, Ordering::SeqCst);
    let connection = db::connect(&core.db_path)?;
    db::set_setting(&connection, "queue_paused", "false")?;
    indexer::start_worker(app, core.inner().clone());
    Ok(())
}

#[tauri::command]
pub async fn force_delete_library(core: State<'_, Arc<AppCore>>) -> Result<()> {
    let core = core.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        if core.model_installing.load(Ordering::SeqCst) {
            return Err(RecallError::Message(
                "Wait for model installation to finish before force deleting the library.".into(),
            ));
        }
        core.paused.store(true, Ordering::SeqCst);
        let start = Instant::now();
        while core.worker_running.load(Ordering::SeqCst) {
            if start.elapsed() > Duration::from_secs(120) {
                return Err(RecallError::Message(
                    "Timed out waiting for the current indexing job to stop. Try again after the current file finishes.".into(),
                ));
            }
            thread::sleep(Duration::from_millis(100));
        }

        if core.db_path.exists() {
            fs::remove_file(&core.db_path)?;
        }
        for suffix in ["wal", "shm"] {
            let sidecar = core.db_path.with_extension(format!("db-{suffix}"));
            if sidecar.exists() {
                fs::remove_file(sidecar)?;
            }
        }
        if core.thumbnail_dir.exists() {
            fs::remove_dir_all(&core.thumbnail_dir)?;
        }
        fs::create_dir_all(&core.thumbnail_dir)?;

        db::migrate(&core.db_path)?;
        ai::ensure_defaults(&core.db_path, &core.model_dir)?;
        if ai::model_dir_is_complete(&core.db_path, &core.model_dir) {
            ai::set_state(&core.db_path, "ready")?;
        }
        let connection = db::connect(&core.db_path)?;
        db::set_setting(&connection, "queue_paused", "false")?;
        core.paused.store(false, Ordering::SeqCst);
        Ok(())
    })
    .await
    .map_err(|error| RecallError::Message(error.to_string()))?
}

#[tauri::command(rename_all = "camelCase")]
pub fn retry_failed_job(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
    job_id: String,
) -> Result<()> {
    let connection = db::connect(&core.db_path)?;
    let now = chrono::Utc::now().to_rfc3339();
    let changed = connection.execute("UPDATE indexing_jobs SET state='pending',error_message=NULL,updated_at=?2 WHERE id=?1 AND state='failed'", rusqlite::params![job_id, now])?
        + connection.execute("UPDATE extraction_stage_jobs SET state='pending',error_message=NULL,updated_at=?2 WHERE id=?1 AND state='failed'", rusqlite::params![job_id, chrono::Utc::now().to_rfc3339()])?;
    if changed == 0 {
        return Err("Failed job was not found".into());
    }
    indexer::start_worker(app, core.inner().clone());
    Ok(())
}

#[tauri::command]
pub fn get_indexing_status(core: State<'_, Arc<AppCore>>) -> Result<IndexingStatus> {
    db::indexing_status(&core.db_path)
}

#[tauri::command]
pub fn list_recent_assets(
    core: State<'_, Arc<AppCore>>,
    limit: Option<i64>,
) -> Result<Vec<AssetSummary>> {
    db::list_recent_assets(&core.db_path, limit.unwrap_or(20))
}

#[tauri::command]
pub fn search_files(
    core: State<'_, Arc<AppCore>>,
    query: String,
    filters: SearchFilters,
) -> Result<Vec<SearchResult>> {
    fusion::search(&core, &query, &filters)
}

#[tauri::command]
pub fn search_files_debug(
    core: State<'_, Arc<AppCore>>,
    query: String,
    filters: SearchFilters,
) -> Result<crate::types::SearchDebugReport> {
    if !cfg!(debug_assertions) {
        return Err("Retrieval diagnostics are unavailable in production builds".into());
    }
    fusion::search_debug(&core, &query, &filters)
}

#[tauri::command]
pub fn get_visual_diagnostics(
    core: State<'_, Arc<AppCore>>,
) -> Result<crate::types::VisualDiagnostics> {
    if !cfg!(debug_assertions) {
        return Err("Visual diagnostics are unavailable in production builds".into());
    }
    let selection = ai::selection(&core.db_path)?;
    let model_id = selection.visual_model_id.clone();
    let runtime = core.visual.read().clone();
    let (
        image_assets,
        images_indexed,
        images_with_embeddings,
        region_embeddings,
        images_with_regions,
        images_classified,
    ) = db::visual_counts(&core.db_path, &model_id)?;
    let (images_tagged, visual_tags) =
        db::visual_tag_counts(&core.db_path, crate::ai::VISUAL_TAGGER_GENERAL)?;
    let (pending_jobs, _processing_jobs, failed_jobs) = db::active_job_counts(&core.db_path)?;
    Ok(crate::types::VisualDiagnostics {
        visual_enabled: selection.visual_enabled(),
        files_installed: ai::is_visual_installed(&core.model_dir, &model_id),
        runtime_loaded: runtime.is_some(),
        tagger_files_installed: ai::is_visual_tagger_installed(&core.model_dir),
        tagger_runtime_loaded: core.visual_tagger.read().is_some(),
        embedding_dims: runtime.as_ref().map(|r| r.dims()),
        prompt_bank_loaded: core.visual_prompts.read().is_some(),
        load_status: ai::visual_status(&core.db_path),
        visual_model_id: model_id,
        image_assets,
        images_indexed,
        images_with_embeddings,
        region_embeddings,
        images_with_regions,
        images_classified,
        images_tagged,
        visual_tags,
        pending_jobs,
        failed_jobs,
    })
}

#[tauri::command]
pub fn reindex_visual_library(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
) -> Result<IndexingStatus> {
    let core = core.inner().clone();
    let selection = ai::selection(&core.db_path)?;
    if !selection.visual_enabled() {
        return Err("Enable and install a visual-search model before re-indexing".into());
    }
    if core.visual.read().is_none() {
        return Err("The visual-search runtime is not loaded".into());
    }
    db::queue_visual_stage_reindex(
        &core.db_path,
        &selection.visual_model_id,
        crate::ai::VISUAL_MODEL_VERSION,
    )?;
    if core.visual_tagger.read().is_some() {
        db::queue_visual_tagging_stage_reindex(
            &core.db_path,
            crate::ai::VISUAL_TAGGER_GENERAL,
            crate::ai::VISUAL_TAGGER_VERSION,
        )?;
    }
    if !core.paused.load(Ordering::SeqCst) {
        indexer::start_worker(app, core.clone());
    }
    db::indexing_status(&core.db_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_asset_thumbnail(core: State<'_, Arc<AppCore>>, asset_id: String) -> Result<Vec<u8>> {
    let connection = db::connect(&core.db_path)?;
    let exists: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM assets WHERE id=?1 AND available=1",
            [asset_id.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        return Err("Thumbnail source is unavailable".into());
    }
    let path = core.thumbnail_dir.join(format!("{asset_id}.png"));
    let canonical_root = core.thumbnail_dir.canonicalize()?;
    let canonical_path = path.canonicalize()?;
    if !canonical_path.starts_with(canonical_root) {
        return Err("Security check failed: thumbnail is outside app data".into());
    }
    Ok(fs::read(canonical_path)?)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_asset_pipeline_status(
    core: State<'_, Arc<AppCore>>,
    asset_id: String,
) -> Result<Vec<crate::types::AssetStageStatus>> {
    db::asset_pipeline_status(&core.db_path, &asset_id)
}

#[tauri::command]
pub fn get_reindex_status(core: State<'_, Arc<AppCore>>) -> Result<IndexingStatus> {
    db::indexing_status(&core.db_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn open_source_file(core: State<'_, Arc<AppCore>>, asset_id: String) -> Result<()> {
    let path = validated_asset_path(&core, &asset_id)?;
    open::that(path)
        .map_err(|e| RecallError::Message(format!("Could not open source file: {e}")))?;
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
pub fn reveal_source_file(core: State<'_, Arc<AppCore>>, asset_id: String) -> Result<()> {
    let path = validated_asset_path(&core, &asset_id)?;
    std::process::Command::new("explorer.exe")
        .arg(format!("/select,{}", path.to_string_lossy()))
        .spawn()?;
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
pub fn copy_source_path(core: State<'_, Arc<AppCore>>, asset_id: String) -> Result<()> {
    let path = validated_asset_path(&core, &asset_id)?;
    Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(path.to_string_lossy().into_owned()))
        .map_err(|e| RecallError::Message(format!("Could not copy path: {e}")))?;
    Ok(())
}

fn validated_asset_path(core: &AppCore, asset_id: &str) -> Result<PathBuf> {
    let connection = db::connect(&core.db_path)?;
    let record: Option<(String,String)> = connection.query_row("SELECT a.absolute_path,f.path FROM assets a JOIN watched_folders f ON f.id=a.folder_id WHERE a.id=?1 AND a.available=1", [asset_id], |r| Ok((r.get(0)?,r.get(1)?))).optional()?;
    let (asset, root) =
        record.ok_or_else(|| RecallError::Message("Indexed source is unavailable".into()))?;
    validate_containment(Path::new(&root), Path::new(&asset))
}

pub fn validate_containment(root: &Path, asset: &Path) -> Result<PathBuf> {
    let canonical_root = root.canonicalize()?;
    let canonical_asset = asset.canonicalize()?;
    if !canonical_asset.starts_with(&canonical_root) {
        return Err("Security check failed: source is outside its approved folder".into());
    }
    Ok(canonical_asset)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lexical_path_prefix_does_not_authorize_sibling() {
        let root = Path::new(r"C:\Users\demo\Documents");
        let sibling = Path::new(r"C:\Users\demo\Documents-private\secret.txt");
        assert!(!sibling.starts_with(root));
    }
}
