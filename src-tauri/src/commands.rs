use std::{
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

use arboard::Clipboard;
use rusqlite::OptionalExtension;
use tauri::{AppHandle, State};

use crate::{
    ai, db,
    error::{RecallError, Result},
    indexer, search,
    types::{
        AssetSummary, BootstrapState, IndexingStatus, ModelStatus, SearchFilters, SearchResult,
        WatchedFolder,
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
            indexer::start_worker(app, core.clone());
            ai::status(&core.db_path)
        }
        Err(error) => {
            let _ = ai::set_error(&core.db_path, &error.to_string());
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

#[tauri::command(rename_all = "camelCase")]
pub fn retry_failed_job(
    app: AppHandle,
    core: State<'_, Arc<AppCore>>,
    job_id: String,
) -> Result<()> {
    let connection = db::connect(&core.db_path)?;
    let changed = connection.execute("UPDATE indexing_jobs SET state='pending',error_message=NULL,updated_at=?2 WHERE id=?1 AND state='failed'", rusqlite::params![job_id,chrono::Utc::now().to_rfc3339()])?;
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
    search::search(&core, &query, &filters)
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
