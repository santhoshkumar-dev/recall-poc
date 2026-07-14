use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    path::Path,
    sync::{atomic::Ordering, Arc},
};

use chrono::Utc;
use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;
use walkdir::{DirEntry, WalkDir};

use crate::{
    db,
    error::{RecallError, Result},
    extract::{self, ProcessOutcome},
    types::{IndexingEvent, WatchedFolder},
    AppCore,
};

const SUPPORTED: &[&str] = &["txt", "md", "pdf", "png", "jpg", "jpeg", "webp"];
const EXCLUDED: &[&str] = &[
    ".git",
    "node_modules",
    ".next",
    "dist",
    "build",
    "target",
    "appdata",
    "caches",
];

pub fn supported_extension(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_string_lossy().to_ascii_lowercase();
    SUPPORTED.contains(&extension.as_str()).then_some(extension)
}

fn allowed_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let name = entry.file_name().to_string_lossy();
    if name.starts_with('.') && name != "." {
        return false;
    }
    !EXCLUDED.contains(&name.to_ascii_lowercase().as_str())
}

pub fn add_and_scan(app: &AppHandle, core: &Arc<AppCore>, folder: &Path) -> Result<()> {
    let canonical = folder.canonicalize()?;
    if !canonical.is_dir() {
        return Err("Selected path is not a folder".into());
    }
    let folder_path = canonical.to_string_lossy().into_owned();
    let connection = db::connect(&core.db_path)?;
    let existing: Option<String> = connection
        .query_row(
            "SELECT id FROM watched_folders WHERE path=?1",
            [&folder_path],
            |r| r.get(0),
        )
        .optional()?;
    let folder_id = existing.unwrap_or_else(|| Uuid::new_v4().to_string());
    connection.execute(
        "INSERT OR IGNORE INTO watched_folders(id,path,created_at) VALUES (?1,?2,?3)",
        params![folder_id, folder_path, Utc::now().to_rfc3339()],
    )?;
    drop(connection);
    scan_folder(app, core, &folder_id)
}

pub fn scan_folder(app: &AppHandle, core: &Arc<AppCore>, folder_id: &str) -> Result<()> {
    let connection = db::connect(&core.db_path)?;
    let folder_path: String = connection.query_row(
        "SELECT path FROM watched_folders WHERE id=?1",
        [folder_id],
        |r| r.get(0),
    )?;
    connection.execute(
        "UPDATE assets SET available=0 WHERE folder_id=?1",
        [folder_id],
    )?;
    app.emit(
        "indexing://folder-started",
        event_for_folder(folder_id, "Scanning folder"),
    )
    .map_err(|e| RecallError::Message(e.to_string()))?;
    let mut seen = 0usize;
    for entry in WalkDir::new(&folder_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(allowed_entry)
        .filter_map(std::result::Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let Some(extension) = supported_extension(entry.path()) else {
            continue;
        };
        let metadata = entry
            .metadata()
            .map_err(|e| RecallError::Message(e.to_string()))?;
        let absolute = entry.path().canonicalize()?.to_string_lossy().into_owned();
        let relative = entry
            .path()
            .strip_prefix(&folder_path)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .into_owned();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs().to_string())
            .unwrap_or_default();
        let digest = file_hash(entry.path())?;
        let existing: Option<(String, Option<String>, String)> = connection
            .query_row(
                "SELECT id,sha256,status FROM assets WHERE absolute_path=?1",
                [&absolute],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let changed = existing
            .as_ref()
            .map(|(_, hash, status)| {
                hash.as_deref() != Some(digest.as_str()) || status != "indexed"
            })
            .unwrap_or(true);
        let asset_id = existing
            .map(|value| value.0)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        connection.execute(r#"INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,sha256,status,available)
          VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,'pending',1)
          ON CONFLICT(absolute_path) DO UPDATE SET folder_id=excluded.folder_id,relative_path=excluded.relative_path,filename=excluded.filename,extension=excluded.extension,mime_type=excluded.mime_type,size_bytes=excluded.size_bytes,modified_at=excluded.modified_at,sha256=excluded.sha256,available=1,status=CASE WHEN assets.sha256<>excluded.sha256 THEN 'pending' ELSE assets.status END"#,
          params![asset_id,folder_id,absolute,relative,entry.file_name().to_string_lossy(),extension,mime_guess::from_path(entry.path()).first_or_octet_stream().essence_str(),metadata.len() as i64,modified,digest])?;
        if changed {
            let now = Utc::now().to_rfc3339();
            connection.execute("INSERT INTO indexing_jobs(id,asset_id,stage,state,created_at,updated_at) VALUES (?1,?2,'index','pending',?3,?3) ON CONFLICT(asset_id) DO UPDATE SET state='pending',error_message=NULL,updated_at=excluded.updated_at", params![Uuid::new_v4().to_string(),asset_id,now])?;
        }
        seen += 1;
        if seen % 50 == 0 {
            let _ = app.emit(
                "indexing://file-progress",
                IndexingEvent {
                    folder_id: Some(folder_id.into()),
                    asset_id: None,
                    filename: None,
                    completed: Some(seen),
                    total: None,
                    message: Some("Discovering files".into()),
                },
            );
        }
    }
    let missing_ids: Vec<String> = {
        let mut statement =
            connection.prepare("SELECT id FROM assets WHERE folder_id=?1 AND available=0")?;
        {
            let rows = statement.query_map([folder_id], |r| r.get(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    for id in missing_ids {
        connection.execute("UPDATE indexing_jobs SET state='cancelled',error_message='Source file is unavailable' WHERE asset_id=?1", [id])?;
    }
    app.emit(
        "indexing://folder-completed",
        IndexingEvent {
            folder_id: Some(folder_id.into()),
            asset_id: None,
            filename: None,
            completed: Some(seen),
            total: Some(seen),
            message: Some("Folder scan complete".into()),
        },
    )
    .map_err(|e| RecallError::Message(e.to_string()))?;
    drop(connection);
    start_worker(app.clone(), core.clone());
    Ok(())
}

pub fn start_worker(app: AppHandle, core: Arc<AppCore>) {
    if core
        .worker_running
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    tauri::async_runtime::spawn_blocking(move || {
        loop {
            if core.paused.load(Ordering::SeqCst) {
                break;
            }
            let next = match db::claim_next_job(&core.db_path) {
                Ok(value) => value,
                Err(_) => break,
            };
            let Some((job_id, asset)) = next else {
                break;
            };
            let _ = app.emit(
                "indexing://file-started",
                IndexingEvent {
                    folder_id: Some(asset.folder_id.clone()),
                    asset_id: Some(asset.id.clone()),
                    filename: Some(asset.filename.clone()),
                    completed: None,
                    total: None,
                    message: None,
                },
            );
            let ai = core.ai.read().clone();
            match extract::process_file(
                Path::new(&asset.absolute_path),
                &asset.extension,
                ai.as_deref(),
                &core.thumbnail_dir,
                &asset.id,
            ) {
                Ok(ProcessOutcome::Chunks(chunks)) => {
                    match db::save_chunks(&core.db_path, &job_id, &asset.id, &chunks) {
                        Ok(()) => {
                            let _ = app.emit(
                                "indexing://file-completed",
                                IndexingEvent {
                                    folder_id: Some(asset.folder_id),
                                    asset_id: Some(asset.id),
                                    filename: Some(asset.filename),
                                    completed: None,
                                    total: None,
                                    message: None,
                                },
                            );
                        }
                        Err(error) => fail(
                            &app,
                            &core,
                            &job_id,
                            &asset.id,
                            &asset.filename,
                            error.to_string(),
                        ),
                    }
                }
                Ok(ProcessOutcome::Skipped(reason)) => {
                    let _ =
                        db::mark_job(&core.db_path, &job_id, &asset.id, "skipped", Some(&reason));
                }
                Ok(ProcessOutcome::ModelRequired) => {
                    let _ = db::mark_job(
                        &core.db_path,
                        &job_id,
                        &asset.id,
                        "pending",
                        Some("Local OCR model required"),
                    );
                    break;
                }
                Err(error) => fail(
                    &app,
                    &core,
                    &job_id,
                    &asset.id,
                    &asset.filename,
                    error.to_string(),
                ),
            }
        }
        core.worker_running.store(false, Ordering::SeqCst);
        let _ = app.emit(
            "indexing://queue-state",
            IndexingEvent {
                folder_id: None,
                asset_id: None,
                filename: None,
                completed: None,
                total: None,
                message: Some("Queue idle".into()),
            },
        );
    });
}

fn fail(
    app: &AppHandle,
    core: &AppCore,
    job_id: &str,
    asset_id: &str,
    filename: &str,
    message: String,
) {
    let _ = db::mark_job(&core.db_path, job_id, asset_id, "failed", Some(&message));
    let _ = app.emit(
        "indexing://file-failed",
        IndexingEvent {
            folder_id: None,
            asset_id: Some(asset_id.into()),
            filename: Some(filename.into()),
            completed: None,
            total: None,
            message: Some(message),
        },
    );
}

fn event_for_folder(folder_id: &str, message: &str) -> IndexingEvent {
    IndexingEvent {
        folder_id: Some(folder_id.into()),
        asset_id: None,
        filename: None,
        completed: None,
        total: None,
        message: Some(message.into()),
    }
}

pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn remove_folder(core: &AppCore, folder_id: &str) -> Result<()> {
    let connection = db::connect(&core.db_path)?;
    let asset_ids: HashSet<String> = {
        let mut statement = connection.prepare("SELECT id FROM assets WHERE folder_id=?1")?;
        {
            let rows = statement.query_map([folder_id], |r| r.get(0))?;
            rows.collect::<std::result::Result<HashSet<_>, _>>()?
        }
    };
    for id in &asset_ids {
        let _ = std::fs::remove_file(core.thumbnail_dir.join(format!("{id}.png")));
    }
    let chunk_ids: Vec<String> = {
        let mut statement = connection.prepare(
            "SELECT c.id FROM chunks c JOIN assets a ON a.id=c.asset_id WHERE a.folder_id=?1",
        )?;
        {
            let rows = statement.query_map([folder_id], |r| r.get(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    for id in chunk_ids {
        connection.execute("DELETE FROM chunks_fts WHERE chunk_id=?1", [id])?;
    }
    connection.execute("DELETE FROM watched_folders WHERE id=?1", [folder_id])?;
    Ok(())
}

pub fn list_folders(core: &AppCore) -> Result<Vec<WatchedFolder>> {
    db::list_folders(&core.db_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supports_only_planned_extensions() {
        assert_eq!(
            supported_extension(Path::new("note.MD")).as_deref(),
            Some("md")
        );
        assert!(supported_extension(Path::new("sheet.xlsx")).is_none());
    }
}
