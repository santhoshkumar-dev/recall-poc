use std::path::Path;

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    error::Result,
    types::{AssetRecord, AssetSummary, BootstrapState, ChunkInput, IndexingStatus, WatchedFolder},
};

pub fn connect(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")?;
    Ok(connection)
}

pub fn migrate(path: &Path) -> Result<()> {
    let connection = connect(path)?;
    connection.execute_batch(r#"
        CREATE TABLE IF NOT EXISTS watched_folders (
          id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets (
          id TEXT PRIMARY KEY,
          folder_id TEXT NOT NULL REFERENCES watched_folders(id) ON DELETE CASCADE,
          absolute_path TEXT NOT NULL UNIQUE,
          relative_path TEXT NOT NULL,
          filename TEXT NOT NULL,
          extension TEXT,
          mime_type TEXT,
          size_bytes INTEGER NOT NULL,
          modified_at TEXT NOT NULL,
          sha256 TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          available INTEGER NOT NULL DEFAULT 1,
          error_message TEXT,
          thumbnail_path TEXT,
          indexed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_assets_folder ON assets(folder_id);
        CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL,
          page_number INTEGER,
          text TEXT NOT NULL,
          embedding BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_asset ON chunks(asset_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS indexing_jobs (
          id TEXT PRIMARY KEY,
          asset_id TEXT NOT NULL UNIQUE REFERENCES assets(id) ON DELETE CASCADE,
          stage TEXT NOT NULL DEFAULT 'index',
          state TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          error_message TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_state ON indexing_jobs(state, created_at);
        CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT OR IGNORE INTO app_settings(key, value) VALUES ('model_state', 'missing');
        INSERT OR IGNORE INTO app_settings(key, value) VALUES ('queue_paused', 'false');
        UPDATE indexing_jobs SET state='pending', error_message='Recovered after application restart' WHERE state='processing';
        UPDATE assets SET status='pending' WHERE status='processing';
    "#)?;
    Ok(())
}

pub fn setting(connection: &Connection, key: &str, default: &str) -> Result<String> {
    Ok(connection
        .query_row(
            "SELECT value FROM app_settings WHERE key=?1",
            [key],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or_else(|| default.to_owned()))
}

pub fn set_setting(connection: &Connection, key: &str, value: &str) -> Result<()> {
    connection.execute("INSERT INTO app_settings(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value])?;
    Ok(())
}

pub fn bootstrap(path: &Path) -> Result<BootstrapState> {
    let connection = connect(path)?;
    Ok(BootstrapState {
        database_ready: true,
        model_state: setting(&connection, "model_state", "missing")?,
        folders: connection.query_row("SELECT COUNT(*) FROM watched_folders", [], |r| r.get(0))?,
        indexed_files: connection.query_row(
            "SELECT COUNT(*) FROM assets WHERE status='indexed' AND available=1",
            [],
            |r| r.get(0),
        )?,
        queue_paused: setting(&connection, "queue_paused", "false")? == "true",
    })
}

pub fn list_folders(path: &Path) -> Result<Vec<WatchedFolder>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(r#"
      SELECT f.id, f.path, f.created_at,
        COUNT(a.id), COALESCE(SUM(CASE WHEN a.status='indexed' AND a.available=1 THEN 1 ELSE 0 END), 0)
      FROM watched_folders f LEFT JOIN assets a ON a.folder_id=f.id AND a.available=1
      GROUP BY f.id ORDER BY f.created_at
    "#)?;
    let rows = statement.query_map([], |r| {
        Ok(WatchedFolder {
            id: r.get(0)?,
            path: r.get(1)?,
            created_at: r.get(2)?,
            available_files: r.get(3)?,
            indexed_files: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn indexing_status(path: &Path) -> Result<IndexingStatus> {
    let connection = connect(path)?;
    let count = |state: &str| -> rusqlite::Result<i64> {
        connection.query_row(
            "SELECT COUNT(*) FROM indexing_jobs WHERE state=?1",
            [state],
            |r| r.get(0),
        )
    };
    let current_file = connection.query_row("SELECT a.filename FROM indexing_jobs j JOIN assets a ON a.id=j.asset_id WHERE j.state='processing' LIMIT 1", [], |r| r.get(0)).optional()?;
    Ok(IndexingStatus {
        paused: setting(&connection, "queue_paused", "false")? == "true",
        pending: count("pending")?,
        processing: count("processing")?,
        indexed: count("indexed")?,
        skipped: count("skipped")?,
        failed: count("failed")?,
        current_file,
    })
}

pub fn list_recent_assets(path: &Path, limit: i64) -> Result<Vec<AssetSummary>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare("SELECT id,filename,extension,absolute_path,status,error_message,indexed_at FROM assets WHERE available=1 ORDER BY COALESCE(indexed_at, modified_at) DESC LIMIT ?1")?;
    let rows = statement.query_map([limit.clamp(1, 100)], |r| {
        Ok(AssetSummary {
            id: r.get(0)?,
            filename: r.get(1)?,
            extension: r.get(2)?,
            source_path: r.get(3)?,
            status: r.get(4)?,
            error_message: r.get(5)?,
            indexed_at: r.get(6)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn claim_next_job(path: &Path) -> Result<Option<(String, AssetRecord)>> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let candidate: Option<(String, AssetRecord)> = transaction
        .query_row(
            r#"
      SELECT j.id,a.id,a.folder_id,a.absolute_path,a.filename,COALESCE(a.extension,'')
      FROM indexing_jobs j JOIN assets a ON a.id=j.asset_id
      WHERE j.state='pending' AND a.available=1 ORDER BY j.created_at LIMIT 1
    "#,
            [],
            |r| {
                Ok((
                    r.get(0)?,
                    AssetRecord {
                        id: r.get(1)?,
                        folder_id: r.get(2)?,
                        absolute_path: r.get(3)?,
                        filename: r.get(4)?,
                        extension: r.get(5)?,
                    },
                ))
            },
        )
        .optional()?;
    if let Some((job_id, asset)) = &candidate {
        let now = Utc::now().to_rfc3339();
        transaction.execute("UPDATE indexing_jobs SET state='processing', attempts=attempts+1, updated_at=?2 WHERE id=?1", params![job_id, now])?;
        transaction.execute(
            "UPDATE assets SET status='processing', error_message=NULL WHERE id=?1",
            [asset.id.as_str()],
        )?;
    }
    transaction.commit()?;
    Ok(candidate)
}

pub fn save_chunks(path: &Path, job_id: &str, asset_id: &str, chunks: &[ChunkInput]) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let old_ids: Vec<String> = {
        let mut statement = transaction.prepare("SELECT id FROM chunks WHERE asset_id=?1")?;
        let rows = statement.query_map([asset_id], |r| r.get(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for id in old_ids {
        transaction.execute("DELETE FROM chunks_fts WHERE chunk_id=?1", [id])?;
    }
    transaction.execute("DELETE FROM chunks WHERE asset_id=?1", [asset_id])?;
    for chunk in chunks {
        let id = uuid::Uuid::new_v4().to_string();
        let blob = chunk
            .embedding
            .as_ref()
            .map(|values| embedding_to_blob(values));
        transaction.execute("INSERT INTO chunks(id,asset_id,chunk_index,page_number,text,embedding) VALUES (?1,?2,?3,?4,?5,?6)", params![id,asset_id,chunk.index,chunk.page_number,chunk.text,blob])?;
        transaction.execute(
            "INSERT INTO chunks_fts(chunk_id,text) VALUES (?1,?2)",
            params![id, chunk.text],
        )?;
    }
    let now = Utc::now().to_rfc3339();
    transaction.execute(
        "UPDATE assets SET status='indexed', indexed_at=?2, error_message=NULL WHERE id=?1",
        params![asset_id, now],
    )?;
    transaction.execute(
        "UPDATE indexing_jobs SET state='indexed', updated_at=?2, error_message=NULL WHERE id=?1",
        params![job_id, now],
    )?;
    transaction.commit()?;
    Ok(())
}

pub fn mark_job(
    path: &Path,
    job_id: &str,
    asset_id: &str,
    state: &str,
    message: Option<&str>,
) -> Result<()> {
    let connection = connect(path)?;
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "UPDATE indexing_jobs SET state=?2,error_message=?3,updated_at=?4 WHERE id=?1",
        params![job_id, state, message, now],
    )?;
    connection.execute(
        "UPDATE assets SET status=?2,error_message=?3 WHERE id=?1",
        params![asset_id, state, message],
    )?;
    Ok(())
}

pub fn embedding_to_blob(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_blob_round_trip() {
        let source = vec![0.25, -2.5, 9.75];
        assert_eq!(blob_to_embedding(&embedding_to_blob(&source)), source);
    }
}
