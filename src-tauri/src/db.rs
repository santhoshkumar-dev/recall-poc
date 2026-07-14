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

/// Latest schema version. Bump when adding a migration step below.
const SCHEMA_VERSION: i64 = 1;

pub fn migrate(path: &Path) -> Result<()> {
    let connection = connect(path)?;

    // v0 baseline: idempotent core schema, always safe to re-run.
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

    // Versioned migrations for additive schema changes on existing installs.
    let mut version: i64 = connection.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version < 1 {
        connection.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS image_embeddings (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              model_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              page_number INTEGER NOT NULL DEFAULT -1,
              dimensions INTEGER NOT NULL,
              embedding BLOB NOT NULL,
              indexed_at TEXT NOT NULL,
              PRIMARY KEY (asset_id, model_id, page_number)
            );
            CREATE INDEX IF NOT EXISTS idx_image_emb_model ON image_embeddings(model_id);
            CREATE TABLE IF NOT EXISTS visual_classifications (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              model_id TEXT NOT NULL,
              label TEXT NOT NULL,
              score REAL NOT NULL,
              rank INTEGER NOT NULL,
              PRIMARY KEY (asset_id, model_id, label)
            );
            CREATE INDEX IF NOT EXISTS idx_visclass_asset ON visual_classifications(asset_id);
            CREATE TABLE IF NOT EXISTS asset_metadata (
              asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
              dates_json TEXT, times_json TEXT, amounts_json TEXT,
              urls_json TEXT, emails_json TEXT, phone_numbers_json TEXT,
              identifiers_json TEXT, metadata_json TEXT,
              updated_at TEXT NOT NULL
            );
        "#)?;
        version = 1;
    }
    let _ = version;
    connection.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
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

pub fn claim_next_job(path: &Path) -> Result<Option<(String, String, AssetRecord)>> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let candidate: Option<(String, String, AssetRecord)> = transaction
        .query_row(
            r#"
      SELECT j.id,j.stage,a.id,a.folder_id,a.absolute_path,a.filename,COALESCE(a.extension,'')
      FROM indexing_jobs j JOIN assets a ON a.id=j.asset_id
      WHERE j.state='pending' AND a.available=1 ORDER BY j.created_at LIMIT 1
    "#,
            [],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    AssetRecord {
                        id: r.get(2)?,
                        folder_id: r.get(3)?,
                        absolute_path: r.get(4)?,
                        filename: r.get(5)?,
                        extension: r.get(6)?,
                    },
                ))
            },
        )
        .optional()?;
    if let Some((job_id, _stage, asset)) = &candidate {
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

/// Concatenated text of an asset's chunks (reading order), for metadata/summary.
pub fn asset_text(path: &Path, asset_id: &str) -> Result<String> {
    let connection = connect(path)?;
    let mut statement = connection
        .prepare("SELECT text FROM chunks WHERE asset_id=?1 ORDER BY chunk_index")?;
    let rows = statement.query_map([asset_id], |r| r.get::<_, String>(0))?;
    let texts = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(texts.join("\n"))
}

/// Append a single extra chunk (e.g. a structured summary) without wiping the
/// asset's existing chunks. Kept in sync with the FTS mirror.
pub fn append_chunk(path: &Path, asset_id: &str, text: &str, embedding: Option<&[f32]>) -> Result<()> {
    let connection = connect(path)?;
    let id = uuid::Uuid::new_v4().to_string();
    let next_index: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(chunk_index)+1,0) FROM chunks WHERE asset_id=?1",
            [asset_id],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let blob = embedding.map(embedding_to_blob);
    connection.execute(
        "INSERT INTO chunks(id,asset_id,chunk_index,page_number,text,embedding) VALUES (?1,?2,?3,NULL,?4,?5)",
        params![id, asset_id, next_index, text, blob],
    )?;
    connection.execute(
        "INSERT INTO chunks_fts(chunk_id,text) VALUES (?1,?2)",
        params![id, text],
    )?;
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

pub fn queue_model_reindex(path: &Path, ocr_changed: bool, embedding_changed: bool) -> Result<()> {
    if !ocr_changed && !embedding_changed {
        return Ok(());
    }
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let now = Utc::now().to_rfc3339();
    let predicate = if embedding_changed {
        "available=1"
    } else {
        "available=1 AND LOWER(COALESCE(extension,'')) IN ('png','jpg','jpeg','webp')"
    };
    transaction.execute(
        &format!("UPDATE assets SET status='pending', error_message=NULL WHERE {predicate}"),
        [],
    )?;
    transaction.execute(
        &format!(
            "UPDATE indexing_jobs
             SET state='pending', stage='index', error_message=NULL, updated_at=?1
             WHERE asset_id IN (SELECT id FROM assets WHERE {predicate})"
        ),
        [now],
    )?;
    transaction.commit()?;
    Ok(())
}

const IMAGE_PREDICATE: &str =
    "available=1 AND LOWER(COALESCE(extension,'')) IN ('png','jpg','jpeg','webp')";

/// Re-queue only image assets for a visual-only pipeline stage.
/// `stage` is 'visual' (regenerate image embedding + classification + metadata,
/// reusing existing text chunks) or 'recategorize' (recompute classification
/// from cached image embeddings, no image encoder rerun).
pub fn queue_visual_reindex(path: &Path, stage: &str) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let now = Utc::now().to_rfc3339();
    transaction.execute(
        &format!("UPDATE assets SET status='pending', error_message=NULL WHERE {IMAGE_PREDICATE}"),
        [],
    )?;
    transaction.execute(
        &format!(
            "UPDATE indexing_jobs
             SET state='pending', stage=?1, error_message=NULL, updated_at=?2
             WHERE asset_id IN (SELECT id FROM assets WHERE {IMAGE_PREDICATE})"
        ),
        params![stage, now],
    )?;
    transaction.commit()?;
    Ok(())
}

/// Whole-image vectors use page_number = -1; PDF pages use their 1-based number.
pub const WHOLE_IMAGE_PAGE: i64 = -1;

pub fn save_image_embedding(
    path: &Path,
    asset_id: &str,
    model_id: &str,
    model_version: &str,
    page_number: i64,
    embedding: &[f32],
) -> Result<()> {
    let connection = connect(path)?;
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "INSERT INTO image_embeddings(asset_id,model_id,model_version,page_number,dimensions,embedding,indexed_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7)
         ON CONFLICT(asset_id,model_id,page_number)
         DO UPDATE SET model_version=excluded.model_version, dimensions=excluded.dimensions,
                       embedding=excluded.embedding, indexed_at=excluded.indexed_at",
        params![
            asset_id,
            model_id,
            model_version,
            page_number,
            embedding.len() as i64,
            embedding_to_blob(embedding),
            now
        ],
    )?;
    Ok(())
}

/// (asset_id, page_number, embedding) for every stored image vector of a model.
pub fn load_image_embeddings(
    path: &Path,
    model_id: &str,
) -> Result<Vec<(String, i64, Vec<f32>)>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT ie.asset_id, ie.page_number, ie.embedding
         FROM image_embeddings ie JOIN assets a ON a.id=ie.asset_id
         WHERE ie.model_id=?1 AND a.available=1 AND a.status='indexed'",
    )?;
    let rows = statement.query_map([model_id], |r| {
        let blob: Vec<u8> = r.get(2)?;
        Ok((r.get(0)?, r.get(1)?, blob_to_embedding(&blob)))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn image_embedding_for(
    path: &Path,
    asset_id: &str,
    model_id: &str,
) -> Result<Option<Vec<f32>>> {
    let connection = connect(path)?;
    connection
        .query_row(
            "SELECT embedding FROM image_embeddings WHERE asset_id=?1 AND model_id=?2 AND page_number=?3",
            params![asset_id, model_id, WHOLE_IMAGE_PAGE],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .optional()?
        .map(|blob| Ok(blob_to_embedding(&blob)))
        .transpose()
}

/// Replace an asset's visual classifications; `scored` is ordered best-first.
pub fn save_visual_classifications(
    path: &Path,
    asset_id: &str,
    model_id: &str,
    scored: &[crate::types::VisualCategory],
) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "DELETE FROM visual_classifications WHERE asset_id=?1 AND model_id=?2",
        params![asset_id, model_id],
    )?;
    for (rank, category) in scored.iter().enumerate() {
        transaction.execute(
            "INSERT INTO visual_classifications(asset_id,model_id,label,score,rank)
             VALUES (?1,?2,?3,?4,?5)",
            params![asset_id, model_id, category.label, category.score, rank as i64],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

pub fn classifications_for(
    path: &Path,
    asset_id: &str,
    model_id: &str,
) -> Result<Vec<crate::types::VisualCategory>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT label, score FROM visual_classifications
         WHERE asset_id=?1 AND model_id=?2 ORDER BY rank",
    )?;
    let rows = statement.query_map(params![asset_id, model_id], |r| {
        Ok(crate::types::VisualCategory {
            label: r.get(0)?,
            score: r.get(1)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Assets whose top visual categories include any of `labels`, best score first.
pub fn assets_by_categories(
    path: &Path,
    model_id: &str,
    labels: &[String],
    limit: i64,
) -> Result<Vec<(String, f32)>> {
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    let connection = connect(path)?;
    let placeholders = labels.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT vc.asset_id, MAX(vc.score) FROM visual_classifications vc
         JOIN assets a ON a.id=vc.asset_id
         WHERE vc.model_id=? AND a.available=1 AND a.status='indexed'
           AND vc.label IN ({placeholders})
         GROUP BY vc.asset_id ORDER BY MAX(vc.score) DESC LIMIT ?"
    );
    let mut statement = connection.prepare(&sql)?;
    let mut binds: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(labels.len() + 2);
    binds.push(&model_id);
    for label in labels {
        binds.push(label);
    }
    binds.push(&limit);
    let rows = statement.query_map(binds.as_slice(), |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn save_asset_metadata(
    path: &Path,
    asset_id: &str,
    metadata: &crate::types::ExtractedMetadata,
) -> Result<()> {
    let connection = connect(path)?;
    let now = Utc::now().to_rfc3339();
    let dates = serde_json::to_string(&metadata.dates)?;
    let times = serde_json::to_string(&metadata.times)?;
    let amounts = serde_json::to_string(&metadata.amounts)?;
    let urls = serde_json::to_string(&metadata.urls)?;
    let emails = serde_json::to_string(&metadata.emails)?;
    let phones = serde_json::to_string(&metadata.phone_numbers)?;
    let identifiers = serde_json::to_string(&metadata.identifiers)?;
    let full = serde_json::to_string(metadata)?;
    connection.execute(
        "INSERT INTO asset_metadata(asset_id,dates_json,times_json,amounts_json,urls_json,emails_json,phone_numbers_json,identifiers_json,metadata_json,updated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
         ON CONFLICT(asset_id) DO UPDATE SET
           dates_json=excluded.dates_json, times_json=excluded.times_json,
           amounts_json=excluded.amounts_json, urls_json=excluded.urls_json,
           emails_json=excluded.emails_json, phone_numbers_json=excluded.phone_numbers_json,
           identifiers_json=excluded.identifiers_json, metadata_json=excluded.metadata_json,
           updated_at=excluded.updated_at",
        params![asset_id, dates, times, amounts, urls, emails, phones, identifiers, full, now],
    )?;
    Ok(())
}

/// Coverage counts for the visual subsystem:
/// (image_assets, images_indexed, images_with_embeddings, images_classified).
pub fn visual_counts(path: &Path, model_id: &str) -> Result<(i64, i64, i64, i64)> {
    let connection = connect(path)?;
    let one = |sql: &str, bind: &[&dyn rusqlite::ToSql]| -> Result<i64> {
        Ok(connection.query_row(sql, bind, |r| r.get(0))?)
    };
    let image_assets = one(
        &format!("SELECT COUNT(*) FROM assets WHERE {IMAGE_PREDICATE}"),
        &[],
    )?;
    let images_indexed = one(
        &format!("SELECT COUNT(*) FROM assets WHERE {IMAGE_PREDICATE} AND status='indexed'"),
        &[],
    )?;
    let with_embeddings = one(
        "SELECT COUNT(DISTINCT asset_id) FROM image_embeddings WHERE model_id=?1",
        &[&model_id],
    )?;
    let classified = one(
        "SELECT COUNT(DISTINCT asset_id) FROM visual_classifications WHERE model_id=?1",
        &[&model_id],
    )?;
    Ok((image_assets, images_indexed, with_embeddings, classified))
}

/// All indexed, available assets as lightweight briefs (id → filename/path).
pub fn indexed_asset_briefs(path: &Path) -> Result<Vec<crate::types::AssetBrief>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT id, folder_id, filename, extension, absolute_path
         FROM assets WHERE available=1 AND status='indexed'",
    )?;
    let rows = statement.query_map([], |r| {
        Ok(crate::types::AssetBrief {
            id: r.get(0)?,
            folder_id: r.get(1)?,
            filename: r.get(2)?,
            extension: r.get(3)?,
            source_path: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn metadata_for(
    path: &Path,
    asset_id: &str,
) -> Result<Option<crate::types::ExtractedMetadata>> {
    let connection = connect(path)?;
    connection
        .query_row(
            "SELECT metadata_json FROM asset_metadata WHERE asset_id=?1",
            [asset_id],
            |r| r.get::<_, String>(0),
        )
        .optional()?
        .map(|json| serde_json::from_str(&json).map_err(Into::into))
        .transpose()
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
