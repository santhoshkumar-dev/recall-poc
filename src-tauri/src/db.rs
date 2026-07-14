use std::path::Path;

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    error::Result,
    types::{
        AssetRecord, AssetSummary, BootstrapState, ChunkInput, IndexingStatus, VisualTag,
        WatchedFolder,
    },
};

pub fn connect(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")?;
    Ok(connection)
}

/// Latest schema version. Bump when adding a migration step below.
const SCHEMA_VERSION: i64 = 6;

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
          embedding BLOB,
          source TEXT
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
        connection.execute_batch(
            r#"
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
        "#,
        )?;
        version = 1;
    }
    if version < 2 {
        connection.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS content_items (
              id TEXT PRIMARY KEY,
              sha256 TEXT NOT NULL UNIQUE,
              first_seen_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS extraction_provenance (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              job_id TEXT REFERENCES indexing_jobs(id) ON DELETE SET NULL,
              stage TEXT NOT NULL,
              extractor_id TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              source_index INTEGER,
              confidence REAL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_provenance_asset ON extraction_provenance(asset_id, stage);
            CREATE TABLE IF NOT EXISTS document_classifications (
              asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
              document_type TEXT NOT NULL,
              confidence REAL NOT NULL,
              evidence_json TEXT NOT NULL,
              provenance_id TEXT REFERENCES extraction_provenance(id) ON DELETE SET NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_docclass_type ON document_classifications(document_type, confidence);
            CREATE TABLE IF NOT EXISTS extracted_entities (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              entity_type TEXT NOT NULL,
              raw_value TEXT NOT NULL,
              normalized_value TEXT NOT NULL,
              confidence REAL NOT NULL,
              provenance_id TEXT REFERENCES extraction_provenance(id) ON DELETE SET NULL,
              UNIQUE(asset_id, entity_type, normalized_value)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_value ON extracted_entities(normalized_value, entity_type);
        "#,
        )?;
        if !column_exists(&connection, "assets", "content_id")? {
            connection.execute_batch(
                "ALTER TABLE assets ADD COLUMN content_id TEXT REFERENCES content_items(id);",
            )?;
        }
        connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_assets_content ON assets(content_id);",
        )?;
        // Backfill stable content identities for databases created before v2.
        let mut statement = connection.prepare(
            "SELECT id, sha256 FROM assets WHERE sha256 IS NOT NULL AND content_id IS NULL",
        )?;
        let rows: Vec<(String, String)> = statement
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        drop(statement);
        for (asset_id, sha256) in rows {
            let content_id = ensure_content_item(&connection, &sha256)?;
            connection.execute(
                "UPDATE assets SET content_id=?2 WHERE id=?1",
                params![asset_id, content_id],
            )?;
        }
        version = 2;
    }
    if version < 3 {
        connection.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS image_embedding_regions (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              model_id TEXT NOT NULL,
              region_id INTEGER NOT NULL,
              region_kind TEXT NOT NULL,
              x INTEGER NOT NULL, y INTEGER NOT NULL,
              width INTEGER NOT NULL, height INTEGER NOT NULL,
              source_width INTEGER NOT NULL, source_height INTEGER NOT NULL,
              pipeline_version TEXT NOT NULL,
              indexed_at TEXT NOT NULL,
              PRIMARY KEY (asset_id, model_id, region_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_regions_asset ON image_embedding_regions(asset_id, model_id);
        "#,
        )?;
        version = 3;
    }
    if version < 4 {
        connection.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS extraction_stage_jobs (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              stage TEXT NOT NULL,
              model_id TEXT NOT NULL,
              pipeline_version TEXT NOT NULL,
              state TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(asset_id, stage, model_id, pipeline_version)
            );
            CREATE INDEX IF NOT EXISTS idx_stage_jobs_state ON extraction_stage_jobs(state, created_at);
            CREATE INDEX IF NOT EXISTS idx_stage_jobs_asset ON extraction_stage_jobs(asset_id, stage);
        "#,
        )?;
        version = 4;
    }
    if version < 5 {
        // Stage jobs are independent of source indexing jobs. Keep both links
        // so old provenance remains valid while derived-stage outputs can point
        // at the job that actually produced them.
        if !column_exists(&connection, "extraction_provenance", "stage_job_id")? {
            connection.execute_batch(
                "ALTER TABLE extraction_provenance ADD COLUMN stage_job_id TEXT REFERENCES extraction_stage_jobs(id) ON DELETE SET NULL;",
            )?;
        }
        connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_provenance_stage_job ON extraction_provenance(stage_job_id);",
        )?;
        version = 5;
    }
    if version < 6 {
        if !column_exists(&connection, "chunks", "source")? {
            connection.execute_batch("ALTER TABLE chunks ADD COLUMN source TEXT;")?;
        }
        connection.execute_batch(
            r#"
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_asset_source ON chunks(asset_id, source) WHERE source IS NOT NULL;
            CREATE TABLE IF NOT EXISTS visual_tags (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              model_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              region_id INTEGER NOT NULL,
              namespace TEXT NOT NULL,
              label TEXT NOT NULL,
              normalized_label TEXT NOT NULL,
              confidence REAL NOT NULL,
              rank INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (asset_id, model_id, region_id, normalized_label)
            );
            CREATE INDEX IF NOT EXISTS idx_visual_tags_label ON visual_tags(normalized_label, confidence);
            CREATE INDEX IF NOT EXISTS idx_visual_tags_asset_model ON visual_tags(asset_id, model_id);
            CREATE INDEX IF NOT EXISTS idx_visual_tags_model_version ON visual_tags(model_id, model_version);
        "#,
        )?;
        version = 6;
    }
    let _ = version;
    connection.execute(
        "UPDATE extraction_stage_jobs SET state='pending', error_message='Recovered after application restart' WHERE state='processing'",
        [],
    )?;
    connection.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
    Ok(())
}

fn column_exists(connection: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row?.eq_ignore_ascii_case(column) {
            return Ok(true);
        }
    }
    Ok(false)
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

/// Return a stable content identity for a SHA-256 digest. File paths stay on
/// `assets`; the identity survives a rename or move within a watched folder.
pub fn ensure_content_item(connection: &Connection, sha256: &str) -> Result<String> {
    let existing: Option<String> = connection
        .query_row(
            "SELECT id FROM content_items WHERE sha256=?1",
            [sha256],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        connection.execute(
            "UPDATE content_items SET updated_at=?2 WHERE id=?1",
            params![id, Utc::now().to_rfc3339()],
        )?;
        return Ok(id);
    }
    let id = uuid::Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "INSERT OR IGNORE INTO content_items(id,sha256,first_seen_at,updated_at) VALUES (?1,?2,?3,?3)",
        params![id, sha256, now],
    )?;
    Ok(connection.query_row(
        "SELECT id FROM content_items WHERE sha256=?1",
        [sha256],
        |r| r.get(0),
    )?)
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
    let asset_count = |state: &str| -> rusqlite::Result<i64> {
        connection.query_row(
            "SELECT COUNT(*) FROM assets WHERE available=1 AND status=?1",
            [state],
            |r| r.get::<_, i64>(0),
        )
    };
    let processing = connection.query_row(
        "SELECT COUNT(DISTINCT asset_id) FROM (
           SELECT asset_id FROM indexing_jobs WHERE state='processing'
           UNION ALL
           SELECT asset_id FROM extraction_stage_jobs WHERE state='processing'
         )",
        [],
        |r| r.get::<_, i64>(0),
    )?;
    let background_pending = connection.query_row(
        "SELECT COUNT(*) FROM extraction_stage_jobs WHERE state='pending'",
        [],
        |r| r.get::<_, i64>(0),
    )?;
    let background_processing = connection.query_row(
        "SELECT COUNT(*) FROM extraction_stage_jobs WHERE state='processing'",
        [],
        |r| r.get::<_, i64>(0),
    )?;
    let failed_stages = connection.query_row(
        "SELECT COUNT(DISTINCT asset_id) FROM extraction_stage_jobs WHERE state='failed'",
        [],
        |r| r.get::<_, i64>(0),
    )?;
    let failed = asset_count("failed")? + failed_stages;
    let current = connection
        .query_row(
            "SELECT filename, stage FROM (
           SELECT a.filename, j.stage FROM indexing_jobs j JOIN assets a ON a.id=j.asset_id WHERE j.state='processing'
           UNION ALL
           SELECT a.filename, j.stage FROM extraction_stage_jobs j JOIN assets a ON a.id=j.asset_id WHERE j.state='processing'
         ) LIMIT 1",
            [],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()?;
    let (current_file, current_stage) = current
        .map(|(file, stage)| (Some(file), Some(stage)))
        .unwrap_or((None, None));
    Ok(IndexingStatus {
        paused: setting(&connection, "queue_paused", "false")? == "true",
        pending: asset_count("pending")?,
        processing,
        indexed: asset_count("indexed")?,
        skipped: asset_count("skipped")?,
        failed,
        background_pending,
        background_processing,
        current_stage,
        current_file,
    })
}

/// Internal queue counts include derived extraction stages. Use this for
/// operational guards and developer diagnostics, not for the user-facing file
/// queue cards.
pub fn active_job_counts(path: &Path) -> Result<(i64, i64, i64)> {
    let connection = connect(path)?;
    let count = |state: &str| -> rusqlite::Result<i64> {
        let source = connection.query_row(
            "SELECT COUNT(*) FROM indexing_jobs WHERE state=?1",
            [state],
            |r| r.get::<_, i64>(0),
        )?;
        let derived = connection.query_row(
            "SELECT COUNT(*) FROM extraction_stage_jobs WHERE state=?1",
            [state],
            |r| r.get::<_, i64>(0),
        )?;
        Ok(source + derived)
    };
    Ok((count("pending")?, count("processing")?, count("failed")?))
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
    if let Some((job_id, stage, asset)) = &candidate {
        let now = Utc::now().to_rfc3339();
        transaction.execute("UPDATE indexing_jobs SET state='processing', attempts=attempts+1, updated_at=?2 WHERE id=?1", params![job_id, now])?;
        if stage == "index" {
            transaction.execute(
                "UPDATE assets SET status='processing', error_message=NULL WHERE id=?1",
                [asset.id.as_str()],
            )?;
        }
    }
    transaction.commit()?;
    Ok(candidate)
}

/// Queue a single derived extraction stage. Stages are versioned by the model
/// and pipeline so an OCR/model change invalidates only its own output.
pub fn queue_extraction_stage(
    path: &Path,
    asset_id: &str,
    stage: &str,
    model_id: &str,
    pipeline_version: &str,
) -> Result<()> {
    let connection = connect(path)?;
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "INSERT INTO extraction_stage_jobs(id,asset_id,stage,model_id,pipeline_version,state,attempts,created_at,updated_at)
         VALUES (?1,?2,?3,?4,?5,'pending',0,?6,?6)
         ON CONFLICT(asset_id,stage,model_id,pipeline_version) DO UPDATE SET
           state='pending',attempts=0,error_message=NULL,updated_at=excluded.updated_at",
        params![uuid::Uuid::new_v4().to_string(), asset_id, stage, model_id, pipeline_version, now],
    )?;
    Ok(())
}

pub fn claim_next_extraction_stage_job(
    path: &Path,
) -> Result<Option<(String, String, AssetRecord)>> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let candidate = transaction.query_row(
        "SELECT j.id,j.stage,a.id,a.folder_id,a.absolute_path,a.filename,COALESCE(a.extension,'')
         FROM extraction_stage_jobs j JOIN assets a ON a.id=j.asset_id
         WHERE j.state='pending' AND a.available=1 AND a.status='indexed'
         ORDER BY CASE j.stage
           WHEN 'visual' THEN 10
           WHEN 'text_embedding' THEN 20
           WHEN 'analysis' THEN 30
           WHEN 'visual_tagging' THEN 40
           WHEN 'visual_regions' THEN 50
           WHEN 'visual_region_tagging' THEN 60
           ELSE 90
         END, j.created_at LIMIT 1",
        [],
        |row| Ok((
            row.get(0)?, row.get(1)?, AssetRecord {
                id: row.get(2)?, folder_id: row.get(3)?, absolute_path: row.get(4)?,
                filename: row.get(5)?, extension: row.get(6)?,
            },
        )),
    ).optional()?;
    if let Some((job_id, _, _)) = &candidate {
        transaction.execute(
            "UPDATE extraction_stage_jobs SET state='processing',attempts=attempts+1,updated_at=?2 WHERE id=?1",
            params![job_id, Utc::now().to_rfc3339()],
        )?;
    }
    transaction.commit()?;
    Ok(candidate)
}

pub fn mark_extraction_stage_job(
    path: &Path,
    job_id: &str,
    state: &str,
    message: Option<&str>,
) -> Result<()> {
    let connection = connect(path)?;
    connection.execute(
        "UPDATE extraction_stage_jobs SET state=?2,error_message=?3,updated_at=?4 WHERE id=?1",
        params![job_id, state, message, Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

pub fn retry_or_fail_extraction_stage_job(
    path: &Path,
    job_id: &str,
    message: &str,
) -> Result<bool> {
    let connection = connect(path)?;
    let (stage, attempts): (String, i64) = connection.query_row(
        "SELECT stage,attempts FROM extraction_stage_jobs WHERE id=?1",
        [job_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let maximum = match stage.as_str() {
        "visual" | "analysis" => 2,
        _ => 3,
    };
    let now = Utc::now().to_rfc3339();
    if attempts < maximum {
        connection.execute(
            "UPDATE extraction_stage_jobs SET state='pending',error_message=?2,updated_at=?3 WHERE id=?1",
            params![job_id, message, now],
        )?;
        Ok(true)
    } else {
        connection.execute(
            "UPDATE extraction_stage_jobs SET state='failed',error_message=?2,updated_at=?3 WHERE id=?1",
            params![job_id, message, now],
        )?;
        Ok(false)
    }
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
        transaction.execute("INSERT INTO chunks(id,asset_id,chunk_index,page_number,text,embedding,source) VALUES (?1,?2,?3,?4,?5,?6,NULL)", params![id,asset_id,chunk.index,chunk.page_number,chunk.text,blob])?;
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
    let mut statement =
        connection.prepare("SELECT text FROM chunks WHERE asset_id=?1 ORDER BY chunk_index")?;
    let rows = statement.query_map([asset_id], |r| r.get::<_, String>(0))?;
    let texts = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(texts.join("\n"))
}

pub fn chunks_without_embeddings(path: &Path, asset_id: &str) -> Result<Vec<(String, String)>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT id,text FROM chunks WHERE asset_id=?1 AND embedding IS NULL ORDER BY chunk_index",
    )?;
    let rows = statement.query_map([asset_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn save_chunk_embeddings(path: &Path, values: &[(String, Vec<f32>)]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    for (id, embedding) in values {
        transaction.execute(
            "UPDATE chunks SET embedding=?2 WHERE id=?1",
            params![id, embedding_to_blob(embedding)],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

/// Append a single extra chunk (e.g. a structured summary) without wiping the
/// asset's existing chunks. Kept in sync with the FTS mirror.
pub fn append_chunk(
    path: &Path,
    asset_id: &str,
    text: &str,
    embedding: Option<&[f32]>,
) -> Result<()> {
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
        "INSERT INTO chunks(id,asset_id,chunk_index,page_number,text,embedding,source) VALUES (?1,?2,?3,NULL,?4,?5,NULL)",
        params![id, asset_id, next_index, text, blob],
    )?;
    connection.execute(
        "INSERT INTO chunks_fts(chunk_id,text) VALUES (?1,?2)",
        params![id, text],
    )?;
    Ok(())
}

/// Replace a generated chunk by source key, keeping OCR/native chunks intact.
pub fn upsert_generated_chunk(
    path: &Path,
    asset_id: &str,
    source: &str,
    text: &str,
    embedding: Option<&[f32]>,
) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let old_ids: Vec<String> = {
        let mut statement =
            transaction.prepare("SELECT id FROM chunks WHERE asset_id=?1 AND source=?2")?;
        let rows = statement.query_map(params![asset_id, source], |row| row.get(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for id in old_ids {
        transaction.execute("DELETE FROM chunks_fts WHERE chunk_id=?1", [id])?;
    }
    transaction.execute(
        "DELETE FROM chunks WHERE asset_id=?1 AND source=?2",
        params![asset_id, source],
    )?;
    if text.trim().is_empty() {
        transaction.commit()?;
        return Ok(());
    }
    let id = uuid::Uuid::new_v4().to_string();
    let next_index: i64 = transaction
        .query_row(
            "SELECT COALESCE(MAX(chunk_index)+1,0) FROM chunks WHERE asset_id=?1",
            [asset_id],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let blob = embedding.map(embedding_to_blob);
    transaction.execute(
        "INSERT INTO chunks(id,asset_id,chunk_index,page_number,text,embedding,source)
         VALUES (?1,?2,?3,NULL,?4,?5,?6)",
        params![id, asset_id, next_index, text, blob, source],
    )?;
    transaction.execute(
        "INSERT INTO chunks_fts(chunk_id,text) VALUES (?1,?2)",
        params![id, text],
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

/// Finish a visual-only refresh job without changing the asset's searchable
/// status. Existing embeddings remain usable if the refresh fails.
pub fn mark_visual_job(
    path: &Path,
    job_id: &str,
    state: &str,
    message: Option<&str>,
) -> Result<()> {
    let connection = connect(path)?;
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "UPDATE indexing_jobs SET state=?2,error_message=?3,updated_at=?4 WHERE id=?1",
        params![job_id, state, message, now],
    )?;
    Ok(())
}

/// Requeue transient failures using a small, stage-specific retry budget.
/// Full extraction is allowed three attempts; derived visual stages two. The
/// durable job record retains the most recent failure across application restarts.
pub fn retry_or_fail_job(path: &Path, job_id: &str, asset_id: &str, message: &str) -> Result<bool> {
    let connection = connect(path)?;
    let (stage, attempts): (String, i64) = connection.query_row(
        "SELECT stage, attempts FROM indexing_jobs WHERE id=?1",
        [job_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let maximum = if stage == "index" { 3 } else { 2 };
    let now = Utc::now().to_rfc3339();
    if attempts < maximum {
        connection.execute(
            "UPDATE indexing_jobs SET state='pending',error_message=?2,updated_at=?3 WHERE id=?1",
            params![job_id, message, now],
        )?;
        if stage == "index" {
            connection.execute(
                "UPDATE assets SET status='pending',error_message=?2 WHERE id=?1",
                params![asset_id, message],
            )?;
        }
        return Ok(true);
    }
    connection.execute(
        "UPDATE indexing_jobs SET state='failed',error_message=?2,updated_at=?3 WHERE id=?1",
        params![job_id, message, now],
    )?;
    if stage == "index" {
        connection.execute(
            "UPDATE assets SET status='failed',error_message=?2 WHERE id=?1",
            params![asset_id, message],
        )?;
    }
    Ok(false)
}

pub fn queue_model_reindex(
    path: &Path,
    ocr_changed: bool,
    embedding_changed: bool,
    embedding_model_id: &str,
    embedding_pipeline_version: &str,
) -> Result<()> {
    if !ocr_changed && !embedding_changed {
        return Ok(());
    }
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let now = Utc::now().to_rfc3339();
    if ocr_changed {
        let predicate = "available=1 AND LOWER(COALESCE(extension,'')) IN ('png','jpg','jpeg','webp','gif','bmp','tif','tiff')";
        transaction.execute(
            &format!("UPDATE assets SET status='pending', error_message=NULL WHERE {predicate}"),
            [],
        )?;
        transaction.execute(
            &format!("UPDATE indexing_jobs SET state='pending', stage='index', error_message=NULL, updated_at=?1 WHERE asset_id IN (SELECT id FROM assets WHERE {predicate})"),
            [now.clone()],
        )?;
    }
    transaction.commit()?;
    // Text-model changes operate on existing OCR chunks. They neither re-run
    // OCR nor touch image vectors.
    if embedding_changed {
        let asset_ids = {
            let connection = connect(path)?;
            let mut statement = connection
                .prepare("SELECT id FROM assets WHERE available=1 AND status='indexed'")?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for asset_id in asset_ids {
            let connection = connect(path)?;
            connection.execute(
                "UPDATE chunks SET embedding=NULL WHERE asset_id=?1",
                [asset_id.as_str()],
            )?;
            drop(connection);
            queue_extraction_stage(
                path,
                &asset_id,
                "text_embedding",
                embedding_model_id,
                embedding_pipeline_version,
            )?;
        }
    }
    Ok(())
}

const IMAGE_PREDICATE: &str =
    "available=1 AND LOWER(COALESCE(extension,'')) IN ('png','jpg','jpeg','webp','gif','bmp','tif','tiff')";

/// Preferred visual refresh path for the extraction graph. Existing source and
/// OCR stages remain complete; only the requested visual-model output is made
/// pending again.
pub fn queue_visual_stage_reindex(path: &Path, model_id: &str, model_version: &str) -> Result<i64> {
    let asset_ids = {
        let connection = connect(path)?;
        let mut statement = connection.prepare(&format!(
            "SELECT id FROM assets WHERE {IMAGE_PREDICATE} AND status='indexed'"
        ))?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for asset_id in &asset_ids {
        queue_extraction_stage(path, asset_id, "visual", model_id, model_version)?;
    }
    Ok(asset_ids.len() as i64)
}

/// On startup, reindex only already-indexed images whose whole-image vector is
/// absent or was produced by an older visual pipeline version. New assets keep
/// their normal full-index job, so this never replaces OCR/text extraction.
pub fn queue_stale_visual_reindex(path: &Path, model_id: &str, model_version: &str) -> Result<i64> {
    let asset_ids = {
        let connection = connect(path)?;
        let mut statement = connection.prepare(
            "SELECT a.id FROM assets a WHERE a.available=1 AND a.status='indexed'
             AND LOWER(COALESCE(a.extension,'')) IN ('png','jpg','jpeg','webp','gif','bmp','tif','tiff')
             AND NOT EXISTS (SELECT 1 FROM image_embeddings ie WHERE ie.asset_id=a.id AND ie.model_id=?1 AND ie.model_version=?2 AND ie.page_number=?3)",
        )?;
        let rows = statement
            .query_map(params![model_id, model_version, WHOLE_IMAGE_PAGE], |row| {
                row.get::<_, String>(0)
            })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for asset_id in &asset_ids {
        queue_extraction_stage(path, asset_id, "visual", model_id, model_version)?;
    }
    Ok(asset_ids.len() as i64)
}

pub fn queue_visual_tagging_stage_reindex(
    path: &Path,
    model_id: &str,
    model_version: &str,
) -> Result<i64> {
    let asset_ids = {
        let connection = connect(path)?;
        let mut statement = connection.prepare(&format!(
            "SELECT id FROM assets WHERE {IMAGE_PREDICATE} AND status='indexed'"
        ))?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for asset_id in &asset_ids {
        queue_extraction_stage(path, asset_id, "visual_tagging", model_id, model_version)?;
    }
    Ok(asset_ids.len() as i64)
}

pub fn queue_stale_visual_tagging_reindex(
    path: &Path,
    model_id: &str,
    model_version: &str,
) -> Result<i64> {
    let asset_ids = {
        let connection = connect(path)?;
        let mut statement = connection.prepare(
            "SELECT a.id FROM assets a WHERE a.available=1 AND a.status='indexed'
             AND LOWER(COALESCE(a.extension,'')) IN ('png','jpg','jpeg','webp','gif','bmp','tif','tiff')
             AND NOT EXISTS (
               SELECT 1 FROM visual_tags vt
               WHERE vt.asset_id=a.id AND vt.model_id=?1 AND vt.model_version=?2
             )",
        )?;
        let rows = statement.query_map(params![model_id, model_version], |row| {
            row.get::<_, String>(0)
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for asset_id in &asset_ids {
        queue_extraction_stage(path, asset_id, "visual_tagging", model_id, model_version)?;
    }
    Ok(asset_ids.len() as i64)
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

pub fn delete_image_embeddings(path: &Path, asset_id: &str, model_id: &str) -> Result<()> {
    let connection = connect(path)?;
    connection.execute(
        "DELETE FROM image_embedding_regions WHERE asset_id=?1 AND model_id=?2",
        params![asset_id, model_id],
    )?;
    connection.execute(
        "DELETE FROM image_embeddings WHERE asset_id=?1 AND model_id=?2",
        params![asset_id, model_id],
    )?;
    Ok(())
}

pub fn save_image_embedding_region(
    path: &Path,
    asset_id: &str,
    model_id: &str,
    region: &crate::extract::ImageRegion,
    pipeline_version: &str,
) -> Result<()> {
    let connection = connect(path)?;
    connection.execute(
        "INSERT INTO image_embedding_regions(asset_id,model_id,region_id,region_kind,x,y,width,height,source_width,source_height,pipeline_version,indexed_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)
         ON CONFLICT(asset_id,model_id,region_id) DO UPDATE SET region_kind=excluded.region_kind,
           x=excluded.x,y=excluded.y,width=excluded.width,height=excluded.height,
           source_width=excluded.source_width,source_height=excluded.source_height,
           pipeline_version=excluded.pipeline_version,indexed_at=excluded.indexed_at",
        params![asset_id, model_id, region.region_id, region.kind, region.x as i64, region.y as i64,
          region.width as i64, region.height as i64, region.source_width as i64, region.source_height as i64,
          pipeline_version, Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

/// (asset_id, page_number, embedding) for every stored image vector of a model.
pub fn load_image_embeddings(path: &Path, model_id: &str) -> Result<Vec<(String, i64, Vec<f32>)>> {
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

pub fn image_embeddings_for_all(
    path: &Path,
    asset_id: &str,
    model_id: &str,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT page_number, embedding FROM image_embeddings WHERE asset_id=?1 AND model_id=?2 ORDER BY page_number DESC",
    )?;
    let rows = statement.query_map(params![asset_id, model_id], |row| {
        let blob: Vec<u8> = row.get(1)?;
        Ok((row.get(0)?, blob_to_embedding(&blob)))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
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
            params![
                asset_id,
                model_id,
                category.label,
                category.score,
                rank as i64
            ],
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

pub fn normalize_visual_tag(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn save_visual_tags(
    path: &Path,
    asset_id: &str,
    model_id: &str,
    model_version: &str,
    tags: &[VisualTag],
) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "DELETE FROM visual_tags WHERE asset_id=?1 AND model_id=?2",
        params![asset_id, model_id],
    )?;
    let now = Utc::now().to_rfc3339();
    for tag in tags {
        let normalized = normalize_visual_tag(&tag.label);
        if normalized.is_empty() {
            continue;
        }
        transaction.execute(
            "INSERT INTO visual_tags(asset_id,model_id,model_version,region_id,namespace,label,normalized_label,confidence,rank,created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                asset_id,
                model_id,
                model_version,
                tag.region_id,
                tag.namespace,
                tag.label,
                normalized,
                tag.confidence,
                tag.rank as i64,
                now
            ],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

pub fn visual_tags_for(path: &Path, asset_id: &str, model_id: &str) -> Result<Vec<VisualTag>> {
    let connection = connect(path)?;
    let mut statement = connection.prepare(
        "SELECT region_id,namespace,label,confidence,rank FROM visual_tags
         WHERE asset_id=?1 AND model_id=?2 ORDER BY rank, confidence DESC",
    )?;
    let rows = statement.query_map(params![asset_id, model_id], |row| {
        Ok(VisualTag {
            region_id: row.get(0)?,
            namespace: row.get(1)?,
            label: row.get(2)?,
            confidence: row.get(3)?,
            rank: row.get::<_, i64>(4)? as usize,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn visual_tag_candidates(
    path: &Path,
    terms: &[String],
    model_id: &str,
) -> Result<Vec<(String, f32)>> {
    let normalized_terms: Vec<String> = terms
        .iter()
        .map(|term| normalize_visual_tag(term))
        .filter(|term| !term.is_empty())
        .collect();
    if normalized_terms.is_empty() {
        return Ok(Vec::new());
    }
    let connection = connect(path)?;
    let mut score_by_asset: std::collections::HashMap<String, f32> =
        std::collections::HashMap::new();
    let mut statement = connection.prepare(
        "SELECT vt.asset_id, MAX(vt.confidence)
         FROM visual_tags vt JOIN assets a ON a.id=vt.asset_id
         WHERE a.available=1 AND a.status='indexed' AND vt.model_id=?1
           AND (
             vt.normalized_label=?2
             OR vt.normalized_label LIKE ?3
             OR vt.normalized_label LIKE ?4
             OR vt.normalized_label LIKE ?5
             OR ?2 LIKE vt.normalized_label || '%'
           )
         GROUP BY vt.asset_id",
    )?;
    for term in normalized_terms {
        let prefix = format!("{term} %");
        let suffix = format!("% {term}");
        let infix = format!("% {term} %");
        let rows = statement.query_map(params![model_id, term, prefix, suffix, infix], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?;
        for row in rows {
            let (id, score) = row?;
            score_by_asset
                .entry(id)
                .and_modify(|value| *value = value.max(score))
                .or_insert(score);
        }
    }
    let mut hits = score_by_asset.into_iter().collect::<Vec<_>>();
    hits.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(hits)
}

pub fn visual_tag_counts(path: &Path, model_id: &str) -> Result<(i64, i64)> {
    let connection = connect(path)?;
    let tagged_assets = connection.query_row(
        "SELECT COUNT(DISTINCT asset_id) FROM visual_tags WHERE model_id=?1",
        [model_id],
        |row| row.get(0),
    )?;
    let tags = connection.query_row(
        "SELECT COUNT(*) FROM visual_tags WHERE model_id=?1",
        [model_id],
        |row| row.get(0),
    )?;
    Ok((tagged_assets, tags))
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

/// Persist reduced provenance together with deterministic document type and
/// entity extraction. This is one SQLite transaction so partial enrichment
/// never exposes a classification without its evidence records.
pub fn save_document_analysis(
    path: &Path,
    asset_id: &str,
    stage_job_id: &str,
    classification: &crate::types::DocumentClassification,
    entities: &[crate::types::ExtractedEntity],
) -> Result<()> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction()?;
    let provenance_id = uuid::Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    transaction.execute(
        "INSERT INTO extraction_provenance(id,asset_id,job_id,stage_job_id,stage,extractor_id,extractor_version,source_kind,source_index,confidence,created_at)
         VALUES (?1,?2,NULL,?3,'document_analysis','recall-deterministic-metadata',?4,'asset',NULL,?5,?6)",
        params![provenance_id, asset_id, stage_job_id, crate::metadata::METADATA_EXTRACTOR_VERSION, classification.confidence, now],
    )?;
    transaction.execute(
        "INSERT INTO document_classifications(asset_id,document_type,confidence,evidence_json,provenance_id,updated_at)
         VALUES (?1,?2,?3,?4,?5,?6)
         ON CONFLICT(asset_id) DO UPDATE SET document_type=excluded.document_type, confidence=excluded.confidence,
           evidence_json=excluded.evidence_json, provenance_id=excluded.provenance_id, updated_at=excluded.updated_at",
        params![asset_id, classification.document_type, classification.confidence, serde_json::to_string(&classification.evidence)?, provenance_id, now],
    )?;
    transaction.execute(
        "DELETE FROM extracted_entities WHERE asset_id=?1",
        [asset_id],
    )?;
    for entity in entities {
        transaction.execute(
            "INSERT INTO extracted_entities(id,asset_id,entity_type,raw_value,normalized_value,confidence,provenance_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![uuid::Uuid::new_v4().to_string(), asset_id, entity.entity_type, entity.raw_value, entity.normalized_value, entity.confidence, provenance_id],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

/// Record an output produced by a derived extraction stage. This deliberately
/// keeps source-job provenance and stage-job provenance separate: the latter
/// survives targeted model refreshes without pretending OCR ran again.
pub fn record_stage_output_provenance(
    path: &Path,
    asset_id: &str,
    stage_job_id: &str,
) -> Result<()> {
    let connection = connect(path)?;
    let (stage, model_id, pipeline_version): (String, String, String) = connection.query_row(
        "SELECT stage,model_id,pipeline_version FROM extraction_stage_jobs WHERE id=?1",
        [stage_job_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    connection.execute(
        "INSERT INTO extraction_provenance(id,asset_id,job_id,stage_job_id,stage,extractor_id,extractor_version,source_kind,source_index,confidence,created_at)
         VALUES (?1,?2,NULL,?3,?4,?5,?6,'asset',NULL,NULL,?7)",
        params![
            uuid::Uuid::new_v4().to_string(),
            asset_id,
            stage_job_id,
            stage,
            model_id,
            pipeline_version,
            Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(())
}

pub fn document_type_candidates(path: &Path, types: &[String]) -> Result<Vec<(String, f32)>> {
    if types.is_empty() {
        return Ok(Vec::new());
    }
    let connection = connect(path)?;
    let placeholders = std::iter::repeat("?")
        .take(types.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT d.asset_id, d.confidence FROM document_classifications d JOIN assets a ON a.id=d.asset_id WHERE a.available=1 AND a.status='indexed' AND d.document_type IN ({placeholders})");
    let mut statement = connection.prepare(&sql)?;
    let values: Vec<&dyn rusqlite::ToSql> = types
        .iter()
        .map(|value| value as &dyn rusqlite::ToSql)
        .collect();
    let rows = statement.query_map(values.as_slice(), |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn entity_candidates(path: &Path, terms: &[String]) -> Result<Vec<(String, f32)>> {
    let normalized: Vec<String> = terms
        .iter()
        .map(|term| {
            term.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_uppercase()
        })
        .filter(|term| term.len() >= 3)
        .collect();
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    let connection = connect(path)?;
    // SQLite bundled with rusqlite does not include an array parameter type, so
    // issue one exact lookup per normalized query token.
    let mut score_by_asset = std::collections::HashMap::new();
    let mut statement = connection.prepare("SELECT e.asset_id, MAX(e.confidence) FROM extracted_entities e JOIN assets a ON a.id=e.asset_id WHERE a.available=1 AND a.status='indexed' AND e.normalized_value=?1 GROUP BY e.asset_id")?;
    for term in normalized {
        let rows = statement.query_map([term], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f32>(1)?))
        })?;
        for row in rows {
            let (id, score) = row?;
            score_by_asset
                .entry(id)
                .and_modify(|value: &mut f32| *value = value.max(score))
                .or_insert(score);
        }
    }
    Ok(score_by_asset.into_iter().collect())
}

/// Coverage counts for the visual subsystem:
/// (image_assets, images_indexed, images_with_embeddings, region_embeddings,
/// images_with_regions, images_classified).
pub fn visual_counts(path: &Path, model_id: &str) -> Result<(i64, i64, i64, i64, i64, i64)> {
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
    let region_embeddings = one(
        "SELECT COUNT(*) FROM image_embeddings WHERE model_id=?1 AND page_number<>?2",
        &[&model_id, &WHOLE_IMAGE_PAGE],
    )?;
    let images_with_regions = one(
        "SELECT COUNT(DISTINCT asset_id) FROM image_embeddings WHERE model_id=?1 AND page_number<>?2",
        &[&model_id, &WHOLE_IMAGE_PAGE],
    )?;
    Ok((
        image_assets,
        images_indexed,
        with_embeddings,
        region_embeddings,
        images_with_regions,
        classified,
    ))
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

    #[test]
    fn visual_reindex_keeps_existing_asset_searchable() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("recall-visual-reindex-{}.db", uuid::Uuid::new_v4()));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/images','now')",
            [],
        )?;
        connection.execute(
            "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
             VALUES ('asset','folder','C:/images/dog.png','dog.png','dog.png','png','image/png',1,'now','indexed',1)",
            [],
        )?;
        connection.execute(
            "INSERT INTO indexing_jobs(id,asset_id,stage,state,created_at,updated_at)
             VALUES ('job','asset','index','indexed','now','now')",
            [],
        )?;
        drop(connection);

        queue_visual_stage_reindex(&path, "mobileclip", "2")?;
        let connection = connect(&path)?;
        let status: String =
            connection.query_row("SELECT status FROM assets WHERE id='asset'", [], |row| {
                row.get(0)
            })?;
        assert_eq!(status, "indexed");
        drop(connection);

        let (_, stage, _) = claim_next_extraction_stage_job(&path)?.expect("visual job");
        assert_eq!(stage, "visual");
        let connection = connect(&path)?;
        let status: String =
            connection.query_row("SELECT status FROM assets WHERE id='asset'", [], |row| {
                row.get(0)
            })?;
        assert_eq!(status, "indexed");
        drop(connection);

        let connection = connect(&path)?;
        let stage_job: String = connection.query_row(
            "SELECT id FROM extraction_stage_jobs WHERE asset_id='asset'",
            [],
            |row| row.get(0),
        )?;
        drop(connection);
        mark_extraction_stage_job(&path, &stage_job, "failed", Some("test failure"))?;
        let connection = connect(&path)?;
        let status: String =
            connection.query_row("SELECT status FROM assets WHERE id='asset'", [], |row| {
                row.get(0)
            })?;
        assert_eq!(status, "indexed");
        drop(connection);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn stage_output_provenance_points_to_the_derived_job() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "recall-stage-provenance-{}.db",
            uuid::Uuid::new_v4()
        ));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/files','now')",
            [],
        )?;
        connection.execute(
            "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
             VALUES ('asset','folder','C:/files/note.txt','note.txt','note.txt','txt','text/plain',1,'now','indexed',1)",
            [],
        )?;
        drop(connection);

        queue_extraction_stage(&path, "asset", "text_embedding", "e5", "chunk-v1")?;
        let (job_id, stage, _) = claim_next_extraction_stage_job(&path)?.expect("stage job");
        assert_eq!(stage, "text_embedding");
        record_stage_output_provenance(&path, "asset", &job_id)?;
        let connection = connect(&path)?;
        let links: (Option<String>, Option<String>) = connection.query_row(
            "SELECT job_id,stage_job_id FROM extraction_provenance WHERE asset_id='asset'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(links, (None, Some(job_id)));
        drop(connection);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn stale_visual_version_requeues_only_indexed_images() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("recall-stale-visual-{}.db", uuid::Uuid::new_v4()));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/images','now')",
            [],
        )?;
        for (id, extension, status) in [
            ("image", "png", "indexed"),
            ("note", "txt", "indexed"),
            ("new", "png", "pending"),
        ] {
            connection.execute(
                "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
                 VALUES (?1,'folder',?2,?2,?2,?3,'application/octet-stream',1,'now',?4,1)",
                params![id, format!("C:/images/{id}.{extension}"), extension, status],
            )?;
            connection.execute(
                "INSERT INTO indexing_jobs(id,asset_id,stage,state,created_at,updated_at) VALUES (?1,?1,'index',?2,'now','now')",
                params![id, if status == "indexed" { "indexed" } else { "pending" }],
            )?;
        }
        drop(connection);
        assert_eq!(queue_stale_visual_reindex(&path, "mobileclip", "2")?, 1);
        let connection = connect(&path)?;
        let job: (String, String) = connection.query_row(
            "SELECT stage,state FROM extraction_stage_jobs WHERE asset_id='image'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert_eq!(job, ("visual".into(), "pending".into()));
        let new_stage: String =
            connection.query_row("SELECT stage FROM indexing_jobs WHERE id='new'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(new_stage, "index");
        drop(connection);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn visual_tags_are_searchable_model_evidence() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("recall-visual-tags-{}.db", uuid::Uuid::new_v4()));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/images','now')",
            [],
        )?;
        connection.execute(
            "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
             VALUES ('cat','folder','C:/images/cat.png','cat.png','cat.png','png','image/png',1,'now','indexed',1)",
            [],
        )?;
        drop(connection);

        save_visual_tags(
            &path,
            "cat",
            "visual-tags/general-v1",
            "1",
            &[
                VisualTag {
                    region_id: -1,
                    namespace: "wd:general".into(),
                    label: "tabby cat".into(),
                    confidence: 0.82,
                    rank: 0,
                },
                VisualTag {
                    region_id: -1,
                    namespace: "wd:general".into(),
                    label: "indoor".into(),
                    confidence: 0.61,
                    rank: 1,
                },
            ],
        )?;

        let hits = visual_tag_candidates(&path, &["cat".to_string()], "visual-tags/general-v1")?;
        assert_eq!(hits, vec![("cat".into(), 0.82)]);
        let stored = visual_tags_for(&path, "cat", "visual-tags/general-v1")?;
        assert_eq!(stored[0].label, "tabby cat");
        let counts = visual_tag_counts(&path, "visual-tags/general-v1")?;
        assert_eq!(counts, (1, 2));
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn stale_visual_tag_version_requeues_only_visual_tagging() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "recall-stale-visual-tags-{}.db",
            uuid::Uuid::new_v4()
        ));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/images','now')",
            [],
        )?;
        for (id, extension, status) in [
            ("image", "png", "indexed"),
            ("note", "txt", "indexed"),
            ("new", "png", "pending"),
        ] {
            connection.execute(
                "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
                 VALUES (?1,'folder',?2,?2,?2,?3,'application/octet-stream',1,'now',?4,1)",
                params![id, format!("C:/images/{id}.{extension}"), extension, status],
            )?;
            connection.execute(
                "INSERT INTO indexing_jobs(id,asset_id,stage,state,created_at,updated_at) VALUES (?1,?1,'index',?2,'now','now')",
                params![id, if status == "indexed" { "indexed" } else { "pending" }],
            )?;
        }
        drop(connection);

        assert_eq!(
            queue_stale_visual_tagging_reindex(&path, "visual-tags/general-v1", "2")?,
            1
        );
        let connection = connect(&path)?;
        let job: (String, String, String, String) = connection.query_row(
            "SELECT stage,state,model_id,pipeline_version FROM extraction_stage_jobs WHERE asset_id='image'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        assert_eq!(
            job,
            (
                "visual_tagging".into(),
                "pending".into(),
                "visual-tags/general-v1".into(),
                "2".into()
            )
        );
        let other_jobs: i64 = connection.query_row(
            "SELECT COUNT(*) FROM extraction_stage_jobs WHERE asset_id<>'image'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(other_jobs, 0);
        drop(connection);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn generated_tag_chunk_replaces_old_fts_text() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("recall-tag-chunk-{}.db", uuid::Uuid::new_v4()));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/images','now')",
            [],
        )?;
        connection.execute(
            "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
             VALUES ('asset','folder','C:/images/photo.png','photo.png','photo.png','png','image/png',1,'now','indexed',1)",
            [],
        )?;
        drop(connection);

        upsert_generated_chunk(&path, "asset", "visual_tags", "cat, sofa", None)?;
        upsert_generated_chunk(&path, "asset", "visual_tags", "dog, park", None)?;

        let connection = connect(&path)?;
        let text: String = connection.query_row(
            "SELECT text FROM chunks WHERE asset_id='asset' AND source='visual_tags'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(text, "dog, park");
        let cat_hits: i64 = connection.query_row(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'cat'",
            [],
            |r| r.get(0),
        )?;
        let dog_hits: i64 = connection.query_row(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'dog'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(cat_hits, 0);
        assert_eq!(dog_hits, 1);
        drop(connection);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn indexing_status_pending_is_file_based_not_stage_based() -> Result<()> {
        let path = std::env::temp_dir().join(format!("recall-status-{}.db", uuid::Uuid::new_v4()));
        migrate(&path)?;
        let connection = connect(&path)?;
        connection.execute(
            "INSERT INTO watched_folders(id,path,created_at) VALUES ('folder','C:/files','now')",
            [],
        )?;
        connection.execute(
            "INSERT INTO assets(id,folder_id,absolute_path,relative_path,filename,extension,mime_type,size_bytes,modified_at,status,available)
             VALUES ('asset','folder','C:/files/cat.png','cat.png','cat.png','png','image/png',1,'now','indexed',1)",
            [],
        )?;
        connection.execute(
            "INSERT INTO indexing_jobs(id,asset_id,stage,state,created_at,updated_at)
             VALUES ('source','asset','index','indexed','now','now')",
            [],
        )?;
        drop(connection);

        queue_extraction_stage(
            &path,
            "asset",
            "visual_tagging",
            "visual-tags/general-v1",
            "1",
        )?;
        let public = indexing_status(&path)?;
        assert_eq!(public.pending, 0);
        assert_eq!(public.indexed, 1);
        assert_eq!(public.background_pending, 1);
        assert_eq!(public.background_processing, 0);
        assert_eq!(active_job_counts(&path)?, (1, 0, 0));

        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
