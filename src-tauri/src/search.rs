//! Retrieval primitives shared by the fusion engine ([`crate::fusion`]).
//!
//! Candidate loading, FTS5 keyword scoring, cosine similarity and snippet
//! extraction live here; ranking/fusion lives in `fusion.rs`.

use std::collections::HashMap;

use crate::{db, error::Result, types::SearchFilters};

#[derive(Clone)]
pub(crate) struct Candidate {
    pub chunk_id: String,
    pub asset_id: String,
    pub folder_id: String,
    pub extension: Option<String>,
    pub page_number: Option<i64>,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

/// Load all indexed text chunks as candidates, applying folder/extension filters.
pub(crate) fn load_candidates(
    connection: &rusqlite::Connection,
    filters: &SearchFilters,
) -> Result<Vec<Candidate>> {
    let mut statement = connection.prepare(
        r#"
      SELECT c.id,c.asset_id,a.folder_id,a.extension,c.page_number,c.text,c.embedding
      FROM chunks c JOIN assets a ON a.id=c.asset_id WHERE a.available=1 AND a.status='indexed'
    "#,
    )?;
    let rows = statement.query_map([], |r| {
        Ok(Candidate {
            chunk_id: r.get(0)?,
            asset_id: r.get(1)?,
            folder_id: r.get(2)?,
            extension: r.get(3)?,
            page_number: r.get(4)?,
            text: r.get(5)?,
            embedding: r
                .get::<_, Option<Vec<u8>>>(6)?
                .map(|blob| db::blob_to_embedding(&blob)),
        })
    })?;
    Ok(rows
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|candidate| matches_filters(candidate, filters))
        .collect())
}

/// Raw (non-negative) BM25 keyword scores per chunk id for an FTS MATCH string.
/// An empty match string yields no results.
pub(crate) fn fts_scores(
    connection: &rusqlite::Connection,
    match_expr: &str,
) -> Result<HashMap<String, f32>> {
    let mut keyword_raw: HashMap<String, f32> = HashMap::new();
    if !match_expr.is_empty() {
        let mut fts = connection.prepare("SELECT chunk_id, -bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?1 LIMIT 250")?;
        let matches = fts.query_map([match_expr], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)? as f32))
        })?;
        for row in matches {
            let (id, score) = row?;
            keyword_raw.insert(id, score.max(0.0));
        }
    }
    Ok(keyword_raw)
}

/// Build a precise FTS5 MATCH expression: a CONJUNCTION of the required terms
/// (all must appear), optionally OR'd with exact phrases for adjacency ranking.
///
/// Uses prefix terms so plurals match (`ticket*` → ticket/tickets), but because
/// every term is required, a single unrelated prefix hit cannot qualify: query
/// "train ticket" needs both `train*` AND `ticket*`, so a "training" screenshot
/// with no ticket word never matches. This replaces the old permissive prefix-OR.
pub(crate) fn fts_conjunction(required_terms: &[String], phrases: &[String]) -> String {
    if required_terms.is_empty() {
        return String::new();
    }
    let conjunction = required_terms
        .iter()
        .map(|t| format!("\"{}\"*", t.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" AND ");
    let mut expr = format!("({conjunction})");
    for phrase in phrases {
        let cleaned = phrase.replace('"', "");
        if cleaned.split_whitespace().count() >= 2 {
            expr.push_str(&format!(" OR \"{cleaned}\""));
        }
    }
    expr
}

pub(crate) fn matches_filters(candidate: &Candidate, filters: &SearchFilters) -> bool {
    if let Some(folder_id) = &filters.folder_id {
        if &candidate.folder_id != folder_id {
            return false;
        }
    }
    filters.extensions.is_empty()
        || candidate
            .extension
            .as_ref()
            .map(|ext| {
                filters
                    .extensions
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0, 0.0, 0.0);
    for (a, b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

pub(crate) fn snippet(text: &str, query: &str) -> String {
    let lower = text.to_lowercase();
    let position = query
        .split_whitespace()
        .find_map(|term| lower.find(&term.to_lowercase()))
        .unwrap_or(0);
    let chars: Vec<char> = text.chars().collect();
    let char_position = text[..position.min(text.len())].chars().count();
    let start = char_position.saturating_sub(80);
    let end = (start + 420).min(chars.len());
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        chars[start..end].iter().collect::<String>(),
        if end < chars.len() { "…" } else { "" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cosine_ranks_identical_vectors() {
        assert!((cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]) - 1.0).abs() < 0.0001);
    }
    #[test]
    fn fts_conjunction_requires_all_terms() {
        let expr = fts_conjunction(&["train".into(), "ticket".into()], &["train ticket".into()]);
        assert_eq!(expr, "(\"train\"* AND \"ticket\"*) OR \"train ticket\"");
    }

    #[test]
    fn fts_conjunction_empty_without_terms() {
        assert_eq!(fts_conjunction(&[], &[]), "");
    }

    #[test]
    fn train_ticket_fts_rejects_training_only_distractor() -> Result<()> {
        let connection = rusqlite::Connection::open_in_memory()?;
        connection.execute_batch(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
             INSERT INTO chunks_fts(chunk_id, text) VALUES
               ('distractor', 'Adobe Photoshop training lesson with image editing tools'),
               ('ticket', 'Indian Railways train ticket booking confirmation');",
        )?;

        let terms = vec!["train".to_string(), "ticket".to_string()];
        let phrases = vec!["train ticket".to_string()];
        let scores = fts_scores(&connection, &fts_conjunction(&terms, &phrases))?;

        assert!(scores.contains_key("ticket"));
        assert!(!scores.contains_key("distractor"));
        Ok(())
    }
}
