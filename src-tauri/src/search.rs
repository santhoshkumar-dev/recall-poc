use std::collections::HashMap;

use crate::{
    db,
    error::Result,
    types::{SearchFilters, SearchResult},
    AppCore,
};

#[derive(Clone)]
struct Candidate {
    chunk_id: String,
    asset_id: String,
    folder_id: String,
    filename: String,
    extension: Option<String>,
    source_path: String,
    page_number: Option<i64>,
    text: String,
    embedding: Option<Vec<f32>>,
}

pub fn search(core: &AppCore, query: &str, filters: &SearchFilters) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let connection = db::connect(&core.db_path)?;
    let mut statement = connection.prepare(r#"
      SELECT c.id,c.asset_id,a.folder_id,a.filename,a.extension,a.absolute_path,c.page_number,c.text,c.embedding
      FROM chunks c JOIN assets a ON a.id=c.asset_id WHERE a.available=1 AND a.status='indexed'
    "#)?;
    let rows = statement.query_map([], |r| {
        Ok(Candidate {
            chunk_id: r.get(0)?,
            asset_id: r.get(1)?,
            folder_id: r.get(2)?,
            filename: r.get(3)?,
            extension: r.get(4)?,
            source_path: r.get(5)?,
            page_number: r.get(6)?,
            text: r.get(7)?,
            embedding: r
                .get::<_, Option<Vec<u8>>>(8)?
                .map(|blob| db::blob_to_embedding(&blob)),
        })
    })?;
    let candidates: Vec<Candidate> = rows
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|candidate| matches_filters(candidate, filters))
        .collect();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let fts_query = fts_query(query);
    let mut keyword_raw: HashMap<String, f32> = HashMap::new();
    if !fts_query.is_empty() {
        let mut fts = connection.prepare("SELECT chunk_id, -bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?1 LIMIT 250")?;
        let matches = fts.query_map([fts_query], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)? as f32))
        })?;
        for row in matches {
            let (id, score) = row?;
            keyword_raw.insert(id, score.max(0.0));
        }
    }
    let keyword_max = keyword_raw
        .values()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);
    let ai = core.ai.read().clone();
    let query_embedding = ai
        .as_ref()
        .map(|runtime| runtime.embed(vec![query.to_owned()]))
        .transpose()?
        .and_then(|mut values| values.pop());
    let semantic_enabled = query_embedding.is_some();
    let mut by_asset: HashMap<String, SearchResult> = HashMap::new();

    for candidate in candidates {
        let keyword_score =
            keyword_raw.get(&candidate.chunk_id).copied().unwrap_or(0.0) / keyword_max;
        let semantic_score = match (&query_embedding, &candidate.embedding) {
            (Some(query), Some(document)) => cosine_similarity(query, document).max(0.0),
            _ => 0.0,
        };
        if !semantic_enabled && keyword_score == 0.0 {
            continue;
        }
        let combined_score = if semantic_enabled {
            semantic_score * 0.75 + keyword_score * 0.25
        } else {
            keyword_score
        };
        if combined_score < 0.08 {
            continue;
        }
        let result = SearchResult {
            asset_id: candidate.asset_id.clone(),
            filename: candidate.filename,
            extension: candidate.extension,
            source_path: candidate.source_path,
            snippet: snippet(&candidate.text, query),
            page_number: candidate.page_number,
            semantic_score,
            keyword_score,
            combined_score,
        };
        if by_asset
            .get(&candidate.asset_id)
            .map(|old| old.combined_score < combined_score)
            .unwrap_or(true)
        {
            by_asset.insert(candidate.asset_id, result);
        }
    }
    let mut results: Vec<SearchResult> = by_asset.into_values().collect();
    results.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));
    results.truncate(30);
    Ok(results)
}

fn matches_filters(candidate: &Candidate, filters: &SearchFilters) -> bool {
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

fn fts_query(query: &str) -> String {
    query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| part.len() > 1)
        .map(|part| format!("\"{}\"*", part.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
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

fn snippet(text: &str, query: &str) -> String {
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
    fn fts_input_is_quoted() {
        assert_eq!(fts_query("train ticket!"), "\"train\"* OR \"ticket\"*");
    }
}
