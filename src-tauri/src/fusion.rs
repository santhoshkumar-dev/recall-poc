//! Multimodal retrieval fusion.
//!
//! Runs the retrieval channels (exact-text, semantic-text, visual, visual
//! category, metadata, filename/folder), normalizes each independently by rank,
//! and combines them with intent-aware Reciprocal-Rank Fusion (RRF). Ranking
//! weights live here and nowhere else.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::{
    db,
    error::Result,
    query_intent,
    search,
    types::{
        AssetBrief, ChannelDiagnostics, ChannelResult, MatchReason, QueryIntent, SearchDebugReport,
        SearchFilters, SearchResult, VisualCategory,
    },
    AppCore,
};

/// RRF constant.
const RRF_K: f32 = 60.0;
/// Per-channel candidate caps.
const TEXT_CANDIDATES: usize = 50;
const VISUAL_CANDIDATES: usize = 50;
const CATEGORY_CANDIDATES: i64 = 40;
const METADATA_CANDIDATES: usize = 40;
/// Final asset cap.
const MAX_RESULTS: usize = 20;

// Qualification floors — an asset must show at least one discriminating signal
// (semantic-text alone is unreliable because E5 similarities cluster high).
// These prevent a haystack of loosely-related screenshots from being returned
// as "matches" when nothing genuinely relevant exists. Calibrate with the raw
// scores shown in the retrieval inspector; refined in the Phase 7 eval.
const VISUAL_FLOOR: f32 = 0.25;
const CATEGORY_FLOOR: f32 = 0.27;
const SEMANTIC_STRONG: f32 = 0.85;
// Channel-inclusion floors: weak guesses don't even enter rank fusion.
const VISUAL_INCLUDE: f32 = 0.20;
const CATEGORY_INCLUDE: f32 = 0.22;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Channel {
    Exact,
    Semantic,
    Visual,
    Category,
    Metadata,
    Filename,
}

impl Channel {
    fn key(self) -> &'static str {
        match self {
            Channel::Exact => "exact_text",
            Channel::Semantic => "semantic_text",
            Channel::Visual => "visual",
            Channel::Category => "visual_category",
            Channel::Metadata => "metadata",
            Channel::Filename => "filename",
        }
    }
}

/// Intent-aware channel weights (fractions summing to ~1.0).
fn weights(intent: QueryIntent) -> HashMap<Channel, f32> {
    use Channel::*;
    let table: &[(Channel, f32)] = match intent {
        QueryIntent::ExactIdentifier => {
            &[(Exact, 0.70), (Metadata, 0.20), (Semantic, 0.05), (Visual, 0.05)]
        }
        QueryIntent::Visual => &[
            (Visual, 0.60),
            (Category, 0.15),
            (Semantic, 0.10),
            (Metadata, 0.10),
            (Filename, 0.05),
        ],
        QueryIntent::Category => &[
            (Category, 0.35),
            (Semantic, 0.25),
            (Exact, 0.20),
            (Metadata, 0.15),
            (Visual, 0.05),
        ],
        QueryIntent::DateFiltered => &[
            (Semantic, 0.30),
            (Category, 0.25),
            (Visual, 0.25),
            (Exact, 0.15),
            (Filename, 0.05),
        ],
        // General / mixed / semantic / amount / others.
        _ => &[
            (Semantic, 0.30),
            (Exact, 0.25),
            (Visual, 0.25),
            (Metadata, 0.15),
            (Filename, 0.05),
        ],
    };
    table.iter().copied().collect()
}

/// A channel's ranked hits (asset_id → raw score), plus its diagnostics.
struct ChannelRun {
    channel: Channel,
    ranked: Vec<(String, f32)>,
    latency_ms: u128,
}

impl ChannelRun {
    /// asset_id → 0-based rank.
    fn ranks(&self) -> HashMap<&str, usize> {
        self.ranked
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.as_str(), i))
            .collect()
    }
}

/// Everything computed for one query — used by both search and the inspector.
pub struct FusionOutcome {
    pub results: Vec<SearchResult>,
    pub intents: Vec<QueryIntent>,
    pub expanded_categories: Vec<String>,
    pub applied_filters: Vec<String>,
    pub channels: Vec<ChannelDiagnostics>,
    pub total_latency_ms: u128,
}

pub fn search(core: &AppCore, query: &str, filters: &SearchFilters) -> Result<Vec<SearchResult>> {
    Ok(execute(core, query, filters)?.results)
}

pub fn search_debug(
    core: &AppCore,
    query: &str,
    filters: &SearchFilters,
) -> Result<SearchDebugReport> {
    let outcome = execute(core, query, filters)?;
    Ok(SearchDebugReport {
        query: query.to_string(),
        intents: outcome.intents,
        expanded_categories: outcome.expanded_categories,
        applied_filters: outcome.applied_filters,
        channels: outcome.channels,
        results: outcome.results,
        total_latency_ms: outcome.total_latency_ms,
    })
}

pub fn execute(core: &AppCore, query: &str, filters: &SearchFilters) -> Result<FusionOutcome> {
    let overall = Instant::now();
    let analysis = query_intent::analyze(query);
    let intent = query_intent::primary_intent(&analysis.intents);

    let mut applied_filters = Vec::new();
    if let Some(folder) = &filters.folder_id {
        applied_filters.push(format!("folder_id = {folder}"));
    }
    if !filters.extensions.is_empty() {
        applied_filters.push(format!("extensions = {}", filters.extensions.join(", ")));
    }
    // File-type intent: if the user named a file type and set no explicit
    // extension filter, apply the implied extensions as a soft filter.
    let implied_exts = if filters.extensions.is_empty() {
        query_intent::implied_extensions(query)
    } else {
        Vec::new()
    };
    if !implied_exts.is_empty() {
        applied_filters.push(format!("implied file types = {}", implied_exts.join(", ")));
    }

    let empty = |lat: u128| FusionOutcome {
        results: Vec::new(),
        intents: analysis.intents.clone(),
        expanded_categories: analysis.expanded_categories.clone(),
        applied_filters: applied_filters.clone(),
        channels: Vec::new(),
        total_latency_ms: lat,
    };
    if query.trim().is_empty() {
        return Ok(empty(overall.elapsed().as_millis()));
    }

    let connection = db::connect(&core.db_path)?;
    let candidates = search::load_candidates(&connection, filters)?;
    let briefs = db::indexed_asset_briefs(&core.db_path)?;
    let allowed: HashSet<String> = briefs.iter().map(|b| b.id.clone()).collect();
    let brief_map: HashMap<String, AssetBrief> =
        briefs.into_iter().map(|b| (b.id.clone(), b)).collect();

    // Per-asset text scoring inputs.
    let keyword_raw = search::fts_scores(&connection, query)?;
    let ai = core.ai.read().clone();
    let query_embedding = ai
        .as_ref()
        .map(|runtime| runtime.embed_query(query.to_owned()))
        .transpose()?;

    // Best chunk per asset (for snippet + page), plus per-asset fts/semantic max.
    let mut best_chunk: HashMap<String, (String, Option<i64>, f32)> = HashMap::new(); // asset -> (text, page, rank_key)
    let mut fts_by_asset: HashMap<String, f32> = HashMap::new();
    let mut semantic_by_asset: HashMap<String, f32> = HashMap::new();
    for c in &candidates {
        let fts = keyword_raw.get(&c.chunk_id).copied().unwrap_or(0.0);
        let sem = match (&query_embedding, &c.embedding) {
            (Some(q), Some(d)) => search::cosine_similarity(q, d).max(0.0),
            _ => 0.0,
        };
        fts_by_asset
            .entry(c.asset_id.clone())
            .and_modify(|v| *v = v.max(fts))
            .or_insert(fts);
        semantic_by_asset
            .entry(c.asset_id.clone())
            .and_modify(|v| *v = v.max(sem))
            .or_insert(sem);
        let rank_key = sem.max(fts);
        let better = best_chunk
            .get(&c.asset_id)
            .map(|(_, _, k)| *k < rank_key)
            .unwrap_or(true);
        if better {
            best_chunk.insert(c.asset_id.clone(), (c.text.clone(), c.page_number, rank_key));
        }
    }

    // --- Build channel runs ---
    let mut runs: Vec<ChannelRun> = Vec::new();

    // Exact-text (FTS).
    let t = Instant::now();
    runs.push(ChannelRun {
        channel: Channel::Exact,
        ranked: top_n(map_to_sorted(&fts_by_asset, |s| s > 0.0), TEXT_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Semantic (E5).
    if query_embedding.is_some() {
        let t = Instant::now();
        runs.push(ChannelRun {
            channel: Channel::Semantic,
            ranked: top_n(map_to_sorted(&semantic_by_asset, |s| s > 0.0), TEXT_CANDIDATES),
            latency_ms: t.elapsed().as_millis(),
        });
    }

    // Visual (MobileCLIP text→image).
    let visual = core.visual.read().clone();
    let mut visual_by_asset: HashMap<String, f32> = HashMap::new();
    if let Some(runtime) = visual.as_ref() {
        let t = Instant::now();
        let model_id = crate::ai::selection(&core.db_path)?.visual_model_id;
        let q = runtime.embed_text(query)?;
        let mut scored: Vec<(String, f32)> = db::load_image_embeddings(&core.db_path, &model_id)?
            .into_iter()
            .filter(|(id, _, _)| allowed.contains(id))
            .map(|(id, _page, vec)| (id, search::cosine_similarity(&q, &vec).max(0.0)))
            .collect();
        // Keep best page per asset.
        let mut best: HashMap<String, f32> = HashMap::new();
        for (id, s) in scored.drain(..) {
            best.entry(id).and_modify(|v| *v = v.max(s)).or_insert(s);
        }
        visual_by_asset = best.clone();
        runs.push(ChannelRun {
            channel: Channel::Visual,
            ranked: top_n(map_to_sorted(&best, |s| s >= VISUAL_INCLUDE), VISUAL_CANDIDATES),
            latency_ms: t.elapsed().as_millis(),
        });
    }

    // Visual category (query expansion → classifications).
    let mut category_by_asset: HashMap<String, f32> = HashMap::new();
    if visual.is_some() && !analysis.expanded_categories.is_empty() {
        let t = Instant::now();
        let model_id = crate::ai::selection(&core.db_path)?.visual_model_id;
        let hits = db::assets_by_categories(
            &core.db_path,
            &model_id,
            &analysis.expanded_categories,
            CATEGORY_CANDIDATES,
        )?;
        for (id, score) in &hits {
            if allowed.contains(id) {
                category_by_asset.insert(id.clone(), *score);
            }
        }
        runs.push(ChannelRun {
            channel: Channel::Category,
            ranked: top_n(
                map_to_sorted(&category_by_asset, |s| s >= CATEGORY_INCLUDE),
                CATEGORY_CANDIDATES as usize,
            ),
            latency_ms: t.elapsed().as_millis(),
        });
    }

    // Metadata (exact identifiers / urls / amounts). Populated once Phase 5
    // fills asset_metadata; before that it simply returns nothing.
    let t = Instant::now();
    let metadata_hits = metadata_channel(core, query, &allowed)?;
    let metadata_by_asset: HashMap<String, f32> = metadata_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::Metadata,
        ranked: top_n(metadata_hits, METADATA_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Filename / folder.
    let t = Instant::now();
    let filename_hits = filename_channel(query, &brief_map);
    let filename_by_asset: HashMap<String, f32> = filename_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::Filename,
        ranked: top_n(filename_hits, TEXT_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // --- Fuse ---
    let weights = weights(intent);
    let mut fused: HashMap<String, f32> = HashMap::new();
    for run in &runs {
        let w = weights.get(&run.channel).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        for (asset_id, rank) in run.ranks() {
            let contribution = w * (1.0 / (RRF_K + rank as f32));
            *fused.entry(asset_id.to_string()).or_insert(0.0) += contribution;
        }
    }

    // Assemble + qualify.
    let mut assembled: Vec<(SearchResult, f32)> = Vec::new();
    for (asset_id, fused_score) in &fused {
        let Some(brief) = brief_map.get(asset_id) else {
            continue;
        };
        // Soft file-type filter from query intent.
        if !implied_exts.is_empty() {
            let ext_ok = brief
                .extension
                .as_ref()
                .map(|e| implied_exts.iter().any(|i| i.eq_ignore_ascii_case(e)))
                .unwrap_or(false);
            if !ext_ok {
                continue;
            }
        }
        let fts = fts_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let semantic = semantic_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let visual_s = visual_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let category_s = category_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let filename_s = filename_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let metadata_s = metadata_by_asset.get(asset_id).copied().unwrap_or(0.0);

        let qualifies = fts > 0.0
            || visual_s >= VISUAL_FLOOR
            || category_s >= CATEGORY_FLOOR
            || filename_s > 0.0
            || metadata_s > 0.0
            || semantic >= SEMANTIC_STRONG;
        if !qualifies {
            continue;
        }

        let mut reasons: Vec<MatchReason> = Vec::new();
        if fts > 0.0 {
            reasons.push(MatchReason::ExactText);
        }
        if semantic >= SEMANTIC_STRONG {
            reasons.push(MatchReason::SemanticText);
        }
        if visual_s >= VISUAL_FLOOR {
            reasons.push(MatchReason::VisualSimilarity);
        }
        if category_s >= CATEGORY_FLOOR {
            reasons.push(MatchReason::VisualCategory);
        }
        if filename_s > 0.0 {
            reasons.push(MatchReason::Filename);
        }
        if metadata_s > 0.0 {
            reasons.push(MatchReason::Metadata);
        }

        let (snippet_text, page_number) = match best_chunk.get(asset_id) {
            Some((text, page, _)) => (search::snippet(text, query), *page),
            None => (String::new(), None),
        };
        let top_categories = top_categories_for(core, asset_id).unwrap_or_default();

        let mut result = SearchResult::text_only(
            asset_id.clone(),
            brief.filename.clone(),
            brief.extension.clone(),
            brief.source_path.clone(),
            snippet_text,
            page_number,
            semantic,
            fts,
            *fused_score,
        );
        result.visual_score = visual_s;
        result.category_score = category_s;
        result.match_reasons = reasons;
        result.top_categories = top_categories;
        assembled.push((result, *fused_score));
    }

    assembled.sort_by(|a, b| b.1.total_cmp(&a.1));
    assembled.truncate(MAX_RESULTS);

    // Normalize combined_score to 0..1 for display (order preserved).
    let max_fused = assembled.first().map(|(_, s)| *s).unwrap_or(1.0).max(f32::EPSILON);
    let results: Vec<SearchResult> = assembled
        .into_iter()
        .map(|(mut r, s)| {
            r.combined_score = (s / max_fused).clamp(0.0, 1.0);
            r
        })
        .collect();

    let channels = runs
        .iter()
        .map(|run| diagnostics(run, &brief_map))
        .collect();

    Ok(FusionOutcome {
        results,
        intents: analysis.intents,
        expanded_categories: analysis.expanded_categories,
        applied_filters,
        channels,
        total_latency_ms: overall.elapsed().as_millis(),
    })
}

/// Metadata channel: match query tokens against extracted identifiers/urls/
/// emails/amounts stored per asset. Empty until Phase 5 populates asset_metadata.
fn metadata_channel(
    core: &AppCore,
    query: &str,
    allowed: &HashSet<String>,
) -> Result<Vec<(String, f32)>> {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
        .collect();
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let mut hits: Vec<(String, f32)> = Vec::new();
    for asset_id in allowed {
        if let Some(meta) = db::metadata_for(&core.db_path, asset_id)? {
            let mut haystack: Vec<String> = Vec::new();
            haystack.extend(meta.identifiers.iter().cloned());
            haystack.extend(meta.urls.iter().cloned());
            haystack.extend(meta.emails.iter().cloned());
            haystack.extend(meta.phone_numbers.iter().cloned());
            haystack.extend(meta.amounts.iter().map(|a| a.raw.clone()));
            let hay = haystack.join(" ").to_lowercase();
            let matched = tokens.iter().filter(|t| hay.contains(*t)).count();
            if matched > 0 {
                hits.push((asset_id.clone(), matched as f32));
            }
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(hits)
}

/// Filename/folder channel: fraction of query tokens present in filename+path.
fn filename_channel(
    query: &str,
    briefs: &HashMap<String, AssetBrief>,
) -> Vec<(String, f32)> {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect();
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<(String, f32)> = Vec::new();
    for brief in briefs.values() {
        let hay = format!("{} {}", brief.filename, brief.source_path).to_lowercase();
        let matched = tokens.iter().filter(|t| hay.contains(*t)).count();
        if matched > 0 {
            hits.push((brief.id.clone(), matched as f32 / tokens.len() as f32));
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    hits
}

fn top_categories_for(core: &AppCore, asset_id: &str) -> Result<Vec<VisualCategory>> {
    let model_id = crate::ai::selection(&core.db_path)?.visual_model_id;
    if model_id == crate::ai::VISUAL_DISABLED {
        return Ok(Vec::new());
    }
    let mut cats = db::classifications_for(&core.db_path, asset_id, &model_id)?;
    cats.truncate(3);
    Ok(cats)
}

fn diagnostics(run: &ChannelRun, briefs: &HashMap<String, AssetBrief>) -> ChannelDiagnostics {
    let results = run
        .ranked
        .iter()
        .enumerate()
        .map(|(rank, (asset_id, raw))| ChannelResult {
            channel: run.channel.key().to_string(),
            asset_id: asset_id.clone(),
            filename: briefs
                .get(asset_id)
                .map(|b| b.filename.clone())
                .unwrap_or_default(),
            rank,
            raw_score: *raw,
            normalized_score: 1.0 / (RRF_K + rank as f32),
        })
        .collect::<Vec<_>>();
    ChannelDiagnostics {
        channel: run.channel.key().to_string(),
        latency_ms: run.latency_ms,
        candidate_count: run.ranked.len(),
        results,
    }
}

/// Sort a score map descending, keeping entries passing `keep`.
fn map_to_sorted<F: Fn(f32) -> bool>(map: &HashMap<String, f32>, keep: F) -> Vec<(String, f32)> {
    let mut v: Vec<(String, f32)> = map
        .iter()
        .filter(|(_, s)| keep(**s))
        .map(|(id, s)| (id.clone(), *s))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

fn top_n(mut v: Vec<(String, f32)>, n: usize) -> Vec<(String, f32)> {
    v.truncate(n);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_prefers_top_ranks() {
        // rank 0 contributes 1/60, rank 9 contributes 1/69.
        assert!(1.0 / (RRF_K + 0.0) > 1.0 / (RRF_K + 9.0));
    }

    #[test]
    fn weights_sum_close_to_one() {
        for intent in [
            QueryIntent::ExactIdentifier,
            QueryIntent::Visual,
            QueryIntent::Category,
            QueryIntent::DateFiltered,
            QueryIntent::SemanticText,
        ] {
            let total: f32 = weights(intent).values().sum();
            assert!((total - 1.0).abs() < 0.001, "intent weights must sum to 1");
        }
    }
}
