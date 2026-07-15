//! Multimodal retrieval fusion with evidence gating.
//!
//! Builds a deterministic [`QueryPlan`], runs the retrieval channels (exact
//! text via conjunction FTS, semantic text, visual, visual tags, metadata,
//! filename), and combines them with intent-aware one-based
//! Reciprocal-Rank Fusion. Text/document results require one strong or two
//! moderate signals; an explicit visual-intent query may return ranked visual
//! candidates without presenting rank as certainty.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[cfg(test)]
use std::sync::Arc;

use crate::{
    ai, db,
    error::Result,
    query_intent::{self, QueryPlan},
    search,
    types::{
        AssetBrief, ChannelDiagnostics, ChannelResult, MatchReason, QueryIntent, SearchDebugReport,
        SearchFilters, SearchResult, VisualCategory,
    },
    AppCore,
};

#[cfg(test)]
use crate::visual::PromptBank;

const RRF_K: f32 = 60.0;
const TEXT_CANDIDATES: usize = 50;
const VISUAL_CANDIDATES: usize = 100;
const METADATA_CANDIDATES: usize = 40;
const MAX_RESULTS: usize = 20;

// --- Calibrated evidence thresholds (tune via the inspector; Phase 7 eval) ---
// Semantic (E5) similarity: moderate only, never a sole qualifier.
const SEM_MODERATE: f32 = 0.85;
// Visual (MobileCLIP text→image) cosine.
const VIS_MAD_FLOOR: f32 = 0.01;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Channel {
    Exact,
    Semantic,
    Visual,
    VisualTag,
    DocumentType,
    Entity,
    Metadata,
    Filename,
}

impl Channel {
    fn key(self) -> &'static str {
        match self {
            Channel::Exact => "exact_text",
            Channel::Semantic => "semantic_text",
            Channel::Visual => "visual",
            Channel::VisualTag => "visual_tag",
            Channel::DocumentType => "document_type",
            Channel::Entity => "entity",
            Channel::Metadata => "metadata",
            Channel::Filename => "filename",
        }
    }
}

fn weights(intent: QueryIntent) -> HashMap<Channel, f32> {
    use Channel::*;
    let table: &[(Channel, f32)] = match intent {
        QueryIntent::ExactIdentifier => &[
            (Exact, 0.55),
            (Entity, 0.20),
            (Metadata, 0.15),
            (Semantic, 0.05),
            (Visual, 0.05),
        ],
        QueryIntent::Visual => &[
            (Visual, 0.75),
            (Exact, 0.08),
            (Semantic, 0.05),
            (Metadata, 0.02),
            (Filename, 0.05),
            (VisualTag, 0.05),
        ],
        // Deterministic document-intent queries: document metadata and exact
        // OCR stay primary. Visual tags can help image-only assets, but generic
        // visual prompt-bank labels are not retrieval evidence.
        QueryIntent::Category => &[
            (DocumentType, 0.30),
            (Exact, 0.25),
            (VisualTag, 0.10),
            (Semantic, 0.20),
            (Metadata, 0.10),
            (Visual, 0.05),
        ],
        QueryIntent::DateFiltered => &[
            (Semantic, 0.35),
            (Visual, 0.30),
            (Exact, 0.20),
            (Metadata, 0.10),
            (Filename, 0.05),
        ],
        _ => &[
            (Semantic, 0.20),
            (Exact, 0.20),
            (DocumentType, 0.15),
            (Entity, 0.15),
            (Visual, 0.10),
            (VisualTag, 0.05),
            (Metadata, 0.10),
            (Filename, 0.05),
        ],
    };
    table.iter().copied().collect()
}

/// Signal strength contributed by a channel for an asset.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Strength {
    None,
    Moderate,
    Strong,
}

struct ChannelRun {
    channel: Channel,
    ranked: Vec<(String, f32)>,
    latency_ms: u128,
}

impl ChannelRun {
    fn ranks(&self) -> HashMap<&str, usize> {
        self.ranked
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.as_str(), i))
            .collect()
    }
}

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
    let plan = query_intent::plan(query);
    let outcome = execute(core, query, filters)?;
    let visual = core.visual.read().clone();
    let prompt = plan.visual_prompts.first().cloned();
    let diagnostic_vector = match (visual.as_ref(), prompt.as_deref()) {
        (Some(runtime), Some(prompt)) => Some(runtime.embed_text(prompt)?),
        _ => None,
    };
    Ok(SearchDebugReport {
        query: query.to_string(),
        visual_query: plan.visual_query,
        visual_prompts: plan.visual_prompts,
        intents: outcome.intents,
        expanded_categories: outcome.expanded_categories,
        applied_filters: outcome.applied_filters,
        channels: outcome.channels,
        results: outcome.results,
        total_latency_ms: outcome.total_latency_ms,
        model_revision: "4966d353f43c64efd99580a758f946950216b6e6".into(),
        image_profile_id: ai::MOBILECLIP_IMAGE_PROFILE_ID.into(),
        text_profile_id: ai::MOBILECLIP_TEXT_PROFILE_ID.into(),
        visual_token_count: match (visual.as_ref(), prompt.as_deref()) {
            (Some(runtime), Some(prompt)) => Some(runtime.token_count(prompt)?),
            _ => None,
        },
        query_embedding_dims: diagnostic_vector.as_ref().map(Vec::len),
        query_embedding_norm: diagnostic_vector
            .as_ref()
            .map(|vector| vector.iter().map(|value| value * value).sum::<f32>().sqrt()),
        query_embedding_finite: diagnostic_vector
            .as_ref()
            .map(|vector| vector.iter().all(|value| value.is_finite())),
    })
}

pub fn execute(core: &AppCore, query: &str, filters: &SearchFilters) -> Result<FusionOutcome> {
    let overall = Instant::now();
    let plan = query_intent::plan(query);

    let mut applied_filters = Vec::new();
    if let Some(folder) = &filters.folder_id {
        applied_filters.push(format!("folder_id = {folder}"));
    }
    if !filters.extensions.is_empty() {
        applied_filters.push(format!("extensions = {}", filters.extensions.join(", ")));
    }
    let implied_exts = if filters.extensions.is_empty() {
        plan.implied_extensions.clone()
    } else {
        Vec::new()
    };
    if !implied_exts.is_empty() {
        applied_filters.push(format!("implied file types = {}", implied_exts.join(", ")));
    }
    if let Some(primary) = &plan.primary_category {
        applied_filters.push(format!("primary category = {primary}"));
    } else if plan.broad_category {
        applied_filters.push("broad category expansion".to_string());
    }

    let finish = |results, channels, filters: Vec<String>| FusionOutcome {
        results,
        intents: plan.intents.clone(),
        expanded_categories: plan.category_labels.clone(),
        applied_filters: filters,
        channels,
        total_latency_ms: overall.elapsed().as_millis(),
    };

    if query.trim().is_empty() {
        return Ok(finish(Vec::new(), Vec::new(), applied_filters));
    }

    let connection = db::connect(&core.db_path)?;
    let selection = ai::selection(&core.db_path)?;
    let text_profile_id = ai::text_embedding_profile_id(&selection.embedding_model_id);
    let candidates = search::load_candidates(&connection, filters, &text_profile_id)?;
    let briefs = db::indexed_asset_briefs(&core.db_path)?;
    // Filters must constrain every channel, including image-only assets that do
    // not appear in the filtered text-chunk candidate set.
    let allowed: HashSet<String> = briefs
        .iter()
        .filter(|brief| brief_matches_filters(brief, filters))
        .map(|brief| brief.id.clone())
        .collect();
    let brief_map: HashMap<String, AssetBrief> = briefs
        .into_iter()
        .filter(|brief| allowed.contains(&brief.id))
        .map(|brief| (brief.id.clone(), brief))
        .collect();

    // --- Text scoring inputs ---
    let fts_match = search::fts_conjunction(&plan.required_terms, &plan.exact_phrases);
    let keyword_raw = search::fts_scores(&connection, &fts_match)?;
    let ai = core.ai.read().clone();
    let query_embedding = ai
        .as_ref()
        .map(|runtime| runtime.embed_query(query.to_owned()))
        .transpose()?;

    let mut best_chunk: HashMap<String, (String, Option<i64>, f32)> = HashMap::new();
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
        if best_chunk
            .get(&c.asset_id)
            .map(|(_, _, k)| *k < rank_key)
            .unwrap_or(true)
        {
            best_chunk.insert(
                c.asset_id.clone(),
                (c.text.clone(), c.page_number, rank_key),
            );
        }
    }

    let mut runs: Vec<ChannelRun> = Vec::new();

    // Exact text (conjunction FTS).
    let t = Instant::now();
    runs.push(ChannelRun {
        channel: Channel::Exact,
        ranked: top_n(map_to_sorted(&fts_by_asset, |s| s > 0.0), TEXT_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Semantic text.
    if query_embedding.is_some() {
        let t = Instant::now();
        runs.push(ChannelRun {
            channel: Channel::Semantic,
            ranked: top_n(
                map_to_sorted(&semantic_by_asset, |s| s > 0.0),
                TEXT_CANDIDATES,
            ),
            latency_ms: t.elapsed().as_millis(),
        });
    }

    // Visual text-to-image similarity (single pass over image embeddings).
    let visual = core.visual.read().clone();
    let mut visual_by_asset: HashMap<String, f32> = HashMap::new();
    let mut visual_z_by_asset: HashMap<String, f32> = HashMap::new();
    let mut visual_region_scores: HashMap<String, Vec<f32>> = HashMap::new();
    let mut visual_best_region: HashMap<String, (i64, f32)> = HashMap::new();
    if let Some(runtime) = visual.as_ref() {
        let model_id = selection.visual_model_id.clone();

        let t = Instant::now();
        let query_vectors = plan
            .visual_prompts
            .iter()
            .map(|prompt| runtime.embed_text(prompt))
            .collect::<Result<Vec<_>>>()?;
        db::for_each_image_embedding(
            &core.db_path,
            &model_id,
            ai::VISUAL_MODEL_VERSION,
            ai::MOBILECLIP_IMAGE_PROFILE_ID,
            runtime.dims(),
            |id, page, emb| {
                if !allowed.contains(&id) {
                    return;
                }
                let vsim = query_vectors
                    .iter()
                    .map(|query_vector| search::cosine_similarity(query_vector, &emb))
                    .fold(f32::MIN, f32::max)
                    .max(0.0);
                visual_region_scores
                    .entry(id.clone())
                    .or_default()
                    .push(vsim);
                visual_best_region
                    .entry(id.clone())
                    .and_modify(|best| {
                        if vsim > best.1 {
                            *best = (page, vsim);
                        }
                    })
                    .or_insert((page, vsim));
            },
        )?;
        for (id, scores) in visual_region_scores {
            visual_by_asset.insert(id, aggregate_region_scores(&scores));
        }
        let (visual_median, visual_mad) =
            robust_location_scale(&visual_by_asset.values().copied().collect::<Vec<_>>());
        for (id, raw) in &visual_by_asset {
            visual_z_by_asset.insert(
                id.clone(),
                (*raw - visual_median) / visual_mad.max(VIS_MAD_FLOOR),
            );
        }
        if !visual_by_asset.is_empty() {
            applied_filters.push(format!(
                "visual calibration median={visual_median:.3}, mad={visual_mad:.3}"
            ));
        }
        let mut visual_hits: Vec<(String, f32)> = visual_by_asset
            .iter()
            .filter(|(_, raw)| raw.is_finite() && **raw > 0.0)
            .map(|(id, raw)| (id.clone(), *raw))
            .collect();
        visual_hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        let vt = t.elapsed().as_millis();
        runs.push(ChannelRun {
            channel: Channel::Visual,
            ranked: top_n(visual_hits, VISUAL_CANDIDATES),
            latency_ms: vt,
        });
    }

    let t = Instant::now();
    let tag_terms = visual_tag_terms(&plan);
    let tag_hits = db::visual_tag_candidates(&core.db_path, &tag_terms, ai::VISUAL_TAGGER_GENERAL)?
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect::<Vec<_>>();
    let visual_tag_by_asset: HashMap<String, f32> = tag_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::VisualTag,
        ranked: top_n(tag_hits, VISUAL_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Metadata.
    let t = Instant::now();
    let document_hits = document_type_channel(core, &plan, &allowed)?;
    let document_by_asset: HashMap<String, f32> = document_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::DocumentType,
        ranked: top_n(document_hits, METADATA_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    let t = Instant::now();
    let entity_hits = db::entity_candidates(&core.db_path, &plan.required_terms)?
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect::<Vec<_>>();
    let entity_by_asset: HashMap<String, f32> = entity_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::Entity,
        ranked: top_n(entity_hits, METADATA_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Metadata.
    let t = Instant::now();
    let metadata_hits = metadata_channel(core, &plan, &allowed)?;
    let metadata_by_asset: HashMap<String, f32> = metadata_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::Metadata,
        ranked: top_n(metadata_hits, METADATA_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // Filename / folder.
    let t = Instant::now();
    let (filename_hits, filename_phrase) = filename_channel(&plan, &brief_map);
    let filename_by_asset: HashMap<String, f32> = filename_hits.iter().cloned().collect();
    runs.push(ChannelRun {
        channel: Channel::Filename,
        ranked: top_n(filename_hits, TEXT_CANDIDATES),
        latency_ms: t.elapsed().as_millis(),
    });

    // --- Fuse (one-based RRF) ---
    let weights = weights(plan.primary_intent);
    let mut fused: HashMap<String, f32> = HashMap::new();
    for run in &runs {
        let w = weights.get(&run.channel).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        for (asset_id, rank) in run.ranks() {
            let contribution = w * (1.0 / (RRF_K + rank as f32 + 1.0));
            *fused.entry(asset_id.to_string()).or_insert(0.0) += contribution;
        }
    }

    // --- Evidence gating ---
    let mut assembled: Vec<(SearchResult, f32)> = Vec::new();
    for (asset_id, fused_score) in &fused {
        let Some(brief) = brief_map.get(asset_id) else {
            continue;
        };
        if !implied_exts.is_empty() {
            let ok = brief
                .extension
                .as_ref()
                .map(|e| implied_exts.iter().any(|i| i.eq_ignore_ascii_case(e)))
                .unwrap_or(false);
            if !ok {
                continue;
            }
        }

        let fts = fts_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let semantic = semantic_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let visual_s = visual_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let visual_z = visual_z_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let visual_tag_s = visual_tag_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let filename_s = filename_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let metadata_s = metadata_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let document_s = document_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let entity_s = entity_by_asset.get(asset_id).copied().unwrap_or(0.0);

        // Strength per channel.
        let mut strengths: Vec<(MatchReason, Strength)> = Vec::new();
        // Exact conjunction hit ⇒ all query terms present ⇒ strong.
        strengths.push((
            MatchReason::ExactText,
            if fts > 0.0 {
                Strength::Strong
            } else {
                Strength::None
            },
        ));
        // Semantic is moderate at most, never a sole qualifier.
        strengths.push((
            MatchReason::SemanticText,
            if semantic >= SEM_MODERATE {
                Strength::Moderate
            } else {
                Strength::None
            },
        ));
        strengths.push((
            MatchReason::VisualSimilarity,
            visual_strength(visual_s, plan.visual_query),
        ));
        strengths.push((
            MatchReason::VisualTag,
            visual_tag_strength(visual_tag_s, plan.visual_query, is_document_query(&plan)),
        ));
        strengths.push((
            MatchReason::Metadata,
            if metadata_s >= 2.0 {
                Strength::Strong
            } else if metadata_s > 0.0 {
                Strength::Moderate
            } else {
                Strength::None
            },
        ));
        let filename_strength = if filename_phrase && filename_s >= 0.999 {
            Strength::Strong
        } else if filename_s > 0.0 {
            Strength::Moderate
        } else {
            Strength::None
        };
        strengths.push((MatchReason::Filename, filename_strength));
        strengths.push((
            MatchReason::DocumentType,
            if document_s >= 0.80 {
                Strength::Strong
            } else if document_s > 0.0 {
                Strength::Moderate
            } else {
                Strength::None
            },
        ));
        strengths.push((
            MatchReason::Entity,
            if entity_s >= 0.80 {
                Strength::Strong
            } else if entity_s > 0.0 {
                Strength::Moderate
            } else {
                Strength::None
            },
        ));

        // Qualify: one strong, OR two independent moderate. Semantic alone never
        // qualifies (it is only ever Moderate, so it needs a second signal).
        let visual_candidate = plan.visual_query && visual_s.is_finite() && visual_s > 0.0;
        let Some(confidence) = confidence_for(strengths.iter().map(|(_, signal)| *signal))
            .or_else(|| visual_candidate.then_some("candidate"))
        else {
            continue;
        };

        let reasons: Vec<MatchReason> = strengths
            .iter()
            .filter(|(_, s)| *s != Strength::None)
            .map(|(r, _)| *r)
            .collect();

        let (snippet_text, page_number) = match best_chunk.get(asset_id) {
            Some((text, page, _)) => (search::snippet(text, query), *page),
            None => (String::new(), None),
        };
        let top_categories = top_categories_for(core, asset_id).unwrap_or_default();
        let top_visual_tags = top_visual_tags_for(core, asset_id).unwrap_or_default();

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
        result.visual_z_score = visual_z;
        result.visual_region_id = visual_best_region.get(asset_id).map(|(region, _)| *region);
        result.match_reasons = reasons;
        result.top_categories = top_categories;
        result.top_visual_tags = top_visual_tags;
        result.confidence = confidence.to_string();
        result.thumbnail_available = brief.thumbnail_available;
        result.alternate_location_count = brief_map
            .values()
            .filter(|candidate| candidate.content_id == brief.content_id)
            .count()
            .saturating_sub(1);
        assembled.push((result, *fused_score));
    }

    assembled.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut seen_content = HashSet::new();
    assembled.retain(|(result, _)| {
        brief_map
            .get(&result.asset_id)
            .map(|brief| seen_content.insert(brief.content_id.clone()))
            .unwrap_or(false)
    });
    assembled.truncate(MAX_RESULTS);

    // combined_score kept only for internal ordering; do not present as a %.
    let max_fused = assembled
        .first()
        .map(|(_, s)| *s)
        .unwrap_or(1.0)
        .max(f32::EPSILON);
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
    Ok(finish(results, channels, applied_filters))
}

fn visual_strength(raw: f32, visual_query: bool) -> Strength {
    if visual_query && raw.is_finite() && raw > 0.0 {
        Strength::Moderate
    } else {
        Strength::None
    }
}

fn visual_tag_strength(score: f32, visual_query: bool, document_query: bool) -> Strength {
    if document_query {
        Strength::None
    } else if score >= 0.30 || (visual_query && score >= 0.25) {
        Strength::Moderate
    } else {
        Strength::None
    }
}

fn robust_location_scale(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, VIS_MAD_FLOOR);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let median = median_of_sorted(&sorted);
    let mut deviations: Vec<f32> = sorted.iter().map(|value| (*value - median).abs()).collect();
    deviations.sort_by(f32::total_cmp);
    // 1.4826 turns MAD into a standard-deviation-like robust scale.
    (median, median_of_sorted(&deviations) * 1.4826)
}

/// Aggregate region-level similarities to one parent asset. The best region is
/// decisive for localized evidence, while the second-best score regularizes a
/// one-crop accident. This is more stable than a raw maximum yet does not dilute
/// a ticket panel or text block inside a tall screenshot.
fn aggregate_region_scores(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let best = sorted[0];
    let supporting = if sorted.len() > 1 { sorted[1] } else { best };
    0.8 * best + 0.2 * supporting
}

fn median_of_sorted(values: &[f32]) -> f32 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn confidence_for(strengths: impl IntoIterator<Item = Strength>) -> Option<&'static str> {
    let (mut strong, mut moderate) = (0, 0);
    for signal in strengths {
        match signal {
            Strength::Strong => strong += 1,
            Strength::Moderate => moderate += 1,
            Strength::None => {}
        }
    }
    if strong > 0 {
        Some("strong")
    } else if moderate >= 2 {
        Some("moderate")
    } else {
        None
    }
}

/// Lazily build / fetch the cached prompt-embedding bank.
#[cfg(test)]
fn prompt_bank(
    core: &AppCore,
    runtime: &crate::visual::VisualRuntime,
    model_id: &str,
) -> Result<Arc<PromptBank>> {
    if let Some(bank) = core.visual_prompts.read().clone() {
        return Ok(bank);
    }
    let dir = ai::visual_directory(&core.model_dir, model_id);
    let built = Arc::new(PromptBank::load_or_build(runtime, &dir)?);
    *core.visual_prompts.write() = Some(built.clone());
    Ok(built)
}

fn metadata_channel(
    core: &AppCore,
    plan: &QueryPlan,
    allowed: &HashSet<String>,
) -> Result<Vec<(String, f32)>> {
    if plan.required_terms.is_empty() {
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
            let matched = plan
                .required_terms
                .iter()
                .filter(|t| hay.contains(*t))
                .count();
            if matched > 0 {
                hits.push((asset_id.clone(), matched as f32));
            }
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(hits)
}

fn document_type_channel(
    core: &AppCore,
    plan: &QueryPlan,
    allowed: &HashSet<String>,
) -> Result<Vec<(String, f32)>> {
    let types = document_types_for(plan);
    let mut hits = db::document_type_candidates(&core.db_path, &types)?
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(hits)
}

fn document_types_for(plan: &QueryPlan) -> Vec<String> {
    let mut types = Vec::new();
    for label in &plan.category_labels {
        let mapped = match label.as_str() {
            "train_ticket" => Some("train_ticket"),
            "flight_ticket" => Some("flight_ticket"),
            "hotel_reservation" => Some("hotel_booking"),
            "invoice" => Some("invoice"),
            "payment_receipt" => Some("receipt"),
            _ => None,
        };
        if let Some(kind) = mapped {
            if !types.iter().any(|existing| existing == kind) {
                types.push(kind.to_string());
            }
        }
    }
    types
}

fn is_document_query(plan: &QueryPlan) -> bool {
    !document_types_for(plan).is_empty()
}

/// Returns (hits, whole_phrase_present_flag_used_for_strong).
fn filename_channel(
    plan: &QueryPlan,
    briefs: &HashMap<String, AssetBrief>,
) -> (Vec<(String, f32)>, bool) {
    if plan.required_terms.is_empty() {
        return (Vec::new(), false);
    }
    let phrase = plan.exact_phrases.first().cloned();
    let uses_phrase = phrase.is_some();
    let mut hits: Vec<(String, f32)> = Vec::new();
    for brief in briefs.values() {
        let hay = format!("{} {}", brief.filename, brief.source_path).to_lowercase();
        let matched = plan
            .required_terms
            .iter()
            .filter(|t| hay.contains(*t))
            .count();
        if matched == 0 {
            continue;
        }
        let mut score = matched as f32 / plan.required_terms.len() as f32;
        if let Some(p) = &phrase {
            if hay.contains(p.as_str()) {
                score = 1.0; // full phrase in filename → strong
            }
        }
        hits.push((brief.id.clone(), score));
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    (hits, uses_phrase)
}

fn top_categories_for(core: &AppCore, asset_id: &str) -> Result<Vec<VisualCategory>> {
    let model_id = ai::selection(&core.db_path)?.visual_model_id;
    if model_id == ai::VISUAL_DISABLED {
        return Ok(Vec::new());
    }
    let mut cats = db::classifications_for(&core.db_path, asset_id, &model_id)?;
    cats.truncate(3);
    Ok(cats)
}

fn top_visual_tags_for(core: &AppCore, asset_id: &str) -> Result<Vec<crate::types::VisualTag>> {
    let mut tags = db::visual_tags_for(&core.db_path, asset_id, ai::VISUAL_TAGGER_GENERAL)?;
    tags.retain(|tag| !tag.namespace.ends_with(":rating"));
    tags.truncate(6);
    Ok(tags)
}

fn visual_tag_terms(plan: &QueryPlan) -> Vec<String> {
    let mut terms = Vec::new();
    for term in &plan.required_terms {
        push_term(&mut terms, term);
        if let Some(singular) = singularize(term) {
            push_term(&mut terms, &singular);
        }
    }
    if plan.required_terms.len() > 1 {
        let phrase = plan.required_terms.join(" ");
        push_term(&mut terms, &phrase);
    }
    terms
}

fn push_term(terms: &mut Vec<String>, term: &str) {
    let normalized = db::normalize_visual_tag(term);
    if !normalized.is_empty() && !terms.iter().any(|existing| existing == &normalized) {
        terms.push(normalized);
    }
}

fn singularize(term: &str) -> Option<String> {
    if term.ends_with("ies") && term.len() > 4 {
        Some(format!("{}y", &term[..term.len() - 3]))
    } else if term.ends_with("es") && term.len() > 3 {
        Some(term[..term.len() - 2].to_string())
    } else if term.ends_with('s') && term.len() > 3 {
        Some(term[..term.len() - 1].to_string())
    } else {
        None
    }
}

fn brief_matches_filters(brief: &AssetBrief, filters: &SearchFilters) -> bool {
    if filters
        .folder_id
        .as_ref()
        .is_some_and(|folder_id| &brief.folder_id != folder_id)
    {
        return false;
    }
    filters.extensions.is_empty()
        || brief.extension.as_ref().is_some_and(|extension| {
            filters
                .extensions
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(extension))
        })
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
            normalized_score: 1.0 / (RRF_K + rank as f32 + 1.0),
        })
        .collect::<Vec<_>>();
    ChannelDiagnostics {
        channel: run.channel.key().to_string(),
        latency_ms: run.latency_ms,
        candidate_count: run.ranked.len(),
        results,
    }
}

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
    fn one_based_rrf_prefers_top_ranks() {
        assert!(1.0 / (RRF_K + 1.0) > 1.0 / (RRF_K + 10.0));
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

    #[test]
    fn semantic_alone_does_not_qualify() {
        // One moderate signal (semantic) → not enough; needs a strong or 2 moderate.
        assert_eq!(confidence_for([Strength::Moderate]), None);
    }

    #[test]
    fn one_strong_signal_qualifies_as_strong() {
        assert_eq!(confidence_for([Strength::Strong]), Some("strong"));
    }

    #[test]
    fn two_moderate_signals_qualify_as_moderate() {
        assert_eq!(
            confidence_for([Strength::Moderate, Strength::Moderate]),
            Some("moderate")
        );
    }

    #[test]
    fn no_evidence_produces_no_result() {
        assert_eq!(confidence_for([Strength::None, Strength::None]), None);
    }

    #[test]
    fn open_visual_candidate_does_not_require_a_distribution_outlier() {
        assert_eq!(visual_strength(0.18, true), Strength::Moderate);
        assert_eq!(visual_strength(0.18, false), Strength::None);
    }

    #[test]
    fn zero_or_invalid_visual_scores_do_not_qualify() {
        assert_eq!(visual_strength(0.0, true), Strength::None);
        assert_eq!(visual_strength(f32::NAN, true), Strength::None);
    }

    #[test]
    fn visual_tags_are_never_strong_or_document_evidence() {
        assert_eq!(visual_tag_strength(0.44, true, false), Strength::Moderate);
        assert_eq!(visual_tag_strength(0.55, false, false), Strength::Moderate);
        assert_eq!(visual_tag_strength(0.90, true, true), Strength::None);
    }

    #[test]
    fn visual_tag_terms_normalize_simple_plurals_only() {
        let cats = query_intent::plan("cats");
        assert_eq!(visual_tag_terms(&cats), vec!["cats", "cat"]);
        let buildings = query_intent::plan("buildings");
        assert_eq!(visual_tag_terms(&buildings), vec!["buildings", "building"]);
    }

    #[test]
    fn region_aggregation_requires_some_support_without_losing_local_match() {
        let localized = aggregate_region_scores(&[0.90, 0.40, 0.10]);
        assert!(localized > 0.75);
        assert!(localized < 0.90);
    }

    #[test]
    #[ignore = "requires RECALL_APP_DATA pointing at an installed local Recall data directory"]
    fn benchmark_installed_open_visual_queries() -> Result<()> {
        use std::path::PathBuf;
        use std::sync::{atomic::AtomicBool, Arc};

        let root = PathBuf::from(
            std::env::var("RECALL_APP_DATA")
                .map_err(|_| "Set RECALL_APP_DATA to the Recall application-data directory")?,
        );
        let db_path = root.join("recall.db");
        db::migrate(&db_path)?;
        let model_dir = std::env::var("RECALL_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.join("models"));
        let selection = ai::selection(&db_path)?;
        let text_runtime = Arc::new(ai::AiRuntime::load_for_selection(&model_dir, &selection)?);
        let visual_runtime = ai::load_visual_for_selection(&model_dir, &selection)?
            .ok_or("Visual model is disabled")?;
        let core = AppCore {
            db_path,
            model_dir,
            thumbnail_dir: root.join("thumbnails"),
            paused: AtomicBool::new(true),
            worker_running: AtomicBool::new(false),
            model_installing: AtomicBool::new(false),
            ai: parking_lot::RwLock::new(Some(text_runtime)),
            visual: parking_lot::RwLock::new(Some(visual_runtime)),
            visual_tagger: parking_lot::RwLock::new(None),
            visual_prompts: parking_lot::RwLock::new(None),
        };

        let mut top_tens: HashMap<&str, Vec<HashSet<String>>> = HashMap::new();
        let mut query_latencies = Vec::new();
        for (query, label) in [
            ("dog", "dog"),
            ("dogs", "dog"),
            ("cat", "cat"),
            ("cats", "cat"),
            ("building", "building"),
            ("buildings", "building"),
        ] {
            let plan = query_intent::plan(query);
            let runtime = core.visual.read().clone().expect("visual runtime");
            let vectors = plan
                .visual_prompts
                .iter()
                .map(|prompt| runtime.embed_text(prompt))
                .collect::<Result<Vec<_>>>()?;
            let model_id = ai::selection(&core.db_path)?.visual_model_id;
            let embeddings = db::load_image_embeddings(
                &core.db_path,
                &model_id,
                ai::VISUAL_MODEL_VERSION,
                ai::MOBILECLIP_IMAGE_PROFILE_ID,
                runtime.dims(),
            )?;
            let bank = prompt_bank(&core, &runtime, &model_id)?;
            let mut scores: Vec<(String, f32, f32, f32)> = embeddings
                .iter()
                .map(|(id, _, embedding)| {
                    let raw = vectors
                        .iter()
                        .map(|vector| search::cosine_similarity(vector, embedding))
                        .fold(f32::MIN, f32::max);
                    let (positive, _, margin) =
                        bank.category_set_margin(embedding, &plan.category_labels);
                    (id.clone(), raw, positive, margin)
                })
                .collect();
            let (median, mad) = robust_location_scale(
                &scores.iter().map(|(_, raw, _, _)| *raw).collect::<Vec<_>>(),
            );
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!(
                "VISUAL_DISTRIBUTION query={query:?} median={median:.3} mad={mad:.3} top={:?}",
                scores.first().map(|(_, raw, positive, margin)| (
                    *raw,
                    (*raw - median) / mad.max(VIS_MAD_FLOOR),
                    *positive,
                    *margin,
                ))
            );
            let query_started = Instant::now();
            let results = search(&core, query, &SearchFilters::default())?;
            query_latencies.push(query_started.elapsed().as_millis());
            eprintln!(
                "VISUAL_QUERY query={query:?} results={} top={:?}",
                results.len(),
                results.first().map(|result| (
                    result.filename.as_str(),
                    result.visual_score,
                    result.visual_z_score,
                    result.category_score,
                ))
            );
            let relevant = |filename: &str| {
                filename
                    .to_ascii_lowercase()
                    .starts_with(&format!("{label}_"))
            };
            assert!(
                results
                    .first()
                    .map(|result| relevant(&result.filename))
                    .unwrap_or(false),
                "{query:?} top result was not labeled {label:?}"
            );
            let precision_at_5 = results
                .iter()
                .take(5)
                .filter(|result| relevant(&result.filename))
                .count() as f32
                / 5.0;
            let connection = db::connect(&core.db_path)?;
            let pattern = format!("{label}_%");
            let relevant_total: i64 = connection.query_row(
                "SELECT COUNT(*) FROM assets WHERE available=1 AND LOWER(filename) LIKE ?1",
                [pattern],
                |row| row.get(0),
            )?;
            let recall_at_10 = results
                .iter()
                .take(10)
                .filter(|result| relevant(&result.filename))
                .count() as f32
                / relevant_total.max(1) as f32;
            eprintln!(
                "VISUAL_METRICS query={query:?} p@5={precision_at_5:.2} r@10={recall_at_10:.2}"
            );
            assert!(
                precision_at_5 >= 0.80,
                "{query:?} precision@5={precision_at_5}"
            );
            assert!(recall_at_10 >= 0.75, "{query:?} recall@10={recall_at_10}");
            top_tens.entry(label).or_default().push(
                results
                    .iter()
                    .take(10)
                    .map(|result| result.asset_id.clone())
                    .collect(),
            );
        }
        for (label, sets) in top_tens {
            let intersection = sets[0].intersection(&sets[1]).count() as f32;
            let union = sets[0].union(&sets[1]).count().max(1) as f32;
            let jaccard = intersection / union;
            eprintln!("VISUAL_STABILITY label={label:?} top10_jaccard={jaccard:.2}");
            assert!(
                jaccard >= 0.60,
                "{label:?} singular/plural jaccard={jaccard}"
            );
        }
        query_latencies.sort_unstable();
        eprintln!(
            "VISUAL_SEARCH_LATENCY corpus_assets=61 p50_ms={} p95_ms={}",
            query_latencies[query_latencies.len() / 2],
            query_latencies[query_latencies.len() - 1]
        );
        Ok(())
    }

    #[test]
    #[ignore = "allocates up to ~195 MiB to measure the exact 512-d scan kernel"]
    fn benchmark_exact_visual_scan_scaling() {
        let dims = crate::visual::encoder::EMBED_DIMS;
        let unit = 1.0_f32 / (dims as f32).sqrt();
        let query = vec![unit; dims];
        for count in [1_000_usize, 10_000, 100_000] {
            let matrix = vec![unit; count * dims];
            let mut samples = Vec::new();
            for _ in 0..5 {
                let started = Instant::now();
                let best = matrix
                    .chunks_exact(dims)
                    .map(|vector| vector.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::MIN, f32::max);
                std::hint::black_box(best);
                samples.push(started.elapsed().as_millis());
            }
            samples.sort_unstable();
            eprintln!(
                "VISUAL_SCAN vectors={count} bytes={} p50_ms={} p95_ms={}",
                matrix.len() * std::mem::size_of::<f32>(),
                samples[samples.len() / 2],
                samples[samples.len() - 1]
            );
        }
    }
}
