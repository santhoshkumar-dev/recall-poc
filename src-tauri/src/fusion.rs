//! Multimodal retrieval fusion with evidence gating.
//!
//! Builds a deterministic [`QueryPlan`], runs the retrieval channels (exact
//! text via conjunction FTS, semantic text, visual, visual-category margin,
//! metadata, filename), and combines them with intent-aware one-based
//! Reciprocal-Rank Fusion. A result is returned only if it clears an evidence
//! bar (one strong OR two moderate signals; semantic-text alone never
//! qualifies). When nothing clears the bar the result set is empty — no weak
//! result is ever normalized into a "100% match".

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    ai, db,
    error::Result,
    query_intent::{self, QueryPlan},
    search,
    types::{
        AssetBrief, ChannelDiagnostics, ChannelResult, MatchReason, QueryIntent, SearchDebugReport,
        SearchFilters, SearchResult, VisualCategory,
    },
    visual::PromptBank,
    AppCore,
};

const RRF_K: f32 = 60.0;
const TEXT_CANDIDATES: usize = 50;
const VISUAL_CANDIDATES: usize = 50;
const CATEGORY_CANDIDATES: usize = 40;
const METADATA_CANDIDATES: usize = 40;
const MAX_RESULTS: usize = 20;

// --- Calibrated evidence thresholds (tune via the inspector; Phase 7 eval) ---
// Semantic (E5) similarity: moderate only, never a sole qualifier.
const SEM_MODERATE: f32 = 0.85;
// Visual (MobileCLIP text→image) cosine.
const VIS_RAW_INCLUDE: f32 = 0.10;
const VIS_RAW_MODERATE: f32 = 0.11;
const VIS_RAW_STRONG: f32 = 0.14;
const VIS_Z_INCLUDE: f32 = 2.5;
const VIS_Z_MODERATE: f32 = 3.0;
const VIS_Z_STRONG: f32 = 3.5;
const VIS_MAD_FLOOR: f32 = 0.01;
// Visual-category margin (primary − best-other).
const CAT_POSITIVE_FLOOR: f32 = 0.10;
const CAT_MARGIN_STRONG: f32 = 0.025;
const CAT_MARGIN_MODERATE: f32 = 0.01;
const CAT_MARGIN_INCLUDE: f32 = 0.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Channel {
    Exact,
    Semantic,
    Visual,
    VisualTag,
    Category,
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
            Channel::Category => "visual_category",
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
            (Visual, 0.45),
            (VisualTag, 0.30),
            (Category, 0.05),
            (Semantic, 0.08),
            (Metadata, 0.07),
            (Filename, 0.05),
        ],
        // Specific/broad category queries: category-led, with strong exact text.
        QueryIntent::Category => &[
            (Category, 0.20),
            (DocumentType, 0.25),
            (Exact, 0.23),
            (VisualTag, 0.10),
            (Semantic, 0.15),
            (Metadata, 0.05),
            (Visual, 0.02),
        ],
        QueryIntent::DateFiltered => &[
            (Semantic, 0.30),
            (Category, 0.25),
            (Visual, 0.25),
            (Exact, 0.15),
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
    let candidates = search::load_candidates(&connection, filters)?;
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

    // Visual + category-margin (single pass over image embeddings).
    let visual = core.visual.read().clone();
    let mut visual_by_asset: HashMap<String, f32> = HashMap::new();
    let mut visual_z_by_asset: HashMap<String, f32> = HashMap::new();
    let mut category_by_asset: HashMap<String, f32> = HashMap::new();
    let mut category_positive_by_asset: HashMap<String, f32> = HashMap::new();
    let mut category_negative_by_asset: HashMap<String, f32> = HashMap::new();
    let mut visual_region_scores: HashMap<String, Vec<f32>> = HashMap::new();
    let mut category_region_scores: HashMap<String, Vec<(f32, f32, f32)>> = HashMap::new();
    let mut visual_best_region: HashMap<String, (i64, f32)> = HashMap::new();
    if let Some(runtime) = visual.as_ref() {
        let model_id = ai::selection(&core.db_path)?.visual_model_id;
        let bank = prompt_bank(core, runtime, &model_id).ok();

        let t = Instant::now();
        let query_vectors = plan
            .visual_prompts
            .iter()
            .map(|prompt| runtime.embed_text(prompt))
            .collect::<Result<Vec<_>>>()?;
        let embeddings = db::load_image_embeddings(&core.db_path, &model_id)?;
        for (id, page, emb) in &embeddings {
            if !allowed.contains(id) {
                continue;
            }
            let vsim = query_vectors
                .iter()
                .map(|query_vector| search::cosine_similarity(query_vector, emb))
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
                        *best = (*page, vsim);
                    }
                })
                .or_insert((*page, vsim));
            if !plan.category_labels.is_empty() {
                if let Some(bank) = bank.as_ref() {
                    let (positive, negative, margin) =
                        bank.category_set_margin(emb, &plan.category_labels);
                    category_region_scores
                        .entry(id.clone())
                        .or_default()
                        .push((positive, negative, margin));
                }
            }
        }
        for (id, scores) in visual_region_scores {
            visual_by_asset.insert(id, aggregate_region_scores(&scores));
        }
        for (id, scores) in category_region_scores {
            // Keep all three values from the same best region; combining the
            // best positive and best negative from different crops distorts the
            // category margin for screenshots.
            if let Some((positive, negative, margin)) = scores
                .into_iter()
                .max_by(|left, right| left.2.total_cmp(&right.2))
            {
                category_positive_by_asset.insert(id.clone(), positive);
                category_negative_by_asset.insert(id.clone(), negative);
                category_by_asset.insert(id, margin);
            }
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
            .filter(|(id, raw)| {
                **raw >= VIS_RAW_INCLUDE
                    && visual_z_by_asset.get(*id).copied().unwrap_or(0.0) >= VIS_Z_INCLUDE
            })
            .map(|(id, raw)| (id.clone(), *raw))
            .collect();
        visual_hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        let vt = t.elapsed().as_millis();
        runs.push(ChannelRun {
            channel: Channel::Visual,
            ranked: top_n(visual_hits, VISUAL_CANDIDATES),
            latency_ms: vt,
        });

        if !plan.category_labels.is_empty() && bank.is_some() {
            let t = Instant::now();
            let mut category_hits: Vec<(String, f32)> = category_by_asset
                .iter()
                .filter(|(id, margin)| {
                    **margin >= CAT_MARGIN_INCLUDE
                        && category_positive_by_asset.get(*id).copied().unwrap_or(0.0)
                            >= CAT_POSITIVE_FLOOR
                })
                .map(|(id, margin)| (id.clone(), *margin))
                .collect();
            category_hits.sort_by(|a, b| b.1.total_cmp(&a.1));
            runs.push(ChannelRun {
                channel: Channel::Category,
                ranked: top_n(category_hits, CATEGORY_CANDIDATES),
                latency_ms: t.elapsed().as_millis(),
            });
        }
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
        let category_s = category_by_asset.get(asset_id).copied().unwrap_or(0.0);
        let category_positive = category_positive_by_asset
            .get(asset_id)
            .copied()
            .unwrap_or(0.0);
        let category_negative = category_negative_by_asset
            .get(asset_id)
            .copied()
            .unwrap_or(0.0);
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
            visual_strength(visual_s, visual_z, plan.visual_query),
        ));
        strengths.push((
            MatchReason::VisualTag,
            visual_tag_strength(visual_tag_s, plan.visual_query, is_document_query(&plan)),
        ));
        // A generic visual ticket label alone is never enough for a sensitive
        // document query. OCR/entity/document-type evidence must corroborate it.
        let cat_strength = if is_document_query(&plan) {
            match category_strength(category_positive, category_s) {
                Strength::Strong => Strength::Moderate,
                value => value,
            }
        } else {
            category_strength(category_positive, category_s)
        };
        strengths.push((MatchReason::VisualCategory, cat_strength));
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
        let Some(confidence) = confidence_for(strengths.iter().map(|(_, signal)| *signal)) else {
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
        result.category_score = category_s;
        result.category_positive_score = category_positive;
        result.category_negative_score = category_negative;
        result.match_reasons = reasons;
        result.top_categories = top_categories;
        result.top_visual_tags = top_visual_tags;
        result.confidence = confidence.to_string();
        assembled.push((result, *fused_score));
    }

    assembled.sort_by(|a, b| b.1.total_cmp(&a.1));
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

fn strength(value: f32, strong: f32, moderate: f32) -> Strength {
    if value >= strong {
        Strength::Strong
    } else if value >= moderate {
        Strength::Moderate
    } else {
        Strength::None
    }
}

fn visual_strength(raw: f32, robust_z: f32, visual_query: bool) -> Strength {
    if visual_query && raw >= VIS_RAW_STRONG && robust_z >= VIS_Z_STRONG {
        Strength::Strong
    } else if raw >= VIS_RAW_MODERATE && robust_z >= VIS_Z_MODERATE {
        Strength::Moderate
    } else {
        Strength::None
    }
}

fn category_strength(positive: f32, margin: f32) -> Strength {
    if positive < CAT_POSITIVE_FLOOR {
        Strength::None
    } else {
        strength(margin, CAT_MARGIN_STRONG, CAT_MARGIN_MODERATE)
    }
}

fn visual_tag_strength(score: f32, visual_query: bool, document_query: bool) -> Strength {
    if document_query {
        Strength::None
    } else if score >= 0.55 || (visual_query && score >= 0.45) {
        Strength::Strong
    } else if score >= 0.30 {
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
    fn robust_visual_outlier_can_qualify_open_visual_query() {
        let values = [0.10, 0.10, 0.10, 0.10, 0.18];
        let (median, mad) = robust_location_scale(&values);
        let z = (0.18 - median) / mad.max(VIS_MAD_FLOOR);
        assert_eq!(visual_strength(0.18, z, true), Strength::Strong);
        assert_eq!(visual_strength(0.18, z, false), Strength::Moderate);
    }

    #[test]
    fn flat_visual_distribution_does_not_qualify() {
        let values = [0.12, 0.12, 0.12, 0.12];
        let (median, mad) = robust_location_scale(&values);
        let z = (0.12 - median) / mad.max(VIS_MAD_FLOOR);
        assert_eq!(visual_strength(0.12, z, true), Strength::None);
    }

    #[test]
    fn category_strength_uses_positive_floor_and_margin() {
        assert_eq!(category_strength(0.09, 0.05), Strength::None);
        assert_eq!(category_strength(0.12, 0.015), Strength::Moderate);
        assert_eq!(category_strength(0.14, 0.03), Strength::Strong);
    }

    #[test]
    fn visual_tag_strength_can_qualify_open_visual_queries() {
        assert_eq!(visual_tag_strength(0.44, true, false), Strength::Moderate);
        assert_eq!(visual_tag_strength(0.45, true, false), Strength::Strong);
        assert_eq!(visual_tag_strength(0.55, false, false), Strength::Strong);
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
        let model_dir = root.join("models");
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

        for query in ["dogs", "cats", "buildings"] {
            let plan = query_intent::plan(query);
            let runtime = core.visual.read().clone().expect("visual runtime");
            let vectors = plan
                .visual_prompts
                .iter()
                .map(|prompt| runtime.embed_text(prompt))
                .collect::<Result<Vec<_>>>()?;
            let model_id = ai::selection(&core.db_path)?.visual_model_id;
            let embeddings = db::load_image_embeddings(&core.db_path, &model_id)?;
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
            let results = search(&core, query, &SearchFilters::default())?;
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
            assert!(!results.is_empty(), "{query:?} returned no visual results");
        }
        Ok(())
    }
}
