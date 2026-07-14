//! Prompt-ensemble zero-shot classification.
//!
//! Each category has several text prompts. We embed every prompt once with the
//! CLIP text encoder, cache the embeddings on disk (keyed by prompt-bank
//! version), and score an image against them. A category's score is the MAX
//! cosine over its prompts (an average-of-top-2 alternative is available via
//! [`SCORE_MODE`]). Scores are visual-match strengths, NOT probabilities.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::VisualCategory;
use crate::visual::{category_prompts, VisualRuntime};

/// Number of top categories persisted per image.
pub const TOP_CATEGORIES: usize = 5;

#[allow(dead_code)] // AvgTop2 is an alternative scoring mode selected via SCORE_MODE.
enum ScoreMode {
    Max,
    AvgTop2,
}
const SCORE_MODE: ScoreMode = ScoreMode::Max;

#[derive(Serialize, Deserialize)]
struct PromptVector {
    label: String,
    vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct PromptCache {
    version: String,
    prompts: Vec<PromptVector>,
}

fn cache_path(visual_dir: &Path, version: &str) -> PathBuf {
    visual_dir.join(format!("prompt_embeddings_v{version}.json"))
}

/// Cached prompt embeddings ready for scoring.
pub struct PromptBank {
    prompts: Vec<PromptVector>,
}

impl PromptBank {
    /// Load the prompt bank for the current version from disk, building and
    /// caching it (one text-encode per prompt) on a miss.
    pub fn load_or_build(runtime: &VisualRuntime, visual_dir: &Path) -> Result<Self> {
        let version = category_prompts::PROMPT_BANK_VERSION;
        let path = cache_path(visual_dir, version);
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(cache) = serde_json::from_slice::<PromptCache>(&bytes) {
                if cache.version == version && !cache.prompts.is_empty() {
                    return Ok(Self {
                        prompts: cache.prompts,
                    });
                }
            }
        }
        // Build: embed every prompt once.
        let mut prompts = Vec::new();
        for (label, prompt) in category_prompts::flattened_prompts() {
            let vector = runtime.embed_text(prompt)?;
            prompts.push(PromptVector {
                label: label.to_string(),
                vector,
            });
        }
        let cache = PromptCache {
            version: version.to_string(),
            prompts,
        };
        if let Ok(bytes) = serde_json::to_vec(&cache) {
            let _ = std::fs::write(&path, bytes);
        }
        Ok(Self {
            prompts: cache.prompts,
        })
    }

    /// Per-label score (max prompt cosine) for a normalized image embedding.
    pub fn label_scores(&self, image_embedding: &[f32]) -> Vec<VisualCategory> {
        use std::collections::HashMap;
        let mut per_label: HashMap<&str, Vec<f32>> = HashMap::new();
        for prompt in &self.prompts {
            let score = dot(image_embedding, &prompt.vector);
            per_label.entry(&prompt.label).or_default().push(score);
        }
        let mut scored: Vec<VisualCategory> = per_label
            .into_iter()
            .map(|(label, mut scores)| {
                scores.sort_by(|a, b| b.total_cmp(a));
                let score = match SCORE_MODE {
                    ScoreMode::Max => scores.first().copied().unwrap_or(0.0),
                    ScoreMode::AvgTop2 => {
                        let n = scores.len().min(2).max(1);
                        scores.iter().take(n).sum::<f32>() / n as f32
                    }
                };
                VisualCategory {
                    label: label.to_string(),
                    score,
                }
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored
    }

    /// Top [`TOP_CATEGORIES`] categories (for persisted classification).
    pub fn classify(&self, image_embedding: &[f32]) -> Vec<VisualCategory> {
        let mut scored = self.label_scores(image_embedding);
        scored.truncate(TOP_CATEGORIES);
        scored
    }

    /// Aggregate classifications across whole images and their regions. A
    /// localized document cue (for example a ticket panel in a tall screenshot)
    /// should be retained without making the parent result appear multiple times.
    pub fn classify_regions<'a>(
        &self,
        embeddings: impl IntoIterator<Item = &'a [f32]>,
    ) -> Vec<VisualCategory> {
        let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for embedding in embeddings {
            for category in self.label_scores(embedding) {
                best.entry(category.label)
                    .and_modify(|score| *score = score.max(category.score))
                    .or_insert(category.score);
            }
        }
        let mut categories = best
            .into_iter()
            .map(|(label, score)| VisualCategory { label, score })
            .collect::<Vec<_>>();
        categories.sort_by(|a, b| b.score.total_cmp(&a.score));
        categories.truncate(TOP_CATEGORIES);
        categories
    }

    /// Discriminative margin for `primary` label: its prompt similarity minus
    /// the best competing category's similarity. Positive when the image looks
    /// more like `primary` than anything else; negative for unrelated images.
    /// Returns (positive_score, best_other_score, margin).
    pub fn category_margin(&self, image_embedding: &[f32], primary: &str) -> (f32, f32, f32) {
        self.category_set_margin(image_embedding, &[primary.to_string()])
    }

    /// Margin between the best requested category and the best category not in
    /// the requested set. Supports both specific and broad category plans.
    pub fn category_set_margin(
        &self,
        image_embedding: &[f32],
        requested: &[String],
    ) -> (f32, f32, f32) {
        let scores = self.label_scores(image_embedding);
        let pos = scores
            .iter()
            .filter(|c| requested.iter().any(|label| label == &c.label))
            .map(|c| c.score)
            .fold(f32::MIN, f32::max)
            .max(0.0);
        let neg = scores
            .iter()
            .filter(|c| !requested.iter().any(|label| label == &c.label))
            .map(|c| c.score)
            .fold(f32::MIN, f32::max)
            .max(0.0);
        (pos, neg, pos - neg)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
