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

    /// Score a (L2-normalized) image embedding against every category, returning
    /// the top [`TOP_CATEGORIES`] best-first.
    pub fn classify(&self, image_embedding: &[f32]) -> Vec<VisualCategory> {
        use std::collections::HashMap;
        // Collect per-label prompt scores (cosine == dot for normalized vectors).
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
        scored.truncate(TOP_CATEGORIES);
        scored
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
