//! MobileCLIP2-S0 image and text encoders (ONNX via `ort`).
//!
//! Loads the paired vision/text ONNX models plus the CLIP BPE tokenizer, and
//! produces L2-normalized 512-d embeddings in a shared cross-modal space —
//! entirely separate from the E5 text space.

use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

use image::RgbImage;
use ndarray::{Array, Array2};
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::error::{RecallError, Result};
use crate::visual::preprocess;

/// CLIP text context length.
const CONTEXT_LENGTH: usize = 77;
/// Embedding dimensionality for MobileCLIP2-S0.
pub const EMBED_DIMS: usize = 512;

/// Bound visual inference to a single image at a time (memory-friendly, and the
/// indexing worker is single-threaded anyway).
static INFER_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
const TEXT_CACHE_CAPACITY: usize = 128;
const TEXT_CACHE_TTL: Duration = Duration::from_secs(20 * 60);

struct CachedTextEmbedding {
    vector: Vec<f32>,
    touched_at: Instant,
    sequence: u64,
}

#[derive(Default)]
struct TextEmbeddingCache {
    values: HashMap<String, CachedTextEmbedding>,
    sequence: u64,
}

impl TextEmbeddingCache {
    fn get(&mut self, query: &str) -> Option<Vec<f32>> {
        let now = Instant::now();
        self.values
            .retain(|_, value| now.duration_since(value.touched_at) <= TEXT_CACHE_TTL);
        let value = self.values.get_mut(query)?;
        self.sequence += 1;
        value.sequence = self.sequence;
        value.touched_at = now;
        Some(value.vector.clone())
    }

    fn insert(&mut self, query: String, vector: Vec<f32>) {
        self.sequence += 1;
        if self.values.len() >= TEXT_CACHE_CAPACITY && !self.values.contains_key(&query) {
            if let Some(oldest) = self
                .values
                .iter()
                .min_by_key(|(_, value)| value.sequence)
                .map(|(key, _)| key.clone())
            {
                self.values.remove(&oldest);
            }
        }
        self.values.insert(
            query,
            CachedTextEmbedding {
                vector,
                touched_at: Instant::now(),
                sequence: self.sequence,
            },
        );
    }
}

pub struct VisualRuntime {
    vision: Mutex<Session>,
    text: Mutex<Session>,
    tokenizer: Tokenizer,
    vision_input: String,
    text_inputs: Vec<String>,
    pad_id: u32,
    dims: usize,
    text_cache: Mutex<TextEmbeddingCache>,
}

impl VisualRuntime {
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Load the runtime from `models/visual/<model_id>/`.
    pub fn load(dir: &Path) -> Result<Self> {
        let vision = build_session(&dir.join("vision_model.onnx"), "vision")?;
        let text = build_session(&dir.join("text_model.onnx"), "text")?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| RecallError::Message(format!("Could not load CLIP tokenizer: {e}")))?;
        // Fixed 77-token context: pad + truncate.
        let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(CONTEXT_LENGTH),
            pad_id,
            pad_token: "<|endoftext|>".to_string(),
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: CONTEXT_LENGTH,
            ..Default::default()
        }));

        let vision_input = input_names(&vision.lock())
            .into_iter()
            .next()
            .ok_or_else(|| RecallError::Message("Vision model has no inputs".into()))?;
        let text_inputs = input_names(&text.lock());

        Ok(Self {
            vision,
            text,
            tokenizer,
            vision_input,
            text_inputs,
            pad_id,
            dims: EMBED_DIMS,
            text_cache: Mutex::new(TextEmbeddingCache::default()),
        })
    }

    /// Embed one image → L2-normalized 512-d vector.
    pub fn embed_image(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let _guard = INFER_LOCK.lock();
        let tensor = preprocess::image_to_tensor(image);
        let value = Tensor::from_array(tensor)
            .map_err(|e| RecallError::Message(format!("Image tensor failed: {e}")))?;
        let mut session = self.vision.lock();
        let outputs = session
            .run(ort::inputs![self.vision_input.clone() => value])
            .map_err(|e| RecallError::Message(format!("Vision inference failed: {e}")))?;
        let vector = pooled_embedding(&outputs, self.dims)?;
        Ok(preprocess::l2_normalize(vector))
    }

    /// Embed a text query in the CLIP text space → L2-normalized 512-d vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let cache_key = text.trim().to_lowercase();
        if !cache_key.is_empty() {
            if let Some(vector) = self.text_cache.lock().get(&cache_key) {
                return Ok(vector);
            }
        }
        let _guard = INFER_LOCK.lock();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RecallError::Message(format!("CLIP tokenization failed: {e}")))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let len = ids.len();
        let ids_arr: Array2<i64> = Array::from_shape_vec((1, len), ids)
            .map_err(|e| RecallError::Message(format!("Token tensor failed: {e}")))?;

        // Feed only the inputs the text model actually declares.
        let mut session = self.text.lock();
        let outputs = {
            let ids_value = Tensor::from_array(ids_arr.clone())
                .map_err(|e| RecallError::Message(format!("Token tensor failed: {e}")))?;
            if self.text_inputs.iter().any(|n| n.contains("attention")) {
                let mask_arr: Array2<i64> = Array::from_shape_vec((1, len), mask)
                    .map_err(|e| RecallError::Message(format!("Mask tensor failed: {e}")))?;
                let mask_value = Tensor::from_array(mask_arr)
                    .map_err(|e| RecallError::Message(format!("Mask tensor failed: {e}")))?;
                let id_name = self.text_inputs[0].clone();
                let mask_name = self
                    .text_inputs
                    .iter()
                    .find(|n| n.contains("attention"))
                    .cloned()
                    .unwrap();
                session
                    .run(ort::inputs![id_name => ids_value, mask_name => mask_value])
                    .map_err(|e| RecallError::Message(format!("Text inference failed: {e}")))?
            } else {
                let id_name = self.text_inputs[0].clone();
                session
                    .run(ort::inputs![id_name => ids_value])
                    .map_err(|e| RecallError::Message(format!("Text inference failed: {e}")))?
            }
        };
        let _ = self.pad_id;
        let vector = pooled_embedding(&outputs, self.dims)?;
        let vector = preprocess::l2_normalize(vector);
        if !cache_key.is_empty() {
            self.text_cache.lock().insert(cache_key, vector.clone());
        }
        Ok(vector)
    }
}

fn build_session(path: &Path, which: &str) -> Result<Mutex<Session>> {
    let session = Session::builder()
        .map_err(|e| RecallError::Message(format!("ORT builder failed: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RecallError::Message(format!("ORT opt-level failed: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| RecallError::Message(format!("ORT threads failed: {e}")))?
        .commit_from_file(path)
        .map_err(|e| {
            RecallError::Message(format!("Could not load MobileCLIP {which} model: {e}"))
        })?;
    Ok(Mutex::new(session))
}

fn input_names(session: &Session) -> Vec<String> {
    session
        .inputs()
        .iter()
        .map(|i| i.name().to_owned())
        .collect()
}

/// Select the pooled embedding output (batch size 1).
///
/// CLIP ONNX exports often emit both a pooled `image_embeds`/`text_embeds`
/// (length == dims) and a `last_hidden_state` (much larger). Prefer an output
/// whose flat length is exactly `dims`; otherwise fall back to the first.
fn pooled_embedding(outputs: &ort::session::SessionOutputs, dims: usize) -> Result<Vec<f32>> {
    let mut first: Option<Vec<f32>> = None;
    for (_, value) in outputs.iter() {
        if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
            if data.len() == dims {
                return Ok(data.to_vec());
            }
            if first.is_none() {
                first = Some(data.to_vec());
            }
        }
    }
    first.ok_or_else(|| RecallError::Message("Model produced no float output".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_embedding_cache_returns_a_copy() {
        let mut cache = TextEmbeddingCache::default();
        cache.insert("train ticket".into(), vec![0.1, 0.2]);
        let mut first = cache.get("train ticket").expect("cached vector");
        first[0] = 9.0;
        assert_eq!(cache.get("train ticket"), Some(vec![0.1, 0.2]));
    }
}
