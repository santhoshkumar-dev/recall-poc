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
    dims: usize,
    text_cache: Mutex<TextEmbeddingCache>,
}

impl VisualRuntime {
    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|error| RecallError::Message(format!("CLIP tokenization failed: {error}")))?
            .get_ids()
            .iter()
            .take_while(|id| **id != 0)
            .count())
    }

    /// Load the runtime from `models/visual/<model_id>/`.
    pub fn load(dir: &Path) -> Result<Self> {
        let vision = build_session(&dir.join("vision_model.onnx"), "vision")?;
        let text = build_session(&dir.join("text_model.onnx"), "text")?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| RecallError::Message(format!("Could not load CLIP tokenizer: {e}")))?;
        // MobileCLIP/OpenCLIP uses an all-zero backing array and writes the
        // tokenized sequence into it. Padding with EOT collapses this export's
        // text space because the model observes EOT at every trailing slot.
        let pad_id = 0;
        let pad_token = tokenizer
            .id_to_token(pad_id)
            .ok_or_else(|| RecallError::Message("CLIP tokenizer has no token id 0".into()))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(CONTEXT_LENGTH),
            pad_id,
            pad_token,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: CONTEXT_LENGTH,
            ..Default::default()
        }));

        let vision_inputs = input_names(&vision.lock());
        let vision_input = required_name(&vision_inputs, "pixel_values", "vision input")?;
        let text_inputs = input_names(&text.lock());
        required_name(&text_inputs, "input_ids", "text input")?;
        require_output(&vision.lock(), "image_embeds")?;
        require_output(&text.lock(), "text_embeds")?;

        Ok(Self {
            vision,
            text,
            tokenizer,
            vision_input,
            text_inputs,
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
        let vector = projected_embedding(&outputs, "image_embeds", self.dims)?;
        preprocess::validate_and_normalize(vector, self.dims)
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
                let id_name = "input_ids".to_string();
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
                let id_name = "input_ids".to_string();
                session
                    .run(ort::inputs![id_name => ids_value])
                    .map_err(|e| RecallError::Message(format!("Text inference failed: {e}")))?
            }
        };
        let vector = projected_embedding(&outputs, "text_embeds", self.dims)?;
        let vector = preprocess::validate_and_normalize(vector, self.dims)?;
        if !cache_key.is_empty() {
            self.text_cache.lock().insert(cache_key, vector.clone());
        }
        Ok(vector)
    }

    #[cfg(test)]
    fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|error| RecallError::Message(format!("CLIP tokenization failed: {error}")))?
            .get_ids()
            .to_vec())
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

fn output_names(session: &Session) -> Vec<String> {
    session
        .outputs()
        .iter()
        .map(|output| output.name().to_owned())
        .collect()
}

fn required_name(names: &[String], expected: &str, role: &str) -> Result<String> {
    names
        .iter()
        .find(|name| name.as_str() == expected)
        .cloned()
        .ok_or_else(|| {
            RecallError::Message(format!(
                "MobileCLIP {role} must be named {expected}; found {}",
                names.join(", ")
            ))
        })
}

fn require_output(session: &Session, expected: &str) -> Result<()> {
    let names = output_names(session);
    required_name(&names, expected, "output").map(|_| ())
}

/// Select the pooled embedding output (batch size 1).
///
/// CLIP ONNX exports often emit both a pooled `image_embeds`/`text_embeds`
/// (length == dims) and a `last_hidden_state` (much larger). Prefer an output
/// whose flat length is exactly `dims`; otherwise fall back to the first.
fn projected_embedding(
    outputs: &ort::session::SessionOutputs,
    expected_name: &str,
    dims: usize,
) -> Result<Vec<f32>> {
    for (name, value) in outputs.iter() {
        if name != expected_name {
            continue;
        }
        if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
            if data.len() != dims {
                return Err(RecallError::Message(format!(
                    "MobileCLIP output {expected_name} has {} values; expected {dims}",
                    data.len()
                )));
            }
            return Ok(data.to_vec());
        }
        return Err(RecallError::Message(format!(
            "MobileCLIP output {expected_name} is not f32"
        )));
    }
    Err(RecallError::Message(format!(
        "MobileCLIP did not produce required output {expected_name}"
    )))
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

    #[test]
    #[ignore = "requires RECALL_MODEL_DIR with the installed MobileCLIP2-S0 artifacts"]
    fn installed_text_encoder_uses_zero_padding_and_separates_concepts() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("RECALL_MODEL_DIR")
                .map_err(|_| "Set RECALL_MODEL_DIR to Recall's models directory")?,
        );
        let runtime = VisualRuntime::load(&root.join("visual").join(crate::ai::MOBILECLIP2_S0))?;
        let ids = runtime.token_ids("a photograph of an animal")?;
        assert_eq!(ids.len(), CONTEXT_LENGTH);
        let eot = ids.iter().position(|id| *id == 49_407).expect("EOT token");
        assert!(ids[eot + 1..].iter().all(|id| *id == 0));

        let animal = runtime.embed_text("a photograph of an animal")?;
        let building = runtime.embed_text("a photograph of a building")?;
        let similarity = animal
            .iter()
            .zip(&building)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        assert!(similarity < 0.90, "collapsed text space: {similarity}");
        Ok(())
    }

    #[test]
    #[ignore = "requires RECALL_MODEL_DIR with the installed MobileCLIP2-S0 artifacts"]
    fn benchmark_installed_visual_model() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("RECALL_MODEL_DIR")
                .map_err(|_| "Set RECALL_MODEL_DIR to Recall's models directory")?,
        );
        let dir = root.join("visual").join(crate::ai::MOBILECLIP2_S0);
        let load_started = Instant::now();
        let runtime = VisualRuntime::load(&dir)?;
        let load_ms = load_started.elapsed().as_millis();
        let image_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("sample-data")
            .join("restaurant-card.jpg");
        let image = image::open(image_path)
            .map_err(|error| RecallError::Message(error.to_string()))?
            .into_rgb8();
        let first_started = Instant::now();
        let first = runtime.embed_image(&image)?;
        let first_ms = first_started.elapsed().as_millis();
        let mut warm = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            std::hint::black_box(runtime.embed_image(&image)?);
            warm.push(started.elapsed().as_millis());
        }
        warm.sort_unstable();
        let artifact_bytes = ["vision_model.onnx", "text_model.onnx", "tokenizer.json"]
            .iter()
            .map(|name| std::fs::metadata(dir.join(name)).map(|meta| meta.len()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        println!(
            "VISUAL_MODEL_BENCHMARK load_ms={load_ms} first_image_ms={first_ms} warm_image_p50_ms={} artifact_bytes={artifact_bytes} dims={}",
            warm[warm.len() / 2],
            first.len()
        );
        Ok(())
    }
}
