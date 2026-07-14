//! ONNX multi-label image tagger runtime.
//!
//! This is the Recall equivalent of Panoptikon's independent tagger stage:
//! the model produces tags directly, which search treats as separate evidence
//! from MobileCLIP similarity.

use std::path::Path;

use image::RgbImage;
use ndarray::{Array, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use serde::Deserialize;

use crate::{
    error::{RecallError, Result},
    types::VisualTag,
};

const INPUT_SIZE: u32 = 448;
const TOP_K: usize = 20;
const MIN_CONFIDENCE: f32 = 0.20;

#[derive(Debug, Clone)]
struct Label {
    name: String,
    namespace: String,
}

#[derive(Debug, Deserialize)]
struct LabelRow {
    name: String,
    category: i64,
}

pub struct VisualTaggerRuntime {
    session: Mutex<Session>,
    input_name: String,
    labels: Vec<Label>,
}

impl VisualTaggerRuntime {
    pub fn load(dir: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|error| RecallError::Message(format!("ORT builder failed: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| RecallError::Message(format!("ORT opt-level failed: {error}")))?
            .with_intra_threads(1)
            .map_err(|error| RecallError::Message(format!("ORT threads failed: {error}")))?
            .commit_from_file(dir.join("model.onnx"))
            .map_err(|error| {
                RecallError::Message(format!("Could not load visual tagger model: {error}"))
            })?;
        let input_name = session
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .next()
            .ok_or_else(|| RecallError::Message("Visual tagger model has no inputs".into()))?;
        let labels = load_labels(&dir.join("selected_tags.csv"))?;
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            labels,
        })
    }

    pub fn tag_image(&self, image: &RgbImage, region_id: i64) -> Result<Vec<VisualTag>> {
        let tensor = image_to_tensor(image);
        let value = Tensor::from_array(tensor)
            .map_err(|error| RecallError::Message(format!("Tagger tensor failed: {error}")))?;
        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => value])
            .map_err(|error| {
                RecallError::Message(format!("Visual tagger inference failed: {error}"))
            })?;
        let mut scores = Vec::new();
        for (_, value) in outputs.iter() {
            if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                scores.extend(data.iter().copied());
            }
        }
        if scores.is_empty() {
            return Err("Visual tagger produced no float output".into());
        }
        let mut scored = scores
            .into_iter()
            .zip(self.labels.iter())
            .filter_map(|(raw, label)| {
                let confidence = if (0.0..=1.0).contains(&raw) {
                    raw
                } else {
                    1.0 / (1.0 + (-raw).exp())
                };
                (confidence >= MIN_CONFIDENCE).then(|| (label, confidence))
            })
            .collect::<Vec<_>>();
        let general_threshold = mcut_threshold(
            scored
                .iter()
                .filter(|(label, _)| label.namespace == "wd:general")
                .map(|(_, score)| *score),
        );
        let character_threshold = mcut_threshold(
            scored
                .iter()
                .filter(|(label, _)| label.namespace == "wd:character")
                .map(|(_, score)| *score),
        )
        .max(0.05);
        scored.retain(|(label, score)| match label.namespace.as_str() {
            "wd:general" => *score >= general_threshold,
            "wd:character" => *score >= character_threshold,
            _ => false,
        });
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.truncate(TOP_K);
        Ok(scored
            .into_iter()
            .enumerate()
            .map(|(rank, (label, confidence))| VisualTag {
                region_id,
                namespace: label.namespace.clone(),
                label: label.name.clone(),
                confidence,
                rank,
            })
            .collect())
    }
}

fn load_labels(path: &Path) -> Result<Vec<Label>> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|error| RecallError::Message(format!("Could not load tag labels: {error}")))?;
    let mut labels = Vec::new();
    for row in reader.deserialize::<LabelRow>() {
        let row = row.map_err(|error| {
            RecallError::Message(format!("Could not parse tag labels: {error}"))
        })?;
        labels.push(Label {
            name: row.name.replace('_', " "),
            namespace: match row.category {
                9 => "wd:rating",
                4 => "wd:character",
                _ => "wd:general",
            }
            .to_string(),
        });
    }
    if labels.is_empty() {
        return Err("Visual tagger label file is empty".into());
    }
    Ok(labels)
}

fn image_to_tensor(image: &RgbImage) -> Array4<f32> {
    let square = pad_square(image);
    let resized = image::imageops::resize(
        &square,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::CatmullRom,
    );
    let side = INPUT_SIZE as usize;
    let mut tensor = Array::zeros((1, 3, side, side));
    for (x, y, pixel) in resized.enumerate_pixels() {
        let (xi, yi) = (x as usize, y as usize);
        // Panoptikon's WD path uses timm's ImageNet normalization, then RGB -> BGR.
        tensor[[0, 0, yi, xi]] = (pixel[2] as f32 / 255.0 - 0.406) / 0.225;
        tensor[[0, 1, yi, xi]] = (pixel[1] as f32 / 255.0 - 0.456) / 0.224;
        tensor[[0, 2, yi, xi]] = (pixel[0] as f32 / 255.0 - 0.485) / 0.229;
    }
    tensor
}

fn pad_square(image: &RgbImage) -> RgbImage {
    let side = image.width().max(image.height()).max(1);
    let mut output = RgbImage::from_pixel(side, side, image::Rgb([255, 255, 255]));
    let x = (side - image.width()) / 2;
    let y = (side - image.height()) / 2;
    image::imageops::replace(&mut output, image, x.into(), y.into());
    output
}

fn mcut_threshold(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut sorted = values.into_iter().collect::<Vec<_>>();
    if sorted.len() < 2 {
        return MIN_CONFIDENCE;
    }
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut best_index = 0usize;
    let mut best_gap = 0.0;
    for index in 0..sorted.len() - 1 {
        let gap = sorted[index] - sorted[index + 1];
        if gap > best_gap {
            best_gap = gap;
            best_index = index;
        }
    }
    ((sorted[best_index] + sorted[best_index + 1]) / 2.0).max(MIN_CONFIDENCE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcut_uses_largest_probability_drop() {
        let threshold = mcut_threshold([0.91, 0.88, 0.55, 0.52]);
        assert!(threshold > 0.70 && threshold < 0.75);
    }

    #[test]
    fn tensor_shape_is_stable() {
        let image = RgbImage::from_pixel(320, 160, image::Rgb([10, 20, 30]));
        let tensor = image_to_tensor(&image);
        assert_eq!(tensor.shape(), &[1, 3, 448, 448]);
    }
}
