use std::{fs, path::Path, sync::Arc};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use parking_lot::Mutex;
use rten::Model;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};

use crate::{
    db,
    error::{RecallError, Result},
    types::{ModelProgressEvent, ModelStatus},
};

const DETECTION_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const RECOGNITION_URL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

pub struct AiRuntime {
    embeddings: Mutex<TextEmbedding>,
    ocr: Mutex<OcrEngine>,
}

impl AiRuntime {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let detection = Model::load_file(model_dir.join("text-detection.rten")).map_err(|e| {
            RecallError::Message(format!("Could not load OCR detection model: {e}"))
        })?;
        let recognition =
            Model::load_file(model_dir.join("text-recognition.rten")).map_err(|e| {
                RecallError::Message(format!("Could not load OCR recognition model: {e}"))
            })?;
        let ocr = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection),
            recognition_model: Some(recognition),
            ..Default::default()
        })
        .map_err(|e| RecallError::Message(format!("Could not initialize OCR: {e}")))?;
        let embeddings = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_cache_dir(model_dir.join("fastembed"))
                .with_show_download_progress(false),
        )
        .map_err(|e| RecallError::Message(format!("Could not initialize embeddings: {e}")))?;
        Ok(Self {
            embeddings: Mutex::new(embeddings),
            ocr: Mutex::new(ocr),
        })
    }

    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embeddings
            .lock()
            .embed(texts, Some(16))
            .map_err(|e| RecallError::Message(format!("Embedding failed: {e}")))
    }

    pub fn ocr_image(&self, image: &image::RgbImage) -> Result<String> {
        let source = ImageSource::from_bytes(image.as_raw(), image.dimensions())
            .map_err(|e| RecallError::Message(format!("OCR input failed: {e}")))?;
        let engine = self.ocr.lock();
        let input = engine
            .prepare_input(source)
            .map_err(|e| RecallError::Message(format!("OCR preparation failed: {e}")))?;
        engine
            .get_text(&input)
            .map_err(|e| RecallError::Message(format!("OCR failed: {e}")))
    }
}

pub fn status(db_path: &Path) -> Result<ModelStatus> {
    let connection = db::connect(db_path)?;
    let state = db::setting(&connection, "model_state", "missing")?;
    let last_error = db::setting(&connection, "model_error", "")?;
    Ok(ModelStatus {
        offline_ready: state == "ready",
        state: state.clone(),
        progress: if state == "ready" { 100 } else { 0 },
        message: match state.as_str() {
            "ready" => "Local models are ready",
            "error" if !last_error.is_empty() => last_error.as_str(),
            "error" => "Model setup needs attention",
            "downloading" => "Downloading local models",
            _ => "Models are not installed",
        }
        .into(),
        embedding_model: "all-MiniLM-L6-v2".into(),
    })
}

pub fn install(app: &AppHandle, db_path: &Path, model_dir: &Path) -> Result<Arc<AiRuntime>> {
    fs::create_dir_all(model_dir)?;
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_error", "")?;
    set_state(db_path, "downloading")?;
    emit(app, 5, "downloading", "Preparing private model storage")?;
    download_if_missing(
        app,
        DETECTION_URL,
        &model_dir.join("text-detection.rten"),
        10,
        30,
    )?;
    download_if_missing(
        app,
        RECOGNITION_URL,
        &model_dir.join("text-recognition.rten"),
        35,
        55,
    )?;
    emit(
        app,
        60,
        "downloading",
        "Downloading the English embedding model (about 90 MB; this can take several minutes)",
    )?;
    let runtime = AiRuntime::load(model_dir)?;
    emit(app, 92, "downloading", "Verifying and loading local models")?;
    write_manifest(model_dir)?;
    set_state(db_path, "ready")?;
    emit(
        app,
        100,
        "ready",
        "Models installed — Recall is offline ready",
    )?;
    Ok(Arc::new(runtime))
}

pub fn set_state(db_path: &Path, value: &str) -> Result<()> {
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_state", value)
}

pub fn set_error(db_path: &Path, message: &str) -> Result<()> {
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_error", message)?;
    db::set_setting(&connection, "model_state", "error")
}

fn emit(app: &AppHandle, progress: u8, state: &str, message: &str) -> Result<()> {
    app.emit(
        "models://progress",
        ModelProgressEvent {
            state: state.into(),
            progress,
            message: message.into(),
        },
    )
    .map_err(|e| RecallError::Message(e.to_string()))?;
    Ok(())
}

fn download_if_missing(
    app: &AppHandle,
    url: &str,
    destination: &Path,
    start: u8,
    finish: u8,
) -> Result<()> {
    if destination
        .metadata()
        .map(|m| m.len() > 1024)
        .unwrap_or(false)
    {
        return Ok(());
    }
    emit(
        app,
        start,
        "downloading",
        &format!(
            "Downloading {}",
            destination
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ),
    )?;
    let response = reqwest::blocking::get(url)
        .map_err(|e| RecallError::Message(format!("Model download failed: {e}")))?
        .error_for_status()
        .map_err(|e| RecallError::Message(format!("Model server returned an error: {e}")))?;
    let bytes = response
        .bytes()
        .map_err(|e| RecallError::Message(format!("Could not read downloaded model: {e}")))?;
    if bytes.len() < 1024 {
        return Err("Downloaded OCR model was unexpectedly small".into());
    }
    let temporary = destination.with_extension("rten.part");
    fs::write(&temporary, &bytes)?;
    fs::rename(temporary, destination)?;
    emit(app, finish, "downloading", "OCR model verified")?;
    Ok(())
}

fn write_manifest(model_dir: &Path) -> Result<()> {
    let mut entries = serde_json::Map::new();
    for name in ["text-detection.rten", "text-recognition.rten"] {
        let bytes = fs::read(model_dir.join(name))?;
        entries.insert(name.into(), serde_json::json!({ "bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(bytes)) }));
    }
    fs::write(
        model_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&entries).map_err(|e| RecallError::Message(e.to_string()))?,
    )?;
    Ok(())
}

pub fn model_dir_is_complete(model_dir: &Path) -> bool {
    [
        "text-detection.rten",
        "text-recognition.rten",
        "manifest.json",
    ]
    .iter()
    .all(|name| model_dir.join(name).is_file())
        && model_dir.join("fastembed").is_dir()
}
