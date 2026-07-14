use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding,
    TextInitOptions, TokenizerFiles, UserDefinedEmbeddingModel,
};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use paddle_ocr_rs::ocr_lite::OcrLite;
use parking_lot::Mutex;
use rten::Model;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};

use crate::{
    db,
    error::{RecallError, Result},
    types::{ModelCatalog, ModelOption, ModelProgressEvent, ModelStatus},
};

const OCRS_DETECTION_URL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const OCRS_DETECTION_SHA256: &str =
    "f15cfb56bd02c4bf478a20343986504a1f01e1665c2b3a0ad66340f054b1b5ca";
const OCRS_RECOGNITION_URL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";
const OCRS_RECOGNITION_SHA256: &str =
    "e484866d4cce403175bd8d00b128feb08ab42e208de30e42cd9889d8f1735a6e";

const PPOCR_TINY_DET_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx";
const PPOCR_TINY_DET_SHA256: &str =
    "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8";
const PPOCR_TINY_REC_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx";
const PPOCR_TINY_REC_SHA256: &str =
    "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6";
const PPOCR_TINY_YAML_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.yml";

const PPOCR_SMALL_DET_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx";
const PPOCR_SMALL_DET_SHA256: &str =
    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e";
const PPOCR_SMALL_REC_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx";
const PPOCR_SMALL_REC_SHA256: &str =
    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634";
const PPOCR_SMALL_YAML_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml";

const E5_MODEL_URL: &str =
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/model_int8.onnx";
const E5_MODEL_SHA256: &str = "4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c";
const E5_TOKENIZER_URL: &str =
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer.json";
const E5_CONFIG_URL: &str =
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/config.json";
const E5_SPECIAL_TOKENS_URL: &str =
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/special_tokens_map.json";
const E5_TOKENIZER_CONFIG_URL: &str =
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer_config.json";
// MobileCLIP2-S0 — community ONNX export (paired vision + text encoders, CLIP
// BPE tokenizer). 512-d shared cross-modal space. SHA256 pinned from the HF LFS
// metadata of plhery/mobileclip2-onnx (onnx/s0/*).
const MOBILECLIP_VISION_URL: &str =
    "https://huggingface.co/plhery/mobileclip2-onnx/resolve/main/onnx/s0/vision_model.onnx";
const MOBILECLIP_VISION_SHA256: &str =
    "13d20ebfa8a8f63890eb2727fe4dc63009ff970f43e0f7d9d2ed999659f70c8a";
const MOBILECLIP_TEXT_URL: &str =
    "https://huggingface.co/plhery/mobileclip2-onnx/resolve/main/onnx/s0/text_model.onnx";
const MOBILECLIP_TEXT_SHA256: &str =
    "df590d47744f2ee9f3ccb67c4414d17419568c05bca0c4d166f2faeedf8b92f3";
const MOBILECLIP_TOKENIZER_URL: &str =
    "https://huggingface.co/plhery/mobileclip2-onnx/resolve/main/tokenizer.json";
const WD_SWINV2_TAGGER_MODEL_URL: &str =
    "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx";
const WD_SWINV2_TAGGER_MODEL_SHA256: &str =
    "e61cc3e30576e50c745bd2224a2d03bec65637a84328301da8717d291d9eb96a";
const WD_SWINV2_TAGGER_LABELS_URL: &str =
    "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv";

pub const OCRS_NATIVE: &str = "ocrs-native";
pub const PPOCRV6_TINY: &str = "ppocrv6-tiny";
pub const PPOCRV6_SMALL: &str = "ppocrv6-small";
pub const MINILM_F32: &str = "minilm-l6-f32";
pub const E5_SMALL: &str = "multilingual-e5-small";
pub const EMBEDDING_GEMMA_Q8: &str = "embedding-gemma-q8";
/// Visual-search model ids. `disabled` = OCR + text search only (no download).
pub const VISUAL_DISABLED: &str = "disabled";
pub const MOBILECLIP2_S0: &str = "mobileclip2-s0";
pub const VISUAL_TAGGER_GENERAL: &str = "visual-tags/general-v1";
/// Bump when the visual encoder itself changes (regenerate image embeddings).
pub const VISUAL_MODEL_VERSION: &str = "2";
/// Bump when visual tagging preprocessing / thresholds change.
pub const VISUAL_TAGGER_VERSION: &str = "1";
/// Bump when chunking (regional chunks / summary) changes materially.
pub const CHUNKING_VERSION: &str = "3";
pub const DEFAULT_OCR_MODEL: &str = PPOCRV6_TINY;
pub const DEFAULT_EMBEDDING_MODEL: &str = E5_SMALL;
pub const DEFAULT_VISUAL_MODEL: &str = VISUAL_DISABLED;
pub const DEFAULT_OCR_MAX_SIDE: u32 = 1280;
const DEFAULT_OCR_MAX_SIDE_SETTING: &str = "1280";
const DEFAULT_PROFILE_VERSION: &str = "2";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSelection {
    pub ocr_model_id: String,
    pub embedding_model_id: String,
    pub ocr_max_side: u32,
    /// Visual-search model id, or `disabled`. Optional: the app runs without it.
    #[serde(default = "default_visual_model")]
    pub visual_model_id: String,
}

fn default_visual_model() -> String {
    DEFAULT_VISUAL_MODEL.to_string()
}

impl ModelSelection {
    pub fn visual_enabled(&self) -> bool {
        self.visual_model_id != VISUAL_DISABLED && !self.visual_model_id.is_empty()
    }
}

enum OcrRuntime {
    Native(Mutex<OcrEngine>),
    Paddle(Mutex<OcrLite>),
}

pub struct AiRuntime {
    embeddings: Mutex<TextEmbedding>,
    ocr: OcrRuntime,
    ocr_max_side: u32,
    embedding_model_id: String,
}

impl AiRuntime {
    pub fn load_for_selection(model_dir: &Path, selection: &ModelSelection) -> Result<Self> {
        validate_selection(selection)?;
        let ocr = match selection.ocr_model_id.as_str() {
            OCRS_NATIVE => {
                let detection =
                    Model::load_file(model_dir.join("text-detection.rten")).map_err(|e| {
                        RecallError::Message(format!("Could not load OCR detection model: {e}"))
                    })?;
                let recognition = Model::load_file(model_dir.join("text-recognition.rten"))
                    .map_err(|e| {
                        RecallError::Message(format!("Could not load OCR recognition model: {e}"))
                    })?;
                let engine = OcrEngine::new(OcrEngineParams {
                    detection_model: Some(detection),
                    recognition_model: Some(recognition),
                    ..Default::default()
                })
                .map_err(|e| RecallError::Message(format!("Could not initialize OCR: {e}")))?;
                OcrRuntime::Native(Mutex::new(engine))
            }
            PPOCRV6_TINY | PPOCRV6_SMALL => {
                let directory = paddle_directory(model_dir, &selection.ocr_model_id);
                let mut engine = OcrLite::new();
                engine
                    .init_models_without_angle_with_dict(
                        path_string(&directory.join("det.onnx"))?.as_str(),
                        path_string(&directory.join("rec.onnx"))?.as_str(),
                        path_string(&directory.join("dict.txt"))?.as_str(),
                        ppocr_thread_count(),
                    )
                    .map_err(|e| {
                        RecallError::Message(format!("Could not initialize PP-OCRv6: {e}"))
                    })?;
                OcrRuntime::Paddle(Mutex::new(engine))
            }
            _ => unreachable!("validated OCR model"),
        };

        let embeddings = load_embedding_runtime(model_dir, &selection.embedding_model_id)?;

        Ok(Self {
            embeddings: Mutex::new(embeddings),
            ocr,
            ocr_max_side: selection.ocr_max_side,
            embedding_model_id: selection.embedding_model_id.clone(),
        })
    }

    pub fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_inner(self.with_e5_prefix(texts, "passage: "))
    }

    pub fn embed_query(&self, query: String) -> Result<Vec<f32>> {
        let mut values = self.embed_inner(self.with_e5_prefix(vec![query], "query: "))?;
        values
            .pop()
            .ok_or_else(|| RecallError::Message("Embedding model returned no query vector".into()))
    }

    fn with_e5_prefix(&self, texts: Vec<String>, prefix: &str) -> Vec<String> {
        if self.embedding_model_id == E5_SMALL {
            texts
                .into_iter()
                .map(|text| format!("{prefix}{text}"))
                .collect()
        } else {
            texts
        }
    }

    fn embed_inner(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embeddings
            .lock()
            .embed(texts, Some(16))
            .map_err(|e| RecallError::Message(format!("Embedding failed: {e}")))
    }

    pub fn ocr_image(&self, image: &image::RgbImage) -> Result<String> {
        match &self.ocr {
            OcrRuntime::Native(engine) => {
                let source = ImageSource::from_bytes(image.as_raw(), image.dimensions())
                    .map_err(|e| RecallError::Message(format!("OCR input failed: {e}")))?;
                let engine = engine.lock();
                let input = engine
                    .prepare_input(source)
                    .map_err(|e| RecallError::Message(format!("OCR preparation failed: {e}")))?;
                engine
                    .get_text(&input)
                    .map_err(|e| RecallError::Message(format!("OCR failed: {e}")))
            }
            OcrRuntime::Paddle(engine) => {
                let result = engine
                    .lock()
                    .detect(image, 32, self.ocr_max_side, 0.5, 0.3, 1.6, false, false)
                    .map_err(|e| RecallError::Message(format!("PP-OCRv6 failed: {e}")))?;
                Ok(result
                    .text_blocks
                    .into_iter()
                    .map(|block| block.text)
                    .filter(|text| !text.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }

    pub fn ocr_max_side(&self) -> u32 {
        self.ocr_max_side
    }
}

pub fn ensure_defaults(db_path: &Path, _model_dir: &Path) -> Result<()> {
    let connection = db::connect(db_path)?;
    if db::setting(&connection, "ocr_model_id", "")?.is_empty() {
        db::set_setting(&connection, "ocr_model_id", DEFAULT_OCR_MODEL)?;
    }
    if db::setting(&connection, "embedding_model_id", "")?.is_empty() {
        db::set_setting(&connection, "embedding_model_id", DEFAULT_EMBEDDING_MODEL)?;
    }
    if db::setting(&connection, "ocr_max_side", "")?.is_empty() {
        db::set_setting(&connection, "ocr_max_side", DEFAULT_OCR_MAX_SIDE_SETTING)?;
    }
    if db::setting(&connection, "visual_model_id", "")?.is_empty() {
        db::set_setting(&connection, "visual_model_id", DEFAULT_VISUAL_MODEL)?;
    }
    if db::setting(&connection, "model_default_profile", "")?.is_empty() {
        let ocr_model = db::setting(&connection, "ocr_model_id", DEFAULT_OCR_MODEL)?;
        let embedding_model =
            db::setting(&connection, "embedding_model_id", DEFAULT_EMBEDDING_MODEL)?;
        let max_side = db::setting(&connection, "ocr_max_side", DEFAULT_OCR_MAX_SIDE_SETTING)?;
        if matches!(ocr_model.as_str(), OCRS_NATIVE | PPOCRV6_SMALL)
            && embedding_model == E5_SMALL
            && max_side == "1600"
        {
            db::set_setting(&connection, "ocr_model_id", DEFAULT_OCR_MODEL)?;
            db::set_setting(&connection, "ocr_max_side", DEFAULT_OCR_MAX_SIDE_SETTING)?;
        }
        db::set_setting(
            &connection,
            "model_default_profile",
            DEFAULT_PROFILE_VERSION,
        )?;
    }
    Ok(())
}

pub fn selection(db_path: &Path) -> Result<ModelSelection> {
    let connection = db::connect(db_path)?;
    let max_side = db::setting(&connection, "ocr_max_side", DEFAULT_OCR_MAX_SIDE_SETTING)?
        .parse::<u32>()
        .unwrap_or(DEFAULT_OCR_MAX_SIDE)
        .clamp(960, 4096);
    Ok(ModelSelection {
        ocr_model_id: db::setting(&connection, "ocr_model_id", DEFAULT_OCR_MODEL)?,
        embedding_model_id: db::setting(
            &connection,
            "embedding_model_id",
            DEFAULT_EMBEDDING_MODEL,
        )?,
        ocr_max_side: max_side,
        visual_model_id: db::setting(&connection, "visual_model_id", DEFAULT_VISUAL_MODEL)?,
    })
}

pub fn status(db_path: &Path) -> Result<ModelStatus> {
    let connection = db::connect(db_path)?;
    let selected = selection(db_path)?;
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
            "downloading" => "Downloading and validating local models",
            _ => "Models are not installed",
        }
        .into(),
        ocr_model: ocr_label(&selected.ocr_model_id).into(),
        embedding_model: embedding_label(&selected.embedding_model_id).into(),
        visual_model: visual_label(&selected.visual_model_id).into(),
        visual_enabled: selected.visual_enabled(),
        ocr_max_side: selected.ocr_max_side,
    })
}

fn visual_label(model_id: &str) -> &'static str {
    match model_id {
        MOBILECLIP2_S0 => "MobileCLIP2-S0",
        _ => "Disabled",
    }
}

pub fn catalog(db_path: &Path, model_dir: &Path) -> Result<ModelCatalog> {
    let selected = selection(db_path)?;
    Ok(ModelCatalog {
        ocr_models: vec![
            model_option(
                OCRS_NATIVE,
                "ocrs Native",
                "Existing offline fallback. Accurate, but slow in unoptimized development builds.",
                12,
                false,
                is_ocr_installed(model_dir, OCRS_NATIVE),
            ),
            model_option(
                PPOCRV6_TINY,
                "PP-OCRv6 Tiny",
                "Fastest option for screenshots and clear English text.",
                7,
                true,
                is_ocr_installed(model_dir, PPOCRV6_TINY),
            ),
            model_option(
                PPOCRV6_SMALL,
                "PP-OCRv6 Small",
                "Higher-accuracy option for difficult images; it indexes more slowly.",
                27,
                false,
                is_ocr_installed(model_dir, PPOCRV6_SMALL),
            ),
        ],
        embedding_models: vec![
            model_option(
                E5_SMALL,
                "Multilingual E5 Small",
                "Reliable 384-dimensional multilingual retrieval for the POC.",
                120,
                true,
                is_embedding_installed(model_dir, E5_SMALL),
            ),
            model_option(
                EMBEDDING_GEMMA_Q8,
                "EmbeddingGemma 300M Quantized",
                "Balanced multilingual quality for post-POC benchmarking; 768 dimensions.",
                190,
                false,
                is_embedding_installed(model_dir, EMBEDDING_GEMMA_Q8),
            ),
            model_option(
                MINILM_F32,
                "MiniLM L6 Full Precision",
                "Legacy 384-dimensional model. Largest download and memory footprint.",
                91,
                false,
                is_embedding_installed(model_dir, MINILM_F32),
            ),
        ],
        visual_models: vec![
            model_option(
                VISUAL_DISABLED,
                "Disabled",
                "OCR and text search only. No additional model download.",
                0,
                false,
                true,
            ),
            model_option(
                MOBILECLIP2_S0,
                "MobileCLIP2-S0",
                "Enables visual and cross-modal search across screenshots and photos. Requires visual indexing of image files.",
                290,
                true,
                is_visual_installed(model_dir, MOBILECLIP2_S0),
            ),
        ],
        active_ocr_model_id: selected.ocr_model_id,
        active_embedding_model_id: selected.embedding_model_id,
        active_visual_model_id: selected.visual_model_id,
        ocr_max_side: selected.ocr_max_side,
    })
}

pub fn install(app: &AppHandle, db_path: &Path, model_dir: &Path) -> Result<Arc<AiRuntime>> {
    let selected = selection(db_path)?;
    install_selection(app, db_path, model_dir, &selected)
}

pub fn install_selection(
    app: &AppHandle,
    db_path: &Path,
    model_dir: &Path,
    selected: &ModelSelection,
) -> Result<Arc<AiRuntime>> {
    validate_selection(selected)?;
    fs::create_dir_all(model_dir)?;
    set_last_error(db_path, "")?;
    set_state(db_path, "downloading")?;
    emit(app, 3, "downloading", "Preparing private model storage")?;

    install_ocr(app, model_dir, &selected.ocr_model_id)?;
    emit(
        app,
        62,
        "downloading",
        &format!(
            "Preparing {} embedding model",
            embedding_label(&selected.embedding_model_id)
        ),
    )?;

    install_embedding(app, model_dir, &selected.embedding_model_id)?;
    if selected.visual_enabled() {
        install_visual(app, model_dir, &selected.visual_model_id)?;
        install_visual_tagger(app, model_dir)?;
    }
    let runtime = Arc::new(AiRuntime::load_for_selection(model_dir, selected)?);
    emit(app, 94, "downloading", "Verifying offline model runtime")?;
    write_ready_marker(model_dir, selected)?;
    save_selection(db_path, selected)?;
    emit(
        app,
        100,
        "ready",
        "Models installed - Recall is offline ready",
    )?;
    Ok(runtime)
}

pub fn model_dir_is_complete(db_path: &Path, model_dir: &Path) -> bool {
    let Ok(connection) = db::connect(db_path) else {
        return false;
    };
    if db::setting(&connection, "model_state", "missing")
        .ok()
        .as_deref()
        != Some("ready")
    {
        return false;
    }
    let Ok(selected) = selection(db_path) else {
        return false;
    };
    is_ocr_installed(model_dir, &selected.ocr_model_id)
        && is_embedding_installed(model_dir, &selected.embedding_model_id)
}

pub fn set_state(db_path: &Path, value: &str) -> Result<()> {
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_state", value)
}

/// Record the outcome of the last visual-runtime load attempt (for diagnostics).
pub fn set_visual_status(db_path: &Path, value: &str) {
    if let Ok(connection) = db::connect(db_path) {
        let _ = db::set_setting(&connection, "visual_load_status", value);
    }
}

pub fn visual_status(db_path: &Path) -> String {
    db::connect(db_path)
        .ok()
        .and_then(|c| db::setting(&c, "visual_load_status", "not attempted").ok())
        .unwrap_or_else(|| "not attempted".into())
}

pub fn set_last_error(db_path: &Path, message: &str) -> Result<()> {
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_error", message)
}

pub fn set_error(db_path: &Path, message: &str) -> Result<()> {
    let connection = db::connect(db_path)?;
    db::set_setting(&connection, "model_error", message)?;
    db::set_setting(&connection, "model_state", "error")
}

fn save_selection(db_path: &Path, selected: &ModelSelection) -> Result<()> {
    let mut connection = db::connect(db_path)?;
    let transaction = connection.transaction()?;
    let max_side = selected.ocr_max_side.to_string();
    for (key, value) in [
        ("ocr_model_id", selected.ocr_model_id.as_str()),
        ("embedding_model_id", selected.embedding_model_id.as_str()),
        ("ocr_max_side", max_side.as_str()),
        ("visual_model_id", selected.visual_model_id.as_str()),
        ("visual_model_version", VISUAL_MODEL_VERSION),
        (
            "prompt_bank_version",
            crate::visual::category_prompts::PROMPT_BANK_VERSION,
        ),
        (
            "metadata_extractor_version",
            crate::metadata::METADATA_EXTRACTOR_VERSION,
        ),
        ("chunking_version", CHUNKING_VERSION),
        ("model_state", "ready"),
        ("model_error", ""),
    ] {
        transaction.execute(
            "INSERT INTO app_settings(key,value) VALUES (?1,?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            rusqlite::params![key, value],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

fn validate_selection(selected: &ModelSelection) -> Result<()> {
    if !matches!(
        selected.ocr_model_id.as_str(),
        OCRS_NATIVE | PPOCRV6_TINY | PPOCRV6_SMALL
    ) {
        return Err("Unknown OCR model selection".into());
    }
    if !matches!(
        selected.embedding_model_id.as_str(),
        MINILM_F32 | E5_SMALL | EMBEDDING_GEMMA_Q8
    ) {
        return Err("Unknown embedding model selection".into());
    }
    if !matches!(
        selected.visual_model_id.as_str(),
        VISUAL_DISABLED | MOBILECLIP2_S0
    ) {
        return Err("Unknown visual-search model selection".into());
    }
    if !(960..=4096).contains(&selected.ocr_max_side) {
        return Err("OCR resolution must be between 960 and 4096 pixels".into());
    }
    Ok(())
}

fn install_ocr(app: &AppHandle, model_dir: &Path, model_id: &str) -> Result<()> {
    match model_id {
        OCRS_NATIVE => {
            download_verified(
                app,
                OCRS_DETECTION_URL,
                &model_dir.join("text-detection.rten"),
                OCRS_DETECTION_SHA256,
                8,
                28,
            )?;
            download_verified(
                app,
                OCRS_RECOGNITION_URL,
                &model_dir.join("text-recognition.rten"),
                OCRS_RECOGNITION_SHA256,
                30,
                58,
            )
        }
        PPOCRV6_TINY | PPOCRV6_SMALL => {
            let directory = paddle_directory(model_dir, model_id);
            fs::create_dir_all(&directory)?;
            let (det_url, det_hash, rec_url, rec_hash, yaml_url) = if model_id == PPOCRV6_TINY {
                (
                    PPOCR_TINY_DET_URL,
                    PPOCR_TINY_DET_SHA256,
                    PPOCR_TINY_REC_URL,
                    PPOCR_TINY_REC_SHA256,
                    PPOCR_TINY_YAML_URL,
                )
            } else {
                (
                    PPOCR_SMALL_DET_URL,
                    PPOCR_SMALL_DET_SHA256,
                    PPOCR_SMALL_REC_URL,
                    PPOCR_SMALL_REC_SHA256,
                    PPOCR_SMALL_YAML_URL,
                )
            };
            download_verified(app, det_url, &directory.join("det.onnx"), det_hash, 8, 23)?;
            download_verified(app, rec_url, &directory.join("rec.onnx"), rec_hash, 24, 53)?;
            let yaml_path = directory.join("inference.yml");
            download_file(app, yaml_url, &yaml_path, 54, 57)?;
            write_paddle_dictionary(&yaml_path, &directory.join("dict.txt"))?;
            emit(app, 59, "downloading", "PP-OCRv6 files verified")
        }
        _ => Err("Unknown OCR model selection".into()),
    }
}

fn install_embedding(app: &AppHandle, model_dir: &Path, model_id: &str) -> Result<()> {
    if model_id != E5_SMALL {
        return Ok(());
    }
    let directory = model_dir.join("embeddings").join(E5_SMALL);
    fs::create_dir_all(&directory)?;
    download_verified(
        app,
        E5_MODEL_URL,
        &directory.join("model_int8.onnx"),
        E5_MODEL_SHA256,
        62,
        84,
    )?;
    for (url, name, start, finish) in [
        (E5_TOKENIZER_URL, "tokenizer.json", 85, 86),
        (E5_CONFIG_URL, "config.json", 86, 87),
        (E5_SPECIAL_TOKENS_URL, "special_tokens_map.json", 87, 88),
        (E5_TOKENIZER_CONFIG_URL, "tokenizer_config.json", 88, 89),
    ] {
        download_file(app, url, &directory.join(name), start, finish)?;
    }
    emit(
        app,
        90,
        "downloading",
        "Multilingual E5 Small INT8 files verified",
    )
}

/// On-disk location of a visual model's files: `models/visual/<model_id>/`.
pub fn visual_directory(model_dir: &Path, model_id: &str) -> PathBuf {
    model_dir.join("visual").join(model_id)
}

pub fn is_visual_installed(model_dir: &Path, model_id: &str) -> bool {
    if model_id == VISUAL_DISABLED {
        return true;
    }
    let dir = visual_directory(model_dir, model_id);
    ["vision_model.onnx", "text_model.onnx", "tokenizer.json"]
        .iter()
        .all(|f| dir.join(f).is_file())
}

pub fn visual_tagger_directory(model_dir: &Path) -> PathBuf {
    model_dir.join("visual-tags").join("general-v1")
}

pub fn is_visual_tagger_installed(model_dir: &Path) -> bool {
    let dir = visual_tagger_directory(model_dir);
    ["model.onnx", "selected_tags.csv"]
        .iter()
        .all(|file| dir.join(file).is_file())
}

fn install_visual(app: &AppHandle, model_dir: &Path, model_id: &str) -> Result<()> {
    if model_id != MOBILECLIP2_S0 {
        return Ok(());
    }
    let directory = visual_directory(model_dir, model_id);
    fs::create_dir_all(&directory)?;
    emit(
        app,
        90,
        "downloading",
        "Preparing MobileCLIP2-S0 visual model",
    )?;
    download_verified(
        app,
        MOBILECLIP_VISION_URL,
        &directory.join("vision_model.onnx"),
        MOBILECLIP_VISION_SHA256,
        90,
        93,
    )?;
    download_verified(
        app,
        MOBILECLIP_TEXT_URL,
        &directory.join("text_model.onnx"),
        MOBILECLIP_TEXT_SHA256,
        93,
        97,
    )?;
    // Only tokenizer.json is read at runtime; config.json (98 B) /
    // preprocessor_config.json values are hardcoded in the encoder, and their
    // tiny size trips download_file's small-file guard, so we don't fetch them.
    download_file(
        app,
        MOBILECLIP_TOKENIZER_URL,
        &directory.join("tokenizer.json"),
        97,
        99,
    )?;
    emit(app, 99, "downloading", "MobileCLIP2-S0 files verified")
}

fn install_visual_tagger(app: &AppHandle, model_dir: &Path) -> Result<()> {
    let directory = visual_tagger_directory(model_dir);
    fs::create_dir_all(&directory)?;
    emit(app, 99, "downloading", "Preparing general visual tagger")?;
    download_verified(
        app,
        WD_SWINV2_TAGGER_MODEL_URL,
        &directory.join("model.onnx"),
        WD_SWINV2_TAGGER_MODEL_SHA256,
        99,
        99,
    )?;
    download_file(
        app,
        WD_SWINV2_TAGGER_LABELS_URL,
        &directory.join("selected_tags.csv"),
        99,
        99,
    )?;
    emit(app, 99, "downloading", "Visual tagger files verified")
}

/// Load the visual runtime for a selection, if a visual model is enabled and
/// its files are present. Returns `Ok(None)` when disabled/absent so the app
/// still boots with text-only search.
pub fn load_visual_for_selection(
    model_dir: &Path,
    selected: &ModelSelection,
) -> Result<Option<Arc<crate::visual::VisualRuntime>>> {
    if !selected.visual_enabled() {
        return Ok(None);
    }
    if !is_visual_installed(model_dir, &selected.visual_model_id) {
        return Ok(None);
    }
    let dir = visual_directory(model_dir, &selected.visual_model_id);
    Ok(Some(Arc::new(crate::visual::VisualRuntime::load(&dir)?)))
}

pub fn load_visual_tagger_for_selection(
    model_dir: &Path,
    selected: &ModelSelection,
) -> Result<Option<Arc<crate::visual::VisualTaggerRuntime>>> {
    if !selected.visual_enabled() || !is_visual_tagger_installed(model_dir) {
        return Ok(None);
    }
    Ok(Some(Arc::new(crate::visual::VisualTaggerRuntime::load(
        &visual_tagger_directory(model_dir),
    )?)))
}

fn load_embedding_runtime(model_dir: &Path, model_id: &str) -> Result<TextEmbedding> {
    if model_id == E5_SMALL {
        let directory = model_dir.join("embeddings").join(E5_SMALL);
        let tokenizer_files = TokenizerFiles {
            tokenizer_file: fs::read(directory.join("tokenizer.json"))?,
            config_file: fs::read(directory.join("config.json"))?,
            special_tokens_map_file: fs::read(directory.join("special_tokens_map.json"))?,
            tokenizer_config_file: fs::read(directory.join("tokenizer_config.json"))?,
        };
        let model = UserDefinedEmbeddingModel::new(
            fs::read(directory.join("model_int8.onnx"))?,
            tokenizer_files,
        )
        .with_pooling(Pooling::Mean)
        .with_quantization(QuantizationMode::Static);
        return TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::default())
            .map_err(|error| {
                RecallError::Message(format!(
                    "Could not initialize Multilingual E5 Small INT8: {error}"
                ))
            });
    }

    let (embedding_model, cache_dir) = embedding_configuration(model_dir, model_id)?;
    TextEmbedding::try_new(
        TextInitOptions::new(embedding_model)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false),
    )
    .map_err(|error| RecallError::Message(format!("Could not initialize embeddings: {error}")))
}
fn embedding_configuration(model_dir: &Path, model_id: &str) -> Result<(EmbeddingModel, PathBuf)> {
    match model_id {
        MINILM_F32 => Ok((EmbeddingModel::AllMiniLML6V2, model_dir.join("fastembed"))),
        EMBEDDING_GEMMA_Q8 => Ok((
            EmbeddingModel::EmbeddingGemma300MQ,
            model_dir.join("embeddings").join(EMBEDDING_GEMMA_Q8),
        )),
        _ => Err("Unknown embedding model selection".into()),
    }
}

fn ppocr_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|count| ((count.get() + 1) / 2).clamp(1, 4))
        .unwrap_or(2)
}

fn download_verified(
    app: &AppHandle,
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    start: u8,
    finish: u8,
) -> Result<()> {
    if destination.is_file() && sha256_file(destination)? == expected_sha256 {
        eprintln!("[download] cached, skipping: {url}");
        return Ok(());
    }
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    let name = destination
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    eprintln!("[download] GET {url}");
    emit(
        app,
        start,
        "downloading",
        &format!("Downloading and verifying {name}"),
    )?;
    let temporary = destination.with_extension("part");
    stream_download(app, url, &temporary, start, finish)?;
    let bytes = temporary.metadata().map(|m| m.len()).unwrap_or(0);
    let actual = sha256_file(&temporary)?;
    if actual != expected_sha256 {
        let _ = fs::remove_file(&temporary);
        eprintln!(
            "[download] CHECKSUM MISMATCH for {name}: expected {expected_sha256}, got {actual}"
        );
        return Err(RecallError::Message(format!(
            "Checksum validation failed for {name}"
        )));
    }
    fs::rename(temporary, destination)?;
    eprintln!("[download] OK {name} ({bytes} bytes, sha256 verified)");
    emit(app, finish, "downloading", &format!("Verified {name}"))
}

fn download_file(
    app: &AppHandle,
    url: &str,
    destination: &Path,
    start: u8,
    finish: u8,
) -> Result<()> {
    if destination
        .metadata()
        .map(|value| value.len() > 100)
        .unwrap_or(false)
    {
        eprintln!("[download] cached, skipping: {url}");
        return Ok(());
    }
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    eprintln!("[download] GET {url}");
    let temporary = destination.with_extension("part");
    stream_download(app, url, &temporary, start, finish)?;
    let bytes = temporary.metadata()?.len();
    if bytes < 100 {
        let _ = fs::remove_file(&temporary);
        eprintln!("[download] file too small ({bytes} bytes): {url}");
        return Err("Downloaded model metadata was unexpectedly small".into());
    }
    fs::rename(temporary, destination)?;
    eprintln!(
        "[download] OK {} ({bytes} bytes)",
        destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    Ok(())
}

fn stream_download(
    app: &AppHandle,
    url: &str,
    destination: &Path,
    start: u8,
    finish: u8,
) -> Result<()> {
    const MAX_ATTEMPTS: u8 = 3;

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|error| {
            RecallError::Message(format!("Could not create model download client: {error}"))
        })?;
    let mut last_error = String::new();

    for attempt in 1..=MAX_ATTEMPTS {
        let existing = destination
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let mut request = client.get(url);
        if existing > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
            emit(app, start, "downloading", "Resuming local model download")?;
        }
        let mut response = match request.send() {
            Ok(response) => response,
            Err(error) => {
                last_error =
                    format!("Model download attempt {attempt}/{MAX_ATTEMPTS} failed: {error}");
                if attempt < MAX_ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_millis(300 * attempt as u64));
                    continue;
                }
                break;
            }
        };
        if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && existing > 0 {
            let _ = fs::remove_file(destination);
            last_error = "The partial model file could not be resumed; restarting it once.".into();
            continue;
        }
        if !response.status().is_success() {
            last_error = format!(
                "Model server returned {} on attempt {attempt}/{MAX_ATTEMPTS}",
                response.status()
            );
            if attempt < MAX_ATTEMPTS {
                std::thread::sleep(std::time::Duration::from_millis(300 * attempt as u64));
                continue;
            }
            break;
        }

        let append = existing > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        let starting_bytes = if append { existing } else { 0 };
        let total = content_range_total(&response).or_else(|| {
            response
                .content_length()
                .map(|length| starting_bytes + length)
        });
        let mut output = if append {
            fs::OpenOptions::new().append(true).open(destination)?
        } else {
            fs::File::create(destination)?
        };
        let mut buffer = [0_u8; 64 * 1024];
        let mut downloaded = starting_bytes;
        let mut last_progress = start;
        let transfer = (|| -> Result<()> {
            loop {
                let count = response.read(&mut buffer).map_err(|error| {
                    RecallError::Message(format!("Could not read model download: {error}"))
                })?;
                if count == 0 {
                    break;
                }
                output.write_all(&buffer[..count])?;
                downloaded += count as u64;
                if let Some(total) = total.filter(|value| *value > 0) {
                    let span = finish.saturating_sub(start) as u64;
                    let progress = start
                        + ((downloaded.saturating_mul(span) / total) as u8).min(finish - start);
                    if progress > last_progress {
                        last_progress = progress;
                        emit(
                            app,
                            progress,
                            "downloading",
                            "Downloading local model files",
                        )?;
                    }
                }
            }
            output.sync_all()?;
            Ok(())
        })();
        match transfer {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error =
                    format!("Model download attempt {attempt}/{MAX_ATTEMPTS} failed: {error}");
                if attempt < MAX_ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_millis(300 * attempt as u64));
                }
            }
        }
    }

    Err(RecallError::Message(if last_error.is_empty() {
        "Model download failed without a response".into()
    } else {
        last_error
    }))
}

fn content_range_total(response: &reqwest::blocking::Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit('/').next())
        .and_then(|value| value.parse::<u64>().ok())
}

fn write_paddle_dictionary(yaml_path: &Path, destination: &Path) -> Result<()> {
    let yaml = fs::read_to_string(yaml_path)?;
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml)
        .map_err(|e| RecallError::Message(format!("Invalid model dictionary metadata: {e}")))?;
    let characters = value
        .get("PostProcess")
        .and_then(|value| value.get("character_dict"))
        .and_then(serde_yaml::Value::as_sequence)
        .ok_or_else(|| {
            RecallError::Message("Model metadata did not contain a character dictionary".into())
        })?;

    let mut keys = Vec::with_capacity(characters.len() + 2);
    keys.push("#".to_owned());
    for value in characters {
        let character = value.as_str().ok_or_else(|| {
            RecallError::Message("Model character dictionary was malformed".into())
        })?;
        keys.push(character.to_owned());
    }
    keys.push(" ".to_owned());
    fs::write(destination, keys.join("\n"))?;
    Ok(())
}

fn write_ready_marker(model_dir: &Path, selected: &ModelSelection) -> Result<()> {
    let directory = model_dir.join("ready");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{}--{}.json",
        selected.ocr_model_id, selected.embedding_model_id
    ));
    let bytes = serde_json::to_vec_pretty(selected)
        .map_err(|e| RecallError::Message(format!("Could not save model manifest: {e}")))?;
    fs::write(path, bytes)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn is_ocr_installed(model_dir: &Path, model_id: &str) -> bool {
    match model_id {
        OCRS_NATIVE => {
            model_dir.join("text-detection.rten").is_file()
                && model_dir.join("text-recognition.rten").is_file()
        }
        PPOCRV6_TINY | PPOCRV6_SMALL => {
            let directory = paddle_directory(model_dir, model_id);
            ["det.onnx", "rec.onnx", "dict.txt"]
                .iter()
                .all(|name| directory.join(name).is_file())
        }
        _ => false,
    }
}

fn is_embedding_installed(model_dir: &Path, model_id: &str) -> bool {
    if model_id == E5_SMALL {
        let directory = model_dir.join("embeddings").join(E5_SMALL);
        return [
            "model_int8.onnx",
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]
        .iter()
        .all(|name| directory.join(name).is_file());
    }
    let directory = match embedding_configuration(model_dir, model_id) {
        Ok((_, directory)) => directory,
        Err(_) => return false,
    };
    directory
        .read_dir()
        .ok()
        .and_then(|mut entries| entries.next())
        .is_some()
}

fn paddle_directory(model_dir: &Path, model_id: &str) -> PathBuf {
    model_dir.join("ocr").join(model_id)
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| RecallError::Message("Model path is not valid Unicode".into()))
}

fn ocr_label(model_id: &str) -> &'static str {
    match model_id {
        OCRS_NATIVE => "ocrs Native",
        PPOCRV6_TINY => "PP-OCRv6 Tiny",
        PPOCRV6_SMALL => "PP-OCRv6 Small",
        _ => "Unknown OCR model",
    }
}

fn embedding_label(model_id: &str) -> &'static str {
    match model_id {
        MINILM_F32 => "MiniLM L6 Full Precision",
        E5_SMALL => "Multilingual E5 Small",
        EMBEDDING_GEMMA_Q8 => "EmbeddingGemma 300M Quantized",
        _ => "Unknown embedding model",
    }
}

fn model_option(
    id: &str,
    label: &str,
    description: &str,
    download_mb: u32,
    recommended: bool,
    installed: bool,
) -> ModelOption {
    ModelOption {
        id: id.into(),
        label: label.into(),
        description: description.into(),
        download_mb,
        recommended,
        installed,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        time::{Duration, Instant},
    };

    fn temporary_paths() -> (PathBuf, PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!("recall-ai-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("temporary test directory must be creatable");
        let database = root.join("recall.db");
        let models = root.join("models");
        (root, database, models)
    }

    fn median_millis(samples: &mut [Duration]) -> u128 {
        samples.sort_unstable();
        samples[samples.len() / 2].as_millis()
    }

    fn normalized(value: &str) -> String {
        value
            .chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
        let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
        dot / (left_norm * right_norm)
    }

    fn benchmark_model_dir() -> PathBuf {
        env::var_os("RECALL_MODEL_DIR")
            .map(PathBuf::from)
            .expect("Set RECALL_MODEL_DIR to Recall's local models directory before benchmarking")
    }

    fn benchmark_image(max_side: u32) -> image::RgbImage {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("src-tauri must have a workspace parent")
            .join("sample-data")
            .join("restaurant-card.jpg");
        let decoded = image::open(path).expect("sample benchmark image must be readable");
        if decoded.width().max(decoded.height()) > max_side {
            decoded
                .resize(max_side, max_side, image::imageops::FilterType::Triangle)
                .into_rgb8()
        } else {
            decoded.into_rgb8()
        }
    }

    #[test]
    fn fresh_install_uses_the_lightweight_default_profile() -> Result<()> {
        let (root, database, models) = temporary_paths();
        db::migrate(&database)?;
        ensure_defaults(&database, &models)?;

        let selected = selection(&database)?;
        assert_eq!(selected.ocr_model_id, DEFAULT_OCR_MODEL);
        assert_eq!(selected.embedding_model_id, DEFAULT_EMBEDDING_MODEL);
        assert_eq!(selected.ocr_max_side, DEFAULT_OCR_MAX_SIDE);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn untouched_legacy_small_profile_is_migrated_to_the_fast_default() -> Result<()> {
        let (root, database, models) = temporary_paths();
        db::migrate(&database)?;
        let connection = db::connect(&database)?;
        db::set_setting(&connection, "ocr_model_id", PPOCRV6_SMALL)?;
        db::set_setting(&connection, "embedding_model_id", E5_SMALL)?;
        db::set_setting(&connection, "ocr_max_side", "1600")?;

        ensure_defaults(&database, &models)?;
        let selected = selection(&database)?;
        assert_eq!(selected.ocr_model_id, PPOCRV6_TINY);
        assert_eq!(selected.ocr_max_side, DEFAULT_OCR_MAX_SIDE);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn untouched_legacy_native_profile_is_migrated_to_the_fast_default() -> Result<()> {
        let (root, database, models) = temporary_paths();
        db::migrate(&database)?;
        let connection = db::connect(&database)?;
        db::set_setting(&connection, "ocr_model_id", OCRS_NATIVE)?;
        db::set_setting(&connection, "embedding_model_id", E5_SMALL)?;
        db::set_setting(&connection, "ocr_max_side", "1600")?;

        ensure_defaults(&database, &models)?;
        let selected = selection(&database)?;
        assert_eq!(selected.ocr_model_id, PPOCRV6_TINY);
        assert_eq!(selected.ocr_max_side, DEFAULT_OCR_MAX_SIDE);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
    #[test]
    fn ppocr_thread_budget_is_bounded_for_responsive_desktop_indexing() {
        assert!((1..=4).contains(&ppocr_thread_count()));
    }

    #[test]
    #[ignore = "requires locally installed benchmark model packs; run with RECALL_MODEL_DIR=<models> cargo test benchmark_installed_ocr_models -- --ignored --nocapture"]
    fn benchmark_installed_ocr_models() -> Result<()> {
        let model_dir = benchmark_model_dir();
        assert!(
            is_embedding_installed(&model_dir, E5_SMALL),
            "The E5 default embedding pack is required to load an OCR benchmark runtime"
        );
        let input = benchmark_image(DEFAULT_OCR_MAX_SIDE);
        let expected = normalized("Green Pepper Kitchen");
        let mut benchmarked = 0;

        for model_id in [PPOCRV6_TINY, PPOCRV6_SMALL] {
            if !is_ocr_installed(&model_dir, model_id) {
                eprintln!("Skipping {model_id}: its local OCR pack is not installed");
                continue;
            }
            let runtime = AiRuntime::load_for_selection(
                &model_dir,
                &ModelSelection {
                    ocr_model_id: model_id.into(),
                    embedding_model_id: E5_SMALL.into(),
                    ocr_max_side: DEFAULT_OCR_MAX_SIDE,
                    visual_model_id: VISUAL_DISABLED.into(),
                },
            )?;
            let _ = runtime.ocr_image(&input)?;
            let mut samples = Vec::new();
            let mut output = String::new();
            for _ in 0..3 {
                let started = Instant::now();
                output = runtime.ocr_image(&input)?;
                samples.push(started.elapsed());
            }
            assert!(
                normalized(&output).contains(&expected),
                "{model_id} did not meet the OCR quality floor for restaurant-card.jpg: {output:?}"
            );
            println!(
                "MODEL_BENCHMARK kind=ocr model={model_id} sample=restaurant-card.jpg median_ms={} quality=pass",
                median_millis(&mut samples)
            );
            benchmarked += 1;
        }
        assert!(
            benchmarked > 0,
            "Install at least one PP-OCRv6 pack to benchmark it"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires locally installed embedding packs; run with RECALL_MODEL_DIR=<models> cargo test benchmark_installed_embedding_models -- --ignored --nocapture"]
    fn benchmark_installed_embedding_models() -> Result<()> {
        let model_dir = benchmark_model_dir();
        let documents = vec![
            "Green Pepper Kitchen is Anika's restaurant recommendation.".to_owned(),
            "The train to Bengaluru leaves on Friday at 22:15.".to_owned(),
            "The SSD warranty expires on 12 July 2029.".to_owned(),
            "Lemon rice needs lemons, curry leaves, and peanuts.".to_owned(),
            "Recall search continues working after Wi-Fi is disconnected.".to_owned(),
            "A receipt is a record of an in-store purchase.".to_owned(),
            "Project notes should remain on the local device.".to_owned(),
            "A grocery list can contain rice and vegetables.".to_owned(),
        ];
        let mut benchmarked = 0;
        for model_id in [E5_SMALL, EMBEDDING_GEMMA_Q8, MINILM_F32] {
            if !is_embedding_installed(&model_dir, model_id) {
                eprintln!("Skipping {model_id}: its local embedding pack is not installed");
                continue;
            }
            let mut embedder = load_embedding_runtime(&model_dir, model_id)?;
            let inputs = if model_id == E5_SMALL {
                documents
                    .iter()
                    .map(|document| format!("passage: {document}"))
                    .collect()
            } else {
                documents.clone()
            };
            let _ = embedder.embed(inputs.clone(), Some(16))?;
            let mut samples = Vec::new();
            let mut embeddings = Vec::new();
            for _ in 0..3 {
                let started = Instant::now();
                embeddings = embedder.embed(inputs.clone(), Some(16))?;
                samples.push(started.elapsed());
            }
            assert_eq!(embeddings.len(), documents.len());
            assert!(
                embeddings.iter().all(|embedding| !embedding.is_empty()),
                "{model_id} returned an empty embedding"
            );
            let query = if model_id == E5_SMALL {
                "query: Which restaurant did Anika recommend?"
            } else {
                "Which restaurant did Anika recommend?"
            };
            let query_embedding = embedder.embed(vec![query.to_owned()], Some(1))?;
            let top_result = embeddings
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    cosine_similarity(&query_embedding[0], left)
                        .total_cmp(&cosine_similarity(&query_embedding[0], right))
                })
                .map(|(index, _)| index);
            assert_eq!(
                top_result,
                Some(0),
                "{model_id} did not rank the restaurant document first"
            );
            println!(
                "MODEL_BENCHMARK kind=embedding model={model_id} documents={} median_ms={} dimensions={} retrieval=pass",
                documents.len(),
                median_millis(&mut samples),
                embeddings[0].len()
            );
            benchmarked += 1;
        }
        assert!(
            benchmarked > 0,
            "Install at least one embedding pack to benchmark it"
        );
        Ok(())
    }
}
