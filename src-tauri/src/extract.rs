use std::{fs, path::Path};

use image::ImageFormat;

use crate::{
    ai::AiRuntime,
    error::{RecallError, Result},
    types::{ChunkInput, PageText},
};

const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const WORDS_PER_CHUNK: usize = 350;
const WORD_OVERLAP: usize = 60;

#[derive(Debug)]
pub enum ProcessOutcome {
    Chunks(Vec<ChunkInput>),
    ModelRequired,
    Skipped(String),
}

pub fn process_file(
    path: &Path,
    extension: &str,
    ai: Option<&AiRuntime>,
    thumbnail_dir: &Path,
    asset_id: &str,
) -> Result<ProcessOutcome> {
    let size = fs::metadata(path)?.len();
    if size == 0 {
        return Ok(ProcessOutcome::Skipped("File is empty".into()));
    }
    if size > MAX_FILE_BYTES {
        return Ok(ProcessOutcome::Skipped(
            "File exceeds the 64 MB POC limit".into(),
        ));
    }
    let pages = match extension {
        "txt" | "md" => vec![PageText {
            page_number: None,
            text: String::from_utf8_lossy(&fs::read(path)?).into_owned(),
        }],
        "pdf" => pdf_extract::extract_text_by_pages(path)
            .map_err(|e| RecallError::Message(format!("PDF extraction failed: {e}")))?
            .into_iter()
            .enumerate()
            .map(|(index, text)| PageText {
                page_number: Some(index as i64 + 1),
                text,
            })
            .collect(),
        "png" | "jpg" | "jpeg" | "webp" => {
            let Some(ai) = ai else {
                return Ok(ProcessOutcome::ModelRequired);
            };
            let decoded = image::open(path)
                .map_err(|e| RecallError::Message(format!("Image decoding failed: {e}")))?;
            fs::create_dir_all(thumbnail_dir)?;
            decoded
                .thumbnail(480, 320)
                .save_with_format(
                    thumbnail_dir.join(format!("{asset_id}.png")),
                    ImageFormat::Png,
                )
                .map_err(|e| RecallError::Message(format!("Thumbnail failed: {e}")))?;
            vec![PageText {
                page_number: None,
                text: ai.ocr_image(&decoded.into_rgb8())?,
            }]
        }
        _ => return Ok(ProcessOutcome::Skipped("Unsupported extension".into())),
    };
    let mut chunks = chunk_pages(&pages);
    if chunks.is_empty() {
        return Ok(ProcessOutcome::Skipped(
            "No searchable text was found".into(),
        ));
    }
    if let Some(ai) = ai {
        let embeddings = ai.embed(chunks.iter().map(|chunk| chunk.text.clone()).collect())?;
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
            chunk.embedding = Some(embedding);
        }
    }
    Ok(ProcessOutcome::Chunks(chunks))
}

pub fn chunk_pages(pages: &[PageText]) -> Vec<ChunkInput> {
    let mut output = Vec::new();
    for page in pages {
        let words: Vec<&str> = page.text.split_whitespace().collect();
        let mut start = 0;
        while start < words.len() {
            let end = (start + WORDS_PER_CHUNK).min(words.len());
            let text = words[start..end].join(" ");
            if !text.trim().is_empty() {
                output.push(ChunkInput {
                    index: output.len() as i64,
                    page_number: page.page_number,
                    text,
                    embedding: None,
                });
            }
            if end == words.len() {
                break;
            }
            start = end.saturating_sub(WORD_OVERLAP);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunking_keeps_page_numbers_and_overlap() {
        let text = (0..500)
            .map(|n| format!("word{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_pages(&[PageText {
            page_number: Some(7),
            text,
        }]);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_number, Some(7));
        assert!(chunks[1].text.starts_with("word290"));
    }
}
