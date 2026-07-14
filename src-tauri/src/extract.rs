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
            let max_side = ai.ocr_max_side();
            let ocr_image = if decoded.width().max(decoded.height()) > max_side {
                decoded
                    .resize(max_side, max_side, image::imageops::FilterType::Triangle)
                    .into_rgb8()
            } else {
                decoded.into_rgb8()
            };
            fs::create_dir_all(thumbnail_dir)?;
            image::imageops::thumbnail(&ocr_image, 480, 320)
                .save_with_format(
                    thumbnail_dir.join(format!("{asset_id}.png")),
                    ImageFormat::Png,
                )
                .map_err(|e| RecallError::Message(format!("Thumbnail failed: {e}")))?;
            vec![PageText {
                page_number: None,
                text: ai.ocr_image(&ocr_image)?,
            }]
        }
        _ => return Ok(ProcessOutcome::Skipped("Unsupported extension".into())),
    };
    let mut chunks = chunk_pages(&pages);
    // For screenshots/images, add finer-grained regional chunks (groups of
    // nearby OCR lines) so a query can match a single region, not just the page.
    if matches!(extension, "png" | "jpg" | "jpeg" | "webp") {
        if let Some(page) = pages.first() {
            let base = chunks.len() as i64;
            for (offset, region) in regional_chunks(&page.text).into_iter().enumerate() {
                chunks.push(ChunkInput {
                    index: base + offset as i64,
                    page_number: None,
                    text: region,
                    embedding: None,
                });
            }
        }
    }
    if chunks.is_empty() {
        return Ok(ProcessOutcome::Skipped(
            "No searchable text was found".into(),
        ));
    }
    if let Some(ai) = ai {
        let embeddings =
            ai.embed_documents(chunks.iter().map(|chunk| chunk.text.clone()).collect())?;
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
            chunk.embedding = Some(embedding);
        }
    }
    Ok(ProcessOutcome::Chunks(chunks))
}

/// Decode an image applying EXIF orientation, returning an RGB buffer.
/// Used by the visual indexing pass so CLIP sees upright pixels.
pub fn decode_oriented_rgb(path: &Path) -> Result<image::RgbImage> {
    let decoded = image::open(path)
        .map_err(|e| RecallError::Message(format!("Image decoding failed: {e}")))?;
    let mut rgb = decoded.into_rgb8();
    if let Some(orientation) = read_exif_orientation(path) {
        rgb = apply_orientation(rgb, orientation);
    }
    Ok(rgb)
}

/// EXIF orientation tag (1..=8), if present.
fn read_exif_orientation(path: &Path) -> Option<u32> {
    let file = fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new()
        .read_from_container(&mut reader)
        .ok()?;
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
}

/// Apply the standard EXIF orientation transforms.
fn apply_orientation(img: image::RgbImage, orientation: u32) -> image::RgbImage {
    use image::imageops::{flip_horizontal, flip_vertical, rotate180, rotate270, rotate90};
    match orientation {
        2 => flip_horizontal(&img),
        3 => rotate180(&img),
        4 => flip_vertical(&img),
        5 => rotate90(&flip_horizontal(&img)),
        6 => rotate90(&img),
        7 => rotate270(&flip_horizontal(&img)),
        8 => rotate270(&img),
        _ => img,
    }
}

/// Group nearby OCR lines into regional chunks. PP-OCR joins detected lines
/// with newlines in reading order; we group consecutive non-empty lines into
/// small regions (a lightweight proxy for spatial proximity). Regions with a
/// single short line are dropped to avoid noise. Skipped entirely when the OCR
/// text is already small (the full chunk suffices).
const LINES_PER_REGION: usize = 5;

pub fn regional_chunks(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() <= LINES_PER_REGION {
        return Vec::new();
    }
    let mut regions = Vec::new();
    for group in lines.chunks(LINES_PER_REGION) {
        let region = group.join("\n");
        if region.chars().filter(|c| c.is_alphanumeric()).count() >= 8 {
            regions.push(region);
        }
    }
    regions
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
