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
const REGION_WHOLE: i64 = -1;
const REGION_MAX_COUNT: usize = 12;
const REGION_MIN_LONG_SIDE: u32 = 1_024;
const REGION_TARGET_ASPECT: f32 = 16.0 / 9.0 * 1.5;
const REGION_MAX_ASPECT: f32 = 16.0 / 9.0 * 2.0;
const REGION_PIXEL_TARGET: u32 = 1_536;
const REGION_PIXEL_MAX: u32 = 4_096;
const MAX_IMAGE_PIXELS: u64 = 80_000_000;

/// A stable, persisted location within the oriented source image. `region_id`
/// maps directly to `image_embeddings.page_number`; negative ids reserve this
/// namespace for image regions while positive ids remain available for PDF pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRegion {
    pub region_id: i64,
    pub kind: &'static str,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub source_width: u32,
    pub source_height: u32,
}

#[derive(Debug, Clone)]
pub struct ImageRegionCrop {
    pub region: ImageRegion,
    pub image: image::RgbImage,
}

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
    embed_text: bool,
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
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff" => {
            let Some(ai) = ai else {
                return Ok(ProcessOutcome::ModelRequired);
            };
            let decoded = decode_oriented_rgb(path)?;
            let max_side = ai.ocr_max_side();
            let ocr_image = if decoded.width().max(decoded.height()) > max_side {
                image::DynamicImage::ImageRgb8(decoded.clone())
                    .resize(max_side, max_side, image::imageops::FilterType::Triangle)
                    .into_rgb8()
            } else {
                decoded.clone()
            };
            fs::create_dir_all(thumbnail_dir)?;
            image::imageops::thumbnail(&decoded, 768, 512)
                .save_with_format(
                    thumbnail_dir.join(format!("{asset_id}.png")),
                    ImageFormat::Png,
                )
                .map_err(|e| RecallError::Message(format!("Thumbnail failed: {e}")))?;
            let mut text = ai.ocr_image(&ocr_image)?;
            // Whole-image OCR preserves broad layout. Region OCR restores text
            // density in tall, panoramic, and very-large screenshots where a
            // single resize would make small text unreadable.
            let crops = image_regions(&decoded);
            if crops.len() > 1 {
                let mut regional_text = Vec::new();
                for crop in crops.iter().skip(1) {
                    let region = resize_for_ocr(&crop.image, max_side);
                    regional_text.push(ai.ocr_image(&region)?);
                }
                text = merge_ocr_text(&text, &regional_text);
            }
            vec![PageText {
                page_number: None,
                text,
            }]
        }
        _ => return Ok(ProcessOutcome::Skipped("Unsupported extension".into())),
    };
    let mut chunks = chunk_pages(&pages);
    // For screenshots/images, add finer-grained regional chunks (groups of
    // nearby OCR lines) so a query can match a single region, not just the page.
    if matches!(
        extension,
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff"
    ) {
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
    if embed_text {
        let Some(ai) = ai else {
            return Ok(ProcessOutcome::Chunks(chunks));
        };
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
    let reader = image::ImageReader::open(path)
        .map_err(|e| RecallError::Message(format!("Image could not be opened: {e}")))?
        .with_guessed_format()
        .map_err(|e| RecallError::Message(format!("Image format could not be read: {e}")))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| RecallError::Message(format!("Image dimensions could not be read: {e}")))?;
    if width as u64 * height as u64 > MAX_IMAGE_PIXELS {
        return Err(RecallError::Message(format!(
            "Image has {width}x{height} pixels, above the {MAX_IMAGE_PIXELS}-pixel safety limit"
        )));
    }
    let decoded = image::open(path)
        .map_err(|e| RecallError::Message(format!("Image decoding failed: {e}")))?;
    // CLIP/OCR consume RGB. Alpha is composited over white rather than silently
    // discarded to black, which preserves common transparent PNG/WebP UI assets.
    let rgba = decoded.into_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = pixel[3] as u16;
        let blend = |channel: u8| ((channel as u16 * alpha + 255 * (255 - alpha)) / 255) as u8;
        rgb.put_pixel(
            x,
            y,
            image::Rgb([blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]),
        );
    }
    if let Some(orientation) = read_exif_orientation(path) {
        rgb = apply_orientation(rgb, orientation);
    }
    Ok(rgb)
}

/// Whole image plus adaptive aspect-ratio or pixel-grid crops. This follows
/// the useful part of Panoptikon's slicing approach while retaining Recall's
/// parent-asset result model.
pub fn image_regions(image: &image::RgbImage) -> Vec<ImageRegionCrop> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let whole = ImageRegion {
        region_id: REGION_WHOLE,
        kind: "whole",
        x: 0,
        y: 0,
        width,
        height,
        source_width: width,
        source_height: height,
    };
    let mut bounds = vec![("whole", 0, 0, width, height)];
    let long = width.max(height);
    let short = width.min(height).max(1);
    let aspect = long as f32 / short as f32;
    if long > REGION_PIXEL_MAX {
        let cols = (width as f32 / REGION_PIXEL_TARGET as f32).ceil() as u32;
        let rows = (height as f32 / REGION_PIXEL_TARGET as f32).ceil() as u32;
        for row in 0..rows {
            for col in 0..cols {
                if bounds.len() >= REGION_MAX_COUNT {
                    break;
                }
                let left = col * width / cols;
                let top = row * height / rows;
                let right = ((col + 1) * width / cols).max(left + 1);
                let bottom = ((row + 1) * height / rows).max(top + 1);
                bounds.push(("grid", left, top, right - left, bottom - top));
            }
        }
    } else if long >= REGION_MIN_LONG_SIDE && aspect > REGION_MAX_ASPECT {
        let slices =
            ((aspect / REGION_TARGET_ASPECT).ceil() as u32).clamp(2, (REGION_MAX_COUNT - 1) as u32);
        for index in 0..slices {
            if width >= height {
                let left = index * width / slices;
                let right = ((index + 1) * width / slices).max(left + 1);
                bounds.push(("aspect", left, 0, right - left, height));
            } else {
                let top = index * height / slices;
                let bottom = ((index + 1) * height / slices).max(top + 1);
                bounds.push(("aspect", 0, top, width, bottom - top));
            }
        }
    }
    bounds
        .into_iter()
        .enumerate()
        .map(|(index, (kind, x, y, w, h))| {
            let region = if index == 0 {
                whole.clone()
            } else {
                ImageRegion {
                    region_id: -(index as i64 + 1),
                    kind,
                    x,
                    y,
                    width: w,
                    height: h,
                    source_width: width,
                    source_height: height,
                }
            };
            ImageRegionCrop {
                image: image::imageops::crop_imm(image, x, y, w, h).to_image(),
                region,
            }
        })
        .collect()
}

fn resize_for_ocr(image: &image::RgbImage, max_side: u32) -> image::RgbImage {
    if image.width().max(image.height()) > max_side {
        image::DynamicImage::ImageRgb8(image.clone())
            .resize(max_side, max_side, image::imageops::FilterType::Triangle)
            .into_rgb8()
    } else {
        image.clone()
    }
}

fn merge_ocr_text(whole: &str, regions: &[String]) -> String {
    let mut lines = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for text in std::iter::once(whole).chain(regions.iter().map(String::as_str)) {
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let key: String = line
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            if key.len() >= 3 && seen.insert(key) {
                lines.push(line.to_string());
            }
        }
    }
    lines.join("\n")
}

/// EXIF orientation tag (1..=8), if present.
fn read_exif_orientation(path: &Path) -> Option<u32> {
    let file = fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
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

    #[test]
    fn extreme_aspect_images_get_parent_and_slices() {
        let image = image::RgbImage::new(200, 1200);
        let regions = image_regions(&image);
        assert!(regions.len() > 1);
        assert_eq!(regions[0].region.region_id, -1);
    }

    #[test]
    fn normal_images_only_use_the_whole_region() {
        let image = image::RgbImage::new(1200, 900);
        let regions = image_regions(&image);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].region.kind, "whole");
        assert_eq!(regions[0].region.region_id, -1);
    }

    #[test]
    fn wide_images_are_sliced_too() {
        let image = image::RgbImage::new(3000, 900);
        let regions = image_regions(&image);
        assert!(regions
            .iter()
            .skip(1)
            .all(|crop| crop.region.kind == "aspect"));
    }

    #[test]
    fn ocr_merge_deduplicates_region_overlap() {
        assert_eq!(
            merge_ocr_text("PNR 123\nCoach A1", &["PNR 123\nSeat 10".into()]),
            "PNR 123\nCoach A1\nSeat 10"
        );
    }
}
