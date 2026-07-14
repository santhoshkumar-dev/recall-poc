//! Deterministic MobileCLIP2-S0 image preprocessing.
//!
//! Mirrors the model's `preprocessor_config.json` (CLIPImageProcessor):
//! resize shortest edge to 256, center-crop 256×256, rescale to [0,1]
//! (mean 0 / std 1), producing an NCHW RGB f32 tensor.

use image::RgbImage;
use ndarray::{Array, Array4};

/// MobileCLIP2-S0 square input side.
pub const INPUT_SIZE: u32 = 256;

/// Resize (shortest edge → 256) then center-crop to 256×256.
fn resize_and_center_crop(image: &RgbImage) -> RgbImage {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return RgbImage::new(INPUT_SIZE, INPUT_SIZE);
    }
    // Scale so the shorter side is exactly INPUT_SIZE.
    let scale = INPUT_SIZE as f32 / w.min(h) as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(INPUT_SIZE);
    let new_h = ((h as f32 * scale).round() as u32).max(INPUT_SIZE);
    let resized = image::imageops::resize(
        image,
        new_w,
        new_h,
        image::imageops::FilterType::CatmullRom,
    );
    let x = (new_w - INPUT_SIZE) / 2;
    let y = (new_h - INPUT_SIZE) / 2;
    image::imageops::crop_imm(&resized, x, y, INPUT_SIZE, INPUT_SIZE).to_image()
}

/// Build the `[1, 3, 256, 256]` f32 tensor for the vision encoder.
pub fn image_to_tensor(image: &RgbImage) -> Array4<f32> {
    let cropped = resize_and_center_crop(image);
    let side = INPUT_SIZE as usize;
    let mut tensor = Array::zeros((1, 3, side, side));
    for (x, y, pixel) in cropped.enumerate_pixels() {
        let (xi, yi) = (x as usize, y as usize);
        // rescale 1/255, mean 0 / std 1 → pixel / 255.
        tensor[[0, 0, yi, xi]] = pixel[0] as f32 / 255.0;
        tensor[[0, 1, yi, xi]] = pixel[1] as f32 / 255.0;
        tensor[[0, 2, yi, xi]] = pixel[2] as f32 / 255.0;
    }
    tensor
}

/// L2-normalize a vector in place-safe form; zero vectors are returned as-is.
pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_correct_shape() {
        let img = RgbImage::from_pixel(640, 480, image::Rgb([120, 130, 140]));
        let tensor = image_to_tensor(&img);
        assert_eq!(tensor.shape(), &[1, 3, 256, 256]);
        // 120/255 ≈ 0.4706
        assert!((tensor[[0, 0, 0, 0]] - 120.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn normalizes_unit_length() {
        let n = l2_normalize(vec![3.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6);
    }
}
