//! Export of the developed image as a JPEG or PNG, drawn on iced's device by
//! the preview's own pipeline (`develop.wgsl`), so it shows the same pixels.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::{ImageBuffer, Rgb};

use iced::{Point, Size};

use crate::lens::Corrections;
use crate::preview::{Adjustments, Crop, PreviewPipeline, PrimitiveImage, Rotation};
use crate::processing::Image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    JpegL,
    JpegM,
    JpegS,
    Png8,
    Png16,
}

impl Format {
    pub const ALL: [Format; 5] =
        [Format::JpegL, Format::JpegM, Format::JpegS, Format::Png8, Format::Png16];

    /// JPEG quality of the large, medium and small settings.
    fn jpeg_quality(self) -> Option<u8> {
        match self {
            Format::JpegL => Some(95),
            Format::JpegM => Some(90),
            Format::JpegS => Some(85),
            Format::Png8 | Format::Png16 => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Format::JpegL | Format::JpegM | Format::JpegS => "jpg",
            Format::Png8 | Format::Png16 => "png",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Format::JpegL => "JPEG (L)",
            Format::JpegM => "JPEG (M)",
            Format::JpegS => "JPEG (S)",
            Format::Png8 => "PNG 8-bit",
            Format::Png16 => "PNG 16-bit",
        })
    }
}

/// Output tiles are rendered at most this size: any device takes them, and
/// each readback stays at 64 MB.
const OUTPUT_TILE: u32 = 2048;
/// f32, because 16-bit PNG needs more than f16's 11 significant bits.
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// How the image is drawn: what the preview shows outside crop mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub adjustments: Adjustments,
    pub rotation: Rotation,
    pub crop: Crop,
    /// Distortion and vignetting to apply (CA is the demosaic stage's).
    pub lens: Corrections,
}

/// Writes the part of `image` inside the crop's frame, drawn as `view`
/// says, to `path`. It is drawn on `device` (iced's) by the preview's own
/// pipeline, at one image pixel per output pixel.
pub fn export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
    view: View,
    format: Format,
    path: &Path,
) -> Result<(), String> {
    // Error scopes are per device, not per thread: an error of iced's in the
    // meantime lands here too (it would otherwise panic).
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let rendered = render(device, queue, image, view);
    let invalid = pollster::block_on(device.pop_error_scope());
    let out_of_memory = pollster::block_on(device.pop_error_scope());
    if out_of_memory.is_some() {
        return Err("Out of GPU memory. Close other GPU-heavy programs and try again.".into());
    }
    if let Some(error) = invalid {
        return Err(format!("GPU error: {error}"));
    }
    let (encoded, width, height) = rendered?;
    write(&encoded, width, height, format, path)
}

/// The export's size in pixels: the frame's, rounded, which keeps every
/// output pixel's centre inside the frame.
pub fn frame_pixels(image: &Image, rotation: Rotation, crop: Crop) -> (u32, u32) {
    let rotated = rotation.size(Size::new(image.width as f32, image.height as f32));
    let frame = crop.frame_size(rotated);
    ((frame.width.round() as u32).max(1), (frame.height.round() as u32).max(1))
}

/// The display-encoded frame, three f32 per pixel, with its size.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
    view: View,
) -> Result<(Vec<f32>, usize, usize), String> {
    let View {
        adjustments,
        rotation,
        crop,
        lens,
    } = view;
    let (out_w, out_h) = frame_pixels(image, rotation, crop);
    let mut pipeline = PreviewPipeline::with_format(device, OUTPUT_FORMAT);
    let (tile_w, tile_h) = (out_w.min(OUTPUT_TILE), out_h.min(OUTPUT_TILE));
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("export"),
        size: wgpu::Extent3d {
            width: tile_w,
            height: tile_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    const BYTES_PER_PIXEL: u32 = 16;
    let row_bytes = (tile_w * BYTES_PER_PIXEL).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export readback"),
        size: u64::from(row_bytes) * u64::from(tile_h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let (width, height) = (out_w as usize, out_h as usize);
    let mut encoded = vec![0.0f32; width * height * 3];
    for y0 in (0..out_h).step_by(OUTPUT_TILE as usize) {
        for x0 in (0..out_w).step_by(OUTPUT_TILE as usize) {
            let (w, h) = ((out_w - x0).min(OUTPUT_TILE), (out_h - y0).min(OUTPUT_TILE));
            let shot = PrimitiveImage {
                image: image.clone(),
                scale: 1.0,
                center: Point::new(x0 as f32 + w as f32 / 2.0, y0 as f32 + h as f32 / 2.0),
                adjustments,
                rotation,
                crop,
                cropping: false,
                lens,
                point: false,
            };
            pipeline.prepare_image(device, queue, &shot, [w as f32, h as f32])?;

            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("export"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Black where the image does not reach.
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
                pipeline.draw_image(&mut pass);
            }
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_bytes),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);

            let slice = readback.slice(..u64::from(row_bytes) * u64::from(h));
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|e| format!("GPU error: {e}"))?;
            receiver
                .recv()
                .map_err(|e| e.to_string())?
                .map_err(|e| format!("Failed to read the export back: {e}"))?;
            {
                let data = slice.get_mapped_range();
                for y in 0..h as usize {
                    let row: &[f32] = bytemuck::cast_slice(&data[y * row_bytes as usize..][..w as usize * 16]);
                    let out = ((y0 as usize + y) * width + x0 as usize) * 3;
                    for (x, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
                        encoded[out + x * 3..][..3].copy_from_slice(&pixel[..3]);
                    }
                }
            }
            readback.unmap();
        }
    }
    Ok((encoded, width, height))
}

/// Writes display-encoded `encoded` (three per pixel, 0..1) as `format`.
pub fn write(encoded: &[f32], width: usize, height: usize, format: Format, path: &Path) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut file = BufWriter::new(file);
    let (w, h) = (width as u32, height as u32);
    let failed = |e: image::ImageError| format!("{}: {e}", path.display());
    match format {
        Format::Png16 => {
            let samples: Vec<u16> = encoded.iter().map(|&c| (c * 65535.0 + 0.5) as u16).collect();
            let buffer = ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, samples).unwrap();
            buffer.write_to(&mut file, image::ImageFormat::Png).map_err(failed)
        }
        Format::Png8 | Format::JpegL | Format::JpegM | Format::JpegS => {
            let samples: Vec<u8> = encoded.iter().map(|&c| (c * 255.0 + 0.5) as u8).collect();
            let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, samples).unwrap();
            if let Some(quality) = format.jpeg_quality() {
                buffer
                    .write_with_encoder(JpegEncoder::new_with_quality(&mut file, quality))
                    .map_err(failed)
            } else {
                buffer.write_to(&mut file, image::ImageFormat::Png).map_err(failed)
            }
        }
    }
}
