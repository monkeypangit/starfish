//! Export of the developed image as a JPEG or PNG, drawn on iced's device by
//! the preview's own pipeline (`develop.wgsl`), so it shows the same pixels.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::{ImageBuffer, Rgb};

use iced::{Point, Size};

use crate::lens::Corrections;
use crate::preview::{Adjustments, Crop, PreviewPipeline, PrimitiveImage, Rotation};
use crate::processing::Image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Jpeg,
    Png8,
    Png16,
}

impl Format {
    pub const ALL: [Format; 3] = [Format::Jpeg, Format::Png8, Format::Png16];

    pub fn extension(self) -> &'static str {
        match self {
            Format::Jpeg => "jpg",
            Format::Png8 | Format::Png16 => "png",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Format::Jpeg => "JPEG",
            Format::Png8 => "PNG 8-bit",
            Format::Png16 => "PNG 16-bit",
        })
    }
}

/// How hard a PNG is compressed: only the time spent, the pixels are the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Low,
    Medium,
    High,
}

impl Compression {
    pub const ALL: [Compression; 3] = [Compression::Low, Compression::Medium, Compression::High];

    fn png(self) -> CompressionType {
        match self {
            Compression::Low => CompressionType::Fast,
            Compression::Medium => CompressionType::Default,
            Compression::High => CompressionType::Best,
        }
    }
}

impl std::fmt::Display for Compression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Compression::Low => "Low",
            Compression::Medium => "Medium",
            Compression::High => "High",
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
    quality: u8,
    compression: Compression,
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
    write(&encoded, width, height, format, quality, compression, path)
}

/// The export's size in pixels: the frame's, rounded, which keeps every
/// output pixel's centre inside the frame.
pub fn frame_pixels(image: &Image, rotation: Rotation, crop: Crop) -> (u32, u32) {
    let rotated = rotation.size(Size::new(image.width as f32, image.height as f32));
    let frame = crop.frame_size(rotated);
    (
        (frame.width.round() as u32).max(1),
        (frame.height.round() as u32).max(1),
    )
}

/// The display-encoded frame, three f32 per pixel, with its size.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
    view: View,
) -> Result<(Vec<f32>, usize, usize), String> {
    let (out_w, out_h) = frame_pixels(image, view.rotation, view.crop);
    let (width, height) = (out_w as usize, out_h as usize);
    let mut encoded = vec![0.0f32; width * height * 3];
    let tiles: Vec<Tile> = (0..out_h)
        .step_by(OUTPUT_TILE as usize)
        .flat_map(|y0| {
            (0..out_w)
                .step_by(OUTPUT_TILE as usize)
                .map(move |x0| Tile {
                    x: x0,
                    y: y0,
                    w: (out_w - x0).min(OUTPUT_TILE),
                    h: (out_h - y0).min(OUTPUT_TILE),
                })
        })
        .collect();
    render_tiles(device, queue, image, view, &tiles, |tile, pixels| {
        for y in 0..tile.h as usize {
            let out = ((tile.y as usize + y) * width + tile.x as usize) * 3;
            let row = &pixels[y * tile.w as usize * 3..][..tile.w as usize * 3];
            encoded[out..out + row.len()].copy_from_slice(row);
        }
    })?;
    Ok((encoded, width, height))
}

/// A rectangle of the export's output, in its pixels.
#[derive(Debug, Clone, Copy)]
struct Tile {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Draws each of `tiles` (each at most [`OUTPUT_TILE`] square) as the export
/// would and hands `sink` its display-encoded pixels, three f32 per pixel.
fn render_tiles(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
    view: View,
    tiles: &[Tile],
    mut sink: impl FnMut(Tile, &[f32]),
) -> Result<(), String> {
    let View {
        adjustments,
        rotation,
        crop,
        lens,
    } = view;
    let mut pipeline = PreviewPipeline::with_format(device, OUTPUT_FORMAT);
    let tile_w = tiles.iter().map(|t| t.w).max().unwrap_or(1);
    let tile_h = tiles.iter().map(|t| t.h).max().unwrap_or(1);
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

    let mut pixels = Vec::new();
    for &tile in tiles {
        let Tile { x: x0, y: y0, w, h } = tile;
        {
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
                clip_highlights: false,
                clip_shadows: false,
                crossover: false,
                grid: false,
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
                pixels.clear();
                for y in 0..h as usize {
                    let row: &[f32] =
                        bytemuck::cast_slice(&data[y * row_bytes as usize..][..w as usize * 16]);
                    for pixel in row.as_chunks::<4>().0 {
                        pixels.extend_from_slice(&pixel[..3]);
                    }
                }
            }
            readback.unmap();
            sink(tile, &pixels);
        }
    }
    Ok(())
}

/// Side of the squares [`estimate_jpeg_size`] encodes, and how many per row
/// and column of its grid.
const SAMPLE: u32 = 64;
const GRID: usize = 16;

/// Estimates the JPEG's size in bytes at `quality`: a 16 x 16 grid of 64 x 64
/// px squares, evenly spaced over the frame (on whole 16 px blocks, as the
/// JPEG's own), are drawn as the export draws them and encoded in memory as
/// one picture, whose bytes per pixel are scaled to the frame. Frames smaller
/// than a square are sampled whole.
pub fn estimate_jpeg_size(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
    view: View,
    quality: u8,
) -> Result<u64, String> {
    let (out_w, out_h) = frame_pixels(image, view.rotation, view.crop);
    let (w, h) = (out_w.min(SAMPLE), out_h.min(SAMPLE));
    let at = |i: usize, room: u32| (room as usize * i / (GRID - 1)) as u32 / 16 * 16;
    let tiles: Vec<Tile> = (0..GRID * GRID)
        .map(|i| Tile {
            x: at(i % GRID, out_w - w),
            y: at(i / GRID, out_h - h),
            w,
            h,
        })
        .collect();
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    // The squares side by side in one picture, so the file's headers (a
    // few hundred bytes each, as much as a small square's pixels) count once.
    let (mw, mh) = (GRID * w as usize, GRID * h as usize);
    let mut mosaic = vec![0u8; mw * mh * 3];
    let mut placed = 0usize;
    let rendered = render_tiles(device, queue, image, view, &tiles, |tile, pixels| {
        let (cx, cy) = ((placed % GRID) * w as usize, (placed / GRID) * h as usize);
        placed += 1;
        for y in 0..tile.h as usize {
            let out = ((cy + y) * mw + cx) * 3;
            let row = &pixels[y * tile.w as usize * 3..][..tile.w as usize * 3];
            for (to, &c) in mosaic[out..][..row.len()].iter_mut().zip(row) {
                *to = (c * 255.0 + 0.5) as u8;
            }
        }
    });
    let invalid = pollster::block_on(device.pop_error_scope());
    let out_of_memory = pollster::block_on(device.pop_error_scope());
    if out_of_memory.is_some() {
        return Err("Out of GPU memory. Close other GPU-heavy programs and try again.".into());
    }
    if let Some(error) = invalid {
        return Err(format!("GPU error: {error}"));
    }
    rendered?;
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(mw as u32, mh as u32, mosaic).unwrap();
    let mut out = Vec::new();
    buffer
        .write_with_encoder(JpegEncoder::new_with_quality(&mut out, quality))
        .map_err(|e| e.to_string())?;
    let per_pixel = out.len() as f64 / (mw * mh) as f64;
    Ok((per_pixel * f64::from(out_w) * f64::from(out_h)) as u64)
}

/// Writes display-encoded `encoded` (three per pixel, 0..1) as `format`.
pub fn write(
    encoded: &[f32],
    width: usize,
    height: usize,
    format: Format,
    quality: u8,
    compression: Compression,
    path: &Path,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut file = BufWriter::new(file);
    let (w, h) = (width as u32, height as u32);
    let failed = |e: image::ImageError| format!("{}: {e}", path.display());
    match format {
        Format::Png16 => {
            let samples: Vec<u16> = encoded
                .iter()
                .map(|&c| (c * 65535.0 + 0.5) as u16)
                .collect();
            let buffer = ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, samples).unwrap();
            buffer
                .write_with_encoder(PngEncoder::new_with_quality(
                    &mut file,
                    compression.png(),
                    PngFilter::Adaptive,
                ))
                .map_err(failed)
        }
        Format::Png8 | Format::Jpeg => {
            let samples: Vec<u8> = encoded.iter().map(|&c| (c * 255.0 + 0.5) as u8).collect();
            let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, samples).unwrap();
            if format == Format::Jpeg {
                buffer
                    .write_with_encoder(JpegEncoder::new_with_quality(&mut file, quality))
                    .map_err(failed)
            } else {
                buffer
                    .write_with_encoder(PngEncoder::new_with_quality(
                        &mut file,
                        compression.png(),
                        PngFilter::Adaptive,
                    ))
                    .map_err(failed)
            }
        }
    }
}
