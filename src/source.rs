//! The developed image on the display device: one texture array whose layers
//! are a grid over the image, each overlapping its neighbours by [`OVERLAP`]
//! pixels, so a bicubic or mipmapped sample anywhere reads a single layer
//! and shows no seams. Beyond the image the edge pixels repeat. Every layer
//! has mipmaps. `develop.wgsl` (`Grid`, `sample_source`,
//! `sample_source_cubic`) reads it; the preview, the export and the local
//! contrast overview share one upload.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use crate::processing::{IMAGE_FORMAT, Image};

/// Pixels each layer shares with its neighbours, per side. Layers start at
/// multiples of it, so mip levels up to log2(OVERLAP) line up across layers.
const OVERLAP: u32 = 32;
/// Longest layer side, overlap included.
const MAX_LAYER: u32 = 4096;

/// `develop.wgsl`'s `Grid`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GridUniform {
    size: [f32; 2],
    step: [f32; 2],
    layer: [f32; 2],
    columns: u32,
    overlap: f32,
}

pub struct Source {
    pub image_id: u64,
    /// Image size in pixels.
    pub size: [u32; 2],
    device: wgpu::Device,
    /// All layers, all mip levels.
    pub view: wgpu::TextureView,
    pub grid: GridUniform,
    /// GPU memory, all mip levels.
    bytes: u64,
}

/// The source uploaded last, for the next user of the same image (the
/// export after the preview).
static LAST: Mutex<Option<Arc<Source>>> = Mutex::new(None);

/// Splits `len` pixels into layers of at most `max` pixels: their count and
/// step, a multiple of [`OVERLAP`].
fn split(len: u32, max: u32) -> (u32, u32) {
    let count = len.div_ceil(max);
    (count, len.div_ceil(count).next_multiple_of(OVERLAP))
}

impl Source {
    /// `image` on `device`: the last upload if it is the same image on the
    /// same device, else a new one. A 24 MP image takes ~260 MB, which an
    /// integrated GPU may not have twice over: the last upload is let go of
    /// first (callers drop theirs before asking for a new one), and running
    /// out of memory is an `Err` rather than an invalid texture.
    pub fn get(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> Result<Arc<Source>, String> {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(source) = last.as_ref().filter(|s| s.image_id == image.id && s.device == *device) {
            return Ok(source.clone());
        }
        *last = None;
        // Per device, not per thread: an error of iced's in the meantime
        // lands here too (it would otherwise panic).
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let source = Self::upload(device, queue, image);
        let invalid = pollster::block_on(device.pop_error_scope());
        let out_of_memory = pollster::block_on(device.pop_error_scope());
        if out_of_memory.is_some() {
            return Err(format!(
                "Out of GPU memory: showing this {:.0} MP image needs about {} MB of video \
                 memory. Close other GPU-heavy programs, or give the GPU more memory \
                 (integrated GPUs: the UMA frame buffer size in the BIOS).",
                f64::from(image.width * image.height) / 1e6,
                source.bytes / 1_000_000
            ));
        }
        if let Some(error) = invalid {
            return Err(format!("GPU error uploading the image: {error}"));
        }
        let source = Arc::new(source);
        *last = Some(source.clone());
        Ok(source)
    }

    fn upload(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> Self {
        let max_layer = device.limits().max_texture_dimension_2d.min(MAX_LAYER);
        let max_step = (max_layer - 2 * OVERLAP) / OVERLAP * OVERLAP;
        let (columns, step_x) = split(image.width, max_step);
        let (rows, step_y) = split(image.height, max_step);
        let (layer_w, layer_h) = (step_x + 2 * OVERLAP, step_y + 2 * OVERLAP);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source"),
            size: wgpu::Extent3d {
                width: layer_w,
                height: layer_h,
                depth_or_array_layers: columns * rows,
            },
            mip_level_count: layer_w.max(layer_h).ilog2() + 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: IMAGE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        for row in 0..rows {
            for column in 0..columns {
                let x0 = i64::from(column * step_x) - i64::from(OVERLAP);
                let y0 = i64::from(row * step_y) - i64::from(OVERLAP);
                write_layer(queue, &texture, row * columns + column, image, (x0, y0), (layer_w, layer_h));
            }
        }
        generate_mips(device, queue, &texture);
        // all mip levels: 4/3 of the first
        let bytes = u64::from(layer_w * layer_h * columns * rows) * u64::from(BYTES_PER_PIXEL) * 4 / 3;

        Self {
            image_id: image.id,
            size: [image.width, image.height],
            device: device.clone(),
            view: texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            bytes,
            grid: GridUniform {
                size: [image.width as f32, image.height as f32],
                step: [step_x as f32, step_y as f32],
                layer: [layer_w as f32, layer_h as f32],
                columns,
                overlap: OVERLAP as f32,
            },
        }
    }
}

const BYTES_PER_PIXEL: u32 = 8;

/// Fills `layer` of `texture` with the image pixels from `origin` on (which
/// may lie beyond the image), repeating the edge pixels beyond it.
fn write_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    image: &Image,
    origin: (i64, i64),
    size: (u32, u32),
) {
    let (width, height) = (i64::from(image.width), i64::from(image.height));
    let (x0, y0) = origin;
    let (x1, y1) = (x0 + i64::from(size.0), y0 + i64::from(size.1));
    // The part on the image goes straight from the image's pixels.
    let (ix0, ix1) = (x0.max(0), x1.min(width));
    let (iy0, iy1) = (y0.max(0), y1.min(height));
    let write = |at: (i64, i64), bytes: &[u8], offset: u64, row_bytes: u32, w: i64, h: i64| {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: (at.0 - x0) as u32,
                    y: (at.1 - y0) as u32,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(h as u32),
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
        );
    };
    write(
        (ix0, iy0),
        bytemuck::cast_slice(&image.pixels),
        (iy0 * width + ix0) as u64 * u64::from(BYTES_PER_PIXEL),
        image.width * BYTES_PER_PIXEL,
        ix1 - ix0,
        iy1 - iy0,
    );

    // The rest repeats the nearest edge pixel: bands above and below, and
    // left and right of the image part.
    let band = |xs: (i64, i64), ys: (i64, i64)| {
        let (w, h) = (xs.1 - xs.0, ys.1 - ys.0);
        if w <= 0 || h <= 0 {
            return;
        }
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for y in ys.0..ys.1 {
            let y = y.clamp(0, height - 1);
            for x in xs.0..xs.1 {
                let i = ((y * width + x.clamp(0, width - 1)) * 4) as usize;
                pixels.extend_from_slice(&image.pixels[i..i + 4]);
            }
        }
        write((xs.0, ys.0), bytemuck::cast_slice(&pixels), 0, w as u32 * BYTES_PER_PIXEL, w, h);
    };
    band((x0, x1), (y0, iy0));
    band((x0, x1), (iy1, y1));
    band((x0, ix0), (iy0, iy1));
    band((ix1, x1), (iy0, iy1));
}

/// Fills mip levels 1.. of every layer by repeated bilinear halving
/// (`mip.wgsl`).
fn generate_mips(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
    let module = device.create_shader_module(wgpu::include_wgsl!("shaders/mip.wgsl"));
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("source mip"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(IMAGE_FORMAT.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("source mip"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for layer in 0..texture.depth_or_array_layers() {
        for level in 1..texture.mip_level_count() {
            let view = |level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            };
            let (source, target) = (view(level - 1), view(level));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("source mip"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("source mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
    queue.submit([encoder.finish()]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_cover_the_image_and_line_up() {
        for len in [1, 31, 32, 4000, 4032, 4033, 6000, 8256, 11648] {
            let (count, step) = split(len, 4032);
            assert!(count * step >= len, "{len}");
            assert!((count - 1) * step < len, "{len}: an empty layer");
            assert!(step <= 4032 && step % OVERLAP == 0, "{len}");
        }
    }

    #[test]
    fn grid_uniform_matches_the_shader() {
        let module = naga::front::wgsl::parse_str(crate::develop::SHADER).unwrap();
        let grid = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("Grid")).unwrap().1;
        assert_eq!(grid.inner.size(module.to_ctx()) as usize, std::mem::size_of::<GridUniform>());
    }
}
