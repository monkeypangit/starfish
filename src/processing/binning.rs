//! Fast, low-quality demosaic: bins each CFA block into one RGB pixel, then
//! bilinearly upscales back to full resolution. Kept as a future preview path;
//! select it with `STARFISH_DEMOSAIC=binning`.
//!
//! Bayer bins 2×2 blocks, X-Trans bins 3×3 blocks (each 3×3 quadrant of the
//! 6×6 X-Trans pattern contains all three colours).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use std::sync::atomic::AtomicBool;

use super::{Demosaic, Gpu, IMAGE_FORMAT, Noise, texture_desc};
use crate::raw::{MAX_CFA_SIZE, RawFrame, SensorKind};

const WORKGROUP_SIZE: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BinParams {
    size: [u32; 2],
    cfa_size: [u32; 2],
    bin: u32,
    _pad: [u32; 3],
    wb: [f32; 4],
    pattern: [[u32; 4]; MAX_CFA_SIZE * MAX_CFA_SIZE / 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpscaleParams {
    bin: f32,
    _pad: [f32; 3],
}

pub struct BinningDemosaic {
    bin_pipeline: wgpu::ComputePipeline,
    upscale_pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

impl BinningDemosaic {
    pub fn new(device: &wgpu::Device) -> Self {
        let bin_module = device.create_shader_module(wgpu::include_wgsl!("../shaders/bin.wgsl"));
        let upscale_module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/upscale.wgsl"));

        // Explicit layout for the bin pass: R32Float is not filterable, which
        // an auto-derived layout would assume.
        let bin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bin"),
            entries: &[
                layout_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                layout_entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                layout_entry(
                    2,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: IMAGE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ],
        });
        let bin_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bin"),
            bind_group_layouts: &[&bin_layout],
            push_constant_ranges: &[],
        });

        let compute =
            |label, module: &wgpu::ShaderModule, layout: Option<&wgpu::PipelineLayout>| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout,
                    module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

        Self {
            bin_pipeline: compute("bin", &bin_module, Some(&bin_pipeline_layout)),
            upscale_pipeline: compute("upscale", &upscale_module, None),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("upscale"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }
}

impl Demosaic for BinningDemosaic {
    fn name(&self) -> &str {
        "Binning + bilinear upscale"
    }

    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        _noise: Option<Noise>,
        _cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>), String> {
        let bin = match raw.cfa.kind() {
            SensorKind::Bayer => 2,
            SensorKind::XTrans => 3,
            SensorKind::Other => return Err("Binning needs a Bayer or X-Trans CFA".into()),
        };
        let (device, queue) = (&gpu.device, &gpu.queue);
        let (width, height) = (raw.width as u32, raw.height as u32);
        let binned_size = (width.div_ceil(bin), height.div_ceil(bin));

        let raw_texture = device.create_texture_with_data(
            queue,
            &texture_desc(
                "raw",
                (width, height),
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&raw.data),
        );
        let binned = device.create_texture(&texture_desc(
            "binned",
            binned_size,
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        ));
        let output = device.create_texture(&texture_desc(
            "demosaiced",
            (width, height),
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        ));

        let mut pattern = [0u32; MAX_CFA_SIZE * MAX_CFA_SIZE];
        pattern[..raw.cfa.colors.len()].copy_from_slice(&raw.cfa.colors);
        let bin_params = BinParams {
            size: [width, height],
            cfa_size: [raw.cfa.width as u32, raw.cfa.height as u32],
            bin,
            _pad: [0; 3],
            wb: [raw.wb[0], raw.wb[1], raw.wb[2], 1.0],
            pattern: bytemuck::cast(pattern),
        };
        let upscale_params = UpscaleParams {
            bin: bin as f32,
            _pad: [0.0; 3],
        };

        let uniform = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let bin_uniform = uniform("bin params", bytemuck::bytes_of(&bin_params));
        let upscale_uniform = uniform("upscale params", bytemuck::bytes_of(&upscale_params));

        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (raw_view, binned_view, output_view) =
            (view(&raw_texture), view(&binned), view(&output));

        let bin_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bin"),
            layout: &self.bin_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, bin_uniform.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&raw_view)),
                entry(2, wgpu::BindingResource::TextureView(&binned_view)),
            ],
        });
        let upscale_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale"),
            layout: &self.upscale_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, upscale_uniform.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&binned_view)),
                entry(2, wgpu::BindingResource::Sampler(&self.sampler)),
                entry(3, wgpu::BindingResource::TextureView(&output_view)),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binning demosaic"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.bin_pipeline);
            pass.set_bind_group(0, &bin_group, &[]);
            pass.dispatch_workgroups(
                binned_size.0.div_ceil(WORKGROUP_SIZE),
                binned_size.1.div_ceil(WORKGROUP_SIZE),
                1,
            );

            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(0, &upscale_group, &[]);
            pass.dispatch_workgroups(
                width.div_ceil(WORKGROUP_SIZE),
                height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        queue.submit([encoder.finish()]);

        Ok((output, None))
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
