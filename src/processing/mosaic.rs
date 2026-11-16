//! The raw mosaic as an image (a debugging aid): every site shows its own
//! white-balanced sample in its colour's channel and nothing in the others.
//! With "Lateral CA in the mosaic" on (`set_ca_fix`) it is the mosaic
//! `CaFix` hands a network, with the CA taken out.

use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use super::ca_fix::{Ca, CaFix};
use super::{Demosaic, Gpu, IMAGE_FORMAT, Noise, texture_desc};
use crate::lens::Measured;
use crate::raw::{MAX_CFA_SIZE, RawFrame};

const WORKGROUP_SIZE: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cfa_size: [u32; 2],
    _pad: [u32; 2],
    wb: [f32; 4],
    pattern: [[u32; 4]; MAX_CFA_SIZE * MAX_CFA_SIZE / 4],
}

pub struct MosaicDemosaic {
    pipeline: wgpu::ComputePipeline,
    /// The CA pass, made on first use.
    ca_fix: Mutex<Option<CaFix>>,
}

impl MosaicDemosaic {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/mosaic.wgsl"));
        // Explicit layout: R32Float is not filterable, which an auto-derived
        // one would assume.
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mosaic"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: IMAGE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mosaic"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mosaic"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            ca_fix: Mutex::new(None),
        }
    }
}

impl Demosaic for MosaicDemosaic {
    fn name(&self) -> &str {
        "Mosaic (no demosaic)"
    }

    fn unload(&self) {
        *self.ca_fix.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        _noise: Option<Noise>,
        _cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>, Option<Measured>), String> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let (width, height) = (raw.width as u32, raw.height as u32);
        let frame = device.create_texture_with_data(
            queue,
            &texture_desc(
                "raw frame",
                (width, height),
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&raw.data),
        );
        // as a network that does not take the CA is given the mosaic
        let measured = if super::ca_fix() {
            raw.mosaic_ca().copied()
        } else {
            None
        };
        // see neural.rs: the clipped samples rebuilt first, with the CA field
        let first = super::highlights() && super::highlights_first() && measured.is_some();
        let rebuilt = match (&measured, first) {
            (Some(m), true) => Some(super::highlights::frame(
                gpu,
                raw,
                None,
                Some(&Ca::new(raw.height, raw.width, m)),
            )?),
            _ => None,
        };
        let fixed = measured.map(|m| {
            let start = Instant::now();
            let mut ca = Ca::new(raw.height, raw.width, &m);
            if first {
                ca = ca.unclipped();
            }
            let mut pass = self.ca_fix.lock().unwrap_or_else(PoisonError::into_inner);
            let fixed = pass
                .get_or_insert_with(|| CaFix::new(device))
                .apply(gpu, rebuilt.as_ref().unwrap_or(&frame), &raw.cfa, &ca, raw.wb);
            crate::log::write(&format!(
                "{}: lateral CA ({}) taken out of the mosaic, queued in {} ms",
                self.name(),
                m.summary(),
                start.elapsed().as_millis()
            ));
            (fixed, m)
        });
        let reconstructed = if super::highlights() && !first {
            Some(super::highlights::frame(
                gpu,
                raw,
                fixed.as_ref().map(|(texture, _)| texture),
                None,
            )?)
        } else {
            None
        };
        let source = reconstructed
            .as_ref()
            .or(fixed.as_ref().map(|(texture, _)| texture))
            .unwrap_or(&frame);
        let output = device.create_texture(&texture_desc(
            "mosaic",
            (width, height),
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        ));

        let mut pattern = [0u32; MAX_CFA_SIZE * MAX_CFA_SIZE];
        pattern[..raw.cfa.colors.len()].copy_from_slice(&raw.cfa.colors);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mosaic params"),
            contents: bytemuck::bytes_of(&Params {
                cfa_size: [raw.cfa.width as u32, raw.cfa.height as u32],
                _pad: [0; 2],
                wb: [raw.wb[0], raw.wb[1], raw.wb[2], 1.0],
                pattern: bytemuck::cast(pattern),
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (source_view, output_view) = (
            source.create_view(&Default::default()),
            output.create_view(&Default::default()),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mosaic"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mosaic"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                width.div_ceil(WORKGROUP_SIZE),
                height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        queue.submit([encoder.finish()]);

        Ok((output, None, fixed.map(|(_, m)| m)))
    }
}
