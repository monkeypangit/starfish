//! Lens correction on the developed image: distortion, chromatic aberration
//! and vignetting from the profile in the raw file (see `crate::lens`),
//! in one resampling pass between the demosaic and the readback. The app
//! asks it for CA only; `develop.wgsl` draws distortion and vignetting.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use super::{Gpu, IMAGE_FORMAT, texture_desc};
use crate::lens::{Corrections, LUT_RHO_MAX, LUT_SIZE, LensCorrection};

const WORKGROUP_SIZE: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size: [f32; 2],
    centre: [f32; 2],
    inv_half_diagonal: f32,
    lut_max: f32,
    lut_last: f32,
    _pad: f32,
}

pub struct LensPass {
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

impl LensPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/lens.wgsl"));
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lens"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("lens"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// Corrects `input` with the corrections in `on` (an [`IMAGE_FORMAT`] texture with `TEXTURE_BINDING`)
    /// into a new texture of the same size.
    pub fn apply(
        &self,
        gpu: &Gpu,
        input: &wgpu::Texture,
        lens: &LensCorrection,
        on: Corrections,
    ) -> wgpu::Texture {
        let device = &gpu.device;
        let (width, height) = (input.width(), input.height());
        let output = device.create_texture(&texture_desc(
            "lens corrected",
            (width, height),
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        ));

        let (half_diagonal, lut) = lens.lut(width, height, on);
        debug_assert_eq!(lut.len(), LUT_SIZE);
        let params = Params {
            size: [width as f32, height as f32],
            centre: [(width - 1) as f32 / 2.0, (height - 1) as f32 / 2.0],
            inv_half_diagonal: 1.0 / half_diagonal,
            lut_max: LUT_RHO_MAX,
            lut_last: (LUT_SIZE - 1) as f32,
            _pad: 0.0,
        };
        let uniform = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let params = uniform("lens params", bytemuck::bytes_of(&params));
        let lut = uniform("lens table", bytemuck::cast_slice(&lut));

        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, params.as_entire_binding()),
                entry(1, lut.as_entire_binding()),
                entry(2, wgpu::BindingResource::TextureView(&input_view)),
                entry(3, wgpu::BindingResource::Sampler(&self.sampler)),
                entry(4, wgpu::BindingResource::TextureView(&output_view)),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lens correction"),
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
        gpu.queue.submit([encoder.finish()]);
        output
    }
}
