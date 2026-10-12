//! The base layer for local contrast: a guided filter of the log luma on a
//! small overview of the whole image (see `shaders/local.wgsl`), drawn from
//! the image's mipmaps (`source::Source`).

use bytemuck::{Pod, Zeroable};

use crate::develop::{self, LensUniform};
use crate::lens::Corrections;
use crate::processing::IMAGE_FORMAT;
use crate::source::{GridUniform, Source};

/// Longest side of the overview, at most.
const OVERVIEW_SIZE: u32 = 1024;
/// Box filter radius, as a fraction of the overview's long side.
const RADIUS: f32 = 0.03;
/// Variance (in stops²) of log luma that counts as an edge to keep.
const EDGE: f32 = 0.1;
/// Intermediate sums need the precision.
const SUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cam_to_srgb: [[f32; 4]; 3],
    radius: i32,
    eps: f32,
    /// The source's mip level the overview is drawn from.
    level: f32,
    _pad: f32,
    grid: GridUniform,
    lens: LensUniform,
}

/// What the base was computed from.
#[derive(PartialEq)]
struct Key {
    image_id: u64,
    cam_to_srgb: [[f32; 4]; 3],
    lens: Corrections,
}

struct Base {
    key: Key,
    view: wgpu::TextureView,
}

pub struct LocalContrast {
    /// fs_h, fs_v_ab, fs_h_ab, fs_final.
    pipelines: [wgpu::RenderPipeline; 4],
    layout: wgpu::BindGroupLayout,
    /// fs_overview, which reads the source.
    overview_pipeline: wgpu::RenderPipeline,
    overview_layout: wgpu::BindGroupLayout,
    /// Trilinear, for the source.
    source_sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    /// For sampling the base.
    pub sampler: wgpu::Sampler,
    /// Until a base is built: one black texel.
    empty: wgpu::TextureView,
    base: Option<Base>,
}

fn texture(
    device: &wgpu::Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("local contrast"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl LocalContrast {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(develop::shader("local", include_str!("shaders/local.wgsl")));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("local contrast"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1, false),
                texture_entry(2, false),
            ],
        });
        let overview_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("local contrast overview"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("local contrast"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let overview_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("local contrast overview"),
            bind_group_layouts: &[&overview_layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry, format: wgpu::TextureFormat| {
            let layout = if entry == "fs_overview" { &overview_pipeline_layout } else { &pipeline_layout };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let pipelines = [
            pipeline("fs_h", SUM_FORMAT),
            pipeline("fs_v_ab", SUM_FORMAT),
            pipeline("fs_h_ab", SUM_FORMAT),
            pipeline("fs_final", IMAGE_FORMAT),
        ];
        let overview_pipeline = pipeline("fs_overview", IMAGE_FORMAT);

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("local contrast"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let source_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let empty_texture = texture(device, (1, 1), IMAGE_FORMAT);
        let empty = empty_texture.create_view(&Default::default());

        Self {
            pipelines,
            layout,
            overview_pipeline,
            overview_layout,
            source_sampler,
            params,
            sampler,
            empty,
            base: None,
        }
    }

    /// The base texture (a black texel before one is built).
    pub fn view(&self) -> &wgpu::TextureView {
        self.base.as_ref().map_or(&self.empty, |base| &base.view)
    }

    /// Forgets the base, when the image goes away.
    pub fn clear(&mut self) {
        self.base = None;
    }

    /// Recomputes the base for `source` unless it is current; whether it
    /// did. `lens` is the lens correction the image is drawn with: its
    /// uniform, table and the corrections in it.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Source,
        cam_to_srgb: [[f32; 4]; 3],
        lens: (LensUniform, &wgpu::TextureView, Corrections),
    ) -> bool {
        let (lens_uniform, lens_lut, lens) = lens;
        let key = Key {
            image_id: source.image_id,
            cam_to_srgb,
            lens,
        };
        let image_size = source.size;
        if self.base.as_ref().is_some_and(|base| base.key == key) {
            return false;
        }

        // Halve until the long side fits.
        let mut shift = 0;
        while image_size[0].max(image_size[1]).div_ceil(1 << shift) > OVERVIEW_SIZE {
            shift += 1;
        }
        let size = (
            image_size[0].div_ceil(1 << shift),
            image_size[1].div_ceil(1 << shift),
        );
        let radius = ((size.0.max(size.1) as f32 * RADIUS).round() as i32).max(1);

        let overview = texture(device, size, IMAGE_FORMAT);
        let sums = [texture(device, size, SUM_FORMAT), texture(device, size, SUM_FORMAT)];
        let base = texture(device, size, IMAGE_FORMAT);
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let (overview_view, sum_views, base_view) =
            (view(&overview), [view(&sums[0]), view(&sums[1])], view(&base));

        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                cam_to_srgb,
                radius,
                eps: EDGE,
                level: shift as f32,
                _pad: 0.0,
                grid: source.grid,
                lens: lens_uniform,
            }),
        );

        let mut encoder = device.create_command_encoder(&Default::default());

        // Each overview pixel is the mip texel over the same block.
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("local contrast overview"),
                layout: &self.overview_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.source_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(lens_lut),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("local contrast overview"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &overview_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.overview_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // (pipeline, source sums, target)
        let passes = [
            (0, &sum_views[1], &sum_views[0]),
            (1, &sum_views[0], &sum_views[1]),
            (2, &sum_views[1], &sum_views[0]),
            (3, &sum_views[0], &base_view),
        ];
        for (index, source, target) in passes {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("local contrast"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&overview_view),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("local contrast"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipelines[index]);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        self.base = Some(Base { key, view: base_view });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_is_valid() {
        let source = format!("{}\n{}", develop::SHADER, include_str!("shaders/local.wgsl"));
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
        // The Rust uniform must fill the WGSL struct exactly.
        let params = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("Params")).unwrap().1;
        assert_eq!(params.inner.size(module.to_ctx()) as usize, std::mem::size_of::<Params>());
    }
}
