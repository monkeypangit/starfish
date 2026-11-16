//! Lateral CA taken out of the raw mosaic before the demosaic
//! (`shaders/ca_fix.wgsl`; the method and its measurements are in
//! training/common.py's `ca_correct`). Red and blue are moved onto green by a
//! guided filter whose guide is green at each sample's scene point, so the
//! network sees a mosaic as if the lens had none, and nothing is drawn.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use super::{Gpu, texture_desc};
use crate::raw::CfaPattern;

const WORKGROUP_SIZE: u32 = 16;
/// common.py's CA_FIX_EPS, CA_FIX_SIGMA and CA_FIX_RATIO_MAX.
const EPS: f32 = 1e-3;
const SIGMA: f32 = 0.5;
const RATIO_MAX: f32 = 4.0;

/// Lateral chromatic aberration: at pixel p of the frame, red is where green
/// is at p + s(rho) (p - centre), s the radial scale offset `k0 + k1 r^2 +
/// k2 r^4` with r = min(rho, reach), rho = |p - centre| * inv_radius; the
/// same for blue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ca {
    /// (row, column) in the frame's coordinates.
    pub centre: [f32; 2],
    pub inv_radius: f32,
    /// `[k0, k1, k2]` of red and of blue.
    pub red: [f32; 3],
    pub blue: [f32; 3],
    /// Red's and blue's.
    pub reach: [f32; 2],
    /// Red's and blue's constant (dy, dx) added to the radial field: a
    /// decentred lens moves a colour over the whole frame.
    pub shift: [[f32; 2]; 2],
    /// A raw sample at or above this is clipped: the fit leaves it out and
    /// it is resampled along the field without the guided move (see the
    /// shader's `fit` and `gather`); `f32::INFINITY` for none, when the
    /// clipped samples were rebuilt first.
    pub clip: f32,
    /// The strength of the guided move: 1, or 0 for a plain resample along
    /// the field (the benchmark's baseline).
    pub guide: f32,
    /// The result may leave the range of the same-colour samples within 2.5
    /// px of its scene point by this share of that range (negative: no bound).
    pub bound: f32,
}

/// `Ca::bound` as the app runs it: `STARFISH_CA_BOUND` (a number, negative
/// for none) overrides `BOUND`.
fn bound() -> f32 {
    static B: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("STARFISH_CA_BOUND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(BOUND)
    })
}

/// The default `Ca::bound`: the result stays within the range of the
/// same-colour samples around its scene point.
const BOUND: f32 = 0.0;

/// The default `Ca::clip`: what counts as clipped everywhere.
pub const CLIP: f32 = super::CLIPPED;

impl Ca {
    /// `m`, measured on a `height` x `width` frame.
    pub fn new(height: usize, width: usize, m: &crate::lens::Measured) -> Self {
        Self {
            centre: [(height as f32 - 1.0) / 2.0, (width as f32 - 1.0) / 2.0],
            inv_radius: 1.0 / m.radius,
            red: m.coefficients[0],
            blue: m.coefficients[1],
            reach: m.reach,
            shift: m.shift,
            clip: CLIP,
            guide: 1.0,
            bound: bound(),
        }
    }

    /// The same, for a frame whose clipped samples were already rebuilt.
    pub fn unclipped(self) -> Self {
        Self {
            clip: f32::INFINITY,
            ..self
        }
    }

    /// `(dy, dx)`: the sample of CFA colour `c` (0 red, 2 blue; green is 0)
    /// at row `y`, column `x` saw the scene at `(y, x) - d`. The shader's `field`.
    pub fn field(&self, c: usize, y: isize, x: isize) -> [f32; 2] {
        let (k, reach, shift) = match c {
            0 => (self.red, self.reach[0], self.shift[0]),
            2 => (self.blue, self.reach[1], self.shift[1]),
            _ => return [0.0; 2],
        };
        let v = [y as f32 - self.centre[0], x as f32 - self.centre[1]];
        let rho = v[0].hypot(v[1]) * self.inv_radius;
        let r2 = rho.min(reach).powi(2);
        let s = k[0] + r2 * (k[1] + r2 * k[2]);
        [s * v[0] + shift[0], s * v[1] + shift[1]]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    cfa_w: u32,
    cfa_h: u32,
    cy: f32,
    cx: f32,
    inv_radius: f32,
    eps: f32,
    red: [f32; 3],
    reach_r: f32,
    blue: [f32; 3],
    reach_b: f32,
    sigma: f32,
    /// RATIO_MAX in the frame's units, for red and blue
    ratio_max: [f32; 2],
    clip: f32,
    shift_r: [f32; 2],
    shift_b: [f32; 2],
    /// `Ca::guide`, then padding
    tune: [f32; 4],
}

/// The passes in order, with the bindings each reads and the one it writes
/// (see the shader).
const PASSES: [(&str, &[u32], u32); 5] = [
    ("green", &[2], 7),
    ("moved", &[3], 8),
    ("fit", &[2, 4], 9),
    ("fit_mean", &[5], 10),
    ("gather", &[2, 3, 4, 6], 11),
];

pub struct CaFix {
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl CaFix {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/ca_fix.wgsl"));
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let pipelines = PASSES
            .iter()
            .map(|&(name, reads, writes)| {
                let mut entries = vec![
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
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                ];
                // f32 textures are not filterable; they are only loaded
                entries.extend(reads.iter().map(|&b| {
                    entry(
                        b,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    )
                }));
                entries.push(entry(
                    writes,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: format(writes),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ));
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(name),
                    entries: &entries,
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(name),
                            bind_group_layouts: &[&layout],
                            push_constant_ranges: &[],
                        }),
                    ),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Self { pipelines }
    }

    /// `frame` (`R32Float`, as decoded, CFA `cfa` from its top left) with
    /// red and blue moved by `ca`, given in the frame's coordinates, into a
    /// new `R32Float` texture with `TEXTURE_BINDING`. `wb`: the white-balance
    /// gains (R, G, B), for the colour ratio's bound, which is set balanced.
    pub fn apply(
        &self,
        gpu: &Gpu,
        frame: &wgpu::Texture,
        cfa: &CfaPattern,
        ca: &Ca,
        wb: [f32; 3],
    ) -> wgpu::Texture {
        let device = &gpu.device;
        let (w, h) = (frame.width(), frame.height());
        let params = Params {
            w,
            h,
            cfa_w: cfa.width as u32,
            cfa_h: cfa.height as u32,
            cy: ca.centre[0],
            cx: ca.centre[1],
            inv_radius: ca.inv_radius,
            eps: EPS,
            red: ca.red,
            reach_r: ca.reach[0],
            blue: ca.blue,
            reach_b: ca.reach[1],
            sigma: SIGMA,
            // a balanced ratio c / G is the frame's times gain_c / gain_g
            ratio_max: [RATIO_MAX * wb[1] / wb[0], RATIO_MAX * wb[1] / wb[2]],
            clip: ca.clip,
            shift_r: ca.shift[0],
            shift_b: ca.shift[1],
            tune: [ca.guide, ca.bound, 0.0, 0.0],
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CA fix params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pattern = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CA fix CFA"),
            contents: bytemuck::cast_slice(&cfa.colors),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // bindings 7..=11 are written by one pass and read as 3..=6 by later ones
        let made: Vec<wgpu::Texture> = (7..=11)
            .map(|b| {
                device.create_texture(&texture_desc(
                    "CA fix",
                    (w, h),
                    format(b),
                    wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                ))
            })
            .collect();
        let views: Vec<wgpu::TextureView> = made
            .iter()
            .map(|t| t.create_view(&Default::default()))
            .collect();
        let frame_view = frame.create_view(&Default::default());
        let view = |b: u32| match b {
            2 => &frame_view,
            3..=6 => &views[b as usize - 3],
            _ => &views[b as usize - 7],
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CA fix"),
        });
        for (pipeline, &(name, reads, writes)) in self.pipelines.iter().zip(&PASSES) {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pattern.as_entire_binding(),
                },
            ];
            entries.extend(
                reads
                    .iter()
                    .chain([&writes])
                    .map(|&b| wgpu::BindGroupEntry {
                        binding: b,
                        resource: wgpu::BindingResource::TextureView(view(b)),
                    }),
            );
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(w.div_ceil(WORKGROUP_SIZE), h.div_ceil(WORKGROUP_SIZE), 1);
        }
        gpu.queue.submit([encoder.finish()]);
        made.into_iter().nth(4).unwrap()
    }
}

/// The format of the texture at a written binding: a has red's and blue's.
fn format(binding: u32) -> wgpu::TextureFormat {
    match binding {
        9 | 10 => wgpu::TextureFormat::Rg32Float,
        _ => wgpu::TextureFormat::R32Float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::Gpu;

    /// Against training's `ca_correct`: `STARFISH_TEST_CA_FIX_BIN` from
    /// `ca_fix_vector.py` (u32 h, w, cfa period; f32 red and blue scale
    /// offsets; f32 white-balance gains; u32 CFA; f32 mosaic and expected
    /// output, both unbalanced, as the app runs the pass).
    #[test]
    #[ignore]
    fn matches_pytorch() {
        let path = std::env::var("STARFISH_TEST_CA_FIX_BIN").expect("STARFISH_TEST_CA_FIX_BIN");
        let bytes = std::fs::read(path).unwrap();
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let float = |i: usize| f32::from_bits(word(i));
        let (h, w, period) = (word(0) as usize, word(1) as usize, word(2) as usize);
        let (sr, sb) = (float(3), float(4));
        let wb = [float(5), float(6), float(7)];
        let cfa = CfaPattern {
            width: period,
            height: period,
            colors: (0..period * period).map(|i| word(8 + i)).collect(),
        };
        let start = 8 + period * period;
        let mosaic: Vec<f32> = (0..h * w).map(|i| float(start + i)).collect();
        let expected: Vec<f32> = (0..h * w).map(|i| float(start + h * w + i)).collect();

        let gpu = Gpu::new().unwrap();
        let frame = gpu.device.create_texture_with_data(
            &gpu.queue,
            &texture_desc(
                "frame",
                (w as u32, h as u32),
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&mosaic),
        );
        let ca = Ca {
            centre: [(h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0],
            inv_radius: 1.0,
            red: [sr, 0.0, 0.0],
            blue: [sb, 0.0, 0.0],
            reach: [f32::MAX; 2],
            shift: [[0.0; 2]; 2],
            clip: CLIP,
            guide: 1.0,
            bound: -1.0,
        };
        let pass = CaFix::new(&gpu.device);
        let run = |ca: &Ca| -> Vec<f32> {
            let fixed = pass.apply(&gpu, &frame, &cfa, ca, wb);
            crate::processing::read_back_f32(&gpu, &fixed).unwrap()
        };
        let got = run(&ca);

        // the border the network never outputs aside, as the training's tests do
        let border = 36;
        // The shader leaves `a` at 0 near clipped samples (fit: RADIUS + 1,
        // then fit_mean RADIUS, then the gather's reach plus the field's
        // shift), which ca_correct does not do: leave out what is within
        // CLIP_REACH of a clipped sample
        const CLIP_REACH: usize = 14;
        let mut near = vec![false; h * w];
        for y in 0..h {
            for x in 0..w {
                if mosaic[y * w + x] >= CLIP {
                    for yy in y.saturating_sub(CLIP_REACH)..(y + CLIP_REACH + 1).min(h) {
                        for xx in x.saturating_sub(CLIP_REACH)..(x + CLIP_REACH + 1).min(w) {
                            near[yy * w + xx] = true;
                        }
                    }
                }
            }
        }
        let (mut err, mut max, mut n) = (0f64, 0f32, 0usize);
        for y in border..h - border {
            for x in border..w - border {
                if near[y * w + x] {
                    continue;
                }
                let e = got[y * w + x] - expected[y * w + x];
                err += (e as f64).powi(2);
                max = max.max(e.abs());
                n += 1;
            }
        }
        let psnr = -10.0 * (err / n as f64).log10();
        println!(
            "CA fix vs PyTorch: {psnr:.1} dB, max {max:.2e} ({:.1}% of the pixels near clipped samples left out)",
            100.0 * near.iter().filter(|&&b| b).count() as f64 / (h * w) as f64
        );
        assert!(psnr > 90.0, "{psnr} dB");

        // against the same photograph without CA, if the vector has it: red
        // and blue sites only (green is untouched)
        if bytes.len() >= 4 * (start + 3 * h * w) {
            let truth: Vec<f32> = (0..h * w).map(|i| float(start + 2 * h * w + i)).collect();
            let score = |data: &[f32]| {
                let (mut err, mut n) = (0f64, 0usize);
                for y in border..h - border {
                    for x in border..w - border {
                        let c = cfa.colors[(y % period) * period + x % period];
                        if c != 1 && !near[y * w + x] {
                            err += f64::from(data[y * w + x] - truth[y * w + x]).powi(2);
                            n += 1;
                        }
                    }
                }
                -10.0 * (err / n as f64).log10()
            };
            println!("red and blue vs the CA-free mosaic: CA left in {:.2} dB", score(&mosaic));
            println!("  corrected {:.2} dB", score(&got));
            let plain = run(&Ca { guide: 0.0, ..ca });
            println!("  plain resample {:.2} dB", score(&plain));
        }
    }

    /// The CA benchmark (`ca_bench.py` in the training repo writes its vectors, synthetic
    /// scenes with a known CA and the same mosaic without it):
    /// `STARFISH_TEST_CA_BENCH=../starfish-neural-demosaic/ca_bench cargo test --release
    /// ca_benchmark -- --ignored --nocapture`. For each scene and variant the
    /// error of the red and blue sites against the CA-free mosaic, as PSNR (raw
    /// units, white = 1), by distance from the clipped samples of the truth:
    /// on one, 1-3 px, 4-8 px, further.
    #[test]
    #[ignore]
    fn ca_benchmark() {
        let dir = std::env::var("STARFISH_TEST_CA_BENCH").unwrap_or_else(|_| "../starfish-neural-demosaic/ca_bench".into());
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .collect();
        files.sort();
        let gpu = Gpu::new().unwrap();
        let pass = CaFix::new(&gpu.device);
        for path in files {
            let bytes = std::fs::read(&path).unwrap();
            let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
            let float = |i: usize| f32::from_bits(word(i));
            let (h, w, period) = (word(0) as usize, word(1) as usize, word(2) as usize);
            let (sr, sb) = (float(3), float(4));
            let wb = [float(5), float(6), float(7)];
            let cfa = CfaPattern {
                width: period,
                height: period,
                colors: (0..period * period).map(|i| word(8 + i)).collect(),
            };
            let start = 8 + period * period;
            let mosaic: Vec<f32> = (0..h * w).map(|i| float(start + i)).collect();
            let truth: Vec<f32> = (0..h * w).map(|i| float(start + 2 * h * w + i)).collect();
            let frame = gpu.device.create_texture_with_data(
                &gpu.queue,
                &texture_desc(
                    "frame",
                    (w as u32, h as u32),
                    wgpu::TextureFormat::R32Float,
                    wgpu::TextureUsages::TEXTURE_BINDING,
                ),
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&mosaic),
            );
            let ca = Ca {
                centre: [(h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0],
                inv_radius: 1.0,
                red: [sr, 0.0, 0.0],
                blue: [sb, 0.0, 0.0],
                reach: [f32::MAX; 2],
                shift: [[0.0; 2]; 2],
                clip: CLIP,
                guide: 1.0,
                bound: -1.0,
            };
            // Chebyshev distance (up to 9) to a clipped sample of the truth
            let mut dist = vec![9u8; h * w];
            let mut reach: Vec<bool> = truth.iter().map(|&v| v >= CLIP).collect();
            for (i, r) in reach.iter().enumerate() {
                if *r {
                    dist[i] = 0;
                }
            }
            for d in 1..9u8 {
                let mut next = reach.clone();
                for y in 0..h {
                    for x in 0..w {
                        if reach[y * w + x] {
                            continue;
                        }
                        'n: for yy in y.saturating_sub(1)..(y + 2).min(h) {
                            for xx in x.saturating_sub(1)..(x + 2).min(w) {
                                if reach[yy * w + xx] {
                                    next[y * w + x] = true;
                                    dist[y * w + x] = d;
                                    break 'n;
                                }
                            }
                        }
                    }
                }
                reach = next;
            }
            let zone = |d: u8| match d {
                0 => 0,
                1..=3 => 1,
                4..=8 => 2,
                _ => 3,
            };
            let border = 20;
            let score = |data: &[f32]| -> [f64; 4] {
                let (mut err, mut n) = ([0f64; 4], [0usize; 4]);
                for y in border..h - border {
                    for x in border..w - border {
                        let c = cfa.colors[(y % period) * period + x % period];
                        if c == 1 {
                            continue;
                        }
                        let z = zone(dist[y * w + x]);
                        err[z] += f64::from(data[y * w + x] - truth[y * w + x]).powi(2);
                        n[z] += 1;
                    }
                }
                [0, 1, 2, 3].map(|z| {
                    if n[z] == 0 {
                        f64::NAN
                    } else {
                        -10.0 * (err[z] / n[z] as f64).max(1e-12).log10()
                    }
                })
            };
            // the share of the red and blue sites within 3 px of a clipped
            // sample that came out more than 15% (and 0.02) under / over the
            // truth: the dark outlines and bright specks the eye picks out
            let outliers = |data: &[f32]| -> (f64, f64) {
                let (mut dark, mut bright, mut n) = (0usize, 0usize, 0usize);
                for y in border..h - border {
                    for x in border..w - border {
                        let c = cfa.colors[(y % period) * period + x % period];
                        if c == 1 || dist[y * w + x] > 3 {
                            continue;
                        }
                        n += 1;
                        let (v, t) = (data[y * w + x], truth[y * w + x]);
                        if v < t - (0.15 * t).max(0.02) {
                            dark += 1;
                        } else if v > t + (0.15 * t).max(0.02) {
                            bright += 1;
                        }
                    }
                }
                (100.0 * dark as f64 / n.max(1) as f64, 100.0 * bright as f64 / n.max(1) as f64)
            };
            let run = |ca: &Ca| {
                crate::processing::read_back_f32(
                    &gpu,
                    &pass.apply(&gpu, &frame, &cfa, ca, wb),
                )
                .unwrap()
            };
            println!(
                "\n{} ({}x{}, {:.1}% of the pixels within 8 px of a clipped sample)",
                path.file_stem().unwrap().to_string_lossy(),
                w,
                h,
                100.0 * dist.iter().filter(|&&d| d < 9).count() as f64 / (h * w) as f64
            );
            println!(
                "{:<30}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
                "variant", "clipped", "1-3 px", "4-8 px", "far", "dark %", "bright %"
            );
            let mut row = |name: &str, data: &[f32]| {
                let s = score(data);
                let (dark, bright) = outliers(data);
                println!(
                    "{name:<30}{:>9.2}{:>9.2}{:>9.2}{:>9.2}{dark:>9.2}{bright:>9.2}",
                    s[0], s[1], s[2], s[3]
                );
            };
            row("uncorrected", &mosaic);
            for (name, variant) in variants(ca) {
                row(name, &run(&variant));
            }
        }

        /// The variants the benchmark runs: the CA with the settings changed.
        fn variants(ca: Ca) -> Vec<(&'static str, Ca)> {
            vec![
                ("plain resample", Ca { guide: 0.0, ..ca }),
                ("guided (clipped samples out)", ca),
                ("guided, no clip rule", Ca { clip: f32::INFINITY, ..ca }),
                ("guided, bound 0", Ca { bound: 0.0, ..ca }),
                ("guided, bound 0.1", Ca { bound: 0.1, ..ca }),
                ("guided, bound 0.25", Ca { bound: 0.25, ..ca }),
                ("guided, bound 0.5", Ca { bound: 0.5, ..ca }),
            ]
        }
    }

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("../shaders/ca_fix.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
