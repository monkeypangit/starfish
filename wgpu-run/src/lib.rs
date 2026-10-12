//! Runs a network exported by training/common.py `export()`.
//!
//! The export directory holds `graph.txt`, one `.wgsl` shader per dispatch and
//! `weights.bin`. Everything about the model -- which kernels, their sizes,
//! weight layouts, buffer sizes and reuse -- was decided at export time; this
//! file only allocates what graph.txt lists and dispatches it in order.
//!
//! The graph is built for one fixed window (e.g. 1056 x 1056). An image of any
//! size is processed as overlapping windows: each keeps `halo` px of context
//! around the core it writes, and windows wrap around the image edge, because
//! the network pads circularly. The result is identical to running the whole
//! image at once. A window whose core is cut short at the image's end only
//! holds image data in its first core + 2 * halo rows and columns, which is
//! all that core needs: the ops are told that size (binding 5) and skip the
//! work beyond it.
//!
//! `run` cuts and places windows on the CPU; `run_texture` does the same on
//! the GPU (a cut shader before the graph's ops, a place shader after), so
//! nothing leaves the device. `run_raw` is `run_texture` with the inputs made
//! on the GPU too: its cut shader pads the raw frame and balances it, and
//! computes sigma, as `noise::inputs` does on the CPU.
//!
//! graph.txt:
//!   storage f16|f32          element type of every buffer
//!   window H W               the size the graph was built for
//!   halo N                   context px kept around each window's core
//!   multiple N               image height and width must be multiples of this
//!   cfa P C0 C1 ..           the P x P colour filter the mosaic must be phased to
//!                            (0 = R, 1 = G, 2 = B), row-major
//!   buffer ID BYTES [FILE]   a buffer, optionally filled from FILE
//!   mosaic ID / sigma ID / output ID
//!   op FILE B0 B1 B2 B3 B4 GX GY GZ   bind buffers B0..B4 to bindings 0..4
//!                                     (0..3 read, 4 written), dispatch GX x GY x GZ;
//!                                     binding 5 is the uniform `Valid`

use std::num::NonZeroU64;
use std::path::Path;
use std::sync::OnceLock;

use wgpu::util::DeviceExt as _;

pub mod noise;

pub struct Model {
    device: wgpu::Device,
    queue: wgpu::Queue,
    f16: bool,
    window: (u32, u32),
    halo: u32,
    multiple: u32,
    cfa: Vec<u32>,
    bufs: Vec<wgpu::Buffer>,
    mosaic: usize,
    sigma: usize,
    output: usize,
    ops: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, [u32; 3])>,
    // the current window's `Valid`, read by every op
    valid: wgpu::Buffer,
    cut: wgpu::ComputePipeline,
    cut_raw: wgpu::ComputePipeline,
    place: wgpu::ComputePipeline,
    // two readback buffers for `run`, so windows can overlap; made on first use
    stage: OnceLock<[wgpu::Buffer; 2]>,
}

/// One window of `run_texture`, as the cut and place shaders see it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Window {
    y0: u32,
    x0: u32, // window start in the image (may wrap)
    h: u32,
    w: u32, // image size
    cy0: u32,
    cy1: u32, // core rows ...
    cx0: u32,
    cx1: u32, // ... and columns written to the output
    oy: u32,
    ox: u32, // image position of the output texture's origin
    gain: [f32; 3],
    a: f32,
    b: f32, // `run_raw`'s white balance and noise model; unused otherwise
}

/// The rows and columns of the window that hold image data, from its top
/// left; the ops skip work wholly beyond them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Valid {
    h: u32,
    w: u32,
}

/// A raw frame for [`Model::run_raw`].
pub struct RawInput<'a> {
    /// The mosaic as decoded, not white balanced: `R32Float`, TEXTURE_BINDING.
    pub frame: &'a wgpu::Texture,
    /// The image the network sees: its pixel (y, x) is frame pixel
    /// (rows[y], cols[x]), so these can crop, pad or mirror the frame. Its
    /// size must be a multiple of `Model::multiple`, its CFA `Model::cfa`.
    pub rows: &'a [u32],
    pub cols: &'a [u32],
    /// White-balance gains (R, G, B) and the noise model sigma^2 = a * x + b,
    /// as `noise::inputs` takes them.
    pub gain: [f32; 3],
    pub a: f64,
    pub b: f64,
}

/// Workgroup edge of the cut and place shaders.
const WG: u32 = 16;

impl Model {
    /// Load an export directory (one precision: `.../xtrans_wgpu/f16`). An f16
    /// export needs a device created with `wgpu::Features::SHADER_F16`.
    pub fn load(dir: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Model, String> {
        let read = |f: &str| {
            std::fs::read(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()))
        };
        Model::load_with(&read, device, queue)
    }

    /// As `load`, with the export's files coming from `read(file name)`, e.g.
    /// out of the binary when the export is embedded.
    pub fn load_with(
        read: &dyn Fn(&str) -> Result<Vec<u8>, String>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Model, String> {
        let text = String::from_utf8(read("graph.txt")?).map_err(|e| e.to_string())?;
        let num = |s: &str| s.parse::<u64>().map_err(|e| format!("graph.txt: {s}: {e}"));

        let (mut f16, mut window, mut halo, mut multiple) = (false, (0, 0), 0, 1);
        let mut cfa = vec![];
        let (mut mosaic, mut sigma, mut output) = (0, 0, 0);
        let mut bufs = vec![];
        let mut ops = vec![];
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            match f.first().copied() {
                None | Some("#") => {}
                Some("storage") => f16 = f[1] == "f16",
                Some("window") => window = (num(f[1])? as u32, num(f[2])? as u32),
                Some("halo") => halo = num(f[1])? as u32,
                Some("multiple") => multiple = num(f[1])? as u32,
                Some("cfa") => {
                    cfa = f[2..]
                        .iter()
                        .map(|s| num(s).map(|v| v as u32))
                        .collect::<Result<_, _>>()?
                }
                Some("mosaic") => mosaic = num(f[1])? as usize,
                Some("sigma") => sigma = num(f[1])? as usize,
                Some("output") => output = num(f[1])? as usize,
                Some("buffer") => {
                    let size = num(f[2])?;
                    if size > device.limits().max_storage_buffer_binding_size as u64 {
                        return Err(format!("buffer {} is {size} bytes, over this device's \
                                            max_storage_buffer_binding_size; export a smaller window", f[1]));
                    }
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    if let Some(file) = f.get(3) {
                        let mut data = read(file)?;
                        data.resize(size as usize, 0);
                        queue.write_buffer(&buf, 0, &data);
                    }
                    bufs.push(buf);
                }
                Some("op") => ops.push(f.iter().skip(1).map(|s| s.to_string()).collect::<Vec<_>>()),
                Some(k) => return Err(format!("graph.txt: unknown line {k}")),
            }
        }
        if f16 && !device.features().contains(wgpu::Features::SHADER_F16) {
            return Err(
                "this export is f16 and the device has no SHADER_F16; load the f32 export".into(),
            );
        }

        let entry = |i: u32| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i < 4 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<Valid>() as u64),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0), entry(1), entry(2), entry(3), entry(4), uniform],
        });
        let valid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("valid"),
            contents: bytemuck::bytes_of(&Valid {
                h: window.0,
                w: window.1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let mut compiled = vec![];
        for op in ops {
            let src = String::from_utf8(read(&op[0])?).map_err(|e| e.to_string())?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&op[0]),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&op[0]),
                layout: Some(&pipe_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let ids: Vec<usize> = op[1..6]
                .iter()
                .map(|s| num(s).map(|v| v as usize))
                .collect::<Result<_, _>>()?;
            let entries: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(i, &b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: bufs[b].as_entire_binding(),
                })
                .chain([wgpu::BindGroupEntry {
                    binding: 5,
                    resource: valid.as_entire_binding(),
                }])
                .collect();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            let groups = [
                num(&op[6])? as u32,
                num(&op[7])? as u32,
                num(&op[8])? as u32,
            ];
            compiled.push((pipeline, bind, groups));
        }

        let (cut, cut_raw, place) = window_pipelines(device, f16, window, halo, &cfa);
        Ok(Model {
            device: device.clone(),
            queue: queue.clone(),
            f16,
            window,
            halo,
            multiple,
            cfa,
            bufs,
            mosaic,
            sigma,
            output,
            ops: compiled,
            valid,
            cut,
            cut_raw,
            place,
            stage: OnceLock::new(),
        })
    }

    /// How many windows an h x w image is run as.
    pub fn windows(&self, h: u32, w: u32) -> usize {
        let halo = self.halo as usize;
        windows(h as usize, self.window.0 as usize, halo).len()
            * windows(w as usize, self.window.1 as usize, halo).len()
    }

    /// Height and width must be multiples of this (the CFA period / packing tile).
    pub fn multiple(&self) -> u32 {
        self.multiple
    }

    /// The network's reach: an output pixel depends on input pixels at most
    /// this far away. Pixels this far from the edge never see the wrap-around.
    pub fn halo(&self) -> u32 {
        self.halo
    }

    /// The CFA the mosaic must be phased to: period x period colours, row-major,
    /// 0 = R, 1 = G, 2 = B. Pixel (y, x) has colour cfa[(y % p) * p + x % p].
    pub fn cfa(&self) -> &[u32] {
        &self.cfa
    }

    /// Demosaic an h x w image. `mosaic` and `sigma` are h*w row-major: the
    /// white-balanced mosaic (CFA phase as at export) and its per-pixel noise
    /// std. Returns 3 x h x w planar RGB in white-balanced camera space.
    ///
    /// Windows are pipelined over the two readback buffers: while the GPU runs
    /// window i, the CPU cuts out window i+1 and copies window i-1 into place.
    pub fn run(&self, mosaic: &[f32], sigma: &[f32], h: u32, w: u32) -> Vec<f32> {
        assert!(
            h % self.multiple == 0 && w % self.multiple == 0,
            "{h}x{w} is not a multiple of {}",
            self.multiple
        );
        assert!(mosaic.len() == (h * w) as usize && sigma.len() == (h * w) as usize);
        let (th, tw) = (self.window.0 as usize, self.window.1 as usize);
        let (hu, wu, halo) = (h as usize, w as usize, self.halo as usize);
        let mut out = vec![0f32; 3 * hu * wu];
        let mut tm = vec![0f32; th * tw];
        let mut ts = vec![0f32; th * tw];
        let place =
            |out: &mut [f32], t: &[f32], (cy0, cy1, cx0, cx1): (usize, usize, usize, usize)| {
                // the core starts `halo` px into its window
                for c in 0..3 {
                    for y in cy0..cy1 {
                        let src = (c * th + y - cy0 + halo) * tw + halo;
                        let dst = (c * hu + y) * wu + cx0;
                        out[dst..dst + cx1 - cx0].copy_from_slice(&t[src..src + cx1 - cx0]);
                    }
                }
            };
        let mut pending = None;
        let mut i = 0;
        for (y0, cy0, cy1) in windows(hu, th, halo) {
            for (x0, cx0, cx1) in windows(wu, tw, halo) {
                // cut the window, wrapping around the image edge as often as it takes
                for y in 0..th {
                    let row = (y0 + y) % hu * wu;
                    let (mut x, mut j) = (x0, 0);
                    while j < tw {
                        let n = (tw - j).min(wu - x);
                        tm[y * tw + j..y * tw + j + n]
                            .copy_from_slice(&mosaic[row + x..row + x + n]);
                        ts[y * tw + j..y * tw + j + n]
                            .copy_from_slice(&sigma[row + x..row + x + n]);
                        j += n;
                        x = 0;
                    }
                }
                // queued: lands after the previous window's commands, before this one's
                self.queue
                    .write_buffer(&self.bufs[self.mosaic], 0, &self.to_storage(&tm));
                self.queue
                    .write_buffer(&self.bufs[self.sigma], 0, &self.to_storage(&ts));
                self.set_valid(cy1 - cy0, cx1 - cx0);
                let idx = self.submit(i % 2);
                if let Some((k, idx, core)) = pending.take() {
                    place(&mut out, &self.read(k, idx), core);
                }
                pending = Some((i % 2, idx, (cy0, cy1, cx0, cx1)));
                i += 1;
            }
        }
        if let Some((k, idx, core)) = pending {
            place(&mut out, &self.read(k, idx), core);
        }
        out
    }

    /// `run` without leaving the GPU. `mosaic` and `sigma` are h x w `R32Float`
    /// textures (TEXTURE_BINDING) holding what `run` takes; the RGB result is
    /// written to `output`, an `Rgba16Float` texture (STORAGE_BINDING), alpha 1.
    /// `output` receives the region of the result starting at `origin` (row,
    /// column) and is its size, so a padded input can be cropped on the way
    /// out. Only queues the work: anything submitted afterwards sees it.
    ///
    /// `cancelled` is asked before each window; once it says yes no more
    /// windows are submitted and this returns `Ok(false)` (`output` is then
    /// incomplete). To keep that prompt at most two windows are queued at a
    /// time, so the call waits for the GPU to catch up. That wait returns
    /// `Err` if it fails, typically because a submission was rejected (the
    /// device's error callback has the cause).
    pub fn run_texture(
        &self,
        mosaic: &wgpu::Texture,
        sigma: &wgpu::Texture,
        output: &wgpu::Texture,
        origin: (u32, u32),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool, wgpu::PollError> {
        let (h, w) = (mosaic.height(), mosaic.width());
        assert!(
            sigma.size() == mosaic.size(),
            "mosaic and sigma sizes differ"
        );
        assert!(
            mosaic.format() == wgpu::TextureFormat::R32Float
                && sigma.format() == wgpu::TextureFormat::R32Float
        );
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (mosaic, sigma) = (view(mosaic), view(sigma));
        self.run_windows(
            (h, w),
            ([0.0; 3], 0.0, 0.0),
            &self.cut,
            &[
                wgpu::BindingResource::TextureView(&mosaic),
                wgpu::BindingResource::TextureView(&sigma),
            ],
            output,
            origin,
            cancelled,
        )
    }

    /// `run_texture` from the raw frame: the balanced mosaic and sigma are made
    /// per window on the GPU, as `noise::inputs` makes them on the CPU, from
    /// the image `rows` x `cols` (see [`RawInput`]). `origin` and the output
    /// are in that image's coordinates.
    pub fn run_raw(
        &self,
        input: &RawInput,
        output: &wgpu::Texture,
        origin: (u32, u32),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool, wgpu::PollError> {
        let (h, w) = (input.rows.len() as u32, input.cols.len() as u32);
        assert!(input.frame.format() == wgpu::TextureFormat::R32Float);
        assert!(
            input.rows.iter().all(|&y| y < input.frame.height())
                && input.cols.iter().all(|&x| x < input.frame.width()),
            "rows or cols outside the frame"
        );
        let map: Vec<u32> = input.rows.iter().chain(input.cols).copied().collect();
        let map = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rows and cols"),
                contents: bytemuck::cast_slice(&map),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let frame = input.frame.create_view(&Default::default());
        self.run_windows(
            (h, w),
            (input.gain, input.a as f32, input.b as f32),
            &self.cut_raw,
            &[
                wgpu::BindingResource::TextureView(&frame),
                map.as_entire_binding(),
            ],
            output,
            origin,
            cancelled,
        )
    }

    /// The window loop of `run_texture` and `run_raw`: `cut` (bound to the
    /// window, `inputs`, and the graph's mosaic and sigma buffers) fills each
    /// window of the h x w image, the ops run, the core goes to `output`.
    #[allow(clippy::too_many_arguments)]
    fn run_windows(
        &self,
        (h, w): (u32, u32),
        (gain, a, b): ([f32; 3], f32, f32),
        cut_pipeline: &wgpu::ComputePipeline,
        inputs: &[wgpu::BindingResource],
        output: &wgpu::Texture,
        origin: (u32, u32),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool, wgpu::PollError> {
        assert!(
            h % self.multiple == 0 && w % self.multiple == 0,
            "{h}x{w} is not a multiple of {}",
            self.multiple
        );
        assert!(
            origin.0 + output.height() <= h && origin.1 + output.width() <= w,
            "output region exceeds the {h}x{w} image"
        );
        assert!(output.format() == wgpu::TextureFormat::Rgba16Float);
        let (th, tw, halo) = (
            self.window.0 as usize,
            self.window.1 as usize,
            self.halo as usize,
        );

        let mut wins = vec![];
        for (y0, cy0, cy1) in windows(h as usize, th, halo) {
            for (x0, cx0, cx1) in windows(w as usize, tw, halo) {
                let [y0, x0, cy0, cy1, cx0, cx1] = [y0, x0, cy0, cy1, cx0, cx1].map(|v| v as u32);
                wins.push(Window {
                    y0,
                    x0,
                    h,
                    w,
                    cy0,
                    cy1,
                    cx0,
                    cx1,
                    oy: origin.0,
                    ox: origin.1,
                    gain,
                    a,
                    b,
                });
            }
        }
        // one uniform buffer, each window at an aligned offset
        let size = std::mem::size_of::<Window>() as u64;
        let stride = size.next_multiple_of(self.device.limits().min_uniform_buffer_offset_alignment as u64);
        let mut params = vec![0u8; stride as usize * wins.len()];
        for (i, win) in wins.iter().enumerate() {
            params[i * stride as usize..][..size as usize].copy_from_slice(bytemuck::bytes_of(win));
        }
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("windows"),
                contents: &params,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let output = output.create_view(&Default::default());
        let bind = |pipeline: &wgpu::ComputePipeline, i: usize, rest: &[wgpu::BindingResource]| {
            let param = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform,
                offset: i as u64 * stride,
                size: NonZeroU64::new(size),
            });
            let entries: Vec<_> = std::iter::once(param)
                .chain(rest.iter().cloned())
                .enumerate()
                .map(|(b, resource)| wgpu::BindGroupEntry {
                    binding: b as u32,
                    resource,
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        // one submission per window keeps each one short (no watchdog resets
        // on big images); queue order sequences the windows' buffer reuse
        let mut previous = None;
        for (i, win) in wins.iter().enumerate() {
            if cancelled() {
                return Ok(false);
            }
            let cut = bind(
                cut_pipeline,
                i,
                &[
                    inputs[0].clone(),
                    inputs[1].clone(),
                    self.bufs[self.mosaic].as_entire_binding(),
                    self.bufs[self.sigma].as_entire_binding(),
                ],
            );
            let place = bind(
                &self.place,
                i,
                &[
                    self.bufs[self.output].as_entire_binding(),
                    wgpu::BindingResource::TextureView(&output),
                ],
            );
            let valid = self.set_valid((win.cy1 - win.cy0) as usize, (win.cx1 - win.cx0) as usize);
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(cut_pipeline);
                pass.set_bind_group(0, &cut, &[]);
                // only what the ops read
                pass.dispatch_workgroups(valid.w.div_ceil(WG), valid.h.div_ceil(WG), 1);
                for (pipeline, bind, g) in &self.ops {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
                pass.set_pipeline(&self.place);
                pass.set_bind_group(0, &place, &[]);
                pass.dispatch_workgroups(
                    (win.cx1 - win.cx0).div_ceil(WG),
                    (win.cy1 - win.cy0).div_ceil(WG),
                    1,
                );
            }
            let submitted = self.queue.submit([enc.finish()]);
            if let Some(previous) = previous.replace(submitted) {
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(previous),
                    timeout: None,
                })?;
            }
        }
        Ok(true)
    }

    /// Queues the `Valid` of a window whose core is `ch` x `cw` px, for the
    /// next submission's ops, and returns it.
    fn set_valid(&self, ch: usize, cw: usize) -> Valid {
        let span = |core: usize, t: u32| (core as u32 + 2 * self.halo).min(t);
        let valid = Valid {
            h: span(ch, self.window.0),
            w: span(cw, self.window.1),
        };
        self.queue
            .write_buffer(&self.valid, 0, bytemuck::bytes_of(&valid));
        valid
    }

    fn stage(&self) -> &[wgpu::Buffer; 2] {
        self.stage.get_or_init(|| {
            let bytes = (3 * (self.window.0 * self.window.1) as u64 * if self.f16 { 2 } else { 4 })
                .div_ceil(4)
                * 4;
            [(); 2].map(|_| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
        })
    }

    /// Every op, then the output into readback buffer `k`; does not wait.
    fn submit(&self, k: usize) -> wgpu::SubmissionIndex {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (pipeline, bind, g) in &self.ops {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(g[0], g[1], g[2]);
            }
        }
        let st = &self.stage()[k];
        enc.copy_buffer_to_buffer(&self.bufs[self.output], 0, st, 0, st.size());
        let idx = self.queue.submit([enc.finish()]);
        st.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        idx
    }

    /// Wait for submission `idx` and return readback buffer `k` as f32.
    fn read(&self, k: usize, idx: wgpu::SubmissionIndex) -> Vec<f32> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();
        let n = 3 * (self.window.0 * self.window.1) as usize;
        let st = &self.stage()[k];
        let out = {
            let v = st.slice(..).get_mapped_range();
            if self.f16 {
                let mut o = vec![0f32; n];
                half::slice::HalfFloatSliceExt::convert_to_f32_slice(
                    &bytemuck::cast_slice::<u8, half::f16>(&v)[..n],
                    &mut o,
                );
                o
            } else {
                bytemuck::cast_slice::<u8, f32>(&v)[..n].to_vec()
            }
        };
        st.unmap();
        out
    }

    fn to_storage(&self, data: &[f32]) -> Vec<u8> {
        if self.f16 {
            let mut v = vec![half::f16::ZERO; data.len()];
            half::slice::HalfFloatSliceExt::convert_from_f32_slice(&mut v[..], data);
            bytemuck::cast_slice(&v).to_vec()
        } else {
            bytemuck::cast_slice(data).to_vec()
        }
    }
}

/// The cut shader (image textures -> the graph's mosaic and sigma buffers, one
/// window, wrapping around the image edge) and the place shader (the window's
/// core of the graph's output buffer -> the output texture).
fn window_pipelines(
    device: &wgpu::Device,
    f16: bool,
    (th, tw): (u32, u32),
    halo: u32,
    cfa: &[u32],
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let header = format!("{}alias T = {};

struct Window {{ y0: u32, x0: u32, h: u32, w: u32, cy0: u32, cy1: u32, cx0: u32, cx1: u32, oy: u32, ox: u32,
                 g0: f32, g1: f32, g2: f32, a: f32, b: f32 }}
@group(0) @binding(0) var<uniform> win: Window;

const TH: u32 = {th}u;  // window
const TW: u32 = {tw}u;
const HALO: u32 = {halo}u;
", if f16 { "enable f16;\n" } else { "" }, if f16 { "f16" } else { "f32" });
    let cut = format!(
        "{header}
@group(0) @binding(1) var mos_in: texture_2d<f32>;
@group(0) @binding(2) var sig_in: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> mos: array<T>;
@group(0) @binding(4) var<storage, read_write> sig: array<T>;

@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    if (g.x >= TW || g.y >= TH) {{ return; }}
    let p = vec2<u32>((win.x0 + g.x) % win.w, (win.y0 + g.y) % win.h);
    let i = g.y * TW + g.x;
    mos[i] = T(textureLoad(mos_in, p, 0).r);
    sig[i] = T(textureLoad(sig_in, p, 0).r);
}}
"
    );
    // noise::inputs, per window: the image pixel's frame pixel, balanced by
    // the gain of its CFA colour, and its sigma
    let period = (cfa.len() as f64).sqrt() as usize;
    let colors = cfa.iter().map(|c| format!("{c}u")).collect::<Vec<_>>().join(", ");
    let cut_raw = format!(
        "{header}
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> map: array<u32>;  // rows (win.h), then cols (win.w)
@group(0) @binding(3) var<storage, read_write> mos: array<T>;
@group(0) @binding(4) var<storage, read_write> sig: array<T>;

const P: u32 = {period}u;
const CFA = array<u32, {n}>({colors});

@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    if (g.x >= TW || g.y >= TH) {{ return; }}
    let y = (win.y0 + g.y) % win.h;
    let x = (win.x0 + g.x) % win.w;
    let v = textureLoad(frame, vec2<u32>(map[win.h + x], map[y]), 0).r;
    let gains = vec3<f32>(win.g0, win.g1, win.g2);
    let gain = gains[CFA[(y % P) * P + x % P]];
    let i = g.y * TW + g.x;
    mos[i] = T(v * gain);
    sig[i] = T(sqrt(win.a * max(v, 0.0) + win.b) * gain);
}}
",
        n = cfa.len()
    );
    let place = format!(
        "{header}
@group(0) @binding(1) var<storage, read> rgb: array<T>;  // 3 x TH x TW
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    if (g.x >= win.cx1 - win.cx0 || g.y >= win.cy1 - win.cy0) {{ return; }}
    // the core starts HALO px into its window
    let i = (g.y + HALO) * TW + g.x + HALO;
    let plane = TH * TW;
    let c = vec3<f32>(f32(rgb[i]), f32(rgb[plane + i]), f32(rgb[2u * plane + i]));
    // image -> output coordinates; whatever falls outside the output is cropped
    let o = vec2<i32>(i32(win.cx0 + g.x) - i32(win.ox), i32(win.cy0 + g.y) - i32(win.oy));
    if (any(o < vec2<i32>(0)) || any(o >= vec2<i32>(textureDimensions(dst)))) {{ return; }}
    textureStore(dst, o, vec4<f32>(c, 1.0));
}}
"
    );

    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(std::mem::size_of::<Window>() as u64),
    };
    // R32Float is not filterable, which an auto-derived layout would assume
    let texture = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let storage = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let output = wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format: wgpu::TextureFormat::Rgba16Float,
        view_dimension: wgpu::TextureViewDimension::D2,
    };
    let pipeline = |label: &str, src: String, types: &[wgpu::BindingType]| {
        let entries: Vec<_> = types
            .iter()
            .enumerate()
            .map(|(i, &ty)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    (
        pipeline(
            "cut",
            cut,
            &[uniform, texture, texture, storage(false), storage(false)],
        ),
        pipeline(
            "cut raw",
            cut_raw,
            &[uniform, texture, storage(true), storage(false), storage(false)],
        ),
        pipeline("place", place, &[uniform, storage(true), output]),
    )
}

/// Windows of `t` px along an axis of `n` px: (window start, core start, core
/// end). Cores tile the axis; each window starts `halo` px before its core,
/// modulo n, so it may wrap around the image edge -- also when n < t.
fn windows(n: usize, t: usize, halo: usize) -> Vec<(usize, usize, usize)> {
    let core = t - 2 * halo;
    (0..n)
        .step_by(core)
        .map(|c| ((c + n - halo % n) % n, c, (c + core).min(n)))
        .collect()
}
