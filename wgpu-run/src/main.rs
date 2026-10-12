//! Check and time an export from training/:
//!
//!   demosaic-run training/xtrans_wgpu                       # against test.bin
//!   demosaic-run training/xtrans_wgpu --raw raws/DSCF9791.RAF --png out.png
//!       [--crop X,Y,W,H] [--exposure E] [--auto-exposure] [--a A --b B]
//!   demosaic-run training/xtrans_wgpu --size 4000x6000 --bench 5
//!   options: --png FILE (sRGB image), --f32 (use the f32 export even when
//!            f16 is supported), --dump FILE (raw f32 output, 3 x h x w),
//!            --gpu (Model::run_texture instead of Model::run; --bench then
//!            times the GPU work alone, inputs uploaded once)

mod raw;

use std::path::Path;
use std::time::Instant;

const FLAGS: &[&str] = &["--f32", "--auto-exposure", "--gpu"];
const OPTS: &[&str] = &[
    "--raw",
    "--crop",
    "--exposure",
    "--a",
    "--b",
    "--png",
    "--size",
    "--bench",
    "--dump",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: demosaic-run EXPORT_DIR [--raw FILE [--crop X,Y,W,H] [--exposure E] \
                 [--auto-exposure] [--a A --b B]] [--size HxW] [--png FILE] [--bench N] [--f32] [--gpu] [--dump FILE]";
    let dir = Path::new(args.get(1).unwrap_or_else(|| exit(usage)));
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            a if FLAGS.contains(&a) => i += 1,
            a if OPTS.contains(&a) && i + 1 < args.len() => i += 2,
            a => exit(&format!("unknown or incomplete option {a}\n{usage}")),
        }
    }
    let flag = |f: &str| args.iter().any(|a| a == f);
    let opt = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .map(|i| args[i + 1].clone())
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no adapter");
    let f16 = adapter.features().contains(wgpu::Features::SHADER_F16) && !flag("--f32");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: if f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        },
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");
    let precision = if f16 { "f16" } else { "f32" };
    println!(
        "{} ({:?}), {precision}",
        adapter.get_info().name,
        adapter.get_info().backend
    );

    let t0 = Instant::now();
    let model = demosaic_run::Model::load(&dir.join(precision), &device, &queue)
        .unwrap_or_else(|e| exit(&e));
    println!("loaded in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    let q = model.multiple();

    // input: a raw file, a constant image of --size, or test.bin
    let mut test = None;
    let mut srgb_from_cam = vec![];
    let (h, w, mos, sig) = if let Some(path) = opt("--raw") {
        let (r, mos, sig) = load_raw(
            &path,
            model.cfa(),
            q as usize,
            &opt,
            flag("--auto-exposure"),
        );
        srgb_from_cam = r.srgb_from_cam.to_vec();
        (r.h as u32, r.w as u32, mos, sig)
    } else if let Some(s) = opt("--size") {
        let v: Vec<u32> = s.split('x').map(|x| x.parse().unwrap()).collect();
        let (h, w) = (v[0] / q * q, v[1] / q * q);
        (
            h,
            w,
            vec![0.3; (h * w) as usize],
            vec![0.01; (h * w) as usize],
        )
    } else {
        let t = read_test(&dir.join("test.bin"));
        srgb_from_cam = t.srgb_from_cam.clone();
        let r = (t.h, t.w, t.mosaic.clone(), t.sigma.clone());
        test = Some(t);
        r
    };

    let gpu = flag("--gpu");
    let t0 = Instant::now();
    let y = if gpu {
        let (m, s, o) = textures(&device, &queue, &mos, &sig, h, w);
        model.run_texture(&m, &s, &o, (0, 0), &|| false).unwrap();
        read_texture(&device, &queue, &o)
    } else {
        model.run(&mos, &sig, h, w)
    };
    println!(
        "{w}x{h}: {:.1} ms (first run, incl. transfers{})",
        t0.elapsed().as_secs_f64() * 1e3,
        if gpu { ", run_texture" } else { "" }
    );

    let n = (h * w) as usize;
    if let Some(t) = &test {
        let maxerr = y
            .iter()
            .zip(&t.reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let ys = to_srgb_lin(&y, &t.srgb_from_cam, n);
        let rs = to_srgb_lin(&t.reference, &t.srgb_from_cam, n);
        let (hu, wu) = (h as usize, w as usize);
        println!(
            "vs PyTorch fp32: max |err| {maxerr:.5} (camera space), PSNR {:.2} dB (linear sRGB)",
            psnr(&ys, &rs, hu, wu, 8)
        );
        println!(
            "vs truth: wgpu {:.2} dB, PyTorch fp32 {:.2} dB",
            psnr(&ys, &t.clean, hu, wu, 8),
            psnr(&rs, &t.clean, hu, wu, 8)
        );
    }
    if let Some(p) = opt("--png") {
        if srgb_from_cam.is_empty() {
            exit("--png needs --raw or test.bin input, which carry a colour matrix");
        }
        save_png(&to_srgb_lin(&y, &srgb_from_cam, n), h, w, &p);
        println!("wrote {p} ({w}x{h})");
    }
    if let Some(p) = opt("--dump") {
        std::fs::write(&p, bytemuck::cast_slice::<f32, u8>(&y)).unwrap();
    }
    if let Some(k) = opt("--bench").map(|s| s.parse::<usize>().unwrap()) {
        if gpu {
            let (m, s, o) = textures(&device, &queue, &mos, &sig, h, w);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let t0 = Instant::now();
            for _ in 0..k {
                model.run_texture(&m, &s, &o, (0, 0), &|| false).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / k as f64;
            println!(
                "bench {w}x{h} run_texture, GPU only: {ms:.1} ms ({:.1} MP/s)",
                n as f64 / 1e3 / ms
            );
        } else {
            let t0 = Instant::now();
            for _ in 0..k {
                model.run(&mos, &sig, h, w);
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / k as f64;
            println!(
                "bench {w}x{h} incl. transfers: {ms:.1} ms ({:.1} MP/s)",
                n as f64 / 1e3 / ms
            );
        }
    }
}

/// run_texture's inputs, uploaded, and its output texture.
fn textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mos: &[f32],
    sig: &[f32],
    h: u32,
    w: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture) {
    use wgpu::util::DeviceExt as _;
    let desc = |format, usage| wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    };
    let input = |data: &[f32]| {
        device.create_texture_with_data(
            queue,
            &desc(
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(data),
        )
    };
    let out = device.create_texture(&desc(
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
    ));
    (input(mos), input(sig), out)
}

/// An Rgba16Float texture as planar f32 RGB, 3 x h x w.
fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, t: &wgpu::Texture) -> Vec<f32> {
    let (w, h) = (t.width(), t.height());
    let row = (8 * w).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: row as u64 * h as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        t.size(),
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let n = (w * h) as usize;
    let mut out = vec![0f32; 3 * n];
    let v = buf.slice(..).get_mapped_range();
    for y in 0..h as usize {
        let texels: &[half::f16] = bytemuck::cast_slice(&v[y * row as usize..][..8 * w as usize]);
        for x in 0..w as usize {
            for c in 0..3 {
                out[c * n + y * w as usize + x] = texels[4 * x + c].to_f32();
            }
        }
    }
    out
}

fn exit(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1)
}

/// Decode a raw, fit (or take) the noise, and build the network's balanced
/// mosaic and sigma, cropped from the origin to a multiple of `q` (whole CFA
/// periods, so the phase is kept).
fn load_raw(
    path: &str,
    cfa: &[u32],
    q: usize,
    opt: &dyn Fn(&str) -> Option<String>,
    auto: bool,
) -> (raw::Raw, Vec<f32>, Vec<f32>) {
    let crop = opt("--crop").map(|s| {
        let v: Vec<usize> = s
            .split(',')
            .map(|x| x.parse().expect("--crop wants X,Y,W,H"))
            .collect();
        assert!(v.len() == 4, "--crop wants X,Y,W,H");
        [v[0], v[1], v[2], v[3]]
    });
    let exposure: f32 = opt("--exposure").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let p = (cfa.len() as f64).sqrt() as usize;
    let mut r = raw::load(path, cfa, p, crop, exposure, auto).unwrap_or_else(|e| exit(&e));
    println!("raw: {}", r.notes);

    let (a_opt, b_opt) = (
        opt("--a").map(|s| s.parse::<f64>().unwrap()),
        opt("--b").map(|s| s.parse::<f64>().unwrap()),
    );
    let (a, b) = match (a_opt, b_opt) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            let (ea, eb) = demosaic_run::noise::estimate_noise(&r.mosaic, r.h, r.w, cfa, p)
                .unwrap_or_else(|e| exit(&e));
            println!("estimated noise: a={ea:.2e} b={eb:.2e}");
            (a_opt.unwrap_or(ea), b_opt.unwrap_or(eb))
        }
    };
    let ((alo, ahi), (blo, bhi)) = (demosaic_run::noise::A_RANGE, demosaic_run::noise::B_RANGE);
    if !(alo..=ahi).contains(&a) {
        println!("  warning: a={a:.2e} is outside the trained range ({alo:e}, {ahi:e})");
    }
    if !(blo..=bhi).contains(&b) {
        println!("  warning: b={b:.2e} is outside the trained range ({blo:e}, {bhi:e})");
    }

    // whole periods from the origin, so the phase is kept
    let (h, w) = (r.h / q * q, r.w / q * q);
    let cropped: Vec<f32> = (0..h)
        .flat_map(|y| &r.mosaic[y * r.w..y * r.w + w])
        .copied()
        .collect();
    let (mos, sig) = demosaic_run::noise::inputs(&cropped, h, w, cfa, p, r.gain, a, b);
    r.h = h;
    r.w = w;
    r.mosaic = vec![];
    (r, mos, sig)
}

struct Test {
    h: u32,
    w: u32,
    srgb_from_cam: Vec<f32>,
    mosaic: Vec<f32>,
    sigma: Vec<f32>,
    reference: Vec<f32>,
    clean: Vec<f32>,
}

/// u32 h, u32 w, f32[9] srgb_from_cam, mosaic[h*w], sigma[h*w],
/// PyTorch output[3*h*w], clean linear sRGB[3*h*w]
fn read_test(path: &Path) -> Test {
    let b = std::fs::read(path).unwrap_or_else(|e| exit(&format!("{}: {e}", path.display())));
    let h = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let w = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let f: Vec<f32> = bytemuck::pod_collect_to_vec(&b[8..]);
    let n = (h * w) as usize;
    let mut it = f.into_iter();
    let mut take = |k: usize| it.by_ref().take(k).collect::<Vec<f32>>();
    Test {
        h,
        w,
        srgb_from_cam: take(9),
        mosaic: take(n),
        sigma: take(n),
        reference: take(3 * n),
        clean: take(3 * n),
    }
}

/// balanced camera RGB (planar) -> linear sRGB clamped to [0, 1]
fn to_srgb_lin(x: &[f32], m: &[f32], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; 3 * n];
    for i in 0..n {
        for r in 0..3 {
            o[r * n + i] =
                (m[3 * r] * x[i] + m[3 * r + 1] * x[n + i] + m[3 * r + 2] * x[2 * n + i])
                    .clamp(0.0, 1.0);
        }
    }
    o
}

/// PSNR over planar 3 x h x w images, `border` px dropped per side.
fn psnr(a: &[f32], b: &[f32], h: usize, w: usize, border: usize) -> f64 {
    let (mut s, mut n) = (0f64, 0usize);
    for c in 0..3 {
        for y in border..h - border {
            for x in border..w - border {
                let i = (c * h + y) * w + x;
                s += ((a[i] - b[i]) as f64).powi(2);
                n += 1;
            }
        }
    }
    -10.0 * (s / n as f64).log10()
}

/// planar linear sRGB -> 8-bit sRGB PNG
fn save_png(lin: &[f32], h: u32, w: u32, path: &str) {
    let n = (h * w) as usize;
    let enc = |v: f32| {
        let s = if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };
    let mut px = vec![0u8; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            px[3 * i + c] = enc(lin[c * n + i]);
        }
    }
    image::save_buffer(path, &px, w, h, image::ColorType::Rgb8).unwrap();
}
