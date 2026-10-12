# AGENTS.md

Starfish: an iced + wgpu raw photo editor. Raw files are decoded with rawler,
demosaiced and denoised on the GPU by neural networks trained here, and shown
in a wgpu preview.

## Layout

```
src/
  main.rs, styling.rs, preview.rs   the iced app and the image preview
  develop.rs                        shaders/develop.wgsl: sampling the image and the per-pixel develop
                                    (colour matrix, local contrast, adjustments, AgX) that preview.wgsl,
                                    and so the export and the histogram, calls
  source.rs                         the image on iced's device: a texture array of overlapping layers
                                    with mipmaps, uploaded once and shared by preview, export, local contrast
  export.rs                         JPEG / PNG export, drawn on iced's device by the preview's pipeline
  raw.rs                            rawler decode -> RawFrame (normalised, NOT clamped at 0)
  library.rs                        folder listing, thumbnails, settings (per-OS config dir)
  log.rs                            starfish.log in the per-OS data dir + panic hook (no deps)
resources/      fonts (+ licences), icon/ (placeholder five-blade iris logo:
                starfish.svg is the source, PNGs and .ico rendered from it), starfish.desktop
  lens.rs, lens/canon.rs            embedded lens profiles (Fuji, Panasonic, Sony, Canon): parse, radial lookup table
  processing/
    mod.rs      Engine: own wgpu device (full adapter limits, SHADER_F16 if present),
                Demosaic trait, readback into an Image
    neural.rs   NeuralDemosaic: phase-shifts/crops the frame onto the model's CFA,
                fits the noise, runs demosaic_run::Model
    binning.rs  fast binning demosaic, kept for a future preview path
    lens.rs     LensPass: distortion + CA + vignetting resample (shaders/lens.wgsl);
                the app asks it for CA only (distortion and vignetting are drawn)
build.rs        embeds training/*_wgpu/{f16,f32}/ into release builds
wgpu-run/       workspace crate demosaic-run: the model-agnostic wgpu executor
  src/lib.rs      Model::load(dir) / load_with(read), model.run(mosaic, sigma, h, w) on the
                  CPU, model.run_texture(mosaic, sigma, output) all on the GPU,
                  model.run_raw(RawInput, output) the same from the raw frame (the app's)
  src/noise.rs    estimate_noise (sigma^2 = a*x + b fit) and inputs() (balanced mosaic + sigma)
  src/main.rs     demosaic-run: test/benchmark tool (test.bin, --raw, --bench)
  src/raw.rs      raw decoding for demosaic-run only
training/       PyTorch joint demosaic + denoise for X-Trans and Bayer
  common.py        everything shared: data constants, CFA tables, network parts,
                   the synthetic sensor, loss, training loop, export, inference
  train_xtrans.py  X-Trans model: training constants + build() with every layer written out
  train_bayer.py   the same for Bayer
  infer_*.py       mosaic + noise a clean photo, demosaic it, print PSNR
  wgsl.py          compiles a trained model into WGSL shaders + graph.txt
  prep.py          data/DIV2K_train_HR/*.png -> data/div2k.npy + data/div2k_index.npy
  requirements.txt the Python packages (torch cu128, numpy, pillow), for training/.venv
  data/            training data, added by hand (README.md says what; the rest is gitignored)
  xtrans.pt, bayer.pt           the trained models
  xtrans_wgpu/, bayer_wgpu/     their exports, which the app loads
```

Design rules for training/: no abstraction beyond plain PyTorch; every model
and training choice is a constant at the top of a file, never a command-line
flag; checkpoints are plain `torch.save` dicts; no checkpoint validation
beyond what `load_state_dict` does. Keep it that way.

## How the app uses the models

- `Engine` picks `NeuralDemosaic` per sensor (Bayer / X-Trans), f16 when the
  device has `SHADER_F16`, else f32. On DX12 that needs DXC: Windows x86_64
  builds enable wgpu's `static-dxc` and select it (FXC, wgpu's default, has no
  f16; `WGPU_DX12_COMPILER` still overrides). The device asks for the
  adapter's full limits and falls back to defaults plus the texture/buffer
  sizes if refused. wgpu errors do not panic: `Gpu` records them
  (`on_uncaptured_error`, device lost), `Engine::checked` turns one into the
  run's `Err` (out of memory gets a readable message) and unloads the models,
  which may hold invalid buffers. `STARFISH_DEMOSAIC=binning` selects
  `BinningDemosaic` instead. Models load on first use; a mutex serialises
  runs, because a `Model` reuses one set of GPU buffers.
- Debug builds read `training/*_wgpu/` at runtime (re-export, no rebuild);
  release builds embed them via `build.rs` (~8 MB).
- The network wants the white-balanced mosaic in its own CFA phase, sizes a
  multiple of the period, and a sigma map, and pads circularly. `neural.rs`
  pads the frame by >= `Model::halo` per side (so no frame pixel sees the
  opposite edge) plus < 1 period to reach the phase and whole periods,
  mirroring across the edges in whole periods (same CFA colour at every
  padded site); the padding goes to `run_raw` as the frame row and column of
  every padded one, and its `origin` crops back to the frame size on the
  GPU. Interior pixels are bitwise what an unpadded run gives; in the outer
  30 px mirror padding beat the old wrap-around by 2 (X-Trans) to 13 dB
  (Bayer), and black padding by 3-13 dB. The noise fit runs on the
  unpadded, phased frame.
- Debugging: the adjustments panel ends with a "Debug" section whose "Quick
  demosaic" checkbox swaps the neural stage for binning as the main method
  (`Engine::develop(.., quick, ..)`, `Image::quick`). It goes through the same
  redevelop path as the noise slider; the quick stage has no noise estimate, so
  the slider shows "—" and the estimate is measured again when switching back.
  `STARFISH_DEMOSAIC=binning` still forces binning for the whole run, and
  `STARFISH_TEST_QUICK=1` does it in the `develop_file` test.
- `raw.rs` must not clamp below black: the training noise is unclamped below 0
  and `estimate_noise` needs the sub-black samples.
- X-Trans is symmetric under a (3, 3) shift, so two phases match the CFA; they
  are not bit-identical (the 6x6 tiling falls differently), differing most in
  clipped highlights. `phase()` takes the first match. A shift of 3 along
  one axis only swaps R and B and is never accepted.
- `Model::run_texture`: balanced mosaic and sigma go up once as `R32Float`
  textures; per window a cut shader fills the graph's input buffers (wrapping
  at the edges), the ops run, and a place shader writes the core into the
  `Rgba16Float` output, with nothing read back. `Model::run` (CPU windowing)
  stays as the reference; both are bitwise identical (`demosaic-run --gpu` vs
  without, `--dump`). With an f32 export the f16 output texture caps the
  test.bin PSNR at ~76 dB instead of ~83.
- The app uses `Model::run_raw`, which is `run_texture` with the inputs made
  on the GPU: the raw frame goes up as decoded (one unpadded `R32Float`
  texture), with the row and column maps of the padding, the white-balance
  gains and the noise fit, and its cut shader does what `noise::inputs` does
  on the CPU. That removed the CPU padding and `noise::inputs` (49 ms X100S,
  ~180 ms Z7) and most of the upload (29 -> 5 ms, 54 -> 25), also on every
  noise-slider re-develop. Not bitwise: sigma is now f32 `sqrt(a*x + b)`
  instead of f64. Against the old path (`develop_file` dumps): f16 output
  identical on the X100S and an RW2, the Z7 differs in 0.04% of values (max
  3e-3, 105 dB); f32 115-118 dB, at most one f16 step of the output.
- The noise fit is on the CPU, its block loops split over the cores by rows
  of blocks (`noise::par_map`, results in order, so bitwise the same fit):
  X100S 75 -> 31 ms, Z7 32 -> 28 (Bayer's block loop was cheap; what is left
  is mostly the quantile sorts). It runs once per file. `Engine::develop`
  reads the result back for the preview (it renders on iced's device, not
  the processing one): 35 ms X100S, ~110 ms Z7.

## UI behaviour worth knowing

- The toolbar shows only name, camera and size; decode/develop times, the noise
  fit, sensor and demosaic are in the Metadata tab ("Processing"), and a pill
  over the preview says "Developing…" / "Updating…" while a stage runs.
- Files or folders dropped on the window open (`window::Event::FileDropped`) on
  Windows, macOS and X11. winit 0.30 has no drag and drop on Wayland, so the
  empty library's hint mentions dropping only where `drop_supported()` (no
  `WAYLAND_DISPLAY` on Linux).
- Window frame: the native title bar on every platform, with nothing added to
  it (an earlier custom libadwaita-style frame was removed). Window icon:
  `resources/icon/starfish-256.png`; the .exe icon (`starfish.ico`) is embedded by
  `build.rs` via winresource; Linux sets app id `starfish` to match
  `resources/starfish.desktop` (Wayland ignores window icons).
- Settings (last folder/photo) live in `%APPDATA%\Starfish`,
  `~/Library/Application Support/Starfish` or `$XDG_CONFIG_HOME/starfish`; the
  log in `%LOCALAPPDATA%\Starfish`, `~/Library/Logs/Starfish` or
  `$XDG_STATE_HOME/starfish` (`library::config_dir`/`data_dir`). Errors shown
  over the preview go through `App::fail`, which logs them. The Metadata tab's
  "Processing" group shows the GPU line (`Gpu::summary`: adapter, backend,
  f16/f32) and the log path, for bug reports.
- Adjustment values are buttons: click to type a number (Enter applies, Esc
  cancels; `Control::typed`, Kelvin for temperature), and Ctrl+scroll over an
  adjustment nudges it (`Control::nudged`; the handler is attached only while
  Ctrl is held, so the panel still scrolls otherwise).
- Re-demosaics (noise slider, quick demosaic; CA would, but is not offered) run one at a
  time, each with its own cancel flag (`App::redeveloping`): a change that
  makes the run in flight out of date cancels it (checked per window), and
  its `CANCELLED` result starts a run with the latest settings. Opening a file
  cancels it as well.
- Adjustment sections collapse (`Section`, `App::collapsed`; not persisted).
  The panel's "reset all" is the icon in the header next to the tabs.
- The preview (`preview.wgsl`) draws one quad over the whole widget and maps
  each pixel back to the image (`to_image`: tilt, then quarter turns). The
  image is a `source::Source`: a texture array whose layers overlap by 32 px
  (edge pixels repeated beyond the image), each with mipmaps, so no sample
  sees a seam. Where an image pixel covers at least one target pixel the
  shader snaps to the centre of the output pixel under it and samples
  bicubically (Catmull-Rom, `sample_source_cubic`), so zoomed in the preview
  shows the export's pixels as flat squares (checked: 4x views have 3/4 equal
  neighbours); zoomed out it samples the matching mip level trilinearly. The
  last upload is shared (`Source::get`, per image and device), so the export
  reuses the preview's. A 24 MP upload is ~260 MB, which an integrated GPU
  may not have twice: on a Vega 7 a file switch failed to create it (invalid
  `source` texture, reported as a histogram validation error). So the old
  image is let go of before the new one is uploaded (`PreviewPipeline::release`
  first, `LAST` cleared), the histogram's pipeline releases it after every
  run, and the upload runs in error scopes: out of memory is a readable `Err`
  (`prepare_image` returns it; the preview logs it once per image, the
  histogram's run brings it to `App::fail`, the export returns it).
- Every develop logs its timings to the log: model load, noise fit, the
  network's run (padded size, windows, wait for the model, upload) and the
  total with readback. The local contrast overview is drawn from the mips
  (`fs_overview`). Inside the frame but off the image is black (the export's
  look; it was the pane's grey outside crop mode).
- Histogram (`histogram.rs`): computed on the GPU from what the export would
  hold, crop, lens corrections and local contrast included. The preview's
  own pipeline draws every `stride`-th output pixel (about 2^17 of them) into
  a small `Rgba32Float` target, with the view's `point` flag so each target
  pixel is the export's pixel at `i * stride + stride / 2` (snapped, bicubic)
  whatever the scale; `histogram.wgsl` bins it with atomics (R, G, B, and the
  luma of the linear values, each by the 8-bit value it rounds to) and 4 KB
  are read back. It runs on iced's device off the UI thread (`Histogram::compute`
  waits 50 ms and returns `None` until the preview has drawn once); the app
  keeps `histogram_for` (image id + `export::View`) and starts a run after any
  update that changes it, one at a time. ~5 ms per run on the 5060 Ti. Checked
  against a CPU count of the exported pixels at the same spots: equal to
  within 6 counts in ~450k (bin-edge rounding). With even strides,
  `(i + 0.5) * stride` fell on pixel edges and float rounding picked either
  pixel; hence the centred offset. The old CPU path (`Samples`,
  `agx::display`) is gone; `agx.rs` keeps the curve maths and the encoding.
- Export: format picker (JPEG q95, PNG 8/16-bit) and a download button in the
  toolbar. `export.rs` draws the frame with the preview's own pipeline
  (`PreviewPipeline::with_format`, `prepare_image`, `draw_image`) on iced's
  device (`preview::gpu`, saved when the preview first builds its pipeline)
  at scale 1, into 2048 px `Rgba32Float` tiles that are read back and encoded
  on the CPU. It builds its own local contrast base. GPU errors are caught
  with error scopes (per device, not per thread). The output is
  `round(frame)` px at exactly one image pixel per pixel. Unrotated, untilted
  exports are bitwise what bilinear sampling gave (pixel centres); turned and
  tilted ones are sharper (bicubic: ~30% more high-frequency energy, same
  geometry to 0.001 px, 51-57 dB after a 1.5-3 px blur). Checked through the
  `develop_file` test (`STARFISH_TEST_EXPORT=out.jpg|out.png`,
  `STARFISH_TEST_CROP=1` for a turn, tilt and 3:2 crop, run on the engine's
  device); through the app's UI only before the texture array.
- Crop mode (crop icon in the toolbar, `App::cropping`): the frame
  (`preview::frame_rect`; the SLR viewfinder with a split-image disc in a
  microprism collar is drawn per pixel in `preview.wgsl`'s `viewfinder`, whose
  constants set the look, and the picture outside the frame is blurred there) stays put and the image moves
  under it: drag inside the frame = move, vertical drag left or right of it
  = zoom, horizontal drag above or below it = tilt (`preview::Drag`), wheel =
  zoom, right drag or the "Straighten" slider = tilt. `preview::Crop` (offset, zoom, angle) is the state; *frame space*
  (frame top left, image-pixel units) is what the view, `preview.wgsl`
  (`to_image`; outside the frame is darkened in crop mode, discarded
  otherwise) and `export.rs` work in. Outside crop mode the preview shows
  only the frame's content, and the export is exactly that at one image pixel
  per output pixel (black beyond the image), so the shader's `to_image` must
  match `Crop::to_image` and `Rotation::unapply`. Not run in the real
  app yet; the unit tests check the maths and that the shader validates.
- Left-pane thumbnails always show the filename under the picture.

## Lens correction (Fuji, Panasonic, Sony, Canon)

Fuji RAFs embed the correction for the lens, focal length and aperture of the
shot. rawler names the tags but does not decode them, so `lens.rs` reads them:
the second TIFF structure (pointer at byte 100, little-endian, offsets from its
own header), sub-IFD 0xf000, `SRATIONAL` arrays laid out `[scale, knots.., values..]`:

- 0xf00b distortion, percent, negative = barrel; 0xf010 vignetting, brightness
  percent (gain = 100/v); 0xf00f CA, `[scale, knots, red.., blue..(, scale)]`,
  radial scale offsets against green.
- Panasonic RW2 (DC-S5M2 and DC-S1M2 tested): IFD0 tag 0x119, 32 bytes = sixteen LE `i16`
  words, `p = word/32768`. Only distortion is understood: word 12 = half-diagonal
  px (3606 at 6000x4000), word 5 = zoom (`1 + p5`), words 8/4/11 = coefficients
  of `r_out = s(1 + p8 s^2 + p4 s^4 + p11 s^6)` in the distorted radius `s`,
  source sampled at `zoom * s`. Found by matching words to a free polynomial fit
  against the embedded JPEG: that reading reaches the free-fit NCC (0.83-0.97)
  and every other word assignment was >= 0.028 worse. Words 2,3,6,7,9,10,13 are
  unexplained (13 = 9 + 500); vignetting/CA from Panasonic are not decoded. Five
  shots: DC-S5M2 (85mm f/1.8, a zoom at 25-98 mm) to derive it, then a DC-S1M2
  with the S 24-60mm F2.8 at 24mm as a check: unfitted NCC 0.94 outer vs 0.96 for
  a free fit (uncorrected 0.0). A 35mm f/4 shot with a full-resolution
  camera JPEG then matched at NCC 0.97 in every radius band (0.18 / -0.11 in the
  outer two uncorrected). The JPEGs also carry a vignetting correction that
  depends on lens, focal length and aperture (x1.0 at 35mm f/4 and 32mm f/4,
  x1.1 at 24mm, x1.24 at 85mm f/1.8, x1.45 at 25mm), but it is not in 0x119 (no
  word correlates, best |r| 0.87) and the other IFD0 tags are body constants or
  per-shot checksums, so Panasonic vignetting is not available from the file.
- Sony ARW (A6700, A7 III tested): the *raw SubIFD* (IFD0 tag 0x14a, not the
  encrypted SR2 block) has 0x7031/0x7034/0x7036 = whether the camera applied
  vignetting/CA/distortion to its JPEG (A6700: all on; A7 III: only distortion,
  and only with the 55mm) and tables 0x7032 vignetting, 0x7035 CA (red then blue),
  0x7037 distortion, `i16` `[n, n values.., padding]`, n = 11 (A6700) or 16
  (A7 III) at radii `i/(n-1)` of the frame half-diagonal. The tables are always
  embedded; the flags say what the in-camera JPEG did, so JPEGs of pairs whose
  flag is off cannot validate that table. Distortion is `value/16384` as a radius
  ratio (negative = barrel). Verified on the 6 A6700 and 3 A7 III 55mm pairs
  (NCC 0.90-0.98 per band, uncorrected -0.1 to 0.8), but the A6700 amplitude is
  right only to ~4-8% (best-fit units 15000-17700; free-form fit of the 15mm
  barrel is 4% stronger than the table beyond rho 0.8: NCC 0.96 free vs 0.76 at
  1/16384). **Vignetting** (gain `2^(value/8192)`) is applied and on by
  default: no measurement could confirm it (see below) but corrected renders come
  out flat by eye, which was accepted. **CA** (red/blue radius scale offset
  `value/2^20`, `SONY_CA_UNIT`) is a guessed unit, marked by
  `LensCorrection::unverified()` so `lens_default` in `main.rs` starts it off
  (moot while the app offers no CA at all, see below). Once confirmed, change the unit
  and clear `unverified` in `from_sony_tables`. Why neither could be measured: in
  the A6700 JPEGs (flags on) the
  corners are not brighter than the raw (measured gain -0.2..0.05 EV, even at 15mm
  f/3.5 where the table is largest) and the per-channel CA fit gave no consistent
  unit (-10..+12). The vignetting unit was picked from plausible stops (4096-8192
  per EV; 8192) and accepted after eyeballing corrected renders; the CA unit is
  ~1e-6 of the radius per word, unconfirmed. A raw+JPEG pair of one static scene
  shot with the in-camera corrections on and off would settle CA, and confirm
  vignetting by measurement.
- Canon CR2 / CR3 (5D IV, M50, R50, R8 tested; code in `src/lens/canon.rs`):
  the data is Canon maker-note tag 0x4015 (ExifTool's `VignettingCorr`). A CR2
  has it in the normal maker note (note: IFD0 of a CR2 can sit at the very end
  of the file, so IFDs are read in place, not from a prefix). **A CR3's CMT3
  box does not have it**: it is in the CTMD (timed metadata) track, first
  sample, record type 9, whose payload is `[u32 size, u32 tag]` + a TIFF with a
  maker note (type 8 is a 106 KB note with other tags, type 7 the Exif).
  0x4016 (u32 words, from CMT3 or the CR2 note) says what the camera applied to
  its JPEG: word 5 peripheral lighting, 6 CA, 7 distortion (1 = on); it fits
  every JPEG inspected (M50 / RF85 / RF-S 55-210: distortion 0, and the JPEGs
  show none; 5D IV: all 0, JPEGs show no vignetting correction either).
  Layout and units are in the `canon.rs` docs. Knots are radii in 6 um units,
  so a pixel is `pitch_nm / 6000` of a unit (pitch is in the header: R8 6000,
  R50/M50 3720, 5D IV 5360). Vignetting gain is `V0 / V`. Distortion is the
  third 16384-based table, source/output radius ratio at the output radius. Canon's
  JPEGs apply it with **no zoom** (which crops in 2-4% more than needed at the
  short edge); the profile zooms by the shared rule (`1 / max ratio` over the
  frame edge) instead, so its framing is wider than the camera's, but the
  geometry is the same. Verified without fitting (as the camera's unzoomed JPEG): RF 24-50 at 35 and
  50mm (profile rms 0.1-0.2%) and RF-S 18-45 at 18mm (0.26%); vignetting within ~5% of the M50,
  RF-S 55-210 and R8 35mm JPEGs (the R8 50mm f/6.3 pair differ from the table
  by 10-30%, with two JPEGs of the same lens/setting differing from each other:
  the JPEG's own tone processing varies with the scene). "Verified" in the
  profile means the camera applied it (flag), so the verification could be
  made; a table the camera did not apply (EF-M distortion, RF 85mm macro,
  RF-S 55-210, every CR2 vignetting) is read the same way but starts off.
  **No CA:** two more 16384-based tables sit before the distortion one; they
  stay within about +/-3% and cannot be a radial scale (that would be ~25 px
  of fringe); colour shading is the guess, and measuring it against the JPEGs
  was swamped by white balance. They are not read. CR2 has no distortion table.
- Adding a maker: write a `LensCorrection::<maker>` that builds the profile
  (`Distortion::Percent` or `Distortion::Polynomial`, vignetting and CA curves) and add it
  to `LensCorrection::read`; the table, GPU pass and checkboxes need no change.
- Fuji: radius is normalised to the half-diagonal, `scale × knot count` pixels (11
  knots 0..1 on older bodies, 9 knots at sqrt(i/8) on newer). Anchored at 0.
- Fuji and Sony sample radius is `z · r · (1 + D(r))`, `D` indexed by the output
  radius, and `z = 1/(1 + max D)` over the frame edge radii from the short-edge
  midpoint (0.5547 for 3:2) to the corner (1): the largest zoom that keeps every
  output pixel inside the source, as the cameras' own JPEGs do. For barrel
  (D < 0) that is the short-edge midpoint, for pincushion the corner. (An
  earlier short-edge-only rule left blank corners on a pincushion lens: XF200
  NCC 0.22 vs 0.93 with this rule, whose z matched the best free zoom.)
- None of this is documented. The Fuji reading was fitted against the camera's corrected
  JPEG preview inside the RAF: distortion NCC 0.18 -> 0.82 (XF16mm) and
  0.41 -> 0.97 (XF18-135mm), vignetting residual <= ~5%. The table is applied
  at every aperture (the camera does too). The CA sign and size rest on weak
  evidence: each channel's own table with `+` improved or kept the match.
- `Engine::develop(.., lens, ..)` takes a `lens::Corrections` (distortion,
  vignetting, CA) and runs `LensPass` on the demosaic output with the ones the
  profile has. The app passes CA only (`Corrections::CHROMATIC`, `App::wanted_lens`):
  CA is to move into the network. Distortion and vignetting are applied where
  the image is drawn (`App::drawn_lens` -> `Preview::lens`, `export::View::lens`):
  the preview uploads the profile's table (`LensCorrection::lut`, `Image::profile`)
  as a 257x1 texture, and `develop.wgsl`'s `lens_source` maps each point of the
  corrected image to its source point and vignetting gain, so the image is
  resampled once (bicubic) for crop, tilt and distortion together. The local
  contrast overview is drawn through the same mapping (`fs_overview`), and the
  histogram is drawn by the same shader. So a
  distortion or vignetting toggle is instant. Against the old all-in-`LensPass` export: vignetting and CA
  (X100S) 80+ dB, max 0.0016 (f16 rounding); distortion (DC-S5M2) 80 dB with
  local contrast off, 57 dB with it, the local contrast base being built from
  the uncorrected image's mips sampled through the distortion instead of
  block-aligned mips of the corrected image (max 0.1 at a few hard saturated
  edges, no visible change). The snap-and-bicubic test in `preview.wgsl` is on
  the corrected image's footprint, the mip level on the source's: with the
  source's, regions the distortion stretches fell back to bilinear in the export.
  The "Lens corrections" section of the adjustments panel has a checkbox per
  available kind that the app offers. Tables that are all zero count as
  absent, so a prime with no distortion gets only a vignetting checkbox.
- **CA is off for every camera, with no checkbox** (`LENS_OFFERED` in
  `main.rs` leaves it out; it is and'ed into every correction the app asks
  for, so `wanted_lens` is always empty and lens toggles never re-demosaic).
  It is to come back inside the demosaic network. Until then the code stays
  as the reference: the makers' CA tables are still read (`chromatic` in
  `LensCorrection`, `lut` columns 1-2), `LensPass` still applies them when
  asked, and the `develop_file` dump (`STARFISH_TEST_LENS=1`) still includes
  CA. To bring the checkbox back, set `chromatic: true` in `LENS_OFFERED`.
- To check a change: `STARFISH_TEST_LENS=0|1 STARFISH_TEST_DUMP=out.f32` on the
  `develop_file` test dumps planar f32 camera RGB, with all three corrections
  from `LensPass`, to compare with the JPEG; its `STARFISH_TEST_EXPORT` is
  developed as the app does it (CA in `LensPass`, the rest drawn). Compare
  16-bit PNGs with something that reads 16 bits: Pillow reads 16-bit RGB as 8.

## Publishing

Code is `MIT OR Apache-2.0` (LICENSE-MIT, LICENSE-APACHE, README). rawler is
LGPL-2.1 (statically linked; binaries ship its notice); `deny.toml` allows
only permissive licences plus that exception (`cargo deny check licenses`).
The DIV2K-trained weights are interim, to be retrained on contributed raws.
`.github/workflows/ci.yml` tests (unit tests only: no GPU) on
Linux (Ubuntu 22.04 for an old glibc: a build on this Arch box needs glibc
2.44), Windows and macOS (arm64 only; no Intel build) on every push and PR; only a manual
run (Actions -> CI -> Run workflow) also packages and uploads the installers,
and a manual run on a `v*` tag also makes a draft release. Packaging is cargo-packager, configured in
`Cargo.toml` (`[package.metadata.packager]`): .deb + AppImage, NSIS, .app +
.dmg, into `dist/`. Its beforePackagingCommand runs cargo-about (`about.toml`,
`about.hbs`) to write THIRD_PARTY_LICENSES.html (gitignored), which ships in
every package with README and the licences. macOS packages are signed with a
Developer ID and notarized: `macos.signingIdentity` makes cargo-packager sign
the .app and .dmg and notarize the .app, from the repo secrets
`APPLE_CERTIFICATE` (base64 .p12), `APPLE_CERTIFICATE_PASSWORD`,
`APPLE_API_KEY` (key ID), `APPLE_API_ISSUER` and `APPLE_API_KEY_P8` (the .p8
text, written to a file for `APPLE_API_KEY_PATH`). cargo-packager only warns
when notarization credentials are missing, so CI notarizes and staples the
.dmg itself and fails unless `stapler validate` and `spctl` accept both.
Windows and Linux packages are unsigned. The bundle
identifier is `io.github.monkeypangit.starfish`; keep it stable. The release binary is
~53 MB (42 MB stripped, but stripping loses names in the log's backtraces).
Windows builds link the C runtime statically (`.cargo/config.toml`,
`+crt-static`): the first test on a clean Windows machine failed with
VCRUNTIME140.dll missing. CI's runner has Visual Studio and would not notice,
so it checks the .exe's imports with dumpbin.

## Commands

```
cargo build --release
STARFISH_TEST_RAW=file.raf STARFISH_TEST_OUT=out.ppm cargo test --release -- --ignored --nocapture
```

The DIV2K training images go in `training/data/DIV2K_train_HR/`, added by
hand (`training/data/README.md`); `prep.py` turns them into
`training/data/div2k.npy` + `div2k_index.npy`, the default `DATA` of training
and export (`--data` still overrides it). That folder is gitignored except its
README. The Python venv is `training/.venv` (gitignored; Python 3.14, the
packages in `training/requirements.txt`: torch 2.11 cu128, as the GPU is
Blackwell sm_120 and needs a CUDA 12.8+ build; everything assumes `cuda`).
Its install command is at the top of `requirements.txt`. Test raws and
`images/` live outside the repo, in `../demosaic/`, which also holds a
prepared `div2k.npy` + `div2k_index.npy` (`--data ../../demosaic/div2k.npy`
instead of running `prep.py`). From `training/`:

```
.venv/bin/python prep.py   # once, after adding data/DIV2K_train_HR/
.venv/bin/python train_xtrans.py   # resumes xtrans.pt
.venv/bin/python train_xtrans.py --export   # -> xtrans_wgpu/
.venv/bin/python infer_xtrans.py ../../demosaic/images/kodim21.png --a 1e-3
```

From the repo root (the workspace builds demosaic-run into `target/`):

```
cargo build --release -p demosaic-run
./target/release/demosaic-run training/xtrans_wgpu                 # vs test.bin
./target/release/demosaic-run training/xtrans_wgpu --raw ../demosaic/raws/DSCF9791.RAF --png out.png
./target/release/demosaic-run training/xtrans_wgpu --size 4000x6000 --bench 10
./target/release/demosaic-run training/xtrans_wgpu --gpu      # the same through run_texture
```

X-Trans models take Fuji `.RAF` files, Bayer models `.nef` / `.rw2`.
`test.bin` is gitignored; it is rewritten by every export.

## The model

`Demosaic` (common.py): `pixel_unshuffle(mosaic, TILE)` + pooled sigma ->
trunk at 1/TILE resolution -> `pixel_shuffle` to a full-res RGB estimate ->
full-res head on `[estimate, scattered mosaic, CFA masks, sigma]`, whose 3x3 is
a `PhaseConv` (one weight bank per CFA phase). Circular padding throughout.
TILE is one CFA period: 6 for X-Trans, 2 for Bayer. Input is the
white-balanced mosaic and its noise std; output is white-balanced camera RGB.

Module names (`net.N`, `full_head.0.conv`, `full_head.2`) match the earlier
codebase, which is how the trained checkpoints were converted.

## Checkpoints

`{'model': EMA weights, 'raw': trained weights, 'opt', 'scaler', 'step'}`.
Training resumes automatically from the file; inference and export read
`model`. Saved every 10000 steps, each save printing held-out loss and PSNR.

The current `xtrans.pt` / `bayer.pt` were converted from the old format and
have **no `opt`/`scaler`**: export and inference work, resuming training
crashes. The X-Trans run may have stopped before 500k steps (the old format
recorded no step count; `step` was set to 500000).

## Training data

DIV2K, crops cut online (`Crops`), the last 40 images held out. Each crop goes
through `random_batch` -> `sensor`: random camera matrix (8-camera bank blend +
hue/chroma jitter), R/B white-balance un-balance, exposure 1/8..2 with
clipping at SAT, mosaic, Poisson-Gaussian noise, a deliberately misestimated
sigma map, then re-balance. 20% of patches are noise-free; every 8th is
blurred; R/B are swapped with p=0.5. Loss: Charbonnier in tone-mapped linear
sRGB, 18 px border dropped. These were all verified to reproduce the earlier
code exactly at the same seed.

## The wgpu export

`export()` writes `<name>_wgpu/{f16,f32}/` (graph.txt, one .wgsl per dispatch,
weights.bin) and `test.bin`. Everything model-specific -- kernel choice
(1x1 / tiled k x k / plain conv / fused head), template constants, weight
repacking, buffer reuse, dispatch sizes -- is decided in `wgsl.py`; the Rust
side only allocates and dispatches what graph.txt lists. Shaders are built for
one window (`WGPU_WINDOW` = 1056 px); `Model::run` processes any image as
windows with a halo that wrap around the image edge, which is exact because the
network pads circularly. f16 needs the adapter's `SHADER_F16`.

Weights are rounded to fp16 for both builds. The trunk was verified bitwise
identical to the earlier hand-built Rust runner (whole image vs 6 windows, f16
and f32, both models); vs PyTorch fp32 on test.bin: X-Trans 62.0 dB (f16) /
83.2 dB (f32), Bayer 67.4 / 83.1. 24 MP (6000x4000), `--gpu --bench`, f16 / f32
on the 5060 Ti: X-Trans ~340 / 470 ms, Bayer ~375 / 510 ms.

Edge windows do only the work their core needs. A window whose core is cut
short by the image's end holds image data only in its first core + 2 * halo
rows and columns (`Valid`, a uniform at binding 5 that `Model` writes before
each window's submission), and the core reads nothing beyond them. Every
kernel scales that size to its own grid (`VALID`, `S` = window px per grid px)
and skips workgroups wholly outside it; the cut shader fills only that part.
Output is bitwise unchanged (all `--dump`s: test.bin and three raws, f16/f32,
`--gpu` or not), also with old shaders that ignore binding 5. Before, the
number of windows was `ceil(padded size / core)` per axis, so sizes just past a
multiple of the core (996 px X-Trans, 1012 Bayer) paid for nearly empty
windows: 15-30% of the work for most sensors after the app's padding. f16,
before -> after: X-Trans 6000x4000 458 -> 340 ms, 6048x4032 483 -> 352, 4896x3264
262 -> 222; Bayer 6240x4160 556 -> 410, 8256x5504 866 -> 722; sizes that tiled
well are unchanged (5976x3984: 316 either way). Throughput is now ~70 MP/s
(X-Trans) and ~63 MP/s (Bayer) whatever the size. The 1x1 kernels' 64-pixel
runs can straddle rows, so a run is skipped only when it starts beyond the
valid columns and ends on that row or on an invalid one.

The head (`HEAD`, `fold_head` in `wgsl.py`) is folded at export: within one CFA
phase the mask channels are constants (added to the bank's bias) and each tap's
scattered mosaic is nonzero in one known plane, so each bank is a 3x3 on
[estimate, m, sigma], 5 channels instead of 10. Each invocation keeps two
pixels' 32 hidden channels in vec4 registers. Until 2026-10 it was a scalar
kernel on all 10 channels that took ~30% of the run (~135 ms of 439 at
5976x3984, X-Trans and Bayer alike) for ~8% of the MACs; now ~10 ms. Its f16
output moved by ~73 dB PSNR on real raws (rounding) and got closer to PyTorch
(61.2 -> 62.0, 66.1 -> 67.4 dB). Making the bank the fastest dispatch index
(neighbouring workgroups sharing cache lines) did not help the old kernel.

The window was 1536 until 2026-10; 1056 (= 96 * 11, packed grid still a
multiple of the 16 px workgroups) gives bitwise identical output, keeps each
per-window submission ~2.2x shorter (Windows resets the GPU after ~2 s) and
halves the buffers (largest: Bayer f32 54 MB, was 113). On the 5060 Ti it was
faster anyway (X-Trans f16 681 -> 650 ms, Bayer f16 798 -> 500 ms, f32 874 ->
670; only X-Trans f32 slower, 815 -> 880), likely because the buffers now fit
the cache better. Keep the largest buffer under `max_storage_buffer_binding_size`
(128 MiB by default) if it is ever raised.

Any change to `wgsl.py` templates or packing: re-export and check
`demosaic-run` against test.bin (the PSNR vs PyTorch must not drop).
