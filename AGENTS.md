# AGENTS.md

Starfish: an iced + wgpu raw photo editor. Raw files are decoded with rawler,
demosaiced and denoised on the GPU by neural networks (trained in the `starfish-neural-demosaic` repo), and shown
in a wgpu preview.

## Layout

```
src/
  main.rs, styling.rs, preview.rs   the iced app and the image preview
  develop.rs                        shaders/develop.wgsl: sampling the image and the per-pixel develop
                                    (colour matrix, local contrast, adjustments, tone curve) that preview.wgsl,
                                    and so the export and the histogram, calls
  source.rs                         the image on iced's device: a texture array of overlapping layers
                                    with mipmaps, uploaded once and shared by preview, export, local contrast
  color_edit.rs, pill.rs            the colour mixer and grading (data, Oklab maths, GPU uniform) and the pill slider widget
  export.rs                         JPEG / PNG export, drawn on iced's device by the preview's pipeline
  raw.rs                            rawler decode -> RawFrame (normalised, NOT clamped at 0); the
                                    camera -> sRGB matrix from rawler's `color_matrix` (D65 first; see "Colour")
  library.rs                        folder listing, thumbnails, settings (per-OS config dir)
  log.rs                            starfish.log in the per-OS data dir + panic hook (no deps)
resources/      fonts (+ licences), icon/ (placeholder five-blade iris logo:
                starfish.svg is the source, PNGs and .ico rendered from it), starfish.desktop
  lens.rs, lens/canon.rs            embedded lens profiles (Fuji, Panasonic, Sony, Canon): parse, radial lookup table
  lens/measure.rs                   lateral CA measured in the developed image (every camera)
  processing/
    mod.rs      Engine: own wgpu device (full adapter limits, SHADER_F16 if present),
                Demosaic trait, readback into an Image
    neural.rs   NeuralDemosaic: phase-shifts/crops the frame onto the model's CFA,
                fits the noise, runs demosaic_run::Model
    binning.rs  fast binning demosaic, kept for a future preview path
    ca_fix.rs   CaFix: the lateral CA taken out of the raw mosaic before the network
                (shaders/ca_fix.wgsl, training's ca_correct)
    lens.rs     LensPass: distortion + CA + vignetting resample (shaders/lens.wgsl);
                the app asks it for nothing (every correction is drawn); the
                develop_file test's dump uses it
build.rs        embeds models/*_wgpu/{f16,f32}/ into release builds (the *_lite, *_tiny ones if exported)
wgpu-run/       workspace crate demosaic-run: the model-agnostic wgpu executor
  src/lib.rs      Model::load(dir) / load_with(read), model.run(mosaic, sigma, h, w) on the
                  CPU, model.run_texture(mosaic, sigma, output) all on the GPU,
                  model.run_raw(RawInput, output) the same from the raw frame (the app's)
  src/noise.rs    estimate_noise (sigma^2 = a*x + b fit) and inputs() (balanced mosaic + sigma)
  src/pad.rs      the CFA phase and the mirror padding onto the model
  src/main.rs     demosaic-run: test/benchmark tool (test.bin, --raw, --bench)
  src/raw.rs      raw decoding for demosaic-run only
models/         the exported networks the app embeds (from ../starfish-neural-demosaic; see "The models")
scripts/        sync-models.sh
```


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
- Debug builds read `models/*_wgpu/` at runtime (re-export, no rebuild);
  release builds embed them via `build.rs` (~8 MB).
- The network wants the white-balanced mosaic in its own CFA phase, sizes a
  multiple of the period, and a sigma map, and pads with zeros (circularly
  until 2026-10-18: see "The wgpu export"). `neural.rs`
  pads the frame by >= `Model::halo` per side (so no frame pixel sees the
  padding or the opposite edge) plus < 1 period to reach the phase and whole periods,
  mirroring across the edges in whole periods (same CFA colour at every
  padded site); the padding goes to `run_raw` as the frame row and column of
  every padded one, and its `origin` crops back to the frame size on the
  GPU. Interior pixels are bitwise what an unpadded run gives; in the outer
  30 px mirror padding beat the old wrap-around by 2 (X-Trans) to 13 dB
  (Bayer), and black padding by 3-13 dB. The noise fit runs on the
  unpadded, phased frame.
- Debugging: the right panel has a "Debug" tab (debug builds only) whose
  "Demosaic" picker lists the networks exported for the open file's sensor,
  then Quick (`processing::Method`: `Network(variant)` / `Quick`; before a
  file is open, every sensor's). It picks the stage
  (`Engine::develop(.., method, ..)`, `Image::method`): a network or binning.
  The networks are found, once per run, by `neural::variants`: the
  `models/<bayer|xtrans><variant>_wgpu/` folders with a graph in them in
  debug builds (so a new export or a `_ft` copy appears on the next start,
  with no code change), the exports `build.rs` embeds in release builds.
  The variant is the part between the sensor and `_wgpu` (`""` shown as
  "Neural network", `_lite` "Light network", `_tiny` "Tiny network", any
  other as its name). Until the picker is used,
  `App::method` is `None` and the develop workers ask `Engine::automatic`:
  `_lite` on an integrated or software (`DeviceType::Cpu`) adapter if the
  sensor has it, else the plain one; the picker then shows what the image was made with
  (`App::shown_method`). The view never calls `Engine::get` for this, so it
  does not create the device on the UI thread. A change goes through the
  redevelop path. The networks fit the same noise, so
  switching between them keeps the estimate; the quick stage has none, so the
  grain slider shows "—" and the estimate is measured again when switching back.
  Choosing one network unloads the others' buffers (`Engine::stage`). A
  variant the next file's sensor lacks fails the develop with the folder it
  looked for.
  `STARFISH_DEMOSAIC=binning` still forces binning for the whole run; in the
  `develop_file` test `STARFISH_TEST_QUICK=1` uses binning,
  `STARFISH_TEST_MOSAIC=1` the mosaic stage and
  `STARFISH_TEST_MODEL=_lite` (any variant) that export.
  The picker's "Mosaic (no demosaic)" (`Method::Mosaic`, `mosaic.rs`,
  `shaders/mosaic.wgsl`) shows the raw mosaic: each site's white-balanced
  sample in its own colour's channel, the others 0. With "Lateral CA in the
  mosaic" on it is the mosaic `CaFix` gives a network (CA taken out), off
  the raw one; nothing is measured in the output afterwards. Like Quick it
  does not denoise (`Method::denoises`).
  With `--features gpl-demosaic` (off by default) the picker also offers
  "RCD (reference)" for Bayer and "Markesteijn 3-pass (reference)" for
  X-Trans (`Method::Reference`, `src/processing/gpl/`): CPU ports of
  darktable's `rcd.c` and `markesteijn.c` (GPL-3, so they are SPDX-marked
  and a build with the feature is GPL-3, unlike the MIT OR Apache-2.0
  default; the reference .c files in the repo root are darktable's, not
  ours). They demosaic the white balanced mosaic (as darktable does; RCD divides
  by the largest gain, its `scaler`, first) in all the cores (24 MP in
  ~0.5-0.9 s) and upload; no denoising. Balancing changed an RW2's RCD output
  by 0.5/255 and its green sharpness by 0.1%.
  With "Lateral CA in the mosaic" on they get the mosaic `CaFix` gives a
  network (read back from the GPU, ~30 ms more), the same pre-demosaic
  correction, so they are comparable with the networks; off, the CA is
  measured in their output and drawn as for Quick. Not bit-identical to darktable:
  RCD mirrors the image 10 px instead of PPG on its border and runs in
  full-width strips, Markesteijn lacks the final `_vng_lininterpolate` over
  the border. `gpl::tests::kodak_psnr` (needs `../demosaic/images`): Kodak
  01 / 05 / 19 / 21 give RCD 33.2 / 34.3 / 37.1 / 34.9 dB and Markesteijn
  32.1 / 33.8 / 36.3 / 33.9 on 8-bit sRGB values mosaicked noise-free, Bayer
  bilinear 25-27. `STARFISH_TEST_REFERENCE=1` in the `develop_file` test.
- Highlight reconstruction in the mosaic (Debug -> "Highlight reconstruction
  in the mosaic", `processing::set_highlights`, `Image::highlights`, off by
  default, `STARFISH_TEST_HIGHLIGHTS=1` in the test; `highlights.rs`): since
  2026-11-15, when the mosaic's CA is measured, it runs *before* `CaFix`, on the
  mosaic with its CA, so the guide green has no clip plateaus; each colour's
  samples are read where they saw the scene point of the sample being rebuilt
  (`reconstruct`'s `ca`, `Ca::field`: offset `d_k - d_c`, rounded; the ring
  around an area likewise, relative to each area pixel's own colour).
  `STARFISH_HIGHLIGHTS_AFTER_CA=1` gives the old order (below) to compare. Unit
  test `follows_the_ca_field` (red 3 px off green, one scene, only red clips):
  clipped-red rms 0.219 read in place, 0.081 along the field. **Not yet judged
  on real highlights in the app.** Before that: after
  `CaFix` and before the demosaic, for the networks, the Mosaic view and the
  reference demosaics alike, the clipped samples (raw >= `processing::CLIPPED` = 0.98 of white, as the
  grain test has it) are rebuilt on the CPU from the unclipped ones. In the
  spirit of darktable's "inpaint opposed" but written from the idea (its
  `highlights.c` was not at hand), not ported: per clipped area (8-connected
  pixels with a clipped sample in their 3x3) the colour is the mean of each
  colour's unclipped white balanced samples within 3 px around it (neutral if
  a colour has none); each clipped sample then becomes the larger of itself
  and the mean over the colours unclipped in its 5x5 of `local * ring_c /
  ring_k` (with all clipped: the smallest brightening of the ring's colour
  that puts every channel at its clip level), at most 4x its clip level.
  Unclipped samples are never touched. The unit tests: a warm blob whose red
  clips (rms error 0.41 -> 0.23; the blob's hue shifts with brightness and the
  ring's ratio is what is used, so peaks stay low) and an all-clipped square
  keeps the surroundings' hue. ~100-220 ms per 24 MP (about 60 ms of it the
  readback and upload of the CaFix texture, as it runs on the CPU); on the
  files in `../demosaic/raws` 4 to 620 areas, up to 730k samples raised
  (1164077238.rw2's lantern: clipped region mean RGB 0.99 / 0.88 / 0.35 ->
  1.12 / 1.06 / 0.42, peaks to 4, R/G 1.235 -> 1.207). The networks were
  trained on mosaics clipped at SAT, so values above white are outside what
  they saw: not judged beyond the numbers above, no retrain. The place
  shader's grain test still treats the raised samples as clipped (they are
  >= 0.99), so no grain there.
- No network takes the lateral CA as an input any more (see "Lateral CA as
  an input" under "The model"; removed from the app and wgpu-run
  2026-10-25: an export with a `ca` line in graph.txt fails to load, asking
  for a re-export). The CA is only ever taken out of the mosaic (below) or
  measured in the output and drawn.
- Lateral CA in the mosaic (2026-10-18; Debug -> "Lateral CA in the
  mosaic", `processing::set_ca_fix`, `Image::ca_fix`, on by default,
  `STARFISH_TEST_CA_FIX=0` in the test): a network gets the frame with it
  taken out first, by `CaFix` (`ca_fix.rs`, `ca_fix::Ca`), with the
  mosaic's measured CA (`RawFrame::mosaic_ca`, `lens::measure_raw`, once
  per file) in the frame's coordinates; the run reports it as corrected
  (`Demosaic::run`'s third value, `Image::corrected_ca`), so nothing is
  measured or drawn ("Lateral CA ..., in the
  mosaic" in the Metadata tab). It is training/common.py's `ca_correct` on
  the GPU (five passes, any CFA, on the unbalanced frame, which gives the
  same result, so `CaFix::apply` takes the white-balance gains only for the
  ratio's bound): green from green samples, then per red/blue
  site the samples whose scene points land near it, each moved along green
  by a local least-squares slope regularised towards the colour ratio (see
  its docstring). Checked against PyTorch (`ca_fix::tests::matches_pytorch`,
  `training/ca_fix_vector.py`, which writes the mosaic unbalanced with the
  gains, as the app gives it): 152 dB X-Trans and Bayer; with the bound left
  in balanced units it fails at 82-85 dB.
  Changed 2026-10-24 (not yet in any trained model): the colour ratio is
  bounded to 0..`CA_FIX_RATIO_MAX` (4), where it exploded under a
  saturated colour with almost no green (kodim05 with a sharp guide: errors
  up to 16x white; the tent guide escaped by averaging; 0.3% of kodim19's
  red pixels and 3-5% of kodim05's and kodim03's hit the bound); and the
  guide green is the tent at every pixel, green sites too, which kept their
  raw sample before. The move adds `a (G(p) - Gs(q))`, on flat areas only
  green's noise: ISO 800, shifts 0.5-1.5 px, green noise in red and blue
  0.38-0.44 -> 0.27-0.34 of their own sd, X-Trans and Bayer alike; error
  without noise (kodim, CA made at 2x) 0.1-0.35 dB more on Bayer, 0.3-0.6 on
  X-Trans. Tried for the guide, Bayer: RCD's green (direction, luma proxy,
  ratio-corrected estimates) on a first corrected mosaic, as a better
  green should cut the residual (true green: +3 dB): +0.4 dB (RCD on the
  CA-free mosaic +1.3), but 0.5-1.0 of the green noise into red and blue,
  and their own noise 1.1-1.3x. Dropped. CA left in the network's output
  (`measure_rgb_dump`, rho 0.28-0.95): X-T3 DSCF7167 <= 0.08 px (0.53
  without), Z5 <= 0.07 (0.55 without), DSCF4823 <= 0.05 (the drawn
  correction: <= 0.06 but Fuji blue mid-frame ~0.2, X-T30 / A6700, below). ~120 ms for the mosaic measurement per 24 MP (CPU), the pass a few
  ms. Kodak with `TEST_CA` through the CA-free models (suite "CA fixed"):
  X-Trans 29.5 -> 32.8 dB (37.3 without CA), Bayer 28.8 -> 33.2 (39.9); the
  green estimate is what limits it (37.0 / 39.1 with the true green).
  Training has no CA since 2026-10-25 (`CA_FIX`, which put CA into 25% of
  the patches at 2x the patch size and took it out with `ca_correct`, is in
  git history before then): its models oversmoothed fine detail even where
  little or nothing was corrected, likely because the network could not
  tell the corrected patches, whose red/blue were off by up to 40-55 sigma
  (99th percentile; median 0.86-1.01 against 0.62-0.64 elsewhere), from the
  rest, and sigma (from the uncorrected samples) did not say so; with it on
  every patch was DIV2K at half scale, too. A simpler approximation of the
  residual CA is planned. Measured on the way, for that: a CA made at 1x
  interpolates the true samples and `ca_correct` partly undoes it, where a
  lens moves the scene and the sensor samples new points; four kodim
  images, a constant field, red/blue after `ca_correct` against the CA-free
  mosaic, made at 2x / 1x: 0.25 px 29.4 / 33.6 dB (X-Trans), 29.9 / 34.9
  (Bayer); 0.5 px 23.9 / 25.7, 24.0 / 26.0; fine detail kept 0.88 /
  0.81-0.82 at half-pixel shifts. Where little CA is needed it leaves red
  and blue as they are (kodim01: 0 px 85-92 dB from the mosaic, 0.1 px fine
  detail 1.00x). `ca_correct` stays as the app's reference
  (`ca_fix_vector.py`) and for the suite's "CA fixed" rows.
  Constant shift (2026-11-08): the measured `Measured::shift` (the fit's
  `t`) is added to the field (`Ca::shift`, `ca_fix.wgsl`'s `field`). It was
  left out as 0.1-0.3 px of unknown cause, but a decentred lens gives red
  and blue opposite shifts over the whole frame: the Fujinon 18-55 files
  `raws/ca-images/DSCF1768` / `DSCF1813` have R (-0.78, +0.93) and B (+0.91,
  -1.08) px at every radius, which the radial model alone left as fringes of
  opposite colour on the two sides of every edge near the centre (the test
  scene's palm and tower grille are clean now). Only the mosaic path has it:
  the drawn path (`LensCorrection::drawn`, a radial LUT; CA fix off, Quick,
  reference demosaics without the mosaic fix) still ignores `t`.
  `DSCF1813`'s sky/foliage leaves block shifts of several px (outliers the
  Huber fit absorbs; 2239 R blocks against ~3800).
  Clipped samples (2026-11-15, settled 2026-11-16, `Ca::clip` =
  `processing::CLIPPED`): a clipped sample says nothing about the scene's
  edge (the channels clip at different brightness, so c = a Gs + b does not
  hold across it, and the fitted `a` gave colour-plane shifts at highlight
  edges). `fit` leaves a c site whose own sample is clipped out (a = 0 with
  fewer than `MIN_VALID` = 3 left) and `gather` uses a = 0 for a clipped q,
  so that sample is resampled along the field only (as `ca_correct` has with
  a = 0); everything else is the filter as it was, the guide included, up to
  the clip edge. `fit_mean` averages `a` over 5x5 as before. Not in
  `ca_correct`: `matches_pytorch` leaves out the pixels within 14 px of a
  mosaic sample >= `CLIPPED` (0.2% of Kodak 19; 152.7 dB elsewhere, as
  before) and scores red and blue against the CA-free mosaic (kodim19: CA
  left in 31.07 dB, corrected 34.67, plain resample 32.40).
  The CA benchmark (`training/ca_bench.py xtrans|bayer` writes synthetic
  scenes into `training/ca_bench/`, gitignored: lamps of 1.5-20x white in four
  colours with a glow and the lens' 0.7 px blur, colour bars above and below
  the clip, kodim05 alone and with three lamps, and `floodlights` /
  `floodlights_soft`, two rows of 14 neutral 30x lamps with a 1.3% halo and
  the CA 1931359854.cr2 measured; each also noisy, a = 5e-4; a known CA of
  3 / -3.5 px at the corner and the CA-free mosaic as truth;
  `ca_fix::tests::ca_benchmark`, `--ignored --nocapture`,
  `STARFISH_TEST_CA_BENCH`) scores the red and blue sites per variant as PSNR
  (raw units) by distance from the truth's clipped samples (on one, 1-3 px,
  4-8 px, further) and as the share of sites within 3 px that came out more
  than 15% under / over the truth. Variants: uncorrected, plain resample
  (`Ca::guide` 0), guided, guided with no clip rule (`Ca::clip` infinite).
  Clipped / 1-3 px / 4-8 px / far dB, dark sites: lamps uncorrected 17.3 /
  13.2 / 39.8 / 44.2, 10.2%; plain 23.9 / 19.6 / 51.5 / 53.9, 6.3%; guided
  26.3 / 21.9 / 58.7 / 65.7, 5.0%; no rule 27.2 / 24.5 / 58.5 / 65.7, 6.3%.
  floodlights_soft: 30.9 / 22.9, 1.9% (no rule 32.0 / 25.0, 3.2%); edges 32.8
  / 30.1 (33.4 / 31.9). Tried and dropped on it: a rule that also left out a
  guide built from a clipped green (a flag in `moved`) switched the guided
  move off over the whole rim (-2.5 dB clipped, -2.3 dB 1-3 px); `a` clamped
  to `0..ratio_max` changed nothing (+-0.01 dB); clamping the result to the
  range of the samples with a gather weight above 0.1 cost 2-5 dB (a flawed
  test: in practice that is the one nearest sample, so no move at all; see
  the bound below); a weight on `a` by the squared correlation of c and its
  guide over a 9x9 window (smoothstep, only where the guide has structure)
  changed nothing at lenient settings and cost 0.1-2 dB at strict ones, on
  clean data.
  The bound (2026-11-16, `Ca::bound` 0, `STARFISH_CA_BOUND`; `gather`,
  `tune.y`): on 1931359854.cr2's floodlights (a 1600x700 crop) 8 red sites
  came out more than 0.02 under the lowest raw red within 4 px, and 7 over
  the highest (0.037 under at most; e.g. raw 0.134 -> 0.049 where its
  neighbours are 0.09-0.11). The cause, replayed in numpy to the GPU's
  digits: the 5x5 fit's slope `a` (0.82, then averaged over another 5x5) is
  the lamp edge's red / green slope, but in the halo around it the red falls
  far slower than the green (the channels' halos differ: axial CA / PSF), and
  the gather, whose Gaussian weights (sigma 0.5 on a 2 px lattice) are all on
  the nearest sample, adds `a (G(p) - Gs(q))` = 0.82 * (0.217 - 0.323) to it.
  The result is now clamped to the range of the same-colour samples within
  2.5 px of its scene point (widened by `bound` times that range; 0 = not at
  all): the sites out of range on that crop go to 0, and in the benchmark
  (lamps; clipped / 1-3 / 4-8 px / far dB, dark %) 26.3 / 21.9 / 58.7 / 65.7,
  5.0% -> 26.4 / 22.0 / 63.0 / 66.7, 3.1%; floodlights_soft dark 1.9 ->
  1.6%; natural_lamps 6.7 -> 4.7%; edges 1.6 -> 0.6%; kodim05 far 42.48 ->
  42.25 dB (bound 0.1 and 0.25 give back some of that and leave more dark
  sites, 4.0 / 4.3% on lamps).
  `ca_fix_vector.py xtrans|bayer kodimNN` writes `matches_pytorch`'s vector,
  which leaves the bound off. The benchmark's scenes do not reproduce the
  real outline by themselves (every corrected variant has fewer dark sites
  than the uncorrected mosaic), but the sites out of range were what the real
  file had. **Not yet judged on real highlights in the app.**
  With the highlight reconstruction on (below) this test is off (`Ca::clip`
  infinite, `Ca::unclipped`): the rebuilt samples sit above the clip level but
  have structure; samples reconstruction could not raise stay plateaus.
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
- Noise reduction is drawn, not demosaiced (Debug's "Grain"
  slider, 0% = the denoised image, 100% = all the removed noise back; it was
  a "Noise reduction" slider the other way round until 2026-10-17): the network always runs at the
  fitted noise, and the place shader writes into the output's alpha the
  *grain*, what the network removed at each pixel's own sample (`mosaic -
  output[colour]`), divided by that sample's sigma and scaled to green's
  noise at the output's level (`sqrt(a * G / g + b) * g`), so it is equally
  strong whatever the CFA colour (raw, the residual at red X-E1 samples was
  5.6x green's and put the CFA into neutral grain on red surfaces). 0 where the
  sample is clipped, under `run_texture` (no noise model) and from the
  binning stage. `develop.wgsl`'s `sample_source*` add `Edit::grain` =
  `Adjustments::grain` (default 100% since 2026-11-14, 50% before) times it to R, G and B, so the slider is
  instant and continuous and the preview, export and histogram agree; the
  local contrast base takes none. It replaced scaling sigma by `(9^s - 1) / 2`
  (0 at 0%, 1x at 50%, 4x at 100%), which asked the network for sigmas
  below the 0.45-2.2x of the truth it was trained on (`SIGMA_ERR_*`): it
  then decided per region whether to denoise, a blotchy threshold. Colour
  noise stays removed at 0%. The grain's mean per CFA position is 3-8% of its
  sd (the network's fixed per-position bias, see "Training data"; measured
  before the pattern loss).
  Debug -> "3x3 Middle 5 filter grain" (`processing::set_filter_grain`, a
  process-wide switch recorded in `Image::filter_grain`; a change
  re-demosaics; `RawInput::filter_grain`, `STARFISH_TEST_GRAIN_FILTER=1` in
  the test, `STARFISH_TEST_GRAIN_DUMP` writes the grain plane): the place
  shader sorts the 3x3 per-pixel grains (`SORT9`, a 25-step sorting
  network, checked on every 0-1 input; branch-free in registers) and writes
  the mean of ranks 3-7 (`middle5`) instead of the pixel's own grain. On by
  default (since 2026-11-14; before, in release builds only). Chosen by eye (2026-10-17) over the
  3x3 median it replaced, which leaves specks and short streaks on noise:
  neighbouring pixels pick the same sample, so its output is flat patches.
  Compared through a picker (none / median / middle 3 / middle 5 / ranks
  3-7 weighted 1-2-3-2-1, all the same cost), X-E1 (DSCF6089) / Z5
  (nikon_z5_66): strength (sd against the unfiltered) 1 / 0.15 / 0.17 /
  0.20 / 0.18 and 1 / 0.35 / 0.36 / 0.37 / 0.36; correlation with the
  unfiltered 0.20 / 0.34 / 0.44 / 0.40 (X-E1), 0.22-0.25 for all (Z5); exact
  ties with a neighbour, the median's flat patches, 38 / 8.5 / 1.0 / 1.0%
  (X-E1), 34 / 7.5 / 0.8 / 0.8% (Z5); a 3x3 Laplacian's sd over the grain's
  (specks) 2.18 / 1.80 / 1.69 / 1.67 (X-E1), 2.67 / 2.61 / 2.62 / 2.56 (Z5).
- Removed 2026-10-17 (the place shader's code is in git history before
  then): two more grain filters and two colour filters, all Debug
  switches. "Directional median" grain: 3-tap horizontal and vertical
  medians weighted as RCD weights its directions (`disc = (E_v + eps^2) /
  (E_h + E_v + 2 eps^2)` on the output grey, a pixel's disc replaced by its
  eight neighbours' mean when further from 0.5); judged best by eye at the
  time (X-E1 / Z5 sd 1.15 / 1.86, correlation 0.46 / 0.59). "Plus median"
  (the pixel and its four cardinal neighbours): on Bayer those are all the
  other CFA kind, so the G / non-G checkerboard won the median (1.6x the
  residual's grid peak). "Chroma median": each channel to `0.5 * (L +
  med(C - L) + (eps + L) * med(C / (eps + L)))`, 3x3 medians, L the grey
  (R + 2G + B) / 4, eps 1e-4 (with `L` alone instead of `eps + L` it
  darkened flat areas 2.4-6% at -9..-11 EV); fine chroma -27 / -50% at
  -10..-8 EV on the X-E1 / Z5, ~20 ms per 24 MP. "Directional luma median":
  the colour put back on the grey's directional median, against the luma
  zippering (X-E1 fine grey 30.66 -> 28.67% of the mean; band means moved
  up to 1.7%, a median clips peaks). The suite's PyTorch copy of the last
  two (`+med` rows) went with them. Also tried and dropped earlier: four
  directions weighted `1 / (c + eps)^2` mixed in by the anisotropy, RCD's
  diagonals as a second pair (thin jagged diagonal lines), residual minus
  its median, grey + median(residual - grey) (~3x the residual's sd at
  edges), and a guided filter on the grey (judged worse than the median).
- The app uses `Model::run_raw`, which is `run_texture` with the inputs made
  on the GPU: the raw frame goes up as decoded (one unpadded `R32Float`
  texture), with the row and column maps of the padding, the white-balance
  gains and the noise fit, and its cut shader does what `noise::inputs` does
  on the CPU. That removed the CPU padding and `noise::inputs` (49 ms X100S,
  ~180 ms Z7) and most of the upload (29 -> 5 ms, 54 -> 25), also on every
  re-develop. Not bitwise: sigma is now f32 `sqrt(a*x + b)`
  instead of f64. Against the old path (`develop_file` dumps): f16 output
  identical on the X100S and an RW2, the Z7 differs in 0.04% of values (max
  3e-3, 105 dB); f32 115-118 dB, at most one f16 step of the output.
- The noise fit is on the CPU, its block loops split over the cores by rows
  of blocks (`noise::par_map`, results in order, so bitwise the same fit):
  X100S 75 -> 31 ms, Z7 32 -> 28 (Bayer's block loop was cheap; what is left
  is mostly the quantile sorts). It runs once per file. `Engine::develop`
  reads the result back for the preview (it renders on iced's device, not
  the processing one): 35 ms X100S, ~110 ms Z7.
- The noise is always estimated from the image (nothing from metadata):
  8x8 green blocks, 24 level bins, each bin's 2% quantile of block variance
  (Colom, Buades & Morel 2014's per-bin Ponomarenko). Changed 2026-10-17:
  blocks reaching 95% of the brightest green sample are dropped (clipped
  blocks have no noise; they made the slope negative, and four test files
  fell back to a flat sigma), and the line is no longer least squares through
  every bin but `floor_line`, a lower envelope: of the lines through two
  bins that predict a positive variance everywhere and have no bin below
  1/1.5 of them, the one with the most blocks within 1.5x, refitted to
  those weighted by count. Texture only raises a bin, and a level range with
  no flat area (blossom, foliage) set the slope: an ISO 320 A7 III
  (9619256316) fitted a = 2.5e-2, now 7.2e-5; ISO 100 files of every test
  camera now fit 1.3-7e-5 (before up to 1.5e-3). Then a and b are clamped to
  `A_RANGE` / `B_RANGE`. Tried: growing the line from the darkest bins up
  (fails when the dark bins are texture, Z5 foliage 2.6e-3), equal-count
  bins with least squares (still 65x too high).
- The adjustments panel's "Noise level (ISO)" slider, in Details
  (2026-11-14, `Control::Iso`, `NoiseSet::stops`), moves the fit to another
  ISO with no per-camera table: in thirds of a stop from the file's EXIF ISO
  (`RawFrame::iso`; without one it shows stops, "+1.0 EV"), -4..+6 stops, the
  label rounded to the third-stop series (`nominal_iso`). `noise_at_stops`
  scales a by `r = 2^stops` and b by `r^2` (read noise before the
  amplifier; after it b would stay put), clamped to the trained maxima.
  Checked on the test raws' fits (2026-11-14): a / ISO is roughly constant
  per camera (Z5 1.6-2.2e-5, S5M2 1.4-1.8e-5 per ISO 100) apart from 2x
  outliers (one file each: little flat area), so a scales as assumed; b sat
  at its floor (1e-8) in 14 of 18 files, so its r^2 is not tested. A
  per-camera table (a per ISO, b's two parts) is the planned refinement.
- Debug's "Shot noise (a)" and "Read noise (b)" sliders override a or b
  after the ISO (moving the ISO slider clears them; `NoiseSet::ab`, log10 over `A_RANGE` /
  `B_RANGE` and a decade below, whose left end is exactly 0: the networks
  saw a = b = 0 on noise-free patches, and the grain is 0 where sigma is;
  `None` = the estimate, which is where they start, reset to,
  and go back to when a file opens). A change re-demosaics through the
  redevelop path (`set_noise` applies it in the worker, so it also holds
  when the estimate is measured there); the Metadata tab shows "Noise set"
  beside "Noise fit" when they differ. "—" for the quick stage.

- Hot and dead pixels (`processing/hot_pixels.rs`; Debug -> "Hot and dead
  pixels", `processing::set_hot_pixels`, `Image::hot_pixel_fix`, on by
  default, `STARFISH_TEST_HOT_PIXELS=0` in the test): before anything reads
  the mosaic, `develop_unchecked` hands the stage a copy of the frame with
  the defective sites replaced by the median of their same-colour neighbours
  within 3 px. Found automatically (`RawFrame::hot_pixels`, cached per file,
  ~100 ms per 24 MP on all cores): a site beyond the neighbours' min-max range
  by more than 3x that range + 0.005 *and* at an extreme (`EXTREME_ONLY`:
  >= 0.99 with unclipped neighbours, or <= 0.002 with neighbours > 0.02);
  without the extreme rule the test files gave 3-4660 sites, with it far
  fewer (Z7 4660 -> 319, DSCF9791 1215 -> 1, 4024496509 479 -> 8). Warm pixels are therefore left
  alone unless picked. Picked by the user: the "Pick pixels" button in the Repair tab
  (`App::picking`) shows the raw mosaic (`Image::raw_view`: each sample in its
  own channel, no fix, CA or lens, identity rotation / crop / adjustments),
  a click marks the site under it (white square from `preview::Markers`, a
  canvas over the preview), a second click unmarks. Marked sites are replaced
  whatever their value (`hot_pixels::replacements`), only while the checkbox
  is on; leaving the mode sets them (`processing::set_picked`, whose version
  `Image::picked_version` records) and re-demosaics once, and saves them per
  camera body + frame size in the `defects` settings file
  (`library::defects`), loaded when a file opens. A body is the model plus
  the EXIF serial number (`RawFrame::body`; the model alone if the file has
  none; marks saved before 2026-11-14 were per model and no longer match). The counts beside the
  checkbox come from `Image::hot_pixels` / `picked`. The quick first preview
  also goes through this.

- Clipped highlights to grey (built and removed 2026-11-16, never committed;
  to be reworked): the channels clip at different scene brightness and the
  white balance gains tint the plateau (pink), so a CPU mask of the raw
  samples >= `processing::CLIPPED`, marked at their scene points (with the CA
  taken out of the mosaic, red and blue moved by `Ca::field`) and feathered by
  two box blurs of radius 2, was uploaded as an `R8Unorm` array in the
  image's layers with mips and mixed the developed colour towards the pixel's
  luma in `develop.wgsl`, after the colour matrix (so at any temperature). No
  dilation (a patch where only red clipped is a quarter marked on Bayer); the
  luma, not the maximum channel, as the maximum brightened the pixels most
  where the cast was strongest. ~70 ms per 24 MP. Not judged by eye in the
  app.

- Colour zipper post filters (built and removed 2026-11-16; never committed, so
  not in git history; `gpl::tests::dump_reference` is what is left of the
  tooling): a compute pass on the demosaic's output, each channel C of a pixel
  to `0.5 * (L + D + (eps + L) * Q)`, L the pixel's own grey (R + 2G + B) / 4,
  D the median of the colour differences `C - L` and Q of the ratios
  `C / (eps + L)` (eps 1e-4), so only the colour is filtered. Variants: 3x3
  medians of the direct neighbours run twice; a 5-tap directional one (taps one
  CFA period apart: 2 Bayer, 6 X-Trans; horizontal and vertical medians blended
  by a discriminator, RCD's second-derivative energy at first, then the
  variance of the five greys along each direction, with RCD's neighbourhood
  check on a 7x7 lattice of greys; 7 and 11 taps and a mean of the middle five
  were tried); and a cascade of four passes of per-CFA-site, per-channel
  symmetric 3-tap medians (p - o, p, p + o), the offsets o (within 6 px)
  searched against the truth, each stage on the previous output (greedy).
  Judged by PSNR against DIV2K crops mosaicked and demosaiced by the
  reference ports (Markesteijn, RCD; sRGB-encoded channels, 12 held-out crops
  under four noise / camera conditions): none 31.07 / 26.82 / 30.86 / 31.28
  (X-Trans), 3x3 twice +1.4-1.6 dB, directional about +0.05-0.9, cascade 4
  +0.1-0.4 dB over the 3x3 (winners short offsets, (1, +-1), (1, +-2),
  (2, +-1), none at the CFA period; tables searched on a mix of clean, noisy and
  camera-matrix conditions were more robust than ones from one). On the
  network's output no filter helped (the search chose none: it is trained for
  that error), and the directional discriminators differed by <= 0.03 dB on
  bilinear output. Not judged by eye in the app. Removed on request, with the
  highlight reconstruction off by default.

**One clip level (2026-11-16):** `processing::CLIPPED` = 0.98 of white (raw,
unbalanced) is what counts as clipped in the highlight reconstruction, the CA
fix's fallback (`Ca::clip`), the CA measurement's clipped blocks
(`lens/measure.rs`) and wgpu-run's grain test. They were 0.99,
0.95, 0.95, 0.98 and 0.99.

## UI behaviour worth knowing

- The right panel has tabs (`SidebarTab::ALL`): Adjust, Crop, Repair, Info
  (metadata) and, in debug builds only, Debug. Opening Crop puts the preview in
  crop mode and leaving it ends it; leaving Repair while picking finishes the
  picking (one re-demosaic). The tab survives opening another file. Hot and
  dead pixels (checkbox, counts, picker) are in Repair, so release builds have
  them; the demosaic picker, grain filter, CA / highlight switches and noise
  sliders are Debug only (the light / tiny choice is to move to an app-wide
  settings menu). Not yet seen in the real app.

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
- Re-demosaics (demosaic picker, the Debug switches) run one at a
  time, each with its own cancel flag (`App::redeveloping`): a change that
  makes the run in flight out of date cancels it (checked per window), and
  its `CANCELLED` result starts a run with the latest settings. Opening a file
  cancels it as well.
- The tone curve (contrast, midpoint) runs on each linear sRGB channel as
  it is: AgX's inset and outset matrices and the Tone section's "AgX"
  checkbox were removed 2026-10-26 (git history has them).
- Colour edits (`color_edit.rs`; 2026-11-14, replacing the list of selective
  colour edits of 2026-11-13, whose trapezoid range bars, names and
  `EditList` are in git history): two fixed sets, both in the "Color mixer"
  and "Color grading" sections and both run last in `develop.wgsl`'s
  `apply_color_edits`, on the linear value after the tone curve and
  saturation, in Oklab, from the colour as it came in (`Adjustments::
  color_mix`, `grading`; `develop::EditUniform` has `colors`, `grade` and the
  `edit_flags` that say which does anything, so with both at their defaults
  there is no Oklab conversion). The **mixer** has eight colours (`MIX_HUES`,
  OkLCh degrees: red 29, orange 62, yellow 105, green 145, aqua 195, blue
  262, purple 300, magenta 340), each with a hue shift (±1 = ±30 degrees),
  a saturation (-1 greys, +1 triples the chroma) and a brightness (±1 =
  ±0.25 of Oklab L). A pixel takes a smoothstep blend of the settings of
  the two colours either side of its hue (`mixer_at` finds them by counting
  the centres at or below it), faded out below chroma 0.02 so greys are left
  alone. The **grading** has shadows, midtones and highlights, weighted by
  Oklab L (shadows 1 below 0.2 falling to 0 at 0.6, highlights 0 at 0.6 rising
  to 1 at 0.9, midtones the rest, so they add to one), each with a target hue
  (OkLCh degrees) and a "hue shift" amount that pushes the colour towards it
  in Oklab a / b by amount^2 * 0.1, which also works on greys. The panel's
  mixer has a row of eight swatches (`styling::swatch`) selecting the colour
  and three pill sliders painted with what they do (`pill::Track::HueRange`,
  `Saturation`, `Lightness`); the grading a three-way selector
  (`styling::segmented`) and two sliders. `color_edit::tests` check the
  maths in Rust (`apply`, a CPU copy); the shader was checked on the GPU
  (`STARFISH_TEST_COLOR_EDITS=1` in the `develop_file` test sets strong
  values) against a numpy port of the maths on the PNG exports of a Panasonic
  raw with and without: mean |difference| 0.4 of 255 levels, 99.9% under 5
  (the port starts from the 8-bit export). **Not yet seen in the real app.**
  Planned: an eyedropper to pick the colour from the image, a mask overlay,
  presets.
- Auto tone (`auto.rs`, on opening a file, from the camera's embedded JPEG; the
  quantiles of the display luma are compared, so alignment does not matter):
  exposure, contrast and midpoint only. Saturation (100%), the two knees and
  vibrance are left at their defaults for every file (`Fit::apply` and
  `auto::reset` set them, so nothing carries over from the last file). A free
  fit of the three to the luma quantiles, from three starting points (the loss
  has local minima: from the defaults alone 3687377708 ended at contrast 1.0
  with luma rms 0.10, against 0.055), gives the exposure cap (>= 0 EV); the
  exposure is then the highest up to it with <= 0.5% of pixels having a channel
  at the display's white (>= 0.985), and contrast and midpoint are refitted to
  the luma with that exposure (the curve is penalised for clipping too),
  twice, and the exposure checked once more. The six R8 files in
  `raws/cr3_jpg/R8` give exposure 0-1.08, contrast 1.75-3.0, luma rms
  0.011-0.055. Where the raw itself clips (4% on 5789757041) 0 EV cannot get
  under 0.5%. 0.13-0.21 s per file, JPEG decode (10-65 ms) not included (the
  samples are power-encoded once and the clip counts run on up to 8 threads;
  `auto::tests::files`). Earlier versions also fitted saturation, vibrance and
  the knees to the preview's chroma and channel quantiles (git history,
  2026-11-14): a global saturation was pulled up by bright, nearly neutral
  areas the camera renders tinted (a sky), and fitting saturation and vibrance
  one after the other let them trade places, as they act alike where the chroma
  is small. Fitting white balance as well was probed (RGB error -18% on
  3687377708, weak minima) and not done.
- Vibrance (the Color section's "Vibrance" slider,
  `Adjustments::vibrance` 0..2, default 0; the range is the constant
  `preview::VIBRANCE_RANGE` = 0.055 (the slider was removed 2026-11-14);
  `develop.wgsl` right where saturation is applied, after the sigmoid):
  the colour (the distance of the pixel's curve output from the grey of its
  luma) is scaled by `saturation * (1 + vibrance / (1 + (chroma / VIBRANCE_RANGE)^2))`,
  chroma being the pixel's own before either, so grey-ish pixels get all of the
  extra and vivid ones hardly any; no skin-tone protection. The auto tone leaves it at 0. The chroma is in the curve's output
  units, which are about 5% below the display values `auto.rs` measures its
  chroma in. Not seen in the real app yet.
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
  `agx::display`) is gone; `tone.rs` keeps the curve maths and the encoding.
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
  `STARFISH_TEST_CROP=1` for a turn, tilt and 3:2 crop, `STARFISH_TEST_GRAIN_AMOUNT=0..1`
  for the grain, 0 if unset; run on the engine's
  device); through the app's UI only before the texture array.
- Crop mode (the Crop tab, `App::cropping` follows it; its aspect picker,
  straighten slider, rotate and reset are in the tab, the old bar under the
  frame is gone): the frame
  (`preview::frame_rect`; a plain outline drawn per pixel in `preview.wgsl`'s `viewfinder`, the picture outside it dimmed and blurred; the Crop tab's "Composition grid" checkbox, `App::grid`, draws rule-of-thirds lines over it, preview only) stays put and the image moves
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
- Clipping overlays (shown while the Exposure / Black point slider is dragged
  (`ExposureDrag` / `BlackPointDrag` until `ClipReleased`; typing or Ctrl+scroll
  shows none), `App::clip_highlights` / `clip_shadows`, not persisted): `preview.wgsl` tests the developed, clamped linear
  value before sRGB encoding and draws any pixel with a channel at white
  (`CLIP_HIGH`, what rounds to 255) black and any at black (`CLIP_LOW`) white. Preview
  only: export and histogram pass false. Off over the eyedropper's raw view. Not run in the real app yet.
- A file opens upright: `RawFrame::orientation` (the EXIF tag through
  `raw_metadata()`, not `RawImage::orientation`, which rawler leaves at Normal
  for everything but DNG) goes through `Rotation::from_exif` (3, 6, 8; mirrored
  ones are ignored) into `App::rotation` when the quick preview arrives. It is
  the starting rotation, not a separate baseline: the rotate buttons turn from
  it and nothing resets it but opening another file. Checked 2026-11-14 on 48
  test raws: the value equals the camera JPEG's own EXIF orientation in all 31
  pairs (Sony, Canon, Panasonic); the embedded previews are never pre-rotated.
  The 6 / 8 direction has not been checked by eye in the app.
- Left-pane thumbnails always show the filename under the picture.

## Colour

`raw::cam_to_srgb` builds camera RGB -> linear sRGB the
dcraw way from rawler's DNG-style `color_matrix` (Adobe's XYZ -> camera
matrices; `raw::xyz_to_cam` takes D65 first, then other daylight ones, then
any), rows normalised so white stays white after white balance. Until
2026-11-14 it read rawler's deprecated `xyz_to_cam` field, which is all
zeros, so every camera got the identity (camera RGB shown as sRGB) and the
default saturation was 150% to make up for it. The default is now 100%
(`DEFAULT_SATURATION`): the saturation whose display chroma quantiles (50 /
75 / 90%, after the auto tone) match the embedded JPEG's had a median of
0.93 over 61 test raws (quartiles 0.78-1.04; Nikon 0.81, Canon 0.88, Sony
0.93, Fuji and Panasonic 1.03), and 1.63 with the identity, where 150% had
been picked by eye. Exports of an X-T30, R8, A7 III and Z5 II file
matched their camera JPEGs' hues by eye. One matrix per camera (no
dual-illuminant interpolation, no forward matrix or HSV table yet).

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
  `LensCorrection::unverified()` (moot: the app measures CA instead of reading
  tables, see below; that measurement puts the unit at ~2^20 / 0.55, 0.4-0.7
  per file). Why neither could be measured from the JPEGs: in
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
  profile has. The app passes none: every correction is applied where
  the image is drawn (`App::drawn_lens` -> `Preview::lens`, `export::View::lens`):
  the preview uploads the table of `Image::profile` (`LensCorrection::lut`)
  as a 257x1 texture, and `develop.wgsl`'s `lens_source` maps each point of the
  corrected image to its source points (green, and red and blue moved by the
  CA) and vignetting gain, so the image is resampled once (bicubic) for crop,
  tilt, distortion and CA together (`sample_lens*`: three fetches where the
  CA moves a channel; the grain is green's). The local contrast overview is
  drawn through the same mapping, green only (`fs_overview`), and the
  histogram is drawn by the same shader. So every lens toggle is instant. Against the old all-in-`LensPass` export: vignetting and CA
  (X100S) 80+ dB, max 0.0016 (f16 rounding); distortion (DC-S5M2) 80 dB with
  local contrast off, 57 dB with it, the local contrast base being built from
  the uncorrected image's mips sampled through the distortion instead of
  block-aligned mips of the corrected image (max 0.1 at a few hard saturated
  edges, no visible change). The snap-and-bicubic test in `preview.wgsl` is on
  the corrected image's footprint, the mip level on the source's: with the
  source's, regions the distortion stretches fell back to bilinear in the export.
  The "Lens corrections" section of the adjustments panel has a checkbox per
  kind `Image::profile` has: CA wherever it could be measured, whatever the
  camera. Tables that are all zero count as absent, so a prime with no
  distortion gets no distortion checkbox.
- **CA is measured, not read** (`lens/measure.rs`, its docs have the
  method), for every camera, and on by default, where the demosaic stage
  did not correct it itself (see "How the app uses the models").
  `Engine::develop` measures it in the read-back image (~85 ms per 24 MP
  file on 16 cores; not for the quick first preview, `develop_preview`, so
  the CA correction arrives with the full develop), and `LensCorrection::drawn` makes `Image::profile`:
  the file's distortion and vignetting with the measured CA as its
  `chromatic` curves, looked up at the *source* radius (`LensPass` looks the
  table's up at the output radius). The Metadata tab's "Lateral CA" row
  shows the shifts at rho 0.5 / 1 ("drawn", or "in the mosaic": the
  mosaic's, taken out by `CaFix`). Debug builds compile the app crate at opt-level 1 (Cargo.toml):
  unoptimised, the two measurements took ~1.5 s per develop. Why not the tables: measured in the
  mosaic, Fuji's red is 0.75-1.5x its table and blue 0.4-0.75x (X100S /
  X100F: blue table all zero, but both X100S files show the same -0.45 /
  +0.6 px curve), Sony 0.4-0.7x of the guessed unit; the A6700's JPEGs, which
  the camera corrects, show < 0.05 px left of the 0.2-0.8 px in their raws,
  so the measurement, not the tables, matches what the camera removes. Why
  in the image, not the mosaic: the networks already pull red and blue part
  of the way onto green (X-T30 X-Trans output: 36-71% of the mosaic's CA
  left; A6700 Bayer ~73%), so the mosaic's CA over-corrected. Exports of the
  X-T30 and A6700 files: CA left after the correction <= 0.06 px except Fuji
  blue mid-frame (~0.2 px, from 0.8). The tables are still read
  (`chromatic` in `LensCorrection::read`'s profiles) and `LensPass` still
  applies them when asked, for reference.
  **A pitfall found on the way:** interpolate shifts with Catmull-Rom.
  Keys' cubic with a = -0.75 (PyTorch's `grid_sample` bicubic) moves a block
  by 2|a| = 1.5x the shift for small shifts, so estimators built on it
  measure real CA ~1.4x too small, and synthetic CA made with it hides the
  error (it cancels). `measures_known_ca` renders its scene exactly.
  Ignored tests: `measure_file` (a raw's mosaic CA against its table),
  `measure_dump` (`raw::tests::dump_mosaic`'s file), `measure_rgb_dump` (a
  `develop_file` dump: the CA left after the demosaic).
- To check a change: `STARFISH_TEST_LENS=0|1 STARFISH_TEST_DUMP=out.f32` on the
  `develop_file` test dumps planar f32 camera RGB, with all three corrections
  from `LensPass` (CA from the table), to compare with the JPEG; its
  `STARFISH_TEST_EXPORT` is developed as the app does it (every correction
  drawn, CA measured). Compare 16-bit PNGs with something that reads 16
  bits: Pillow reads 16-bit RGB as 8.

## The web viewer (removed in v0.2.0)

A static page that ran the networks in the browser (wasm32 + WebGPU, GitHub Pages) was
removed from this repo in v0.2.0; v0.1.0 and the history before it have it
(`web/`, `.github/workflows/pages.yml`). A simpler demo is planned in the
`starfish-neural-demosaic` repo. `wgpu-run` still has a few wasm-only branches
(`noise::par_map`, `device.poll` use) left from it.

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

Training, its data and its Python venv are in `../starfish-neural-demosaic`. Test raws and
`images/` live outside the repo, in `../demosaic/`.

The workspace builds demosaic-run into `target/`; point it at `models/<export>` or a fresh export in the training repo (its `test.bin` is there):

```
cargo build --release -p demosaic-run
./target/release/demosaic-run ../starfish-neural-demosaic/xtrans_wgpu                 # vs test.bin
./target/release/demosaic-run ../starfish-neural-demosaic/xtrans_wgpu --raw ../demosaic/raws/DSCF9791.RAF --png out.png
./target/release/demosaic-run ../starfish-neural-demosaic/xtrans_wgpu --size 4000x6000 --bench 10
./target/release/demosaic-run ../starfish-neural-demosaic/xtrans_wgpu --gpu      # the same through run_texture
./target/release/demosaic-run ../starfish-neural-demosaic/xtrans_wgpu --raw ../demosaic/raws/DSCF6089.RAF --crop 2000,2200,516,516 --inputs in.bin
      # the network's inputs (as test.bin starts: h, w, srgb_from_cam, mosaic, sigma), for PyTorch
```

X-Trans models take Fuji `.RAF` files, Bayer models `.nef` / `.rw2`.
`test.bin` is gitignored; it is rewritten by every export.

## The models (training lives in a separate repo)

The networks are trained and exported by `starfish-neural-demosaic` (a sibling repo,
`../starfish-neural-demosaic`; split off at v0.2.0). Its AGENTS.md has everything
this file used to say about the model architectures, training data, loss,
checkpoints, fine-tuning, the suite and what was tried and dropped; the section
names this file still cites ("The model", "Training data", "Tried and dropped",
"Checkpoints") are there. This repo holds the exports the app runs in
`models/{bayer,xtrans}[_lite|_tiny]_wgpu/{f16,f32}/` (copied from the training
repo by `scripts/sync-models.sh`; `build.rs` embeds them, debug builds read
them) and the executor `wgpu-run/`. Paths below written `training/x` are in the
training repo (`training/<name>_wgpu` exports: `models/` here).
`wgpu-run` checks an export against its `test.bin` (kept in the training repo).


## The wgpu export

Exports since 2026-10-23 have no CA input, and since 2026-10-25 wgpu-run
no longer runs the older ones that do (a `ca` line in graph.txt is a load
error): their `ca` buffer, `RawInput::ca`, `Model::takes_ca`, `run`'s field
argument, test.bin's field and `demosaic-run --ca-check` are gone (git
history before then has them).

`export()` writes `<name>_wgpu/{f16,f32}/` (graph.txt, one .wgsl per dispatch,
weights.bin) and `test.bin`. Everything model-specific -- kernel choice
(1x1 / tiled k x k / plain conv), template constants, weight
repacking, buffer reuse, dispatch sizes -- is decided in `wgsl.py`; the Rust
side only allocates and dispatches what graph.txt lists. Shaders are built for
one window (`WGPU_WINDOW` = 1056 px); `Model::run` processes any image as
windows with a halo that wrap around the image edge. f16 needs the adapter's
`SHADER_F16`.

**Zero padding** (2026-10-18; `common.conv`, `PhaseConv`, `wgsl.py`'s
`CONV` / `CONV_TILED` loads and `HEAD` taps): every conv pads with zeros, in
training, PyTorch inference and the shaders. It was circular, which in
PyTorch is a copy of every activation before every conv (~30% of a training
step; Bayer clean model fwd+bwd 42 -> 29.5 ms, X-Trans 29 -> 20.8). Nothing
counted sees the padding: the loss and the PSNR drop `BORDER` >= `HALO` >=
the reach per side (`evaluate` drops it from the PSNR too now, so its numbers
are higher than the earlier ones; a checkpoint scored 32.57 dB with the
border and circular, 30.75 with zeros, the same held-out loss), so the same
weights give the same loss and gradients, and circular and zero-padded
PyTorch agree on every pixel the loss counts (Bayer exactly, X-Trans 1e-5).
The shaders: a model's export differs from the circular one only in the
window's outer ring, which the halo keeps out of every core, so the
output is bitwise the same (Bayer and X-Trans clean models, `demosaic-run
--dump`, f16, test.bin sized), and old circular exports keep working
unchanged until they are re-exported. Checked directly with `halo 0` in a
copy of graph.txt and an image exactly one window big (no wrap-around in the
cut): every pixel but the outermost matches zero-padded PyTorch at the f16
level (max 5e-3 Bayer, 1e-2 X-Trans); the outermost ring differs because
`fold_head` puts the masks of taps beyond the edge in the bias, where PyTorch
has none. The cut shader still wraps the image round: the output within
`halo` px of the image's edge is therefore not PyTorch's (which sees zeros
there), and `demosaic-run` drops `halo` px (not 8) before its PSNR and max
error. The app pads by >= halo, so never outputs those pixels.

Weights are rounded to fp16 for both builds. The current models' accuracy
against PyTorch and their timings are in "The model". The original models'
trunk was verified bitwise identical to the earlier hand-built Rust runner
(whole image vs 6 windows, f16 and f32); the numbers below were measured on
those models.

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

The window was 1536 until 2026-10; 1056 (= 96 * 11, packed grid still a
multiple of the 16 px workgroups) gives bitwise identical output, keeps each
per-window submission ~2.2x shorter (Windows resets the GPU after ~2 s) and
halves the buffers (largest: Bayer f32 54 MB, was 113). On the 5060 Ti it was
faster anyway (X-Trans f16 681 -> 650 ms, Bayer f16 798 -> 500 ms, f32 874 ->
670; only X-Trans f32 slower, 815 -> 880), likely because the buffers now fit
the cache better. Keep the largest buffer under `max_storage_buffer_binding_size`
(128 MiB by default) if it is ever raised.

The graph's mosaic and sigma buffers are kept to the end (`write` marks
them as outliving every op): `run_texture`'s place shader reads them after
the graph for the grain. Nothing but the first op reads them, so the
output would otherwise take the mosaic's buffer (as in the first head-free
export: the grain, noise reduction below 100%, was garbage while D was
right). `write` also deletes the directory's old `.wgsl` files first, so an
export with fewer ops leaves none behind for build.rs to embed.

Any change to `wgsl.py` templates or packing: re-export and check
`demosaic-run` against test.bin (the PSNR vs PyTorch must not drop).
