"""Everything train_xtrans.py and train_bayer.py share: the CFA tables, the
network's building blocks, the synthetic sensor, the loss, the training loop,
the wgpu export and a minimal inference.

The two train_*.py files hold the model itself and its training constants; the
constants below define the *data*: how a clean photo is turned into the noisy
raw mosaic the network learns to undo. They are identical for both CFAs.
"""
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import wgsl

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------ data constants

DATA_DIR = HERE / 'data'     # where the user puts the training data (data/README.md)
DIV2K = DATA_DIR / 'DIV2K_train_HR'   # the DIV2K training PNGs, added by hand
DATA = DATA_DIR / 'div2k.npy'         # prep.py output: every DIV2K image, decoded, flat uint8
HOLDOUT_IMAGES = 40          # the last 40 images (DIV2K 0761-0800) are never trained on
EVAL_PATCHES = 256           # held-out crops scored at every checkpoint
EVAL_SEED = 1234
WORKERS = 4                  # DataLoader processes cutting crops

# Poisson-Gaussian noise, var = a*x + b in raw units, a and b log-uniform per patch
A_RANGE = (1e-5, 1e-2)
B_RANGE = (1e-8, 1e-4)
ZERO_NOISE_P = 0.2           # fraction of patches with no noise and an all-zero sigma map

# The sigma map is built from a, b multiplied by these (log-uniform) factors
# while the noise itself uses the true a, b: at inference sigma is estimated.
SIGMA_ERR_A_RANGE = (1 / 2, 2.0)
SIGMA_ERR_B_RANGE = (1 / 5, 5.0)

# White-balance gains (raw -> balanced) for R and B; G is the 1.0 anchor
WB_R_RANGE = (1.3, 2.8)
WB_B_RANGE = (1.2, 2.6)

EXPOSURE_RANGE = (1 / 8, 2.0)  # linear multiplier, log-uniform: -3 .. +1 EV
SAT = 1.0                    # sensor white level in raw units

BLUR_EVERY = 8               # every 8th patch of a batch gets a Gaussian blur ...
BLUR_MAX_SIGMA = 2.0         # ... with sigma uniform in (0, 2] px
RB_SWAP_P = 0.5              # probability of swapping R and B of the source patch

HUE_MAX_DEG = 6.0            # camera-matrix jitter: hue rotation about the neutral axis
SAT_RANGE = (0.85, 1.20)     # camera-matrix jitter: chroma scale

# Loss: Charbonnier in tone-mapped linear sRGB, border dropped
LOSS_BORDER = 18
LOSS_EPS = 1e-6
WARMUP = 500                 # steps of linear LR warm-up before the cosine decay
WEIGHT_DECAY = 1e-4

# The test vector written next to an export: a held-out image through a fixed sensor
TEST_IMAGE = -HOLDOUT_IMAGES  # first held-out image
TEST_SIZE = (768, 1152)      # h, w; multiples of 6
TEST_WB = (2.0, 1.6)
TEST_A, TEST_B = 1e-3, 1e-6

# The wgpu build is compiled for windows of this many px a side (a multiple of
# every TILE); images of any size are processed as overlapping windows, one GPU
# submission each. 1056 = 96 * 11 keeps the packed grid a multiple of the 16 px
# workgroups for both TILEs. Smaller than the earlier 1536 so one submission
# stays short on slow GPUs (Windows resets the GPU after ~2 s) and the buffers
# small, at a few percent more halo work.
WGPU_WINDOW = 1056

# ------------------------------------------------------------------ the CFAs
# 0 = R, 1 = G, 2 = B

BAYER = torch.tensor([[0, 1],
                      [1, 2]])

# dcraw/LibRaw's X-Trans phase
XTRANS = torch.tensor([[1, 1, 0, 1, 1, 2],
                       [1, 1, 2, 1, 1, 0],
                       [2, 0, 1, 0, 2, 1],
                       [1, 1, 2, 1, 1, 0],
                       [1, 1, 0, 1, 1, 2],
                       [0, 2, 1, 2, 0, 1]])


def cfa_masks(cfa, h, w):
    """1,3,h,w one-hot colour of every site; h, w multiples of the CFA period."""
    p = cfa.shape[0]
    onehot = F.one_hot(cfa.long(), 3).permute(2, 0, 1).float()     # 3,p,p
    return onehot.repeat(1, h // p, w // p)[None]


# ------------------------------------------------------------- network parts

def conv(i, o, k):
    """k x k conv with circular padding: the CFA is periodic, so a wrapped
    border site still has the colour the network expects there."""
    return nn.Conv2d(i, o, k, padding=k // 2, padding_mode='circular')


def act():
    return nn.LeakyReLU(0.1, inplace=True)


def icnr(c, tile):
    """Init a conv that feeds pixel_shuffle(tile) so all tile^2 output slots of a
    channel start identical: no checkerboard on the tile grid at step 0."""
    t2 = tile * tile
    o, i, kh, kw = c.weight.shape
    sub = torch.empty(o // t2, i, kh, kw)
    nn.init.kaiming_uniform_(sub, a=5 ** 0.5)
    bias = torch.empty(o // t2).uniform_(-(i * kh * kw) ** -0.5, (i * kh * kw) ** -0.5)
    with torch.no_grad():
        c.weight.copy_(sub.repeat_interleave(t2, dim=0))
        c.bias.copy_(bias.repeat_interleave(t2))


class PhaseConv(nn.Module):
    """A k x k conv with one weight bank per CFA phase, each bank applied only
    at its own pixels. Phase of pixel (y, x) is (y % per) * per + x % per.

    Weight layout is bank-major, (per^2 * o, i, k, k), which is what the wgpu
    runner reads. All banks start identical.
    """

    def __init__(self, i, o, k, per):
        super().__init__()
        self.per, self.o = per, o
        self.conv = nn.Conv2d(i, o * per * per, k)
        with torch.no_grad():
            self.conv.weight.copy_(self.conv.weight[:o].repeat(per * per, 1, 1, 1))
            self.conv.bias.copy_(self.conv.bias[:o].repeat(per * per))

    def forward(self, x):
        p, n, o = self.per, self.per * self.per, self.o
        k = self.conv.kernel_size[0]
        B, C, H, W = x.shape
        xp = F.pad(x, (k // 2,) * 4, mode='circular')
        # im2col: every pixel's k x k neighbourhood stacked into channels, in
        # the order of weight[:, ci, dy, dx]
        cols = torch.stack([xp[..., dy:dy + H, dx:dx + W]
                            for dy in range(k) for dx in range(k)], 2).flatten(1, 2)
        # sort pixels by phase: B, C*k*k, n, H/p, W/p
        cols = F.pixel_unshuffle(cols, p).unflatten(1, (C * k * k, n))
        w = self.conv.weight.reshape(n, o, C * k * k)
        y = torch.einsum('bkqhw,qok->boqhw', cols, w)
        y = y + self.conv.bias.view(n, o).t()[None, :, :, None, None]
        return F.pixel_shuffle(y.flatten(1, 2), p)                  # B, o, H, W


class Demosaic(nn.Module):
    """pack the mosaic into tile^2 planes at 1/tile resolution -> trunk `net`
    -> pixel_shuffle back to a full-resolution RGB estimate -> `full_head`
    refines it, seeing the measured samples, the CFA masks and sigma directly.

    m, sigma: B,1,H,W white-balanced mosaic and its noise std. -> B,3,H,W in
    white-balanced camera space.
    """

    def __init__(self, cfa, tile, net, full_head):
        super().__init__()
        self.tile = tile
        self.net = net
        self.full_head = full_head
        self.register_buffer('cfa', cfa, persistent=False)

    def forward(self, m, sigma):
        t = self.tile
        x = torch.cat([F.pixel_unshuffle(m, t), F.avg_pool2d(sigma, t)], 1)
        estimate = F.pixel_shuffle(self.net(x), t)                  # B,3,H,W
        masks = cfa_masks(self.cfa, m.shape[-2], m.shape[-1]).to(m)
        scattered = m * masks                    # each sample in its own colour plane
        return self.full_head(torch.cat(
            [estimate, scattered, masks.expand(m.shape[0], -1, -1, -1), sigma], 1))


# -------------------------------------------------------- the synthetic sensor

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def apply_ccm(x, m):
    """x: B,3,H,W; m: B,3,3 -> B,3,H,W."""
    return torch.einsum('bij,bjhw->bihw', m.to(x), x)


# DNG ColorMatrix (XYZ D65 -> camera) of eight cameras; training draws a random
# blend of them per patch so the network does not learn one camera's colours.
SRGB_TO_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
CAM_XYZ_BANK = torch.tensor([
    [[1.0405, -0.3755, -0.1270], [-0.5461, 1.3787, 0.1793], [-0.1040, 0.2015, 0.6785]],  # Nikon D850
    [[0.5488, -0.0909, -0.0437], [-0.5769, 1.3284, 0.2732], [-0.1041, 0.2135, 0.8524]],  # Nikon D700
    [[0.6722, -0.0635, -0.0963], [-0.4287, 1.2460, 0.2028], [-0.0908, 0.2162, 0.5668]],  # Canon 5D III
    [[0.5271, -0.0712, -0.0347], [-0.6153, 1.3653, 0.2763], [-0.1601, 0.2366, 0.7242]],  # Sony A7
    [[0.7610, -0.2124, -0.0512], [-0.4971, 1.2600, 0.2560], [-0.0776, 0.1503, 0.6015]],  # Panasonic GH5
    [[0.7368, -0.2260, -0.0587], [-0.3745, 1.1455, 0.2323], [-0.0528, 0.1345, 0.6994]],  # Fujifilm X-T3
    [[0.8756, -0.3050, -0.0902], [-0.4732, 1.2678, 0.2338], [-0.0902, 0.1140, 0.7042]],  # Nikon Z7
    [[0.7178, -0.1584, -0.1010], [-0.4519, 1.2557, 0.2247], [-0.0844, 0.1755, 0.6301]],  # Canon R5
])
# linear sRGB -> camera, rows normalised to 1 so neutral stays neutral
CAM_FROM_SRGB_BANK = CAM_XYZ_BANK @ SRGB_TO_XYZ
CAM_FROM_SRGB_BANK = CAM_FROM_SRGB_BANK / CAM_FROM_SRGB_BANK.sum(2, keepdim=True)


def loguniform(n, lo, hi, dev):
    return (torch.rand(n, 1, 1, 1, device=dev) * math.log(hi / lo) + math.log(lo)).exp()


def random_ccm(n, dev):
    """(cam_from_srgb, srgb_from_cam), each n,3,3: a sharpened random blend of
    the bank, then a small hue rotation and chroma scale (both keep neutral)."""
    bank = CAM_FROM_SRGB_BANK.to(dev)
    w = torch.rand(n, len(bank), device=dev) ** 3
    M = torch.einsum('nk,kij->nij', w / w.sum(1, keepdim=True), bank)
    s = (torch.rand(n, device=dev) * (SAT_RANGE[1] - SAT_RANGE[0]) + SAT_RANGE[0])[:, None, None]
    chroma = s * torch.eye(3, device=dev) + (1 - s) * torch.full((3, 3), 1 / 3, device=dev)
    # rotation about the neutral axis (Rodrigues)
    K = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]], device=dev) / math.sqrt(3)
    th = ((torch.rand(n, device=dev) * 2 - 1) * math.radians(HUE_MAX_DEG))[:, None, None]
    hue = torch.eye(3, device=dev) + th.sin() * K + (1 - th.cos()) * (K @ K)
    M = chroma @ hue @ M
    M = M / M.sum(2, keepdim=True)
    return M, torch.linalg.inv_ex(M)[0]


def sensor(lin, cfa, cam_from_srgb, srgb_from_cam, gain, e, a, b, sa, sb):
    """Photograph a linear-sRGB image with a simulated raw sensor.

    lin: B,3,H,W linear sRGB. cam_from_srgb / srgb_from_cam: B,3,3. gain: B,3,1,1
    WB gains (raw -> balanced). e: exposure; a, b: true noise; sa, sb: the noise
    coefficients the sigma map is built from. All B,1,1,1.

    camera matrix -> exposure -> un-balance to raw -> clip at SAT -> mosaic ->
    noise -> clip -> sigma -> balance mosaic and sigma per site.

    Returns (noisy B,1,H,W, sigma B,1,H,W, target B,3,H,W, srgb_from_cam B,3,3):
    noisy, sigma and target in white-balanced camera space; srgb_from_cam takes
    that to linear sRGB at the original exposure.
    """
    masks = cfa_masks(cfa, lin.shape[-2], lin.shape[-1]).to(lin)
    raw = (apply_ccm(lin, cam_from_srgb) * e / gain).clamp(max=SAT)
    m = (raw * masks).sum(1, keepdim=True)
    noisy = (m + torch.randn_like(m) * (a * m.clamp(min=0) + b).sqrt()).clamp(max=SAT)
    sigma = (sa * noisy.clamp(min=0) + sb).sqrt()
    site_gain = (gain * masks).sum(1, keepdim=True)
    # target: balanced, clamped at the common white level so blown highlights are neutral
    target = (raw * gain).clamp(max=SAT)
    return noisy * site_gain, sigma * site_gain, target, srgb_from_cam / e.view(-1, 1, 1)


def random_batch(clean_u8, cfa):
    """B,3,H,W uint8 sRGB crops (on the GPU) -> a training batch from `sensor`
    with every nuisance drawn at random per patch."""
    dev, n = clean_u8.device, clean_u8.shape[0]
    lin = srgb_to_linear(clean_u8.float() / 255)
    lin = torch.where(torch.rand(n, 1, 1, 1, device=dev) < RB_SWAP_P, lin.flip(1), lin)
    lin = blur_every(lin)
    cam_from_srgb, srgb_from_cam = random_ccm(n, dev)
    gr = loguniform(n, *WB_R_RANGE, dev)
    gain = torch.cat([gr, torch.ones_like(gr), loguniform(n, *WB_B_RANGE, dev)], 1)
    e = loguniform(n, *EXPOSURE_RANGE, dev)
    noisy = (torch.rand(n, 1, 1, 1, device=dev) >= ZERO_NOISE_P).float()
    a = loguniform(n, *A_RANGE, dev) * noisy
    b = loguniform(n, *B_RANGE, dev) * noisy
    sa = a * loguniform(n, *SIGMA_ERR_A_RANGE, dev)
    sb = b * loguniform(n, *SIGMA_ERR_B_RANGE, dev)
    return sensor(lin, cfa, cam_from_srgb, srgb_from_cam, gain, e, a, b, sa, sb)


def blur_every(x):
    """Gaussian-blur every BLUR_EVERY-th patch, sigma uniform in (0, BLUR_MAX_SIGMA].
    The others get sigma 1e-3, whose kernel is an exact delta."""
    n, c, h, w = x.shape
    sel = (torch.arange(n, device=x.device) % BLUR_EVERY == 0).float()
    s = (torch.rand(n, device=x.device) * BLUR_MAX_SIGMA * sel).clamp(min=1e-3)
    r = math.ceil(3 * BLUR_MAX_SIGMA)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k[None] ** 2 / (2 * s[:, None] ** 2))
    g = (g / g.sum(1, keepdim=True)).repeat_interleave(c, 0)        # n*c, K
    y = F.pad(x.reshape(1, n * c, h, w), (r, r, r, r), mode='reflect')
    y = F.conv2d(y, g[:, None, None, :], groups=n * c)
    y = F.conv2d(y, g[:, None, :, None], groups=n * c)
    return y.reshape(n, c, h, w)


# ------------------------------------------------------------------- the loss

def tone(x):
    """Signed display gamma 1/2.2 with a linear toe below sRGB's knee, value and
    slope continuous there; odd-symmetric so out-of-gamut values stay finite."""
    g, k = 1 / 2.2, 0.0031308
    c = k ** g * (1 - g) / (1 - k ** g * (1 - g))
    s = (1 + c) * g * k ** (g - 1)
    ax = x.abs()
    y = torch.where(ax < k, ax * s, (1 + c) * ax.clamp(min=k).pow(g) - c)
    return y * x.sign()


def charbonnier(out, target, srgb_from_cam):
    """Charbonnier in tone-mapped linear sRGB, LOSS_BORDER px dropped."""
    b = LOSS_BORDER
    d = (tone(apply_ccm(out.float(), srgb_from_cam)) -
         tone(apply_ccm(target.float(), srgb_from_cam)))[..., b:-b, b:-b]
    return (d * d + LOSS_EPS).sqrt().mean()


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean())


# ------------------------------------------------------------------- the data

class Crops(torch.utils.data.Dataset):
    """Random size x size crops of the training images, uniform over their
    pixels, under a random rotation / flip. Returns 3,size,size uint8."""

    def __init__(self, data, size, length):
        self.data, self.size, self.length = data, size, length
        self.index = np.load(str(data)[:-4] + '_index.npy')         # offset, h, w per image
        self.index = self.index[:-HOLDOUT_IMAGES]
        h, w = self.index[:, 1], self.index[:, 2]
        self.cum = np.cumsum((h - size + 1) * (w - size + 1)).tolist()  # crop origins
        self.flat = None                                            # opened per worker

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.flat is None:
            self.flat = np.load(self.data, mmap_mode='r')
        k = random.choices(range(len(self.index)), cum_weights=self.cum)[0]
        o, h, w = (int(v) for v in self.index[k])
        y, x = random.randrange(h - self.size + 1), random.randrange(w - self.size + 1)
        im = self.flat[o:o + h * w * 3].reshape(h, w, 3)
        crop = torch.from_numpy(np.array(im[y:y + self.size, x:x + self.size])).permute(2, 0, 1)
        crop = torch.rot90(crop, random.randrange(4), [-2, -1])
        if random.random() < 0.5:
            crop = crop.flip(-1)
        return crop.contiguous()


def held_out_image(data, k):
    """Held-out image k (negative = from the end) as h,w,3 uint8."""
    index = np.load(str(data)[:-4] + '_index.npy')
    o, h, w = (int(v) for v in index[k])
    return np.array(np.load(data, mmap_mode='r')[o:o + h * w * 3].reshape(h, w, 3))


def eval_set(data, size, cfa, dev):
    """EVAL_PATCHES fixed crops of the held-out images through random_batch,
    under a fixed seed: the same tensors at every checkpoint."""
    index = np.load(str(data)[:-4] + '_index.npy')
    flat = np.load(data, mmap_mode='r')
    rng = np.random.default_rng(EVAL_SEED)
    crops = []
    for k in rng.integers(len(index) - HOLDOUT_IMAGES, len(index), EVAL_PATCHES):
        o, h, w = (int(v) for v in index[k])
        y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
        im = flat[o:o + h * w * 3].reshape(h, w, 3)
        crops.append(torch.from_numpy(np.array(im[y:y + size, x:x + size])).permute(2, 0, 1))
    crops = torch.stack(crops).to(dev)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(EVAL_SEED)
        return [random_batch(c, cfa) for c in crops.split(32)]


@torch.no_grad()
def evaluate(net, batches):
    """(loss, psnr) on the eval set; psnr in linear sRGB clamped to [0, 1]."""
    loss, p = 0.0, 0.0
    for noisy, sigma, target, srgb_from_cam in batches:
        with torch.autocast('cuda', dtype=torch.float16):
            out = net(noisy, sigma)
        loss += charbonnier(out, target, srgb_from_cam).item()
        srgb = lambda z: apply_ccm(z.float(), srgb_from_cam).clamp(0, 1)
        p += psnr(srgb(out), srgb(target)).item()
    return loss / len(batches), p / len(batches)


# --------------------------------------------------------------- the training

def train(build, cfa, tile, ckpt, steps, batch, patch, lr, ema_decay, clip_grad,
          data=DATA):
    """Train build() for `steps` steps. `ckpt` holds everything, and a run picks
    up from it if it exists:

        model   the EMA weights: what inference and the export use
        raw     the weights being trained
        opt     AdamW state
        scaler  fp16 GradScaler state
        step    steps done

    Saved every 10000 steps and at the end, each save printing the held-out
    PSNR; at the end the EMA weights are exported for the wgpu runner.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    cfa = cfa.to(dev)

    net = build().to(dev).to(memory_format=torch.channels_last)
    ema = build().to(dev).to(memory_format=torch.channels_last).eval().requires_grad_(False)
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    start = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev)
        net.load_state_dict(ck['raw'])
        ema.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start = ck['step']
        print(f'resuming {ckpt} at step {start}')
    print(f'{sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters')

    def save(step):
        torch.save({'model': ema.state_dict(), 'raw': net.state_dict(),
                    'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step},
                   str(ckpt) + '.tmp')
        os.replace(str(ckpt) + '.tmp', ckpt)       # never leave a half-written file
        loss, p = evaluate(ema, held_out)
        print(f'{step:7d}  saved {ckpt.name}   held-out: loss {loss:.5f}  psnr {p:.2f} dB')

    held_out = eval_set(data, patch, cfa, dev)
    loader = torch.utils.data.DataLoader(
        Crops(data, patch, (steps - start) * batch), batch_size=batch,
        num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=True)
    fnet = torch.compile(net)
    fbatch = torch.compile(random_batch)
    ema_p, net_p = list(ema.parameters()), list(net.parameters())

    t0, run_loss = time.time(), torch.zeros((), device=dev)
    for i, crops in enumerate(loader, start):
        # linear warm-up, then cosine decay to 0 at the last step
        for g in opt.param_groups:
            g['lr'] = lr * min(1.0, i / WARMUP) * 0.5 * (1 + math.cos(math.pi * i / steps))
        noisy, sigma, target, srgb_from_cam = fbatch(crops.to(dev, non_blocking=True), cfa)
        noisy = noisy.contiguous(memory_format=torch.channels_last)
        sigma = sigma.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            out = fnet(noisy, sigma)
        loss = charbonnier(out, target, srgb_from_cam)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net_p, clip_grad)
        scaler.step(opt)
        scaler.update()
        # EMA; the decay ramps up over the first steps so the average does not
        # remember the random init
        d = min(ema_decay, (1 + i) / (10 + i))
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, net_p, 1 - d)

        run_loss += loss.detach()
        if (i + 1) % 1000 == 0:
            print(f'{i + 1:7d}  loss {run_loss.item() / 1000:.5f}  '
                  f'lr {opt.param_groups[0]["lr"]:.2e}  '
                  f'{(i + 1 - start) * batch / (time.time() - t0):.0f} img/s')
            run_loss.zero_()
        if (i + 1) % 10000 == 0:
            save(i + 1)
    if steps % 10000 and start < steps:
        save(steps)
    export(build, cfa, tile, ckpt, data)


# -------------------------------------------------------------- wgpu export

@torch.no_grad()
def export(build, cfa, tile, ckpt, data=DATA):
    """ckpt's EMA weights -> <ckpt>_wgpu/ for the Rust executor in wgpu-run/:

    f16/, f32/   graph.txt + one .wgsl per dispatch + weights.bin, the network
                 compiled for WGPU_WINDOW x WGPU_WINDOW px windows (see wgsl.py);
                 f16 needs the adapter's SHADER_F16 feature, f32 runs anywhere
    test.bin     u32 h, u32 w, f32[9] srgb_from_cam, then f32 mosaic[h*w],
                 sigma[h*w], fp32 PyTorch output[3*h*w] (balanced camera space),
                 clean[3*h*w] (linear sRGB): a held-out image through a fixed sensor
    """
    dev = 'cuda'
    cfa = cfa.to(dev)
    net = build().to(dev).eval()
    net.load_state_dict(torch.load(ckpt, map_location=dev)['model'])
    out = Path(ckpt).with_name(Path(ckpt).stem + '_wgpu')

    # weights rounded to fp16 once, so the f16 and f32 builds compute with the same values
    arr = lambda t: t.detach().half().float().cpu().numpy()
    convs = list(net.net.children())
    trunk = [(f'net.{j}', arr(c.weight), arr(c.bias),
              j + 1 < len(convs) and isinstance(convs[j + 1], nn.LeakyReLU))
             for j, c in enumerate(convs) if isinstance(c, nn.Conv2d)]
    bank = lambda m: (arr(m.conv.weight), arr(m.conv.bias), m.per ** 2) \
        if isinstance(m, PhaseConv) else (arr(m.weight), arr(m.bias), 1)
    layers = {'trunk': trunk, 'head_in': bank(net.full_head[0]),
              'head_out': bank(net.full_head[2])}
    # Receptive-field radius: each trunk 3x3 reaches one tile, plus one tile for
    # a pixel's place inside its own, plus the head's 3x3; rounded up to a whole
    # tile so every window starts on the CFA phase and the packing grid.
    k3 = sum(c.kernel_size[0] == 3 for c in convs if isinstance(c, nn.Conv2d))
    halo = ceil_to(tile * (k3 + 1) + 1, tile)
    for precision in ('f16', 'f32'):
        n_ops, nbytes = wgsl.write(out / precision, precision, layers, cfa.cpu().numpy(),
                                   tile, WGPU_WINDOW, halo)
    print(f'exported {out}/: {n_ops} dispatches per {WGPU_WINDOW} px window, halo {halo} px, '
          f'{nbytes / 1e6:.0f} MB of f32 buffers')

    h, w = TEST_SIZE
    im = torch.from_numpy(held_out_image(data, TEST_IMAGE)[:h, :w]).permute(2, 0, 1)[None]
    lin = srgb_to_linear(im.to(dev).float() / 255)
    torch.manual_seed(0)
    noisy, sigma, srgb_from_cam, ref = fixed_sensor(net, lin, cfa, TEST_WB, 1.0, TEST_A, TEST_B)
    with open(out / 'test.bin', 'wb') as f:
        f.write(struct.pack('<II', h, w))
        for t in (srgb_from_cam, noisy, sigma, ref, lin):
            f.write(t.float().cpu().numpy().astype('<f4').tobytes())


def ceil_to(n, m):
    return -(-n // m) * m


def fixed_sensor(net, lin, cfa, wb, e, a, b, autocast=False):
    """One linear-sRGB image through the sensor with fixed settings (camera
    matrix 0 of the bank, the given WB, exposure and noise, sigma told the true
    a and b), then through the network. -> (noisy, sigma, srgb_from_cam, out)."""
    dev = lin.device
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=dev)
    cam_from_srgb = CAM_FROM_SRGB_BANK[:1].to(dev)
    gain = torch.tensor([wb[0], 1.0, wb[1]], device=dev).view(1, 3, 1, 1)
    noisy, sigma, _, srgb_from_cam = sensor(
        lin, cfa, cam_from_srgb, torch.linalg.inv(cam_from_srgb), gain,
        one(e), one(a), one(b), one(a), one(b))
    with torch.autocast('cuda', dtype=torch.float16, enabled=autocast):
        out = net(noisy, sigma).float()
    return noisy, sigma, srgb_from_cam, out


# ------------------------------------------------------------------ inference

INFER_BORDER = 8   # px dropped per side before PSNR (circular padding wraps the edges)


@torch.no_grad()
def infer(build, cfa, tile, ckpt, image, a, b, exposure, wb, out_path):
    """Mosaic + noise a clean photo, demosaic it, print PSNR, write out_path."""
    dev = 'cuda'
    cfa = cfa.to(dev)
    net = build().to(dev).eval()
    net.load_state_dict(torch.load(ckpt, map_location=dev)['model'])
    im = Image.open(image).convert('RGB')
    w, h = im.width // tile * tile, im.height // tile * tile
    im = torch.from_numpy(np.array(im.crop((0, 0, w, h)))).permute(2, 0, 1)[None]
    lin = srgb_to_linear(im.to(dev).float() / 255)

    _, _, srgb_from_cam, out = fixed_sensor(net, lin, cfa, wb, exposure, a, b, autocast=True)
    out = apply_ccm(out, srgb_from_cam).clamp(0, 1)
    r = INFER_BORDER
    p = psnr(out[..., r:-r, r:-r], lin.clamp(0, 1)[..., r:-r, r:-r]).item()
    print(f'{w}x{h}  a={a:g} b={b:g} exposure={exposure:g}  '
          f'psnr {p:.2f} dB')
    arr = (linear_to_srgb(out[0]).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    Image.fromarray(arr).save(out_path)
    print('wrote', out_path)
