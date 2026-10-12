"""Train the Bayer demosaic + denoise network, then export it for wgpu.

    python train_bayer.py                    # train (resumes bayer.pt if it exists)
    python train_bayer.py --export           # just export bayer.pt -> bayer_wgpu/
"""
import argparse

import torch.nn as nn

from common import BAYER, DATA, HERE, Demosaic, PhaseConv, act, conv, export, icnr, train

CFA = BAYER
TILE = 2                     # pack 2x2 = one Bayer period into 4 planes at 1/2 res
CKPT = HERE / 'bayer.pt'

STEPS = 500_000
BATCH = 32
PATCH = 144                  # crop size, a multiple of TILE
LR = 2e-4
EMA = 0.999
CLIP_GRAD = 5.0


def build():
    # Trunk at 1/2 resolution. Input: 4 packed mosaic planes + pooled sigma.
    net = nn.Sequential(
        conv(5, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 48, 3), act(),
        conv(48, 3 * TILE * TILE, 3),        # -> pixel_shuffle(2): full-res RGB estimate
    )
    icnr(net[-1], TILE)
    # Full-res head on [estimate, scattered mosaic, masks, sigma] = 10 channels;
    # its 3x3 has one weight bank per Bayer phase (4).
    full_head = nn.Sequential(
        PhaseConv(10, 32, 3, per=2), act(),
        conv(32, 3, 1),
    )
    return Demosaic(CFA, TILE, net, full_head)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA, help='prep.py output')
    ap.add_argument('--export', action='store_true', help=f'only export {CKPT.name}')
    args = ap.parse_args()
    if args.export:
        export(build, CFA, TILE, CKPT, args.data)
    else:
        train(build, CFA, TILE, CKPT, STEPS, BATCH, PATCH, LR, EMA, CLIP_GRAD, args.data)
