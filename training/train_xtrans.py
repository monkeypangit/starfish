"""Train the X-Trans demosaic + denoise network, then export it for wgpu.

    python train_xtrans.py                   # train (resumes xtrans.pt if it exists)
    python train_xtrans.py --export          # just export xtrans.pt -> xtrans_wgpu/
"""
import argparse

import torch.nn as nn

from common import DATA, HERE, XTRANS, Demosaic, PhaseConv, act, conv, export, icnr, train

CFA = XTRANS
TILE = 6                     # pack 6x6 = one X-Trans period into 36 planes at 1/6 res
CKPT = HERE / 'xtrans.pt'

STEPS = 500_000
BATCH = 32
PATCH = 144                  # crop size, a multiple of TILE
LR = 2e-4
EMA = 0.999
CLIP_GRAD = 5.0


def build():
    # Trunk at 1/6 resolution. Input: 36 packed mosaic planes + pooled sigma.
    # At tile 6 a 1x1 is already a 6x6 spatial mixer, so three 3x3 layers carry
    # the cross-tile structure and the 1x1s are cheap width.
    net = nn.Sequential(
        conv(37, 144, 3), act(),
        conv(144, 288, 1), act(),
        conv(288, 288, 1), act(),
        conv(288, 144, 3), act(),
        conv(144, 288, 1), act(),
        conv(288, 288, 1), act(),
        conv(288, 144, 3), act(),
        conv(144, 288, 1), act(),
        conv(288, 3 * TILE * TILE, 1),       # -> pixel_shuffle(6): full-res RGB estimate
    )
    icnr(net[-1], TILE)
    # Full-res head on [estimate, scattered mosaic, masks, sigma] = 10 channels;
    # its 3x3 has one weight bank per X-Trans phase (36).
    full_head = nn.Sequential(
        PhaseConv(10, 32, 3, per=6), act(),
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
