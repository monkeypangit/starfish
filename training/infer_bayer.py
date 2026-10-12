"""Mosaic and noise a clean photo as a Bayer sensor would, demosaic it with
bayer.pt, print the PSNR against the original and write the result.

    python infer_bayer.py photo.png [--a 1e-3] [--b 1e-6] [--exposure 1] [--out out.png]
"""
import argparse

from common import infer
from train_bayer import CFA, CKPT, TILE, build

WB = (2.0, 1.6)              # white-balance gains R, B (raw -> balanced), daylight-ish

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--a', type=float, default=1e-3, help='shot noise, var = a*x + b')
    ap.add_argument('--b', type=float, default=1e-6, help='read noise variance')
    ap.add_argument('--exposure', type=float, default=1.0, help='linear multiplier before the sensor clips')
    ap.add_argument('--out', default='out.png')
    args = ap.parse_args()
    infer(build, CFA, TILE, CKPT, args.image, args.a, args.b, args.exposure, WB, args.out)
