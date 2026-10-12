"""Decode every data/DIV2K_train_HR/*.png once into data/div2k.npy (one flat
uint8 array) and data/div2k_index.npy (an (offset, h, w) row per image, sorted
by file name). Training cuts random crops from these at run time. See
data/README.md for where the images come from.

    python prep.py [path/to/DIV2K_train_HR]   # default: data/DIV2K_train_HR
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA, DIV2K

FILES = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else DIV2K).glob('*.png'))


def size(f):
    with Image.open(f) as im:                 # header only
        return im.height, im.width


def fill(job):
    k, off, h, w = job
    out = np.load(DATA, mmap_mode='r+')
    out[off:off + h * w * 3] = np.asarray(Image.open(FILES[k]).convert('RGB')).reshape(-1)
    out.flush()


if __name__ == '__main__':
    if not FILES:
        sys.exit(f'No PNGs found in {sys.argv[1] if len(sys.argv) > 1 else DIV2K}; see data/README.md')
    with Pool(min(os.cpu_count(), 16)) as pool:
        hw = np.array(pool.map(size, FILES), dtype=np.int64)
        n = hw[:, 0] * hw[:, 1] * 3
        index = np.stack([np.concatenate([[0], np.cumsum(n)[:-1]]), hw[:, 0], hw[:, 1]], 1)
        np.lib.format.open_memmap(DATA, 'w+', np.uint8, (int(n.sum()),))
        pool.map(fill, [(k, *index[k]) for k in range(len(FILES))])
    np.save(str(DATA)[:-4] + '_index.npy', index)
    print(f'{DATA}: {len(FILES)} images, {n.sum() / 1e9:.2f} GB')
