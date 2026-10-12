# Training data

The training scripts read their data from this folder. Its contents are not
in git (see `.gitignore`); only this file is.

1. Download the DIV2K training images, `DIV2K_train_HR.zip` (800 PNGs, about
   3.5 GB), from <https://data.vision.ee.ethz.ch/cvl/DIV2K/>. DIV2K is
   provided for academic research; check its terms before using it.
2. Unzip it here, so the images are in `training/data/DIV2K_train_HR/*.png`.
3. From `training/`, run `python prep.py` once. It decodes every image into
   `training/data/div2k.npy` (about 6 GB) and `training/data/div2k_index.npy`,
   which training and export read.

The images are sorted by file name; the last 40 (0761-0800) are held out for
evaluation and the export's test image.
