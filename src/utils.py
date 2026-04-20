import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import mindspore as ms
import mindspore.dataset as ds


def set_random_seed(seed: int = 42):
    ms.set_seed(seed)
    ds.config.set_seed(seed)
    np.random.seed(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return 0.0 if self.count == 0 else self.sum / self.count

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def tensor_to_image_uint8(x) -> np.ndarray:
    if hasattr(x, "asnumpy"):
        x = x.asnumpy()
    x = np.squeeze(x, axis=0)
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    x = np.round(x * 255.0).astype(np.uint8)
    return x


def calc_psnr_ssim(sr_uint8: np.ndarray, hr_uint8: np.ndarray):
    psnr = peak_signal_noise_ratio(hr_uint8, sr_uint8, data_range=255)
    ssim = structural_similarity(hr_uint8, sr_uint8, data_range=255, channel_axis=2)
    return float(psnr), float(ssim)


def save_image(img_uint8: np.ndarray, path: str):
    Image.fromarray(img_uint8).save(path)


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
