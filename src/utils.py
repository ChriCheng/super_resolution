import csv
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


def extract_img_name(name):
    if hasattr(name, "asnumpy"):
        name = name.asnumpy()

    if isinstance(name, np.ndarray):
        if name.size == 1:
            name = name.reshape(-1)[0]
        else:
            name = name.tolist()[0]

    if isinstance(name, (list, tuple)):
        name = name[0]

    if isinstance(name, bytes):
        name = name.decode("utf-8")

    name = str(name).strip().strip("'\"")
    name = os.path.basename(name)
    return name


def tensor_to_image_uint8(x) -> np.ndarray:
    if hasattr(x, "asnumpy"):
        x = x.asnumpy()
    x = np.squeeze(x, axis=0)
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    x = np.round(x * 255.0).astype(np.uint8)
    return x


def rgb_to_y_channel(img_uint8: np.ndarray) -> np.ndarray:
    if img_uint8.ndim == 2:
        return img_uint8.astype(np.float64)

    img = img_uint8.astype(np.float64)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


def shave_border(img: np.ndarray, shave: int) -> np.ndarray:
    if shave <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * shave or w <= 2 * shave:
        return img
    if img.ndim == 3:
        return img[shave:-shave, shave:-shave, :]
    return img[shave:-shave, shave:-shave]


def calc_psnr_ssim(
    sr_uint8: np.ndarray,
    hr_uint8: np.ndarray,
    scale: int = 4,
    test_y_channel: bool = True,
    shave: int = None,
):
    if shave is None:
        shave = scale

    sr_eval = sr_uint8
    hr_eval = hr_uint8

    if shave > 0:
        sr_eval = shave_border(sr_eval, shave)
        hr_eval = shave_border(hr_eval, shave)

    if test_y_channel:
        sr_eval = rgb_to_y_channel(sr_eval)
        hr_eval = rgb_to_y_channel(hr_eval)
        psnr = peak_signal_noise_ratio(hr_eval, sr_eval, data_range=255)
        ssim = structural_similarity(hr_eval, sr_eval, data_range=255)
    else:
        psnr = peak_signal_noise_ratio(hr_eval, sr_eval, data_range=255)
        ssim = structural_similarity(hr_eval, sr_eval, data_range=255, channel_axis=2)

    return float(psnr), float(ssim)


def save_image(img_uint8: np.ndarray, path: str):
    Image.fromarray(img_uint8).save(path, format="PNG")


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)