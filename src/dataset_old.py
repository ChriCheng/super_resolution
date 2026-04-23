import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import mindspore.dataset as ds


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: str) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return files


class DIV2KPatchDataset:
    """Random patch dataset.

    Read DIV2K HR images and generate LR-HR pairs on the fly using bicubic downsampling.
    """

    def __init__(self, hr_dir: str, scale: int = 4, patch_size: int = 192, repeat: int = 20, augment: bool = True):
        self.files = list_images(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.repeat = repeat
        self.augment = augment

        if patch_size % scale != 0:
            raise ValueError("patch_size must be divisible by scale")

    def __len__(self):
        return len(self.files) * self.repeat

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            scale_factor = max(ps / w, ps / h)
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - ps)
        top = random.randint(0, h - ps)
        return img.crop((left, top, left + ps, top + ps))

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.TRANSPOSE)
        return img

    @staticmethod
    def _to_chw_float(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.files[index % len(self.files)]
        hr = Image.open(img_path).convert("RGB")
        hr = self._random_crop(hr)
        if self.augment:
            hr = self._augment(hr)

        lr_size = (hr.size[0] // self.scale, hr.size[1] // self.scale)
        lr = hr.resize(lr_size, Image.BICUBIC)

        lr = self._to_chw_float(lr)
        hr = self._to_chw_float(hr)
        return lr, hr


class Set5EvalDataset:
    """Full-image evaluation dataset.

    The dataset expects high-resolution ground-truth images only.
    LR images are produced internally using bicubic x4 downsampling.
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        self.files = list_images(hr_dir)
        self.scale = scale

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _to_chw_float(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def __getitem__(self, index: int):
        img_path = self.files[index]
        hr = Image.open(img_path).convert("RGB")
        w, h = hr.size
        w = (w // self.scale) * self.scale
        h = (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        return self._to_chw_float(lr), self._to_chw_float(hr), img_path.name


def create_train_loader(hr_dir: str,
                        batch_size: int = 16,
                        scale: int = 4,
                        patch_size: int = 192,
                        repeat: int = 20,
                        num_parallel_workers: int = 1):
    dataset = DIV2KPatchDataset(hr_dir=hr_dir, scale=scale, patch_size=patch_size, repeat=repeat, augment=True)
    loader = ds.GeneratorDataset(dataset, column_names=["lr", "hr"], shuffle=True,
                                 num_parallel_workers=num_parallel_workers, python_multiprocessing=False)
    loader = loader.batch(batch_size, drop_remainder=True)
    return loader


def create_eval_loader(hr_dir: str, scale: int = 4):
    dataset = Set5EvalDataset(hr_dir=hr_dir, scale=scale)
    loader = ds.GeneratorDataset(dataset, column_names=["lr", "hr", "name"], shuffle=False,
                                 num_parallel_workers=1, python_multiprocessing=False)
    loader = loader.batch(1)
    return loader
