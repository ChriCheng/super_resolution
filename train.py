import argparse
import json
import os
import sys
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn

from src.dataset import create_train_loader
from src.model import ESPCN
from src.utils import AverageMeter, ensure_dir, set_random_seed


class ProgressBar:
    """Minimal terminal progress bar without external dependencies."""

    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self._render()

    def update(self, n: int = 1, postfix: str = ""):
        self.current = min(self.current + n, self.total)
        self._render(postfix=postfix)

    def close(self, postfix: str = ""):
        self.current = self.total
        self._render(postfix=postfix)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, postfix: str = ""):
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "=" * filled + "." * (self.width - filled)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0.0
        line = (
            f"\r{self.desc} [{bar}] {self.current}/{self.total} "
            f"({ratio * 100:6.2f}%) {rate:5.2f} it/s"
        )
        if postfix:
            line += f"  {postfix}"
        sys.stdout.write(line)
        sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESPCN x4 on DIV2K with MindSpore")
    parser.add_argument("--div2k_hr_dir", type=str, required=True, help="Path to DIV2K_train_HR")
    parser.add_argument("--save_dir", type=str, default="./outputs/train_espcn_x4")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--repeat", type=int, default=20, help="How many random patches per image each epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_target", type=str, default="GPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--mode", type=str, default="graph", choices=["graph", "pynative"])
    return parser.parse_args()


def main():
    args = parse_args()

    ms.set_context(mode=ms.GRAPH_MODE if args.mode == "graph" else ms.PYNATIVE_MODE)
    ms.set_device(args.device_target, args.device_id)
    set_random_seed(args.seed)
    ensure_dir(args.save_dir)

    with open(os.path.join(args.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_loader = create_train_loader(
        hr_dir=args.div2k_hr_dir,
        batch_size=args.batch_size,
        scale=4,
        patch_size=args.patch_size,
        repeat=args.repeat,
    )
    steps_per_epoch = train_loader.get_dataset_size()

    model = ESPCN(scale=4)
    criterion = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

    def forward_fn(lr, hr):
        sr = model(lr)
        loss = criterion(sr, hr)
        return loss, sr

    grad_fn = ms.value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters, has_aux=True)

    best_loss = float("inf")
    log_path = os.path.join(args.save_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,avg_loss\n")

    for epoch in range(1, args.epochs + 1):
        model.set_train(True)
        loss_meter = AverageMeter()
        progress = ProgressBar(total=steps_per_epoch, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in train_loader.create_tuple_iterator(num_epochs=1):
            lr, hr = batch
            (loss, _), grads = grad_fn(lr, hr)
            optimizer(grads)
            loss_value = float(loss.asnumpy())
            loss_meter.update(loss_value)
            progress.update(postfix=f"loss={loss_value:.6f} avg={loss_meter.avg:.6f}")

        avg_loss = loss_meter.avg
        progress.close(postfix=f"loss={avg_loss:.6f}")
        print(f"Epoch [{epoch}/{args.epochs}]  loss={avg_loss:.6f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.8f}\n")

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.ckpt")
        ms.save_checkpoint(model, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ms.save_checkpoint(model, os.path.join(args.save_dir, "best.ckpt"))

    print(f"Training finished. Best training loss: {best_loss:.6f}")
    print(f"Best checkpoint saved to: {os.path.join(args.save_dir, 'best.ckpt')}")


if __name__ == "__main__":
    main()
