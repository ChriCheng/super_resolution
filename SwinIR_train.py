import argparse
import json
import os
import sys
import time

import mindspore as ms
import mindspore.nn as nn

from src.dataset import create_eval_loader, create_train_loader
from src.model import ESPCN
from src.utils import (
    AverageMeter,
    calc_psnr_ssim,
    ensure_dir,
    set_random_seed,
    tensor_to_image_uint8,
)


class ProgressBar:
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
    parser = argparse.ArgumentParser(description="Train SwinIR-style x4 on DIV2K with MindSpore")
    parser.add_argument("--div2k_hr_dir", type=str, required=True, help="Path to DIV2K_train_HR")
    parser.add_argument("--div2k_val_hr_dir", type=str, default=None, help="Path to DIV2K_valid_HR")
    parser.add_argument("--save_dir", type=str, default="./outputs/train_swinir_x4")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--repeat", type=int, default=20, help="Random patches per image per epoch")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device_target", type=str, default="GPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--mode", type=str, default="graph", choices=["graph", "pynative"])
    return parser.parse_args()


def validate(model, val_loader, scale: int = 4):
    model.set_train(False)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in val_loader.create_tuple_iterator(num_epochs=1):
        lr, hr, _ = batch
        sr = model(lr)

        sr_uint8 = tensor_to_image_uint8(sr)
        hr_uint8 = tensor_to_image_uint8(hr)

        psnr, ssim = calc_psnr_ssim(
            sr_uint8,
            hr_uint8,
            scale=scale,
            test_y_channel=True,
        )
        psnr_meter.update(psnr)
        ssim_meter.update(ssim)

    return psnr_meter.avg, ssim_meter.avg


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

    val_loader = None
    if args.div2k_val_hr_dir:
        val_loader = create_eval_loader(args.div2k_val_hr_dir, scale=4)

    model = ESPCN(scale=4)
    criterion = nn.L1Loss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

    def forward_fn(lr, hr):
        sr = model(lr)
        loss = criterion(sr, hr)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters)

    best_psnr = -1.0
    best_loss = float("inf")
    log_path = os.path.join(args.save_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_psnr,val_ssim,best_psnr\n")

    for epoch in range(1, args.epochs + 1):
        model.set_train(True)
        loss_meter = AverageMeter()
        progress = ProgressBar(total=steps_per_epoch, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in train_loader.create_tuple_iterator(num_epochs=1):
            lr, hr = batch
            loss, grads = grad_fn(lr, hr)
            optimizer(grads)

            loss_value = float(loss.asnumpy())
            loss_meter.update(loss_value)
            progress.update(postfix=f"loss={loss_value:.6f} avg={loss_meter.avg:.6f}")

        avg_loss = loss_meter.avg
        progress.close(postfix=f"loss={avg_loss:.6f}")

        val_psnr = float("nan")
        val_ssim = float("nan")

        if val_loader is not None:
            val_psnr, val_ssim = validate(model, val_loader, scale=4)
            print(
                f"Epoch [{epoch}/{args.epochs}]  "
                f"train_loss={avg_loss:.6f}  val_psnr={val_psnr:.4f}  val_ssim={val_ssim:.6f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                ms.save_checkpoint(model, os.path.join(args.save_dir, "best.ckpt"))
        else:
            print(f"Epoch [{epoch}/{args.epochs}]  train_loss={avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                ms.save_checkpoint(model, os.path.join(args.save_dir, "best.ckpt"))

        ms.save_checkpoint(model, os.path.join(args.save_dir, "last.ckpt"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ms.save_checkpoint(model, os.path.join(args.save_dir, f"epoch_{epoch:03d}.ckpt"))

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.8f},{val_psnr},{val_ssim},{best_psnr}\n")

    summary = {
        "best_psnr": best_psnr,
        "best_loss": best_loss,
        "epochs": args.epochs,
        "save_dir": args.save_dir,
    }
    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if val_loader is not None:
        print(f"Training finished. Best validation PSNR: {best_psnr:.4f} dB")
    else:
        print(f"Training finished. Best training loss: {best_loss:.6f}")
    print(f"Best checkpoint saved to: {os.path.join(args.save_dir, 'best.ckpt')}")


if __name__ == "__main__":
    main()