import argparse
import json
import os

import numpy as np
from PIL import Image
import mindspore as ms

from src.dataset import create_eval_loader
from src.utils import (
    calc_psnr_ssim,
    ensure_dir,
    extract_img_name,
    save_image,
    tensor_to_image_uint8,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bicubic x4 baseline")
    parser.add_argument(
        "--eval_hr_dir",
        "--set5_hr_dir",
        dest="eval_hr_dir",
        type=str,
        required=True,
        help="Path to HR images for evaluation",
    )
    parser.add_argument("--save_dir", type=str, default="./outputs/eval_bicubic_x4")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device_target", type=str, default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--mode", type=str, default="graph", choices=["graph", "pynative"])
    parser.add_argument("--test_y_channel", action="store_true", default=True)
    parser.add_argument("--no_test_y_channel", action="store_false", dest="test_y_channel")
    return parser.parse_args()


def bicubic_upsample_from_lr_tensor(lr, out_h, out_w):
    lr_uint8 = tensor_to_image_uint8(lr)
    sr_img = Image.fromarray(lr_uint8).resize((out_w, out_h), Image.BICUBIC)
    sr_uint8 = np.asarray(sr_img).astype(np.uint8)
    return sr_uint8


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE if args.mode == "graph" else ms.PYNATIVE_MODE)
    ms.set_device(args.device_target, args.device_id)

    ensure_dir(args.save_dir)
    ensure_dir(os.path.join(args.save_dir, "sr_images"))

    loader = create_eval_loader(args.eval_hr_dir, scale=args.scale)
    rows = []

    for batch in loader.create_tuple_iterator(num_epochs=1):
        lr, hr, name = batch

        hr_uint8 = tensor_to_image_uint8(hr)
        h, w = hr_uint8.shape[:2]

        sr_uint8 = bicubic_upsample_from_lr_tensor(lr, h, w)
        psnr, ssim = calc_psnr_ssim(
            sr_uint8,
            hr_uint8,
            scale=args.scale,
            test_y_channel=args.test_y_channel,
        )

        img_name = extract_img_name(name)
        save_path = os.path.join(args.save_dir, "sr_images", img_name)
        save_image(sr_uint8, save_path)

        rows.append({
            "image": img_name,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 6),
        })
        print(f"{img_name}: PSNR={psnr:.4f} dB, SSIM={ssim:.6f}")

    avg_psnr = float(np.mean([r["psnr"] for r in rows]))
    avg_ssim = float(np.mean([r["ssim"] for r in rows]))
    rows.append({"image": "Average", "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6)})
    write_csv(rows, os.path.join(args.save_dir, "bicubic_x4_metrics.csv"))

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "average_psnr": avg_psnr,
            "average_ssim": avg_ssim,
            "scale": args.scale,
            "test_y_channel": args.test_y_channel,
            "eval_hr_dir": args.eval_hr_dir,
        }, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Bicubic x4 Average PSNR: {avg_psnr:.4f} dB")
    print(f"Bicubic x4 Average SSIM: {avg_ssim:.6f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()