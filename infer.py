import argparse
import os

from PIL import Image
import numpy as np
import mindspore as ms

from src.model import ESPCN
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for one image using ESPCN x4")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="./outputs/infer/result.png")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device(args.device_target, args.device_id)
    ensure_dir(os.path.dirname(args.output) or ".")

    net = ESPCN(scale=4)
    ms.load_param_into_net(net, ms.load_checkpoint(args.ckpt_path))
    net.set_train(False)

    img = Image.open(args.input).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    sr = net(ms.Tensor(arr, ms.float32)).asnumpy()[0]
    sr = np.transpose(sr, (1, 2, 0))
    sr = np.clip(sr, 0.0, 1.0)
    sr = (sr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(sr).save(args.output)
    print(f"Saved SR image to {args.output}")


if __name__ == "__main__":
    main()
