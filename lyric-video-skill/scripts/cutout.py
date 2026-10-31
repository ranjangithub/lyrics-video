#!/usr/bin/env python3
"""
cutout.py — cut the person out of a photo (transparent PNG) with an ONNX matting model.

    python cutout.py PHOTO -o OUT.png [--model isnet|birefnet|u2net]

Models live in ../models/ (download once from https://github.com/danielgatis/rembg/releases/tag/v0.0.0):
  isnet-general-use.onnx (default, 1024px, crisp edges, ~5 s/photo on CPU)
  birefnet-portrait-epoch_150.onnx (best hair/edges, slow: ~60 s/photo, ~3 GB RAM)
  u2net_human_seg.onnx (fast, softer)
"""
import argparse, os

import numpy as np
from PIL import Image, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "isnet": ("isnet-general-use.onnx", 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "birefnet": ("birefnet-portrait-epoch_150.onnx", 1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "u2net": ("u2net_human_seg.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def open_image(path):
    if path.lower().endswith((".heic", ".heif")):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def matte(img, model="isnet"):
    import onnxruntime as ort
    fn, size, mean, std = MODELS[model]
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
    sess = ort.InferenceSession(os.path.join(HERE, "..", "models", fn), so, providers=["CPUExecutionProvider"])
    x = np.asarray(img.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    if model == "isnet":
        x = (x - np.array(mean)) / np.array(std)
    else:
        x = (x / max(x.max(), 1e-6) - np.array(mean)) / np.array(std)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    m = out[0, 0] if out.ndim == 4 else out[0]
    if model == "birefnet":
        m = 1 / (1 + np.exp(-m))
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    a = Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    # tighten the matte a little and keep only the biggest blob's neighbourhood
    arr = np.asarray(a, np.float32) / 255
    arr = np.clip((arr - 0.08) / 0.84, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="isnet", choices=list(MODELS))
    a = ap.parse_args()
    img = open_image(a.photo)
    m = matte(img, a.model)
    rgba = img.copy(); rgba.putalpha(m)
    bb = m.point(lambda v: 255 if v > 20 else 0).getbbox()
    if bb:
        rgba = rgba.crop(bb)
    rgba.save(a.out)
    print(a.out, rgba.size)


if __name__ == "__main__":
    main()
