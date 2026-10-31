#!/usr/bin/env python3
"""
separate.py — isolate the vocals so alignment hears the singer, not the band.

    python separate.py SONG.mp3 -o work/vocals.wav [--model PATH.onnx]

Uses sherpa-onnx with the UVR MDX-Net Voc_FT model (ONNX, CPU). Setup:
    pip install sherpa-onnx onnxruntime soundfile
    curl -L -o models/UVR-MDX-NET-Voc_FT.onnx \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
Roughly 0.3–0.5× real time on 2 CPU cores.
"""
import argparse, os, subprocess, sys, time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "UVR-MDX-NET-Voc_FT.onnx")


def load_44k_stereo(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", "44100", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="work/vocals.wav")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--block", type=float, default=10.0, help="seconds per block; lower if memory is tight")
    a = ap.parse_args()

    import sherpa_onnx
    if not os.path.exists(a.model):
        sys.exit(f"Model not found: {a.model} (see the download command at the top of this script)")
    cfg = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=a.model),
            num_threads=a.threads, debug=False, provider="cpu"))
    sep = sherpa_onnx.OfflineSourceSeparation(cfg)
    x = load_44k_stereo(a.audio)
    t = time.time()
    # process in short blocks (the model needs ~200 MB RAM per second of audio) with a 1 s crossfade so memory stays small on long songs
    SR, blk, ov = 44100, int(a.block * 44100), 44100
    n = x.shape[1]
    vocals = np.zeros((2, n), np.float32)
    weight = np.zeros(n, np.float32)
    s = 0
    while s < n:
        e = min(n, s + blk + ov)
        out = sep.process(sample_rate=SR, samples=np.ascontiguousarray(x[:, s:e]))
        v = np.asarray(out.stems[0].data, np.float32)  # Voc_FT: stem 0 = vocals
        v = v.reshape(2, -1) if v.ndim == 2 else np.vstack([v, v])
        v = v[:, : e - s]
        w = np.ones(v.shape[1], np.float32)
        if s > 0:
            w[:ov] = np.linspace(0, 1, ov)
        if e < n:
            w[-ov:] = np.linspace(1, 0, ov)
        vocals[:, s:s + v.shape[1]] += v * w
        weight[s:s + v.shape[1]] += w
        print(f"  {e / SR:6.1f}s / {n / SR:.1f}s", flush=True)
        if e == n:
            break
        s = e - ov
    vocals /= np.maximum(weight, 1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sf.write(a.out, vocals.T, SR)
    print(f"[separate] {x.shape[1] / 44100:.1f}s of audio in {time.time() - t:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
