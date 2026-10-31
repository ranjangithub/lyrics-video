# Models

Not stored in git (about 700 MB in total). Download once into this folder:

```bash
cd models

# Vocal separation (~67 MB)
curl -L -o UVR-MDX-NET-Voc_FT.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx

# Word timing: Omnilingual ASR, 1,600+ languages (~290 MB)
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2 | tar xj

# Person cut-outs for photos (~175 MB each)
curl -L -o isnet-general-use.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx
curl -L -o u2net_human_seg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx
```

Without the models the skill still works: `scripts/align.py` times lyrics from the audio alone (less precisely),
and photos are used without cut-outs.
