#!/usr/bin/env python3
"""
beats.py — beat grid, strong hits and an energy curve for syncing the visuals.

    python beats.py SONG.mp3 -o work/beats.json

Output:
  tempo        estimated BPM
  beats        beat times (s)
  downbeats    every 4th beat, phase chosen where the kicks are strongest
  hits         the strongest percussive onsets (good moments for flashes / shakes)
  energy       loudness curve, 10 values per second, normalised 0..1
"""
import argparse, json, os

import numpy as np
import librosa

SR = 22050
HOP = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="work/beats.json")
    a = ap.parse_args()

    y, _ = librosa.load(a.audio, sr=SR, mono=True)
    _, yp = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=HOP)
    tempo, bf = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)

    # downbeat phase: which of the 4 beat positions carries the most low-end energy
    S = np.abs(librosa.stft(yp, n_fft=2048, hop_length=HOP))
    low = S[: int(150 / (SR / 2048))].sum(0)
    if len(bf) >= 8:
        phase = int(np.argmax([low[bf[p::4]].mean() for p in range(4)]))
    else:
        phase = 0
    downbeats = beats[phase::4]

    # strong hits: onset peaks well above the local level, at least 0.25 s apart
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=20, post_avg=20, delta=env.std() * 1.2, wait=10)
    vals = env[pk]
    keep = pk[vals >= np.percentile(vals, 60)] if len(pk) else pk
    hits = librosa.frames_to_time(keep, sr=SR, hop_length=HOP)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=SR // 10)[0]
    rms = (rms - rms.min()) / (np.percentile(rms, 98) - rms.min() + 1e-9)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {
        "tempo": round(tempo, 1),
        "beats": [round(float(t), 3) for t in beats],
        "downbeats": [round(float(t), 3) for t in downbeats],
        "hits": [round(float(t), 3) for t in hits],
        "energy_fps": 10,
        "energy": [round(float(v), 3) for v in np.clip(rms, 0, 1)],
        "duration": round(len(y) / SR, 3),
    }
    json.dump(out, open(a.out, "w"), indent=None)
    print(f"[beats] tempo≈{out['tempo']} bpm, {len(beats)} beats, {len(hits)} hits -> {a.out}")


if __name__ == "__main__":
    main()
