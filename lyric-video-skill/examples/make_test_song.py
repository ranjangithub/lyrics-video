"""Original test lyrics (written for this repo) sung by espeak-ng over synthetic drums. Usage: python make_test_song.py OUT_DIR"""
"""Build a ~30s test 'song': synthetic drums + bass + TTS vocal lines at known times.
Writes test_song.mp3, test_lyrics.txt, ground_truth.json"""
import json, subprocess, sys, numpy as np, soundfile as sf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tts

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SR = tts.init(b"hi", rate=135, pitch=35)
bpm = 100; beat = 60 / bpm; bar = 4 * beat
dur = 34.0
n = int(dur * SR)
mix = np.zeros(n, np.float32)
rng = np.random.default_rng(1)

def add(sig, t, g=1.0):
    i = int(t * SR); j = min(n, i + len(sig))
    if i < n: mix[i:j] += g * sig[: j - i]

tt = np.arange(int(0.35 * SR)) / SR
kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
sn_t = np.arange(int(0.2 * SR)) / SR
snare = rng.normal(0, 1, len(sn_t)) * np.exp(-sn_t * 22) * 0.5 + np.sin(2*np.pi*190*sn_t)*np.exp(-sn_t*25)*0.4
hat = rng.normal(0, 1, int(0.05 * SR)) * np.exp(-np.arange(int(0.05*SR)) / SR * 80) * 0.15
t = 0.0; k = 0
while t < dur - 0.4:
    add(kick, t, 0.9)
    if k % 4 in (1, 3): add(snare, t, 0.7)
    add(hat, t); add(hat, t + beat / 2)
    t += beat; k += 1
# bass drone (D) following bars, ducks a bit
tb = np.arange(n) / SR
bass = 0.18 * np.sin(2 * np.pi * 73.4 * tb) * (0.6 + 0.4 * np.cos(2 * np.pi * tb / bar) ** 2)
mix += bass.astype(np.float32)

lines = [  # (romanized lyric the user would paste, devanagari for TTS, start time)
    ("Hum rukne waale nahin",        "हम रुकने वाले नहीं",        bar * 2),
    ("Toofan se takraayenge",        "तूफ़ान से टकराएँगे",          bar * 3),
    ("Mitti se uthe hain hum",       "मिट्टी से उठे हैं हम",        bar * 4),
    ("Aasmaan jhuka denge",          "आसमान झुका देंगे",          bar * 5),
    ("Ek awaaz, ek junoon",          "एक आवाज़, एक जुनून",        bar * 7),
    ("Ab rok sake to rok",           "अब रोक सके तो रोक",         bar * 8),
    ("Hum rukne waale nahin",        "हम रुकने वाले नहीं",         bar * 9),
]
truth = []
for rom, dev, st in lines:
    v = tts.say(dev)
    # trim silence
    a = np.where(np.abs(v) > 0.02)[0]
    v = v[a[0]: a[-1]]
    add(v * 0.9, st)
    truth.append({"line": rom, "start": round(st, 3), "end": round(st + len(v) / SR, 3)})

mix /= np.abs(mix).max() * 1.05
sf.write(f"{OUT}/test_song.wav", mix, SR)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/test_song.wav", "-b:a", "192k", f"{OUT}/test_song.mp3"], check=True)
open(f"{OUT}/test_lyrics.txt", "w").write("\n".join(l[0] for l in lines) + "\n")
json.dump(truth, open(f"{OUT}/ground_truth.json", "w"), indent=1)
print(json.dumps(truth, indent=1))
