#!/usr/bin/env python3
"""
ctc_align.py — word-level lyric timing with Meta's Omnilingual ASR (CTC, 1600+ languages,
including Hindi in Devanagari and English), run on the isolated vocal stem via onnxruntime.

    python ctc_align.py VOCALS.wav LYRICS.txt -o work/words.json --audio SONG.mp3
    python ctc_align.py VOCALS.wav --transcribe          # just print what it hears (sanity check)

Model (≈290 MB, int8, CPU):
    curl -L -o omni.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
    tar xjf omni.tar.bz2 -C models/

How it works: the model gives per-20 ms character probabilities; Viterbi forced alignment
finds the most likely moment each lyric character is sung. Between lines we allow a free
"anything" state so ad-libs, humming or unlisted backing vocals don't drag the alignment.
Lyrics must be written in a script the model knows (Devanagari, Latin, …).
"""
import argparse, glob, json, os, re, subprocess, sys, unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
FRAME = 0.02  # seconds per CTC frame


def find_model():
    c = glob.glob(os.path.join(HERE, "..", "models", "sherpa-onnx-omnilingual-*", "model*.onnx"))
    return c[0] if c else None


def load16k(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def logits_for(path, model, block=10.0, ctx=1.5):
    """Log-probs [T, V] for the whole file, computed in overlapping blocks."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    x = load16k(path)
    n = len(x)
    B, C = int(block * SR), int(ctx * SR)
    T = int(np.ceil(n / (FRAME * SR)))
    out = None
    s = 0
    while s < n:
        a, b = max(0, s - C), min(n, s + B + C)
        seg = x[a:b]
        seg = (seg - seg.mean()) / (seg.std() + 1e-7)
        lg = sess.run(None, {"x": seg[None].astype(np.float32)})[0][0]
        if out is None:
            out = np.full((T, lg.shape[1]), -1e4, np.float32)
        f0 = int(round((s - a) / (FRAME * SR)))
        nf = min(int(round(min(B, n - s) / (FRAME * SR))), T - int(round(s / (FRAME * SR))))
        g0 = int(round(s / (FRAME * SR)))
        nf = min(nf, lg.shape[0] - f0)
        out[g0:g0 + nf] = lg[f0:f0 + nf]
        print(f"  [ctc] {min(n, s + B) / SR:6.1f}s / {n / SR:.1f}s", file=sys.stderr, flush=True)
        s += B
    return out


def load_vocab(model):
    tok = {}
    for line in open(os.path.join(os.path.dirname(model), "tokens.txt"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        t, i = line.rsplit(" ", 1)
        tok[t] = int(i)
    return tok


def norm_text(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\sऀ-ॿ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def greedy(lp, tok):
    inv = {i: t for t, i in tok.items()}
    ids = lp.argmax(1)
    out, prev = [], -1
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            out.append((t * FRAME, inv.get(int(i), "?")))
        prev = i
    return out


# Hindi singing is often emitted in Urdu or Gurmukhi script by this model. A Devanagari lyric
# character therefore matches its look-alike sound in those scripts too.
URDU = {
    "क": "کك", "ख": "کخ", "ग": "گ", "घ": "گ", "च": "چ", "छ": "چ", "ज": "جز", "झ": "ج", "ट": "ٹت",
    "ठ": "ٹ", "ड": "ڈد", "ढ": "ڈ", "ण": "ن", "त": "تط", "थ": "ت", "द": "دڈ", "ध": "د", "न": "نں",
    "प": "پ", "फ": "پف", "ब": "ب", "भ": "ب", "म": "م", "य": "یيے", "र": "رڑ", "ल": "ل", "व": "وف",
    "श": "ش", "ष": "ش", "स": "سصث", "ह": "ہھهح", "ा": "اآ", "ी": "یي", "ि": "یي", "ु": "و", "ू": "و",
    "े": "ےیي", "ै": "ےی", "ो": "و", "ौ": "و", "ं": "نں", "ँ": "ں", "अ": "اآ", "आ": "آا", "इ": "ا",
    "ई": "ای", "उ": "او", "ऊ": "او", "ए": "ا", "ओ": "او", "औ": "او", "ऐ": "ا", "ृ": "ر",
}
SILENT = {"\u094D", "\u093C"}  # virama, nukta


def equiv_ids(ch, tok):
    ids = [tok[ch]] if ch in tok else []
    if "\u0900" <= ch <= "\u097F":
        g = chr(ord(ch) + 0x100)  # Gurmukhi block mirrors Devanagari
        if g in tok:
            ids.append(tok[g])
        for u in URDU.get(ch, ""):
            if u in tok:
                ids.append(tok[u])
    return sorted(set(ids))


def force_align(lp, lines, tok, garbage_pen=1.5, skip_pen=12.0):
    """Viterbi CTC alignment of every lyric character. Line gaps get a 'garbage' state that may
    absorb anything (ad-libs, humming), so one mis-heard section doesn't drag the rest.
    A whole lyric line may also be skipped (cost skip_pen) when it isn't actually sung.
    Returns words_meta [(line, word)] and span {word_id: (first_frame, last_frame)}."""
    seq, word_of, line_of, words_meta = [], [], [], []
    cache = {}
    for li, l in enumerate(lines):
        for wi, w in enumerate(norm_text(l).split(" ")):
            wid = len(words_meta); words_meta.append((li, wi))
            for ch in w:
                if ch in SILENT:
                    continue
                if ch not in cache:
                    cache[ch] = equiv_ids(ch, tok)
                if cache[ch]:
                    seq.append(ch); word_of.append(wid); line_of.append(li)
    L, T = len(seq), lp.shape[0]
    uniq = sorted(set(seq))
    col = {c: i for i, c in enumerate(uniq)}
    E = np.stack([np.logaddexp.reduce(lp[:, cache[c]], axis=1) for c in uniq], 1)  # [T, U]
    blank = lp[:, 0]
    garb = lp.max(1) - garbage_pen
    S = 2 * L + 1
    lab_col = np.array([col[c] for c in seq])
    free = np.zeros(S, bool); free[0] = free[-1] = True
    for k in range(L - 1):
        if line_of[k] != line_of[k + 1]:
            free[2 * k + 2] = True
    can_skip = np.zeros(S, bool)
    for k in range(1, L):
        if seq[k] != seq[k - 1]:
            can_skip[2 * k + 1] = True
    # leading blank of every line, plus the final blank: line i may be skipped by jumping
    # from line i's leading blank to line i+1's leading blank
    starts = [0] + [2 * k for k in range(1, L) if line_of[k] != line_of[k - 1]] + [S - 1]
    j_dst = np.array(starts[1:]); j_src = np.array(starts[:-1])
    NEG = np.float32(-1e9)
    bp = np.zeros((T, S), np.int16)  # how many states back the best predecessor is

    def emis(t):
        e = np.empty(S, np.float32)
        e[0::2] = np.where(free[0::2], np.maximum(blank[t], garb[t]), blank[t])
        e[1::2] = E[t, lab_col]
        return e

    dp = np.full(S, NEG, np.float32)
    e = emis(0); dp[0], dp[1] = e[0], e[1]
    for t in range(1, T):
        m1 = np.empty_like(dp); m1[0] = NEG; m1[1:] = dp[:-1]
        m2 = np.full_like(dp, NEG); m2[2:] = dp[:-2]; m2[~can_skip] = NEG
        best = np.maximum(dp, np.maximum(m1, m2))
        back = np.where(best == dp, 0, np.where(best == m1, 1, 2)).astype(np.int16)
        jump = dp[j_src] - skip_pen
        better = jump > best[j_dst]
        best[j_dst[better]] = jump[better]
        back[j_dst[better]] = (j_dst - j_src)[better]
        bp[t] = back
        dp = best + emis(t)
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    span = {}
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            wid = word_of[s // 2]
            a_, b_ = span.get(wid, (t, t))
            span[wid] = (min(a_, t), max(b_, t))
        s -= int(bp[t, s])
    return words_meta, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocals")
    ap.add_argument("lyrics", nargs="?")
    ap.add_argument("-o", "--out", default="work/words.json")
    ap.add_argument("--audio", help="original song (path stored for rendering); defaults to the vocals file")
    ap.add_argument("--model", default=find_model())
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--garbage-pen", type=float, default=4.0, help="cost per frame of unexplained singing between lines")
    ap.add_argument("--skip-pen", type=float, default=30.0, help="cost of leaving a lyric line out entirely")
    ap.add_argument("--cache", default=None, help="npy file to cache logits")
    a = ap.parse_args()
    if not a.model:
        sys.exit("Omnilingual model not found under models/ — see the download command at the top.")

    tok = load_vocab(a.model)
    cache = a.cache or os.path.splitext(a.out)[0] + ".logits.npy"
    if os.path.exists(cache):
        lp = np.load(cache)
    else:
        lp = logits_for(a.vocals, a.model)
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.save(cache, lp)

    lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    sys.path.insert(0, HERE)
    from align import parse_lyrics, split_words
    lines = parse_lyrics(open(a.lyrics, encoding="utf-8").read()) if a.lyrics else []

    if a.transcribe or not a.lyrics:
        g = greedy(lp, tok)
        line, t0 = "", None
        for t, c in g:
            if t0 is None: t0 = t
            if c == " " and line and t - last > 0.6:
                print(f"{t0:7.2f}  {line}"); line, t0 = "", None; last = t; continue
            line += c; last = t
        if line: print(f"{t0:7.2f}  {line}")
        return

    texts = [l["text"] for l in lines]
    meta, span = force_align(lp, texts, tok, a.garbage_pen, a.skip_pen)

    res, skipped = [], []
    wid = 0
    for li, l in enumerate(texts):
        disp = split_words(l)
        nn = len(norm_text(l).split(" "))
        ws = []
        for wi in range(nn):
            a_, b_ = span.get(wid, (None, None)); wid += 1
            if a_ is None:
                continue
            ws.append({"start": round(a_ * FRAME, 3), "end": round((b_ + 1) * FRAME + 0.04, 3)})
        if not ws:
            skipped.append(l)
            continue
        if len(ws) == len(disp):
            for w, d in zip(ws, disp): w["text"] = d
        else:  # punctuation-only tokens etc.: spread display words over the found span
            st, en = ws[0]["start"], ws[-1]["end"]
            k = len(disp)
            ws = [{"text": d, "start": round(st + (en - st) * i / k, 3), "end": round(st + (en - st) * (i + 1) / k, 3)}
                  for i, d in enumerate(disp)]
        for i in range(len(ws) - 1):
            ws[i]["end"] = min(ws[i]["end"], ws[i + 1]["start"])
        res.append({"text": l, "start": ws[0]["start"], "end": ws[-1]["end"], "words": ws})

    dur = lp.shape[0] * FRAME
    out = {"audio": os.path.abspath(a.audio or a.vocals), "duration": round(dur, 3), "method": "ctc",
           "accuracy": "word-level (forced alignment)", "lines": res}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.splitext(a.out)[0] + ".lrc", "w", encoding="utf-8") as f:
        for l in res:
            m = int(l["start"] // 60)
            f.write(f"[{m:02d}:{l['start'] - 60 * m:05.2f}]{l['text']}\n")
    print(f"[ctc_align] {len(res)} lines -> {a.out}")
    if skipped:
        print(f"[ctc_align] {len(skipped)} lyric line(s) not found in the singing and left out:")
        for l in skipped:
            print(f"    - {l}")
    for l in res:
        print(f"  {l['start']:7.2f}–{l['end']:7.2f}  {l['text']}")


if __name__ == "__main__":
    main()
