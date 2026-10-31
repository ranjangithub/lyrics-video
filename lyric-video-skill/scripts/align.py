#!/usr/bin/env python3
"""
align.py — match pasted lyrics to an MP3 and output word-level timings.

    python align.py SONG.mp3 LYRICS.txt -o work/words.json [--method auto|whisper|lrc|vocal]

Methods (auto picks the best one available):
  lrc     The lyrics already carry [mm:ss.xx] line timestamps (LRC format).
          Line starts are exact; words are spread inside each line.
  whisper Forced alignment with stable-ts (Whisper). Most accurate for real songs.
          Needs `pip install stable-ts` and network access to download model weights.
  vocal   No ML model. Finds sung phrases from the audio's harmonic/syllabic energy and
          fits the lyric lines onto them. Approximate: good on clean mixes, weaker on
          dense ones. Always available.

Always also writes a .lrc file next to the JSON so timings can be eyeballed and fixed,
then fed back in with --method lrc.
"""
import argparse, json, math, os, re, sys, unicodedata

import numpy as np

SR = 22050
HOP = 256
FPS_A = SR / HOP  # analysis frames per second (~86)

# ---------------------------------------------------------------- lyrics parsing
TS_RE = re.compile(r"\[(\d{1,2}):(\d{2}(?:[.:]\d{1,3})?)\]")


def parse_lyrics(text):
    """Return list of dicts {text, start(optional)}. Skips blank lines and [Chorus]-style tags."""
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        stamps = TS_RE.findall(raw)
        body = TS_RE.sub("", raw).strip()
        if not body:
            continue
        if re.fullmatch(r"[\[\(].*[\]\)]", body) and not stamps:  # [Chorus], (x2) …
            continue
        if stamps:
            for mm, ss in stamps:  # LRC allows several stamps for a repeated line
                lines.append({"text": body, "start": int(mm) * 60 + float(ss.replace(":", "."))})
        else:
            lines.append({"text": body})
    if any("start" in l for l in lines):
        if not all("start" in l for l in lines):
            sys.exit("Some lyric lines have [mm:ss] timestamps and some don't — add them to all or none.")
        lines.sort(key=lambda l: l["start"])
    return lines


def is_devanagari(ch):
    return "ऀ" <= ch <= "ॿ"


def syllables(word):
    """Rough syllable count for timing weights (Latin/Hinglish and Devanagari)."""
    w = word.lower()
    if any(is_devanagari(c) for c in w):
        n = 0
        for i, c in enumerate(w):
            cat = unicodedata.category(c)
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if "ऄ" <= c <= "औ":  # independent vowel
                n += 1
            elif "क" <= c <= "ह" or "क़" <= c <= "य़":  # consonant
                if nxt != "्":  # not followed by virama (half form)
                    n += 1
        return max(1, n)
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("ee", "le")):
        n -= 1
    return max(1, n)


def split_words(line):
    return [t for t in re.split(r"\s+", line.strip()) if re.search(r"\w", t)]


# ---------------------------------------------------------------- audio analysis
def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def vocal_score(y, clean=False):
    """Frame-wise 0..1 score of 'someone is singing here'.
    Mixed audio: harmonic energy in the vocal band x syllable-rate (3-9 Hz) modulation.
    clean=True (separated vocal stem): plain log energy in the vocal band."""
    import librosa
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d
    if clean:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
        e = np.log10(S[(freqs >= 150) & (freqs <= 5000)].sum(0) + 1e-6)
        lo, hi = np.percentile(e, 10), np.percentile(e, 99)
        e = np.clip((e - lo) / (hi - lo + 1e-9), 0, 1)
        e = gaussian_filter1d(e, 1.5)
        return e / (e.max() + 1e-9)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    H, P = librosa.decompose.hpss(S, margin=(2.0, 1.0))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = (freqs >= 250) & (freqs <= 3500)
    e = np.log1p(H[band].sum(0) / (P[band].sum(0) + H[band].sum(0) + 1e-9) * H[band].sum(0))
    e = (e - np.percentile(e, 5)) / (np.percentile(e, 98) - np.percentile(e, 5) + 1e-9)
    e = np.clip(e, 0, 1)

    # syllabic modulation: band-passed envelope fluctuation in ~3–9 Hz
    env = gaussian_filter1d(e, 1.5)
    slow = uniform_filter1d(env, int(FPS_A / 2.5))
    fast = uniform_filter1d(env, int(FPS_A / 12))
    mod = uniform_filter1d(np.abs(fast - slow), int(FPS_A * 0.4))
    mod = np.clip(mod / (np.percentile(mod, 95) + 1e-9), 0, 1)

    score = gaussian_filter1d(e * (0.35 + 0.65 * mod), 2)
    return score / (score.max() + 1e-9)


def regions(mask, min_len, merge_gap):
    """Boolean frame mask -> list of [start_frame, end_frame) regions."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def otsu(x):
    hist, edges = np.histogram(x, bins=64, range=(0, 1))
    w = hist.cumsum(); m = (hist * edges[:-1]).cumsum()
    tot_w, tot_m = w[-1], m[-1]
    best, thr = -1, 0.3
    for k in range(1, 63):
        w0, w1 = w[k], tot_w - w[k]
        if w0 == 0 or w1 == 0:
            continue
        mu0, mu1 = m[k] / w0, (tot_m - m[k]) / w1
        v = w0 * w1 * (mu0 - mu1) ** 2
        if v > best:
            best, thr = v, edges[k]
    return thr


# ---------------------------------------------------------------- method: vocal (no model)
def align_vocal(y, lines, clean=False):
    score = vocal_score(y, clean)
    thr = max(0.12, otsu(score) * 0.9)
    # phrases: bridge the dips between syllables (<0.2 s), drop blips (<0.2 s: drum bleed etc.)
    units = regions(score > thr, min_len=int(0.2 * FPS_A), merge_gap=int(0.2 * FPS_A))
    if not units:
        sys.exit("Could not find any sung phrases in the audio.")

    W = [sum(syllables(w) for w in split_words(l["text"])) for l in lines]
    L = len(lines)

    def split_at_dip(k):
        a, b = units[k]
        lo, hi = a + (b - a) // 5, b - (b - a) // 5
        cut = lo + int(np.argmin(score[lo:hi]))
        units[k:k + 1] = [[a, cut], [cut, b]]

    # a phrase far longer than any lyric line probably holds two lines sung back-to-back
    rate0 = sum(u[1] - u[0] for u in units) / FPS_A / sum(W)
    max_line = rate0 * max(W) * FPS_A
    k = 0
    while k < len(units):
        if units[k][1] - units[k][0] > 1.7 * max_line and units[k][1] - units[k][0] > 16:
            split_at_dip(k)
        else:
            k += 1
    # and we need at least as many phrases as lines
    while len(units) < L:
        k = max(range(len(units)), key=lambda i: units[i][1] - units[i][0])
        if units[k][1] - units[k][0] < 8:
            break
        split_at_dip(k)

    U = len(units)
    ust = np.array([u[0] for u in units]) / FPS_A
    uen = np.array([u[1] for u in units]) / FPS_A
    udur = uen - ust
    mean_u = udur.mean()

    def solve(rate):
        INF = 1e18
        dp = np.full((L + 1, U + 1), INF); bp = {}
        dp[0][0] = 0
        for j in range(1, U + 1):  # leading skips
            dp[0][j] = dp[0][j - 1] + 1.2 * udur[j - 1] / mean_u
            bp[(0, j)] = ("skip", j - 1)
        for i in range(1, L + 1):
            exp = rate * W[i - 1]
            for j in range(1, U + 1):
                # option A: phrase j is skipped (noise / ad-lib / instrument)
                best = dp[i][j - 1] + 1.2 * udur[j - 1] / mean_u
                arg = ("skip", j - 1)
                # option B: line i covers units k..j-1
                for k in range(j - 1, max(-1, j - 9), -1):
                    if dp[i - 1][k] >= INF:
                        continue
                    span = uen[j - 1] - ust[k]
                    voiced = udur[k:j].sum()
                    gaps = span - voiced
                    c = math.log(max(span, 0.05) / exp) ** 2 + 0.8 * gaps / max(exp, 0.3)
                    # big silence before this line is natural; tiny one hints a line break
                    tot = dp[i - 1][k] + c
                    if tot < best:
                        best, arg = tot, ("take", k)
                dp[i][j] = best; bp[(i, j)] = arg
        # backtrack (allow trailing skips)
        j = int(np.argmin([dp[L][jj] + 1.2 * udur[jj:].sum() / mean_u for jj in range(U + 1)]))
        i, out = L, [None] * L
        while i > 0:
            kind, k = bp[(i, j)]
            if kind == "skip":
                j = k
            else:
                out[i - 1] = (k, j)
                i -= 1; j = k
        return out

    rate = udur.sum() / sum(W)
    for _ in range(3):  # re-estimate seconds-per-syllable from the chosen assignment
        assign = solve(rate)
        spans = [(uen[b - 1] - ust[a]) for a, b in assign]
        rate = float(np.median([s / w for s, w in zip(spans, W)]))

    onsets = vocal_onsets(score)
    result = []
    for li, (a, b) in enumerate(assign):
        segs = [(ust[k], uen[k]) for k in range(a, b)]
        result.append(spread_words(lines[li]["text"], segs, onsets))
    return result, "vocal"


def vocal_onsets(score):
    d = np.diff(score, prepend=0)
    peaks = [i for i in range(1, len(d) - 1) if d[i] > 0.03 and d[i] >= d[i - 1] and d[i] >= d[i + 1]]
    return np.array(peaks) / FPS_A


def spread_words(text, segs, onsets=None, snap=0.09):
    """Distribute words over the voiced segments in proportion to syllables."""
    words = split_words(text)
    wts = np.array([syllables(w) for w in words], float)
    total_voiced = sum(e - s for s, e in segs)
    cum = np.concatenate([[0], np.cumsum(wts)]) / wts.sum() * total_voiced

    def at(v):  # map voiced-time offset -> absolute time
        for s, e in segs:
            if v <= e - s + 1e-9:
                return s + v
            v -= e - s
        return segs[-1][1]

    out = []
    for i, w in enumerate(words):
        st, en = at(cum[i]), at(cum[i + 1])
        if onsets is not None and len(onsets) and i > 0:
            k = np.argmin(np.abs(onsets - st))
            if abs(onsets[k] - st) < snap and onsets[k] > out[-1]["start"] + 0.06:
                st = float(onsets[k])
        out.append({"text": w, "start": round(float(st), 3), "end": round(float(en), 3)})
    for i in range(len(out) - 1):
        out[i]["end"] = max(out[i]["start"] + 0.05, min(out[i]["end"], out[i + 1]["start"]))
    return {"text": text, "start": out[0]["start"], "end": out[-1]["end"], "words": out}


# ---------------------------------------------------------------- method: lrc
def align_lrc(y, lines, duration):
    score = vocal_score(y) if y is not None else None
    res = []
    for i, l in enumerate(lines):
        st = l["start"]
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else duration
        est = 0.2 * sum(syllables(w) for w in split_words(l["text"])) + 0.4
        en = min(nxt - 0.05, st + max(est * 1.8, 1.0))
        if score is not None:  # end the line where singing actually stops
            a, b = int(st * FPS_A), int(min(nxt, duration) * FPS_A)
            seg = score[a:b]
            if len(seg) > 10:
                thr = max(0.12, otsu(score) * 0.9)
                on = np.where(seg > thr)[0]
                if len(on):
                    en = min(en, st + (on[-1] + 1) / FPS_A + 0.1) if on[-1] > len(seg) * 0.2 else en
        res.append(spread_words(l["text"], [(st, max(en, st + 0.3))]))
    return res, "lrc"


# ---------------------------------------------------------------- method: whisper (stable-ts)
def align_whisper(path, lines, model_name, language):
    import stable_whisper  # noqa
    model = stable_whisper.load_model(model_name)
    text = "\n".join(l["text"] for l in lines)
    r = model.align(path, text, language=language)
    aw = [w for seg in r.segments for w in seg.words]
    res, k = [], 0
    for l in lines:
        ws = []
        for w in split_words(l["text"]):
            if k < len(aw):
                ws.append({"text": w, "start": round(aw[k].start, 3), "end": round(aw[k].end, 3)})
                k += 1
        if len(ws) != len(split_words(l["text"])):
            raise RuntimeError("whisper word count mismatch")
        res.append({"text": l["text"], "start": ws[0]["start"], "end": ws[-1]["end"], "words": ws})
    return res, "whisper"


# ---------------------------------------------------------------- output helpers
def fmt_lrc(t):
    m = int(t // 60)
    return f"[{m:02d}:{t - 60 * m:05.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("lyrics", help="text file with one lyric line per line (plain or LRC)")
    ap.add_argument("-o", "--out", default="work/words.json")
    ap.add_argument("--method", default="auto", choices=["auto", "whisper", "lrc", "vocal"])
    ap.add_argument("--model", default="base", help="whisper model size for --method whisper")
    ap.add_argument("--lang", default="hi", help="language code for whisper (hi, en, pa, …)")
    ap.add_argument("--vocals", help="separated vocal stem (from separate.py) - much better vocal-method timing")
    a = ap.parse_args()

    lines = parse_lyrics(open(a.lyrics, encoding="utf-8").read())
    if not lines:
        sys.exit("No lyric lines found.")
    y = load_audio(a.audio)
    duration = len(y) / SR
    has_ts = "start" in lines[0]

    method = a.method
    res = None
    if method == "lrc" or (method == "auto" and has_ts):
        if not has_ts:
            sys.exit("--method lrc needs [mm:ss.xx] timestamps on every line.")
        res, used = align_lrc(y, lines, duration)
    elif method in ("whisper", "auto"):
        try:
            res, used = align_whisper(a.audio, lines, a.model, a.lang)
        except Exception as e:  # not installed, weights blocked, mismatch…
            if method == "whisper":
                sys.exit(f"Whisper alignment failed: {e}")
            print(f"[align] whisper unavailable ({type(e).__name__}: {str(e)[:120]}); "
                  f"falling back to vocal-energy alignment", file=sys.stderr)
    if res is None:
        if a.vocals:
            res, used = align_vocal(load_audio(a.vocals), lines, clean=True)
            used = "vocal-stem"
        else:
            res, used = align_vocal(y, lines)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {"audio": os.path.abspath(a.audio), "duration": round(duration, 3), "method": used,
           "accuracy": {"lrc": "line starts exact", "whisper": "word-level",
                        "vocal": "approximate — review the .lrc",
                        "vocal-stem": "good line timing from isolated vocals — spot-check"}[used],
           "lines": res}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lrc = os.path.splitext(a.out)[0] + ".lrc"
    with open(lrc, "w", encoding="utf-8") as f:
        for l in res:
            f.write(f"{fmt_lrc(l['start'])}{l['text']}\n")
    print(f"[align] method={used}  lines={len(res)}  -> {a.out}, {lrc}")
    for l in res:
        print(f"  {l['start']:7.2f}–{l['end']:7.2f}  {l['text']}")


if __name__ == "__main__":
    main()
