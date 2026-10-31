#!/usr/bin/env python3
"""
make_plan.py — turn aligned words into a default style plan (the renderer's input).

    python make_plan.py work/words.json work/beats.json -o work/style_plan.json \
        [--title "SONG NAME"] [--subtitle "line under the title"] [--hd]

The plan is a plain JSON file meant to be edited (by Claude or by hand) before rendering:
which words go together on screen, which ones are hero words, and how each chunk animates.
See references/style_plan.md for every field.
"""
import argparse, json, re


def display(text):
    """Latin text is shown upper-case; Devanagari is left alone."""
    return text.upper()


def syl(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    return max(1, len(re.findall(r"[aeiouy]+", w))) if w else 2


def chunk_line(words, max_words=2):
    """Group a line's words into on-screen chunks of 1–3 words, ~0.35 s minimum."""
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        dur = cur[-1]["end"] - cur[0]["start"]
        short = all(len(x["text"]) <= 3 for x in cur)
        limit = 3 if short else max_words
        if len(cur) >= limit and dur >= 0.3:
            chunks.append(cur); cur = []
    if cur:
        if chunks and (cur[-1]["end"] - cur[0]["start"] < 0.25) and len(chunks[-1]) + len(cur) <= 3:
            chunks[-1] += cur
        else:
            chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words")
    ap.add_argument("beats")
    ap.add_argument("-o", "--out", default="work/style_plan.json")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--hd", action="store_true", help="1920x1080 instead of 1280x720")
    ap.add_argument("--max-words", type=int, default=2)
    a = ap.parse_args()

    W = json.load(open(a.words, encoding="utf-8"))
    B = json.load(open(a.beats))
    lines = W["lines"]

    seen = {}
    for l in lines:
        key = re.sub(r"\W+", " ", l["text"].lower()).strip()
        seen[key] = seen.get(key, 0) + 1

    intro_anims = ["punch", "glitch", "rise", "punch", "glitch"]
    chunks = []
    for li, l in enumerate(lines):
        key = re.sub(r"\W+", " ", l["text"].lower()).strip()
        is_hook = seen[key] > 1
        groups = chunk_line(l["words"], a.max_words)
        for gi, g in enumerate(groups):
            last = gi == len(groups) - 1
            # payoff word of every other line, and the whole hook when it repeats
            hero = (last and (is_hook or li % 2 == 1)) or (is_hook and gi == 0)
            if hero:
                anim = "slam"
            elif len(g) >= 3:
                anim = "stamp"
            else:
                anim = intro_anims[(li + gi) % len(intro_anims)]
            chunks.append({
                "text": display(" ".join(w["text"] for w in g)),
                "start": round(g[0]["start"] - 0.05, 3),
                "end": round(g[-1]["end"], 3),
                "anim": anim,
                "emphasis": "hero" if hero else "normal",
                "color": "red" if (hero and is_hook) else "white",
                "words": [{"text": display(w["text"]), "start": w["start"]} for w in g],
            })

    # hold each chunk on screen until the next one (but not through long instrumental gaps)
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1]["start"] if i + 1 < len(chunks) else W["duration"]
        c["end"] = round(min(nxt - 0.02, max(c["end"] + 0.6, c["start"] + 0.5)), 3)

    first = chunks[0]["start"] if chunks else W["duration"]
    title = None
    if a.title and first > 2.5:
        title = {"text": a.title.upper(), "sub": a.subtitle.upper(), "start": 0.3, "end": round(first - 0.25, 3)}

    plan = {
        "video": {
            "width": 1920 if a.hd else 1280, "height": 1080 if a.hd else 720, "fps": 30,
            "theme": "crimson-grunge", "font": "anton",
            "shake_on_hits": True, "flash_on_slam": True,
        },
        "title_card": title,
        "chunks": chunks,
        "beats": B["beats"], "hits": B["hits"], "energy": B["energy"], "energy_fps": B["energy_fps"],
        "audio": W["audio"], "duration": W["duration"],
    }
    json.dump(plan, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[plan] {len(chunks)} chunks ({sum(c['emphasis']=='hero' for c in chunks)} hero) -> {a.out}")


if __name__ == "__main__":
    main()
