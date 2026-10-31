#!/usr/bin/env python3
"""
render.py — render a dramatic kinetic-typography lyric video from a style plan.

    python render.py work/style_plan.json -o out/lyric_video.mp4 [--workers 2]
    python render.py work/style_plan.json --stills 5.2,9.8,14.1 -o work/stills   # PNG checks
    python render.py work/style_plan.json --range 4 12 -o out/preview.mp4       # quick preview

Pure Python (Pillow + NumPy) piped into ffmpeg. No browser, no GPU.
"""
import argparse, json, math, os, random, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props as propslib  # noqa: E402
import media as medialib  # noqa: E402
import scenes as sceneslib  # noqa: E402
import comp as complib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "assets", "fonts")
FONT_FILES = {
    "anton": "Anton-Latin.woff",
    "anton-ext": "Anton-LatinExt.woff",
    "bebas": "BebasNeue-Latin.woff",
    "teko": "Teko-Latin-Bold.woff",
    "devanagari": "Teko-Devanagari-Bold.woff",
}

THEMES = {
    # base dark, base light, smoke tint, ember colours, text fills, outline, glow
    "crimson-grunge": dict(base0=(10, 7, 7), base1=(58, 34, 30), smoke=(170, 18, 22),
                           embers=[(255, 70, 40), (255, 140, 60), (255, 230, 220)],
                           white=(255, 228, 226), red=(236, 34, 38), gold=(255, 196, 70),
                           outline=(26, 4, 6), glow=(220, 20, 30), flash=(255, 60, 50)),
    "midnight-steel": dict(base0=(6, 8, 12), base1=(30, 38, 52), smoke=(30, 90, 170),
                           embers=[(120, 190, 255), (200, 230, 255), (255, 255, 255)],
                           white=(236, 242, 255), red=(90, 170, 255), gold=(255, 210, 120),
                           outline=(4, 8, 18), glow=(40, 120, 255), flash=(140, 200, 255)),
    "saffron-fire": dict(base0=(14, 8, 2), base1=(70, 40, 12), smoke=(220, 110, 10),
                         embers=[(255, 170, 40), (255, 220, 120), (255, 255, 230)],
                         white=(255, 244, 222), red=(255, 120, 20), gold=(255, 210, 90),
                         outline=(30, 10, 0), glow=(255, 110, 0), flash=(255, 170, 60)),
}


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def ease_in_quad(x):
    x = clamp(x); return x * x


def ease_out_back(x, s=2.2):
    x = clamp(x) - 1; return x * x * ((s + 1) * x + s) + 1


# ------------------------------------------------------------------ procedural textures
def value_noise(h, w, scale, rng, octaves=4):
    out = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        sh, sw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
        small = rng.random((sh, sw)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(img, np.float32) / 255
        tot += amp; amp *= 0.5; scale /= 2.2
    return out / tot


def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


class Scene:
    def __init__(self, plan, seed=7):
        v = plan["video"]
        self.W, self.H, self.fps = v["width"], v["height"], v["fps"]
        self.theme = THEMES.get(v.get("theme", "crimson-grunge"), THEMES["crimson-grunge"])
        self.plan = plan
        rng = np.random.default_rng(seed)
        W, H, th = self.W, self.H, self.theme
        pad = 1.18
        self.BW, self.BH = int(W * pad), int(H * pad)

        # grunge base: concrete-like texture, darker toward the edges
        n = value_noise(self.BH, self.BW, self.BW / 6, rng, 5)
        speck = (rng.random((self.BH, self.BW)) > 0.996).astype(np.float32)
        speck = np.asarray(Image.fromarray((speck * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        n = norm(n) ** 1.6 * 0.85 + speck * 0.35
        b0, b1 = np.array(th["base0"], np.float32), np.array(th["base1"], np.float32)
        self.base = (b0 + (b1 - b0) * n[..., None]).astype(np.float32)

        # two smoke fields that drift in opposite directions (parallax)
        self.smoke = []
        for k in range(2):
            s = norm(value_noise(self.BH, self.BW, self.BW / (2.5 + k), rng, 4))
            s = np.clip((s - 0.45) * 2.4, 0, 1) ** 1.5
            self.smoke.append(s.astype(np.float32))

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * r ** 1.8, 0.15, 1)[..., None].astype(np.float32)

        self.grain = [(rng.normal(0, 3.5, (H // 2, W // 2)).astype(np.float32)) for _ in range(6)]

        # distress texture for letters: mostly opaque, with worn holes and red speckle
        d = norm(value_noise(H, W, W / 34, rng, 4))
        fine = rng.random((H, W)).astype(np.float32)
        self.wear = np.clip(1.0 - np.clip((d - 0.7) * 5, 0, 1) - (fine > 0.992) * 0.9, 0.0, 1.0)
        self.spots = np.clip((norm(value_noise(H, W, W / 30, rng, 3)) - 0.55) * 3, 0, 1) * (fine > 0.45)

        # particles
        prng = random.Random(seed)
        self.particles = []
        for _ in range(int(140 * W / 1280)):
            self.particles.append(dict(
                x=prng.random() * W, y=prng.random() * H, vy=-(12 + prng.random() * 55) * H / 720,
                vx=(prng.random() - 0.3) * 25, r=0.8 + prng.random() ** 3 * 3.2,
                c=prng.choice(th["embers"]), ph=prng.random() * 6.28, fl=2 + prng.random() * 6,
                streak=prng.random() < 0.12))

        self.font_cache = {}
        self.sprite_cache = {}

    # -------------------------------------------------------------- fonts / sprites
    def font_for(self, text, size):
        name = self.plan["video"].get("font", "anton")
        if any("ऀ" <= c <= "ॿ" for c in text):
            name = "devanagari"
        elif name == "anton" and any(ord(c) > 127 and c not in "·•—–’‘“”…×" for c in text):
            name = "anton-ext"
        key = (name, size)
        if key not in self.font_cache:
            self.font_cache[key] = ImageFont.truetype(os.path.join(FONTS, FONT_FILES[name]), size)
        return self.font_cache[key]

    def layout(self, chunk):
        """Choose font size and per-word positions (1 or 2 rows) to fill the frame boldly."""
        words = chunk["words"] or [{"text": chunk["text"], "start": chunk["start"]}]
        hero = chunk.get("emphasis") == "hero"
        scale = float(chunk.get("size", 1.0))
        max_w = self.W * (0.88 if hero else 0.8)
        max_h = self.H * (0.42 if hero else 0.27) * scale
        texts = [w["text"] for w in words]

        def measure(size, rows):
            f = self.font_for(" ".join(texts), size)
            space = f.getlength(" ") * 0.9
            widths = [f.getlength(t) for t in texts]
            row_w = [sum(widths[i] for i in r) + space * (len(r) - 1) for r in rows]
            asc, desc = f.getmetrics()
            return f, widths, row_w, asc + desc, space

        def best_for(rows):
            lo, hi = 20, int(self.H * 0.9)
            while hi - lo > 2:
                mid = (lo + hi) // 2
                f, _, row_w, lh, _ = measure(mid, rows)
                tot_h = lh * len(rows) * 0.92
                if max(row_w) <= max_w and tot_h <= max_h * len(rows) ** 0.35:
                    lo = mid
                else:
                    hi = mid
            return lo

        options = [[list(range(len(texts)))]]
        if len(texts) >= 2:
            for k in range(1, len(texts)):
                options.append([list(range(k)), list(range(k, len(texts)))])
        scored = [(best_for(r) * (1.0 if len(r) == 1 else 0.86), r) for r in options]
        _, rows = max(scored, key=lambda s: s[0])
        size = best_for(rows)
        f, widths, row_w, lh, space = measure(size, rows)
        line_h = lh * 0.92
        total_h = line_h * len(rows)
        # centre on the actual ink, not the font's ascender box (matters for Devanagari)
        asc, desc = f.getmetrics()
        bb = f.getbbox(" ".join(texts))
        ink_shift = (asc + desc) / 2 - (bb[1] + bb[3]) / 2
        pos = [None] * len(texts)
        for ri, r in enumerate(rows):
            x = (self.W - row_w[ri]) / 2
            y = (self.H - total_h) / 2 + ri * line_h + ink_shift
            for i in r:
                pos[i] = (x, y, widths[i]); x += widths[i] + space
        return f, size, pos, words

    def word_sprite(self, text, size, color_key, hero, seed, clean=False):
        key = (text, size, color_key, hero, clean)
        if key in self.sprite_cache:
            return self.sprite_cache[key]
        th = self.theme
        f = self.font_for(text, size)
        stroke = max(2, int(size * 0.035))
        glow_r = int(size * (0.22 if hero else 0.14))
        pad = glow_r * 2 + stroke * 2
        tw = int(f.getlength(text)) + 2 * pad
        asc, desc = f.getmetrics()
        th_ = asc + desc + 2 * pad
        mask = Image.new("L", (tw, th_), 0)
        ImageDraw.Draw(mask).text((pad, pad), text, font=f, fill=255)
        outline = Image.new("L", (tw, th_), 0)
        ImageDraw.Draw(outline).text((pad, pad), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)

        m = np.asarray(mask, np.float32) / 255
        o = np.asarray(outline, np.float32) / 255
        rng = np.random.default_rng(seed)
        oy = rng.integers(0, max(1, self.H - th_)) if self.H > th_ else 0
        ox = rng.integers(0, max(1, self.W - tw)) if self.W > tw else 0

        rows = (oy + np.arange(th_)) % self.H
        cols = (ox + np.arange(tw)) % self.W

        def crop(tex):  # wrap-around crop so long words never run off the texture
            return tex[np.ix_(rows, cols)]

        wear, spots = crop(self.wear), crop(self.spots)
        if clean:  # small captions: no worn holes or speckle, they'd hurt legibility
            wear, spots = np.ones_like(wear), np.zeros_like(spots)
        fill = np.array(th[color_key], np.float32)
        # distressed fill: worn holes + darker red speckle, slight vertical shading
        shade = np.linspace(1.05, 0.88, th_, dtype=np.float32)[:, None]
        spk = np.array(th["glow"], np.float32)
        rgb = fill[None, None] * shade[..., None]
        rgb = rgb * (1 - spots[..., None] * 0.55) + spk * spots[..., None] * 0.55
        fill_a = m
        hole = m * (1 - wear)  # worn-through spots let the background show

        glow = Image.fromarray((o * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(glow_r))
        g = np.asarray(glow, np.float32) / 255 * (1.0 if hero else 0.75)
        gc = np.array(th["glow"], np.float32)
        oc = np.array(th["outline"], np.float32)

        # stack: glow -> dark outline -> distressed fill
        a = np.clip(g * 0.9, 0, 1)
        col = gc[None, None] * np.ones_like(rgb)
        a2 = o
        col = col * (1 - a2[..., None]) + oc * a2[..., None]
        a = a2 + a * (1 - a2)
        col = col * (1 - fill_a[..., None]) + rgb * fill_a[..., None]
        a = fill_a + a * (1 - fill_a)
        a = a * (1 - hole * 0.9)
        out = np.dstack([np.clip(col, 0, 255), np.clip(a * 255, 0, 255)]).astype(np.uint8)
        spr = Image.fromarray(out, "RGBA")
        self.sprite_cache[key] = (spr, pad)
        return spr, pad

    # -------------------------------------------------------------- background
    def background(self, t, energy, pulse):
        W, H, th = self.W, self.H, self.theme
        z = 1.0 + 0.012 * pulse
        dx = int((self.BW - W) / 2 + math.sin(t * 0.07) * (self.BW - W) * 0.35)
        dy = int((self.BH - H) / 2 + math.cos(t * 0.05) * (self.BH - H) * 0.35)
        bg = self.base[dy:dy + H, dx:dx + W].copy()
        sc = np.array(th["smoke"], np.float32)
        for k, s in enumerate(self.smoke):
            sx = int((t * (14 if k == 0 else -10)) % (self.BW - W))
            sy = int((self.BH - H) / 2 + math.sin(t * 0.11 + k) * (self.BH - H) * 0.4)
            amt = (0.30 + 0.45 * energy) * (0.75 if k else 1.0) * (1 + 0.6 * pulse)
            layer = s[sy:sy + H, sx:sx + W, None]
            bg = bg + sc * layer * amt
        bg *= self.vignette
        g = self.grain[int(t * self.fps) % len(self.grain)]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:H, :W, None]
        bg += g
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        if z > 1.001:
            w2, h2 = int(W * z), int(H * z)
            img = img.resize((w2, h2), Image.BILINEAR).crop(((w2 - W) // 2, (h2 - H) // 2, (w2 - W) // 2 + W, (h2 - H) // 2 + H))
        return img.convert("RGBA")

    def particles_layer(self, t, energy):
        lay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for p in self.particles:
            x = (p["x"] + p["vx"] * t + 18 * math.sin(t * 0.8 + p["ph"])) % self.W
            y = (p["y"] + p["vy"] * t * (0.8 + 0.6 * energy)) % self.H
            fl = 0.55 + 0.45 * math.sin(t * p["fl"] + p["ph"])
            a = int(255 * fl * (0.5 + 0.5 * energy))
            r = p["r"]
            c = p["c"]
            if p["streak"]:
                d.line([(x, y), (x - p["vx"] * 0.05, y - p["vy"] * 0.06)], fill=c + (a,), width=max(1, int(r)))
            else:
                if r > 1.8:
                    d.ellipse([x - r * 2.6, y - r * 2.6, x + r * 2.6, y + r * 2.6], fill=c + (a // 6,))
                d.ellipse([x - r, y - r, x + r, y + r], fill=c + (a,))
        return lay


# ------------------------------------------------------------------ animation of a word sprite
def count_text(final, tw, dur=0.75):
    """Number that rolls up to `final` (keeps commas, decimals and a suffix like '*')."""
    import re
    m = re.match(r"^([\d,]+(?:\.\d+)?)(.*)$", final)
    if not m:
        return final
    num, suffix = m.group(1), m.group(2)
    dec = len(num.split(".")[1]) if "." in num else 0
    target = float(num.replace(",", ""))
    v = target * ease_out_cubic(tw / dur)
    txt = f"{v:,.{dec}f}" if "," in num else f"{v:.{dec}f}"
    return txt + (suffix if tw >= dur else "")


def word_state(anim, tw, chunk_t, chunk_left, rng):
    """Return (scale, dx, dy, alpha, rot, rgb_split, slices) for a word visible tw seconds."""
    s, dx, dy, a, rot, split, slices = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0
    if anim in ("slam", "stamp"):
        d = 0.13 if anim == "slam" else 0.09
        big = 2.3 if anim == "slam" else 1.7
        k = ease_in_quad(tw / d)
        s = big + (1 - big) * k
        a = clamp(tw / 0.05)
        if tw > d:  # tiny rebound after impact
            s = 1 + 0.035 * math.exp(-(tw - d) * 18) * math.cos((tw - d) * 40)
    elif anim == "punch":
        s = 0.45 + 0.55 * ease_out_back(tw / 0.24)
        a = clamp(tw / 0.08)
    elif anim == "rise":
        e = ease_out_cubic(tw / 0.3)
        dy = (1 - e) * 70; a = e; s = 0.92 + 0.08 * e
    elif anim == "glitch":
        a = 1.0 if tw > 0.28 else (0.35 + 0.65 * (rng.random() > 0.35))
        if tw < 0.3:
            split = (1 - tw / 0.3) * 26
            slices = 4
            dx = (rng.random() - 0.5) * 18 * (1 - tw / 0.3)
        s = 1.0
    elif anim == "shake":
        s = 0.7 + 0.3 * ease_out_back(tw / 0.2)
        a = clamp(tw / 0.06)
        dx = (rng.random() - 0.5) * 6; dy = (rng.random() - 0.5) * 6
    else:  # "fade"
        a = clamp(tw / 0.2)
    # slow push-in while holding keeps it alive
    s *= 1 + 0.035 * clamp(chunk_t / 2.5)
    # exit
    if chunk_left < 0.12:
        e = clamp(1 - chunk_left / 0.12)
        a *= 1 - e
        s *= 1 + 0.08 * e
    return s, dx, dy, a, rot, split, slices


def paste_sprite(frame, spr, cx, cy, scale, alpha, split, slices, rng, theme):
    if alpha <= 0.01:
        return
    w, h = spr.size
    if abs(scale - 1) > 0.004:
        spr = spr.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        w, h = spr.size
    if alpha < 0.999:
        r, g, b, a = spr.split()
        a = a.point(lambda v: int(v * alpha))
        spr = Image.merge("RGBA", (r, g, b, a))
    if slices:
        arr = np.asarray(spr).copy()
        for _ in range(slices):
            y0 = rng.integers(0, h); hh = rng.integers(h // 14 + 1, h // 5 + 2)
            arr[y0:y0 + hh] = np.roll(arr[y0:y0 + hh], int(rng.integers(-w // 12, w // 12 + 1)), axis=1)
        spr = Image.fromarray(arr, "RGBA")
    x, y = int(cx - w / 2), int(cy - h / 2)
    if split > 0.5:
        r, g, b, a = spr.split()
        z = Image.new("L", spr.size, 0)
        red = Image.merge("RGBA", (r, z, z, a.point(lambda v: int(v * 0.8))))
        cyan = Image.merge("RGBA", (z, g, b, a.point(lambda v: int(v * 0.8))))
        _safe(frame, red, x - int(split), y)
        _safe(frame, cyan, x + int(split), y)
    _safe(frame, spr, x, y)


def _safe(frame, spr, x, y):
    W, H = frame.size
    w, h = spr.size
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    cx0, cy0 = max(0, -x), max(0, -y)
    cx1, cy1 = min(w, W - x), min(h, H - y)
    frame.alpha_composite(spr.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))


# ------------------------------------------------------------------ animated bar chart
def draw_chart(layer, ch, t, sc, rng):
    """Leaderboard bar chart: bars grow in turn, values count up, the highlighted row glows."""
    W, H = sc.W, sc.H
    tr, left = t - ch["start"], ch["end"] - t
    fade = clamp(tr / 0.2) * clamp(left / 0.15)
    if fade <= 0:
        return
    th = sc.theme
    items = ch["items"]
    vmax = float(ch.get("max", max(i["value"] for i in items)))
    tsize = int(H * 0.075)
    tspr, _ = sc.word_sprite(ch["title"], tsize, "white", True, seed=501, clean=True)
    e = ease_out_back(tr / 0.35, 1.6)
    paste_sprite(layer, tspr, W / 2, H * 0.17, 0.85 + 0.15 * e, fade, 0, 0, rng, th)
    if ch.get("subtitle"):
        sspr, _ = sc.word_sprite(ch["subtitle"], int(H * 0.034), "white", False, seed=502, clean=True)
        paste_sprite(layer, sspr, W / 2, H * 0.255, 1.0, fade * clamp((tr - 0.25) / 0.3), 0, 0, rng, th)
    x0, x1 = W * 0.12, W * 0.88
    rows = len(items)
    y_top, y_bot = H * 0.40, H * 0.86
    step = (y_bot - y_top) / max(1, rows - 1) if rows > 1 else 0
    bar_h = int(H * 0.045)
    lsize = int(H * 0.05)
    d = ImageDraw.Draw(layer)
    for k, it in enumerate(items):
        delay = 0.45 + 0.28 * k
        g = ease_out_cubic((tr - delay) / 0.9)
        ra = fade * clamp((tr - delay + 0.15) / 0.2)
        if ra <= 0:
            continue
        y = y_top + step * k
        hl = it.get("highlight", False)
        col = th["red"] if hl else (150, 150, 162)
        # label + value
        lspr, _ = sc.word_sprite(it["label"].upper(), lsize, "white", hl, seed=510 + k, clean=True)
        paste_sprite(layer, lspr, x0 + lspr.size[0] / 2 - (lspr.size[0] - lsize * 0.2) * 0 - _pad(lspr, lsize), y - bar_h * 1.15,
                     1.0, ra, 0, 0, rng, th)
        val = count_text(str(it["value"]), max(0.0, tr - delay), 0.9)
        vspr, _ = sc.word_sprite(val, int(lsize * 1.15), "gold" if hl else "white", hl, seed=520 + k, clean=True)
        paste_sprite(layer, vspr, x1 - vspr.size[0] / 2 + _pad(vspr, lsize), y - bar_h * 1.2, 1.0, ra, 0, 0, rng, th)
        # track + bar
        a8 = int(255 * ra)
        d.rounded_rectangle([x0, y, x1, y + bar_h], radius=bar_h // 2, fill=(255, 255, 255, int(28 * ra)))
        xe = x0 + (x1 - x0) * (it["value"] / vmax) * g
        if xe > x0 + bar_h:
            if hl:  # glow under the winning bar
                glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
                ImageDraw.Draw(glow).rounded_rectangle([x0, y - 4, xe, y + bar_h + 4], radius=bar_h // 2,
                                                       fill=tuple(th["glow"]) + (int(200 * ra),))
                layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(bar_h * 0.6)))
            d.rounded_rectangle([x0, y, xe, y + bar_h], radius=bar_h // 2, fill=tuple(col) + (a8,))
            if hl:  # bright leading edge
                d.rounded_rectangle([max(x0, xe - bar_h * 1.2), y, xe, y + bar_h], radius=bar_h // 2,
                                    fill=(255, 210, 120, int(230 * ra)))


def _pad(spr, size):
    """Sprites carry glow padding; nudge so the visible text aligns to the column edge."""
    return (spr.size[1] - size * 1.25) / 2


# ------------------------------------------------------------------ frame renderer
class Renderer:
    def __init__(self, plan):
        self.plan = plan
        self.sc = Scene(plan)
        self.beats = np.array(plan.get("beats", []), float)
        self.hits = np.array(plan.get("hits", []), float)
        self.energy = plan.get("energy", [])
        self.efps = plan.get("energy_fps", 10)
        self.layouts = {}
        # impact times: every slam/stamp word landing
        self.plan_dir = plan.get("_dir", ".")
        self.media = medialib.build_layers(plan, self.plan_dir, self.sc.theme)
        self.scenes = sceneslib.build(plan, self.sc.theme)
        self.shots = complib.build(plan, self.sc.theme, self.plan_dir)
        cache = {}
        self.props = [propslib.Prop(p, self.sc.W, self.sc.H, cache) for p in plan.get("props", [])]
        imp = []
        for c in plan["chunks"]:
            if c.get("anim") == "slam":
                imp.append((c["start"] + 0.13, 1.0 if c.get("emphasis") == "hero" else 0.6))
            elif c.get("anim") == "stamp":
                for w in c["words"]:
                    imp.append((max(c["start"], w["start"] - 0.04) + 0.09, 0.35))
        self.impacts = imp

    def energy_at(self, t):
        if not self.energy:
            return 0.5
        i = int(t * self.efps)
        return float(self.energy[min(max(i, 0), len(self.energy) - 1)])

    def pulse_at(self, t):
        if not len(self.beats):
            return 0.0
        prev = self.beats[self.beats <= t]
        if not len(prev):
            return 0.0
        return math.exp(-(t - prev[-1]) * 9)

    def shake_at(self, t):
        mag = 0.0
        for ti, s in self.impacts:
            if 0 <= t - ti < 0.35:
                mag = max(mag, s * 22 * math.exp(-(t - ti) * 11))
        if self.plan["video"].get("shake_on_hits", True) and len(self.hits):
            near = self.hits[(self.hits <= t) & (self.hits > t - 0.2)]
            if len(near):
                mag = max(mag, 5 * math.exp(-(t - near[-1]) * 14))
        return mag * self.sc.W / 1280

    def flash_at(self, t):
        f = 0.0
        for m in self.media:
            if m.fin > 0 and 0 <= t - m.start < 0.25:
                f = max(f, 0.28 * math.exp(-(t - m.start) * 16))
        if not self.plan["video"].get("flash_on_slam", True):
            return f
        for ti, s in self.impacts:
            if 0 <= t - ti < 0.2 and s >= 0.6:
                f = max(f, 0.32 * s * math.exp(-(t - ti) * 16))
        return f

    def frame(self, t):
        sc = self.sc
        rng = np.random.default_rng(int(t * 1000) + 11)
        e = self.energy_at(t)
        pulse = self.pulse_at(t)
        img = sc.background(t, e, pulse)
        if self.scenes:
            sceneslib.composite(img, self.scenes, t, e, pulse)
        if self.media:
            medialib.composite(img, self.media, t)
        comp_flash = complib.draw(img, self.shots, t, e, pulse, "back") if self.shots else 0.0
        img.alpha_composite(sc.particles_layer(t, e))
        if self.props:
            propslib.draw_props(img, self.props, t)

        text_layer = Image.new("RGBA", (sc.W, sc.H), (0, 0, 0, 0))
        for chs in self.plan.get("charts", []):
            if chs["start"] <= t < chs["end"]:
                draw_chart(text_layer, chs, t, sc, rng)
        tc = self.plan.get("title_card")
        if tc and tc["start"] <= t < tc["end"]:
            self.draw_title(text_layer, tc, t, rng)
        for ci, c in enumerate(self.plan["chunks"]):
            if not (c["start"] <= t < c["end"]):
                continue
            if ci not in self.layouts:
                self.layouts[ci] = sc.layout(c)
            f, size, pos, words = self.layouts[ci]
            hero = c.get("emphasis") == "hero"
            ck = c.get("color", "white")
            anim = c.get("anim", "punch")
            for wi, w in enumerate(words):
                appear = c["start"] if anim != "stamp" else max(c["start"], w["start"] - 0.04)
                tw = t - appear
                if tw < 0:
                    continue
                shown = count_text(w["text"], tw) if anim == "count" else w["text"]
                spr, pad = sc.word_sprite(shown, size, ck, hero, seed=ci * 31 + wi)
                s, dx, dy, a, rot, split, slices = word_state("punch" if anim == "count" else anim, tw,
                                                              t - c["start"], c["end"] - t, rng)
                x, y, ww = pos[wi]
                asc, desc = f.getmetrics()
                cx = x + ww / 2 + dx
                cy = y + (asc + desc) / 2 + dy + (c.get("y", 0.5) - 0.5) * sc.H
                # group scale around frame centre so multi-word slams stay composed
                cx = sc.W / 2 + (cx - sc.W / 2) * (s if anim == "slam" else 1)
                ccy = sc.H * c.get("y", 0.5)
                cy = ccy + (cy - ccy) * (s if anim == "slam" else 1)
                paste_sprite(text_layer, spr, cx, cy, s, a, split, slices, rng, sc.theme)
            if c.get("over"):  # small caption above the main text (e.g. "VS PAKISTAN • 2012")
                asc, desc = f.getmetrics()
                bb = f.getbbox(" ".join(w["text"] for w in words))
                top = min(p[1] for p in pos) + bb[1] + (c.get("y", 0.5) - 0.5) * sc.H
                osize = max(20, int(size * float(c.get("over_size", 0.16))))
                ospr, _ = sc.word_sprite(c["over"], osize, c.get("over_color", "white"), False, seed=ci * 31 + 98, clean=True)
                tw = t - c["start"]
                oa = clamp((tw - 0.1) / 0.25) * clamp((c["end"] - t) / 0.12)
                ody = (1 - ease_out_cubic((tw - 0.1) / 0.3)) * -18
                paste_sprite(text_layer, ospr, sc.W / 2, top - osize * 0.75 + ody, 1.0, oa, 0, 0, rng, sc.theme)
            if c.get("sub"):  # small caption under the main text (e.g. "INTERNATIONAL RUNS")
                asc, desc = f.getmetrics()
                bottom = max(p[1] for p in pos) + (asc + desc) * 0.86 + (c.get("y", 0.5) - 0.5) * sc.H
                ssize = max(20, int(size * float(c.get("sub_size", 0.24))))
                sspr, _ = sc.word_sprite(c["sub"], ssize, c.get("sub_color", "white"), False, seed=ci * 31 + 99, clean=True)
                tw = t - c["start"]
                sa = clamp((tw - 0.22) / 0.25) * (clamp((c["end"] - t) / 0.12))
                sdy = (1 - ease_out_cubic((tw - 0.22) / 0.3)) * 18
                paste_sprite(text_layer, sspr, sc.W / 2, bottom + ssize * 0.6 + sdy, 1.0, sa, 0, 0, rng, sc.theme)

        sh = self.shake_at(t)
        ox, oy = (rng.random() - 0.5) * 2 * sh, (rng.random() - 0.5) * 2 * sh
        img.alpha_composite(text_layer, (0, 0)) if sh < 0.5 else _safe(img, text_layer, int(ox), int(oy))
        if self.shots:
            complib.draw(img, self.shots, t, e, pulse, "front")

        fl = max(self.flash_at(t), comp_flash)
        if fl > 0.01:
            over = Image.new("RGBA", img.size, sc.theme["flash"] + (int(255 * fl),))
            img.alpha_composite(over)
        # fade in / out of the whole video
        dur = self.plan["duration"]
        k = min(clamp(t / 0.6), clamp((dur - t) / 1.0))
        out = img.convert("RGB")
        if k < 1:
            out = Image.eval(out, lambda v: int(v * k))
        return out

    def draw_title(self, layer, tc, t, rng):
        sc = self.sc
        fake = {"text": tc["text"], "words": [{"text": tc["text"], "start": tc["start"]}],
                "emphasis": "hero", "size": 1.25, "start": tc["start"]}
        if "title" not in self.layouts:
            self.layouts["title"] = sc.layout(fake)
        f, size, pos, words = self.layouts["title"]
        spr, pad = sc.word_sprite(tc["text"], size, "white", True, seed=999)
        tw = t - tc["start"]
        s, dx, dy, a, _, split, slices = word_state("glitch", tw, tw, tc["end"] - t, rng)
        if tw > 0.3 and rng.random() < 0.06:  # occasional flicker glitch while holding
            split, slices = 10, 2
        s *= 1 + 0.05 * clamp(tw / max(0.5, tc["end"] - tc["start"]))
        x, y, ww = pos[0]
        asc, desc = f.getmetrics()
        paste_sprite(layer, spr, x + ww / 2 + dx, y + (asc + desc) / 2 + dy - sc.H * 0.04, s, a, split, slices, rng, sc.theme)
        if tc.get("sub"):
            sub_size = max(18, int(size * 0.2))
            sspr, _ = sc.word_sprite(tc["sub"], sub_size, "white", False, seed=998)
            sa = clamp((tw - 0.5) / 0.4) * a
            paste_sprite(layer, sspr, sc.W / 2, sc.H * 0.5 + (asc + desc) * 0.5 * s + sub_size * 0.2, 1.0, sa, 0, 0, rng, sc.theme)


# ------------------------------------------------------------------ encoding
def encode_range(args):
    plan_path, t0, t1, out, crf = args
    plan = json.load(open(plan_path, encoding="utf-8"))
    plan["_dir"] = os.path.dirname(os.path.abspath(plan_path))
    r = Renderer(plan)
    W, H, fps = plan["video"]["width"], plan["video"]["height"], plan["video"]["fps"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    for fi in range(f0, f1):
        p.stdin.write(r.frame(fi / fps).tobytes())
        if (fi - f0) % (fps * 5) == 0:
            print(f"  [{os.path.basename(out)}] {fi / fps:6.1f}s / {t1:.1f}s", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("-o", "--out", default="out/lyric_video.mp4")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--range", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--stills", help="comma-separated times; writes PNGs into the -o directory")
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args()

    plan = json.load(open(a.plan, encoding="utf-8"))
    plan["_dir"] = os.path.dirname(os.path.abspath(a.plan))
    medialib.prepare_media(plan, plan["_dir"])
    if a.stills:
        os.makedirs(a.out, exist_ok=True)
        r = Renderer(plan)
        for ts in a.stills.split(","):
            t = float(ts)
            p = os.path.join(a.out, f"still_{t:07.2f}.png")
            r.frame(t).save(p)
            print(p)
        return

    t0, t1 = (a.range if a.range else (0.0, plan["duration"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="lyricvid_")
    n = max(1, min(a.workers, int((t1 - t0) // 4) or 1))
    edges = np.linspace(t0, t1, n + 1)
    fps = plan["video"]["fps"]
    edges = [round(e * fps) / fps for e in edges]
    jobs = [(a.plan, edges[i], edges[i + 1], os.path.join(tmp, f"part{i:02d}.mp4"), a.crf) for i in range(n)]
    print(f"[render] {t1 - t0:.1f}s at {plan['video']['width']}x{plan['video']['height']}@{fps} with {n} worker(s)")
    if n == 1:
        parts = [encode_range(jobs[0])]
    else:
        with ProcessPoolExecutor(n) as ex:
            parts = list(ex.map(encode_range, jobs))
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-ss", str(t0), "-t", str(t1 - t0), "-i", plan["audio"],
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"[render] done -> {a.out}")


if __name__ == "__main__":
    main()
