#!/usr/bin/env python3
"""
comp.py — designed media "shots" with transitions (instead of plain photo/clip backdrops).

Plan field "shots": a list of dicts. Kinds:

  cutout  a person cut out of a photo (PNG with alpha, see cutout.py) standing inside the video's own world
          {"kind": "cutout", "src": "cut/x.png", "start": 3.9, "end": 7.2, "x": 0.5, "height": 0.95,
           "style": "duotone"|"grade", "rays": true, "front": false, "drift": 0.02, "in": "reveal", "out": "zoom"}
          front: true draws the person IN FRONT of the lyrics (text-behind-subject); rays/rim stay behind.
  full    full-frame cinematic clip or photo: cover-cropped to 16:9 around `focus`, denoised, graded, letterboxed
          {"kind": "full", "src": "clip.mov", "start": .., "end": .., "clip_in": 0, "speed": 0.7,
           "focus": [0.5, 0.4], "zoom": [1.0, 1.08], "letterbox": true, "dim": 0.85, "in": "whip", "out": "cut"}
  panels  2–3 slanted vertical strips (great for portrait phone clips) sliding in one after another
          {"kind": "panels", "start": .., "end": .., "items": [{"src": .., "clip_in": 0, "speed": 0.8,
           "focus": [0.5, 0.4]}, ...], "stagger": 0.35, "slant": 7, "in": "slide", "out": "whip"}

Transitions (in/out): cut, fade, whip (motion-blurred push), zoom (punch/zoom-through), glitch (RGB slices),
burn (light-leak flash), rise / sink (from/to below), reveal (bottom-up wipe with a fiery edge), slide.
`tin` / `tout` set their durations (default 0.28 s). Each hard entrance also fires a short flash.
"""
import hashlib, math, os, subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

FPS = 30
VIDEO_EXT = (".mov", ".mp4", ".m4v", ".mkv", ".webm")


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def eoc(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def eob(x, s=1.7):
    x = clamp(x) - 1; return x * x * ((s + 1) * x + s) + 1


def is_video(p):
    return p.lower().endswith(VIDEO_EXT)


def open_image(path):
    if path.lower().endswith((".heic", ".heif")):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    return ImageOps.exif_transpose(Image.open(path))


# ------------------------------------------------------------------ frame sources
def _key(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def extract(src, clip_in, speed, dur, w, h, focus, root):
    """Cover-crop a video to w x h around focus, cleaned up for a 'shot on camera' look; cached JPEG frames."""
    d = os.path.join(root, _key(os.path.abspath(src), clip_in, speed, round(dur, 3), w, h, focus))
    need = int(dur * FPS) + 1
    if os.path.isdir(d) and len(os.listdir(d)) >= need - 1:
        return d
    os.makedirs(d, exist_ok=True)
    fx, fy = focus
    vf = (f"setpts=PTS/{speed},fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={w}:{h}:(iw-{w})*{fx}:(ih-{h})*{fy},hqdn3d=5:4:6:5,gblur=sigma=0.6,unsharp=5:5:0.7")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(clip_in), "-t", f"{(dur + 0.2) * speed:.3f}",
                    "-i", src, "-vf", vf, "-q:v", "2", os.path.join(d, "%05d.jpg")], check=True)
    return d


class Source:
    """A clip or still, delivered as RGB float arrays of a fixed size."""

    def __init__(self, item, start, end, w, h, root):
        self.start = start
        self.video = is_video(item["src"])
        focus = tuple(item.get("focus", (0.5, 0.45)))
        if self.video:
            self.dir = extract(item["src"], item.get("clip_in", 0.0), float(item.get("speed", 1.0)), end - start,
                               w, h, focus, root)
            self.frames = sorted(os.listdir(self.dir))
            self.cache = {}
            mid = np.asarray(Image.open(os.path.join(self.dir, self.frames[len(self.frames) // 2])).convert("RGB"), np.float32)
            self._levels(mid)
        else:
            img = open_image(item["src"]).convert("RGB")
            k = max(w / img.width, h / img.height)
            img = img.resize((int(img.width * k + 1), int(img.height * k + 1)), Image.LANCZOS)
            x0 = int((img.width - w) * focus[0]); y0 = int((img.height - h) * focus[1])
            img = img.crop((x0, y0, x0 + w, y0 + h)).filter(ImageFilter.GaussianBlur(0.7)).filter(
                ImageFilter.UnsharpMask(3, 80, 2))
            self.still = np.asarray(img, np.float32)
            self._levels(self.still)
            self.still = self._apply(self.still)

    def _levels(self, a):
        """Washed-out TV/phone footage: stretch to full range (1st–99.5th percentile of luma)."""
        lum = a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11
        self.lo, self.hi = float(np.percentile(lum, 1.5)), float(np.percentile(lum, 99.5))
        if self.hi - self.lo < 40:
            self.lo, self.hi = 0.0, 255.0

    def _apply(self, a):
        return np.clip((a - self.lo) * (255.0 / (self.hi - self.lo)), 0, 255)

    def get(self, t):
        if not self.video:
            return self.still
        i = min(len(self.frames) - 1, max(0, int((t - self.start) * FPS)))
        if i not in self.cache:
            if len(self.cache) > 4:
                self.cache.clear()
            self.cache[i] = self._apply(np.asarray(Image.open(os.path.join(self.dir, self.frames[i])).convert("RGB"), np.float32))
        return self.cache[i]


# ------------------------------------------------------------------ looks
def grade(a, theme, dim=1.0, strength=1.0):
    lum = (a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11)[..., None]
    des = lum + (a - lum) * 0.55
    dark = np.array(theme["outline"], np.float32) * 1.4
    light = np.array([255, 222, 200], np.float32)
    tone = dark + (light - dark) * (lum / 255)
    g = des * 0.55 + tone * 0.45
    g = (g - 118) * 1.38 + 108
    g = a * (1 - strength) + g * strength
    return np.clip(g * dim, 0, 255)


def duotone(a, theme):
    lum = (a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11) / 255
    lum = np.clip((lum - 0.06) / 0.86, 0, 1) ** 0.9
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    cols = np.array([[8, 2, 4], list(theme["glow"]), [255, 140, 90], [255, 238, 222]], np.float32)
    out = np.stack([np.interp(lum, stops, cols[:, c]) for c in range(3)], -1)
    return out


# ------------------------------------------------------------------ transitions on an RGBA layer (H, W, 4 float)
def hblur(a, r):
    r = int(r)
    if r < 2:
        return a
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode="edge"), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def shift(a, dx, dy):
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    dx, dy = int(round(dx)), int(round(dy))
    xs0, xs1 = max(0, -dx), min(W, W - dx); ys0, ys1 = max(0, -dy), min(H, H - dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = a[ys0:ys1, xs0:xs1]
    return out


def scale(a, s):
    if abs(s - 1) < 0.004:
        return a
    H, W = a.shape[:2]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    w2, h2 = max(1, int(W * s)), max(1, int(H * s))
    img = img.resize((w2, h2), Image.BILINEAR)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(img, ((W - w2) // 2, (H - h2) // 2))
    return np.asarray(canvas, np.float32)


def apply_transition(a, kind, u, entering, rng, W, H, direction=1):
    """u: 0 at the start of the transition -> 1 at the end. Returns (layer, extra_glow 0..1)."""
    glow = 0.0
    if kind in (None, "cut"):
        return a, 0.0
    p = u if entering else 1 - u           # visibility 0..1
    if kind == "fade":
        a = a.copy(); a[..., 3] *= clamp(p)
    elif kind == "whip":
        e = eoc(u) if entering else u ** 2
        off = (1 - e) * W * direction if entering else -e * W * direction
        a = shift(a, off, 0)
        a = hblur(a, abs(1 - e if entering else e) * 60)
    elif kind == "zoom":
        s = 1.35 - 0.35 * eoc(u) if entering else 1 + 0.45 * u ** 2
        a = scale(a, s); a = a.copy(); a[..., 3] *= clamp(p * 1.4)
    elif kind == "punch":
        a = scale(a, 0.75 + 0.25 * eob(u)) if entering else scale(a, 1 - 0.2 * u)
        a = a.copy(); a[..., 3] *= clamp(p * 2)
    elif kind in ("rise", "sink"):
        e = eoc(u) if entering else u ** 2
        dy = (1 - e) * H * 0.45 if entering else e * H * 0.45
        a = shift(a, 0, dy); a = a.copy(); a[..., 3] *= clamp(p * 1.6)
    elif kind == "slide":
        e = eoc(u) if entering else u ** 2
        a = shift(a, (1 - e) * W * 0.6 * direction if entering else -e * W * 0.6 * direction, 0)
        a = a.copy(); a[..., 3] *= clamp(p * 1.5)
    elif kind == "glitch":
        a = a.copy()
        if p < 1:
            for _ in range(7):
                y0 = int(rng.integers(0, H)); hh = int(rng.integers(H // 40 + 1, H // 7))
                a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], int(rng.integers(-W // 8, W // 8)), axis=1)
            r = a.copy(); r[..., 1:3] = 0
            a[..., 0] = np.roll(r[..., 0], int(18 * (1 - p)), axis=1)
            a[..., 3] *= 0.5 + 0.5 * (rng.random() < p + 0.25)
    elif kind == "burn":
        a = a.copy(); a[..., 3] *= clamp(p * 1.3)
        glow = math.sin(math.pi * clamp(u)) * 0.9
    elif kind == "reveal":  # bottom-up wipe with a fiery edge
        e = eoc(u) if entering else 1 - u ** 2
        edge = H * (1 - e * 1.05)
        yy = np.arange(H, dtype=np.float32)[:, None]
        m = np.clip((yy - edge) / 18 + 0.5, 0, 1)
        band = np.exp(-((yy - edge) / 10) ** 2)
        a = a.copy(); a[..., 3] *= m
        a[..., :3] = a[..., :3] + band[..., None] * np.array([255, 150, 50], np.float32) * (a[..., 3:4] > 0) * 0.9
        glow = 0.25 * math.sin(math.pi * clamp(u))
    return a, glow


# ------------------------------------------------------------------ shots
class Shot:
    def __init__(self, spec, W, H, theme, root):
        self.s = spec; self.W, self.H = W, H; self.theme = theme
        self.start, self.end = spec["start"], spec["end"]
        self.kind = spec["kind"]
        self.tin = spec.get("tin", 0.28); self.tout = spec.get("tout", 0.28)
        self.front = spec.get("front", False)
        self.dir = spec.get("dir", 1)
        if self.kind == "full":
            self.bw, self.bh = int(W * 1.15), int(H * 1.15)
            self.src = Source(spec, self.start, self.end, self.bw, self.bh, root)
        elif self.kind == "panels":
            n = len(spec["items"])
            self.pw = int(W / n * 1.02); self.ph = int(H * 1.12)
            self.srcs = [Source(it, self.start, self.end, self.pw + 80, self.ph, root) for it in spec["items"]]
        elif self.kind == "cutout":
            img = open_image(spec["src"]).convert("RGBA")
            th = int(H * spec.get("height", 0.95))
            k = th / img.height
            img = img.resize((max(1, int(img.width * k)), th), Image.LANCZOS)
            if img.width > W * 1.1:
                k2 = W * 1.1 / img.width
                img = img.resize((int(img.width * k2), int(img.height * k2)), Image.LANCZOS)
            a = np.asarray(img, np.float32)
            rgb = np.asarray(Image.fromarray(a[..., :3].astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).filter(
                ImageFilter.UnsharpMask(3, 90, 2)), np.float32)
            rgb = duotone(rgb, theme) if spec.get("style", "duotone") == "duotone" else grade(rgb, theme, 1.0)
            self.sub = np.dstack([rgb, a[..., 3]])
            alpha = Image.fromarray(a[..., 3].astype(np.uint8))
            rim = np.asarray(alpha.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(10)), np.float32)
            self.rim = np.clip(rim - a[..., 3] * 0.6, 0, 255)
            self.glow_big = np.asarray(alpha.resize((alpha.width // 4, alpha.height // 4)).filter(
                ImageFilter.GaussianBlur(18)).resize(alpha.size), np.float32)
            self.rays = spec.get("rays", True)
            if self.rays:
                rr = np.random.default_rng(int(self.start * 100))
                self.ray_ang = rr.uniform(0, 2 * math.pi, 26); self.ray_w = rr.uniform(0.02, 0.08, 26)
                self.ray_a = rr.uniform(0.3, 1.0, 26)

    def active(self, t):
        return self.start <= t < self.end

    # ---- body (no transition)
    def body(self, t, energy, pulse):
        W, H = self.W, self.H
        u = (t - self.start) / max(0.01, self.end - self.start)
        if self.kind == "full":
            fr = self.src.get(t)
            z0, z1 = self.s.get("zoom", [1.0, 1.08])
            z = z0 + (z1 - z0) * u
            bh, bw = fr.shape[:2]
            cw, ch = bw / (1.15 * z), bh / (1.15 * z)
            x0 = (bw - cw) / 2 + self.s.get("pan", 0) * (bw - cw) * (u - 0.5)
            img = Image.fromarray(fr.astype(np.uint8)).crop((int(x0), int((bh - ch) / 2), int(x0 + cw), int((bh + ch) / 2)))
            a = np.asarray(img.resize((W, H), Image.BILINEAR), np.float32)
            a = grade(a, self.theme, self.s.get("dim", 0.85), self.s.get("grade", 1.0))
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
            a *= np.clip(1.15 - 0.45 * r, 0.3, 1)[..., None]
            if self.s.get("letterbox", True):
                lb = int(H * 0.11)
                a[:lb] = 0; a[H - lb:] = 0
            return np.dstack([a, np.full((H, W), 255, np.float32)])
        if self.kind == "panels":
            n = len(self.srcs)
            out = np.zeros((H, W, 4), np.float32)
            stagger = self.s.get("stagger", 0.35)
            slant = math.radians(self.s.get("slant", 7))
            canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            for i, sr in enumerate(self.srcs):
                ti = t - self.start - i * stagger
                if ti < 0:
                    continue
                e = eoc(ti / 0.4)
                fr = sr.get(t)
                drift = int((fr.shape[1] - self.pw) * (0.5 + 0.4 * (u - 0.5)))
                tile = fr[:, drift:drift + self.pw]
                tile = grade(tile, self.theme, self.s.get("dim", 0.85))
                tile = np.dstack([tile, np.full(tile.shape[:2], 255, np.float32)])
                timg = Image.fromarray(tile.astype(np.uint8), "RGBA")
                # slanted parallelogram mask + red borders
                m = Image.new("L", timg.size, 0); dm = ImageDraw.Draw(m)
                off = math.tan(slant) * timg.height / 2
                gap = 6
                dm.polygon([(off + gap, 0), (timg.width - gap + off, 0), (timg.width - gap - off, timg.height),
                            (gap - off, timg.height)], fill=255)
                timg.putalpha(m)
                bd = ImageDraw.Draw(timg)
                col = tuple(self.theme["red"]) + (255,)
                bd.line([(off + gap, 0), (gap - off, timg.height)], fill=col, width=4)
                bd.line([(timg.width - gap + off, 0), (timg.width - gap - off, timg.height)], fill=col, width=4)
                x = int(i * W / n - (W * 0.01)); y = int((H - timg.height) / 2 + (1 - e) * H * (1 if i % 2 else -1))
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(timg, (x, y), timg)
                if e < 1:
                    la = np.asarray(lay, np.float32).copy(); la[..., 3] *= e
                    lay = Image.fromarray(la.astype(np.uint8), "RGBA")
                canvas.alpha_composite(lay)
            return np.asarray(canvas, np.float32)
        if self.kind == "cutout":
            sub = self.sub
            sh, sw = sub.shape[:2]
            drift = self.s.get("drift", 0.025)
            z = 1.0 + 0.06 * u
            cx = W * self.s.get("x", 0.5) + (u - 0.5) * W * drift
            base_y = H * self.s.get("bottom", 1.02)
            out = np.zeros((H, W, 4), np.float32)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            w2, h2 = int(sw * z), int(sh * z)
            x0, y0 = int(cx - w2 / 2), int(base_y - h2)
            return (x0, y0, w2, h2, z)
        return None

    def draw(self, frame, t, energy, pulse, rng, which):
        """which: 'back' or 'front'. Returns flash amount contributed."""
        W, H = self.W, self.H
        tr, left = t - self.start, self.end - t
        tin_kind = self.s.get("in", "cut"); tout_kind = self.s.get("out", "cut")
        glow = 0.0
        if self.kind == "cutout":
            x0, y0, w2, h2, z = self.body(t, energy, pulse)
            # visibility envelope from transitions (cutouts use reveal/rise/zoom/fade/burn)
            def place(arr):
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA").resize((w2, h2), Image.BILINEAR)
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(img, (x0, y0), img)
                return np.asarray(lay, np.float32)
            if which == "back":
                back = np.zeros((H, W, 4), np.float32)
                # light rays from behind the head
                if self.rays:
                    hx, hy = x0 + w2 * self.s.get("head_x", 0.5), y0 + h2 * self.s.get("head_y", 0.22)
                    q = 4  # rays are soft: compute at quarter resolution
                    yy, xx = np.mgrid[0:H // q, 0:W // q].astype(np.float32) * q
                    ang = np.arctan2(yy - hy, xx - hx); dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
                    rays = np.zeros(yy.shape, np.float32)
                    rot = t * 0.12
                    for a0, wd, am in zip(self.ray_ang, self.ray_w, self.ray_a):
                        d = np.abs(((ang - a0 - rot) + math.pi) % (2 * math.pi) - math.pi)
                        rays += am * np.exp(-(d / wd) ** 2)
                    rays *= np.exp(-dist / (H * 0.9)) * (0.55 + 0.35 * pulse)
                    rays = np.asarray(Image.fromarray(np.clip(rays * 80, 0, 255).astype(np.uint8)).resize(
                        (W, H), Image.BILINEAR), np.float32) / 80
                    col = np.array(self.theme["glow"], np.float32) * 0.6 + np.array([255, 170, 90], np.float32) * 0.4
                    back[..., :3] = rays[..., None] * col
                    back[..., 3] = np.clip(rays * 200, 0, 220)
                gl = np.zeros((self.glow_big.shape[0], self.glow_big.shape[1], 4), np.float32)
                gl[..., :3] = np.array(self.theme["glow"], np.float32)
                gl[..., 3] = self.glow_big * 0.85
                g2 = place(gl)
                back = _over(g2, back)
                rim = np.zeros((self.rim.shape[0], self.rim.shape[1], 4), np.float32)
                rim[..., :3] = np.array([255, 120, 60], np.float32); rim[..., 3] = self.rim
                back = _over(place(rim), back)
                if not self.front:
                    back = _over(place(self.sub), back)
                layer = back
            else:
                if not self.front:
                    return 0.0
                layer = place(self.sub)
        else:
            if which == "front":
                return 0.0
            layer = self.body(t, energy, pulse)
        if tr < self.tin:
            layer, g = apply_transition(layer, tin_kind, tr / self.tin, True, rng, W, H, self.dir)
            glow = max(glow, g)
        elif left < self.tout:
            layer, g = apply_transition(layer, tout_kind, 1 - left / self.tout, False, rng, W, H, -self.dir)
            glow = max(glow, g)
        frame.alpha_composite(Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8), "RGBA"))
        flash = 0.0
        if which == "back" and tin_kind not in ("fade",) and 0 <= tr < 0.2:
            flash = 0.3 * math.exp(-tr * 16)
        return max(flash, glow * 0.5)


def _over(top, bottom):
    ta = top[..., 3:4] / 255; ba = bottom[..., 3:4] / 255
    oa = ta + ba * (1 - ta)
    rgb = (top[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) / np.maximum(oa, 1e-6)
    return np.dstack([rgb, oa[..., 0] * 255])


def build(plan, theme, plan_dir):
    W, H = plan["video"]["width"], plan["video"]["height"]
    root = os.path.join(plan_dir, "comp_cache")
    return [Shot(s, W, H, theme, root) for s in plan.get("shots", [])]


def draw(img, shots, t, energy, pulse, which):
    fl = 0.0
    rng = np.random.default_rng(int(t * 1000) + 5)
    for s in shots:
        if s.active(t):
            fl = max(fl, s.draw(img, t, energy, pulse, rng, which))
    return fl
