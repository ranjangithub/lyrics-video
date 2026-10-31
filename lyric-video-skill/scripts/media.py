#!/usr/bin/env python3
"""
media.py — user photos and video clips as full-screen backdrops behind the lyrics.

In a style plan:
  "media": [
    {"src": "media/portrait.png", "start": 120.8, "end": 124.0, "focus": [0.55, 0.3], "zoom": [1.0, 1.15]},
    {"src": "media/clip.mov", "start": 192.0, "end": 203.6, "clip_in": 0.0},
    {"src": "media/clip.mov", "start": 203.6, "end": 205.7, "clip_in": 13.5, "fade_in": 0}
  ]

Fields:
  src        image (png/jpg/webp) or video (mov/mp4/…); phone rotation is handled by ffmpeg
  start/end  song seconds the media is on screen
  clip_in    (video) source second that plays at `start`; the clip plays at normal speed, muted
  focus      [x, y] 0..1 point of the source to keep in frame when cropping (default centre)
  zoom       [from, to] slow push over the shot (Ken Burns). Default [1.04, 1.14] images, [1.08, 1.12] video
  grade      "theme" (default: desaturate, tint shadows/highlights to the theme, add contrast) or "none"
  dim        0..1 brightness kept (default 0.62) so the lyrics stay readable
  fade_in / fade_out   seconds (default 0.18 / 0.3). A cut-in also fires a short flash.

Video frames are pre-extracted once (prepare_media) into work/media_cache so parallel render
workers just read JPEGs.
"""
import hashlib, os, subprocess

import numpy as np
from PIL import Image

VIDEO_EXT = (".mov", ".mp4", ".m4v", ".mkv", ".webm", ".avi")
FPS = 30


def _open_image(path):
    if path.lower().endswith((".heic", ".heif")):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    from PIL import ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def is_video(path):
    return path.lower().endswith(VIDEO_EXT)


def _cache_dir(spec, W, H, root):
    key = (f"{os.path.abspath(spec['src'])}|{spec.get('clip_in', 0)}|{spec['end'] - spec['start']}|{W}x{H}|"
           f"{spec.get('focus')}|{spec.get('speed', 1.0)}|{spec.get('fit', 'cover')}")
    return os.path.join(root, hashlib.md5(key.encode()).hexdigest()[:12])


def prepare_media(plan, plan_dir):
    """Extract video frames for every video media entry (idempotent). Call once before rendering."""
    W, H = plan["video"]["width"], plan["video"]["height"]
    root = os.path.join(plan_dir, "media_cache")
    for m in plan.get("media", []):
        if not is_video(m["src"]):
            continue
        d = _cache_dir(m, W, H, root)
        n_need = int((m["end"] - m["start"]) * FPS) + 2
        if os.path.isdir(d) and len(os.listdir(d)) >= n_need - 2:
            continue
        os.makedirs(d, exist_ok=True)
        BW, BH = int(W * 1.2), int(H * 1.2)
        fx, fy = m.get("focus", [0.5, 0.5])
        sp = float(m.get("speed", 1.0))
        pre = f"setpts=PTS/{sp},fps={FPS}"
        if m.get("fit") == "contain":  # portrait clip: sharp in the middle, blurred copy filling the sides
            fc = (f"[0:v]{pre},split[a][b];[a]scale={BW}:{BH}:force_original_aspect_ratio=increase,"
                  f"crop={BW}:{BH},boxblur=24:2,eq=brightness=-0.12:saturation=0.8[bg];"
                  f"[b]scale=-2:{BH}[fg];[bg][fg]overlay=(W-w)/2:0[v]")
            filt = ["-filter_complex", fc, "-map", "[v]"]
        else:
            filt = ["-vf", f"{pre},scale={BW}:{BH}:force_original_aspect_ratio=increase,"
                           f"crop={BW}:{BH}:(iw-{BW})*{fx}:(ih-{BH})*{fy}"]
        dur = (m["end"] - m["start"] + 0.2) * sp
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(m.get("clip_in", 0.0)), "-t", f"{dur:.3f}",
                        "-i", m["src"], *filt, "-q:v", "3", os.path.join(d, "%05d.jpg")], check=True)
        print(f"[media] extracted {len(os.listdir(d))} frames for {os.path.basename(m['src'])} @ {m['start']}s")


class MediaLayer:
    def __init__(self, spec, W, H, plan_dir, theme):
        self.s = spec; self.W, self.H = W, H
        self.start, self.end = spec["start"], spec["end"]
        self.video = is_video(spec["src"])
        self.fin = spec.get("fade_in", 0.18); self.fout = spec.get("fade_out", 0.3)
        self.dim = spec.get("dim", 0.62)
        self.zoom = spec.get("zoom", [1.08, 1.12] if self.video else [1.04, 1.14])
        self.grade = spec.get("grade", "theme")
        self.theme = theme
        if self.video:
            self.dir = _cache_dir(spec, W, H, os.path.join(plan_dir, "media_cache"))
            self.frames = sorted(os.listdir(self.dir)) if os.path.isdir(self.dir) else []
        else:
            img = _open_image(spec["src"])
            BW, BH = int(W * 1.2), int(H * 1.2)
            if spec.get("fit") == "contain":
                k = max(BW / img.width, BH / img.height)
                bg = img.resize((int(img.width * k + 1), int(img.height * k + 1)), Image.BILINEAR)
                bg = bg.crop(((bg.width - BW) // 2, (bg.height - BH) // 2, (bg.width - BW) // 2 + BW, (bg.height - BH) // 2 + BH))
                from PIL import ImageEnhance, ImageFilter
                bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(BW / 60))).enhance(0.75)
                k2 = min(BW / img.width, BH / img.height)
                fg = img.resize((int(img.width * k2), int(img.height * k2)), Image.LANCZOS)
                bg.paste(fg, ((BW - fg.width) // 2, (BH - fg.height) // 2))
                img = bg
            k = max(BW / img.width, BH / img.height)
            img = img.resize((max(BW, int(img.width * k + 0.5)), max(BH, int(img.height * k + 0.5))), Image.LANCZOS)
            fx, fy = spec.get("focus", [0.5, 0.5])
            x0 = int((img.width - BW) * fx); y0 = int((img.height - BH) * fy)
            self.base = img.crop((x0, y0, x0 + BW, y0 + BH))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = np.clip(1.12 - 0.5 * r ** 2, 0.25, 1)[..., None].astype(np.float32)

    def active(self, t):
        return self.start <= t < self.end

    def alpha(self, t):
        a = 1.0
        if self.fin > 0:
            a = min(a, (t - self.start) / self.fin)
        if self.fout > 0:
            a = min(a, (self.end - t) / self.fout)
        return max(0.0, min(1.0, a))

    def _source(self, t):
        if self.video:
            if not self.frames:
                return None
            i = min(len(self.frames) - 1, max(0, int((t - self.start) * FPS)))
            return Image.open(os.path.join(self.dir, self.frames[i])).convert("RGB")
        return self.base

    def render(self, t):
        src = self._source(t)
        if src is None:
            return None
        u = (t - self.start) / max(0.01, self.end - self.start)
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * u
        BW, BH = src.size
        cw, ch = BW / (1.2 * z) * 1.0, BH / (1.2 * z)
        fx, fy = self.s.get("focus", [0.5, 0.5]) if not self.video else (0.5, 0.5)
        cx = min(max(fx * BW, cw / 2), BW - cw / 2)
        cy = min(max(fy * BH, ch / 2), BH - ch / 2)
        frame = src.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((self.W, self.H), Image.BILINEAR)
        a = np.asarray(frame, np.float32)
        if self.grade == "theme":
            lum = (a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11)[..., None]
            des = lum + (a - lum) * 0.5
            dark = np.array(self.theme.get("outline", (40, 6, 6)), np.float32) * 1.6
            light = np.array([255, 214, 186], np.float32)
            tone = dark + (light - dark) * (lum / 255)
            a = des * 0.5 + tone * 0.5
            a = (a - 118) * 1.22 + 118
            sm = np.array(self.theme.get("smoke", (170, 18, 22)), np.float32)
            a = a + sm * 0.10
        a = a * self.dim * self.vig
        return np.clip(a, 0, 255)


def build_layers(plan, plan_dir, theme):
    W, H = plan["video"]["width"], plan["video"]["height"]
    return [MediaLayer(m, W, H, plan_dir, theme) for m in plan.get("media", [])]


def composite(img, layers, t):
    """Blend the active media layer(s) over the background image (PIL RGBA, modified in place)."""
    act = [m for m in layers if m.active(t)]
    if not act:
        return
    base = np.asarray(img.convert("RGB"), np.float32)
    for m in act:
        lay = m.render(t)
        if lay is None:
            continue
        al = m.alpha(t) * 0.96
        base = base * (1 - al) + lay * al
    img.paste(Image.fromarray(base.astype(np.uint8)).convert("RGBA"))
