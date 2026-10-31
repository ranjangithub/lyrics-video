#!/usr/bin/env python3
"""
scenes.py — animated, generic cricket scenes drawn entirely in code (no photos, no real people).

  stadium   night stadium from the upper stands: floodlight towers, lit oval, LED boards,
            a sea of fans in blue/saffron with twinkling phone lights and camera flashes
  crowd     backlit silhouettes of fans jumping on the beat, arms up, waving tricolours, confetti
  finale    stadium + fireworks bursting on the beat + front-row fan silhouettes + confetti

In a style plan:
  "scenes": [{"type": "stadium", "start": 76.3, "end": 85.4},
             {"type": "crowd",   "start": 85.4, "end": 94.5},
             {"type": "finale",  "start": 238.0, "end": 250.3}]
Optional per scene: "fade" (crossfade seconds, default 0.45), "seed".
Scenes replace the grunge background while active; lyrics, props and particles stay on top.
"""
import math, random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TRICOLOUR = [(255, 153, 51), (245, 245, 240), (19, 136, 8)]


def _ell(u, v, cu, cv, ru, rv):
    return ((u - cu) / ru) ** 2 + ((v - cv) / rv) ** 2


def _blob(r, col):
    """Additive glow sprite (float32 HxWx3)."""
    n = int(r * 2 + 1)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - r
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r / 2.6) ** 2))
    return g[..., None] * np.array(col, np.float32)[None, None]


def _add(arr, spr, x, y):
    h, w = spr.shape[:2]
    x0, y0 = int(x - w / 2), int(y - h / 2)
    H, W = arr.shape[:2]
    a0, b0 = max(0, x0), max(0, y0)
    a1, b1 = min(W, x0 + w), min(H, y0 + h)
    if a1 <= a0 or b1 <= b0:
        return
    arr[b0:b1, a0:a1] += spr[b0 - y0:b1 - y0, a0 - x0:a1 - x0]


class Stadium:
    """Static stadium base (bigger than the frame for camera moves) + dynamic lights."""

    def __init__(self, W, H, seed=11):
        self.W, self.H = W, H
        S = 1.32
        BW, BH = int(W * S), int(H * S)
        self.BW, self.BH = BW, BH
        rng = np.random.default_rng(seed)
        vv, uu = np.mgrid[0:BH, 0:BW].astype(np.float32)
        u, v = uu / BW, vv / BH
        img = np.zeros((BH, BW, 3), np.float32)
        # sky: deep night with a warm haze at the stadium rim
        sky_t = np.clip(v / 0.42, 0, 1)[..., None]
        img[:] = np.array([5, 5, 14]) * (1 - sky_t) + np.array([46, 20, 34]) * sky_t
        # bowl geometry
        cu, cv = 0.5, 0.70
        out_e = _ell(u, v, cu, 0.67, 0.80, 0.40)
        fld_e = _ell(u, v, cu, cv, 0.37, 0.135)
        stands = (out_e <= 1) & (fld_e > 1.0)
        # stands: tiers, darker toward the top, slight blue cast
        r_norm = np.clip((np.sqrt(np.maximum(out_e, 0)) - 0.45) / 0.55, 0, 1)
        tier = (np.sin(r_norm * 2 * math.pi * 6) > 0.92) * 0.6
        base = np.array([30, 26, 38], np.float32) * (1.1 - 0.6 * r_norm)[..., None]
        img = np.where(stands[..., None], base * (1 - tier[..., None] * 0.6), img)
        # roof rim with LED strip
        rim = (out_e > 0.94) & (out_e <= 1.02) & (v < 0.62)
        img = np.where(rim[..., None], np.array([12, 10, 16], np.float32), img)
        # crowd: lots of 1-px fans in jersey colours
        n = int(70000 * (BW * BH) / (1690 * 950))
        cu_ = rng.uniform(0, 1, n * 3); cv_ = rng.uniform(0.25, 1, n * 3)
        keep = (_ell(cu_, cv_, cu, 0.67, 0.80, 0.40) < 0.93) & (_ell(cu_, cv_, cu, cv, 0.37, 0.135) > 1.08)
        cu_, cv_ = cu_[keep][:n], cv_[keep][:n]
        pal = np.array([[45, 95, 210], [70, 130, 235], [30, 70, 170], [255, 150, 50], [235, 235, 230],
                        [40, 140, 60], [200, 60, 50], [120, 110, 120]], np.float32)
        pw = np.array([0.24, 0.16, 0.14, 0.14, 0.1, 0.06, 0.06, 0.1])
        cols = pal[rng.choice(len(pal), len(cu_), p=pw)]
        depth = np.clip((np.sqrt(_ell(cu_, cv_, cu, 0.67, 0.80, 0.40)) - 0.45) / 0.55, 0, 1)
        bright = (1.15 - 0.5 * depth)[:, None] * rng.uniform(0.6, 1.05, (len(cu_), 1))
        self.cx_ = (cu_ * BW).astype(int).clip(0, BW - 2); self.cy_ = (cv_ * BH).astype(int).clip(0, BH - 2)
        self.ccol = (cols * bright).astype(np.float32)
        self.cang = np.arctan2((cv_ - 0.67) / 0.40, (cu_ - cu) / 0.80)  # position around the bowl
        # field: striped green, lit in the middle, LED boards ring + rope + pitch
        stripes = (np.floor((u - cu) / 0.035) % 2) * 0.09
        fld = np.array([34, 118, 52], np.float32) * (1 + stripes)[..., None]
        light = np.clip(1.35 - 0.75 * np.sqrt(np.maximum(fld_e, 0)), 0.5, 1.35)
        img = np.where((fld_e <= 1)[..., None], fld * light[..., None], img)
        boards = (fld_e > 1.0) & (fld_e < 1.09)
        bcol = np.where(((u * 40).astype(int) % 5 < 3)[..., None], np.array([40, 120, 255], np.float32),
                        np.array([255, 120, 30], np.float32))
        img = np.where(boards[..., None], bcol * 0.85, img)
        rope = (fld_e > 0.90) & (fld_e < 0.925)
        img = np.where(rope[..., None], np.array([210, 210, 200], np.float32), img)
        pitch = (np.abs(u - cu) < 0.009 + (v - cv) * 0.012) & (np.abs(v - cv) < 0.024)
        img = np.where(pitch[..., None], np.array([196, 170, 118], np.float32) * 1.15, img)
        crease = pitch & ((np.abs(v - (cv - 0.019)) < 0.0018) | (np.abs(v - (cv + 0.019)) < 0.0018))
        img = np.where(crease[..., None], np.array([250, 250, 245], np.float32), img)
        self.base = np.clip(img, 0, 255)

        # roof LED strip & floodlight towers (additive bloom layer, pulses with the beat)
        bloom = np.zeros_like(self.base)
        for k in range(160):
            a = math.pi * (0.04 + 0.92 * k / 159)
            x = (cu - 0.80 * 0.965 * math.cos(a)) * BW
            y = (0.67 - 0.40 * 0.965 * math.sin(a)) * BH
            if y < 0.6 * BH:
                _add(self.base, _blob(2.5, (255, 240, 210)), x, y)
                _add(bloom, _blob(9, (90, 70, 50)), x, y)
        self.towers = []
        for tu, top in ((0.08, 0.10), (0.30, 0.04), (0.70, 0.04), (0.92, 0.10)):
            x, y = tu * BW, top * BH
            d = Image.fromarray(self.base.astype(np.uint8)); dr = ImageDraw.Draw(d)
            foot = 0.40 * BH if tu in (0.30, 0.70) else 0.5 * BH
            dr.line([(x, y + BH * 0.03), (x, foot)], fill=(14, 12, 18), width=max(3, int(BW * 0.004)))
            dr.line([(x - BW * 0.012, foot), (x, y + BH * 0.05), (x + BW * 0.012, foot)], fill=(14, 12, 18), width=2)
            pw_, ph_ = BW * 0.055, BH * 0.035
            dr.rectangle([x - pw_ / 2 - 3, y - ph_ / 2 - 3, x + pw_ / 2 + 3, y + ph_ / 2 + 3], fill=(20, 18, 24))
            for i in range(6):
                for j in range(3):
                    cx = x - pw_ / 2 + pw_ * (i + 0.5) / 6; cy = y - ph_ / 2 + ph_ * (j + 0.5) / 3
                    dr.ellipse([cx - pw_ / 16, cy - ph_ / 8, cx + pw_ / 16, cy + ph_ / 8], fill=(255, 250, 235))
            self.base = np.asarray(d, np.float32)
            _add(bloom, _blob(BW * 0.06, (255, 230, 190)), x, y)
            _add(bloom, _blob(BW * 0.16, (120, 70, 60)), x, y)
            self.towers.append((x, y))
        # soft light cones toward the field
        cone = Image.new("L", (BW, BH), 0); cd = ImageDraw.Draw(cone)
        for x, y in self.towers:
            cd.polygon([(x - BW * 0.02, y), (x + BW * 0.02, y), (cu * BW + (x - cu * BW) * 0.25 + BW * 0.12, cv * BH),
                        (cu * BW + (x - cu * BW) * 0.25 - BW * 0.12, cv * BH)], fill=26)
        cone = np.asarray(cone.filter(ImageFilter.GaussianBlur(BW * 0.02)), np.float32)[..., None] / 255
        bloom += cone * np.array([255, 235, 200], np.float32) * 0.9
        self.bloom = bloom
        # phone lights for twinkling (subset of crowd positions)
        m = min(1400, len(cu_))
        idx = rng.choice(len(cu_), m, replace=False)
        self.ph_x, self.ph_y = cu_[idx] * BW, cv_[idx] * BH
        self.ph_phase = rng.uniform(0, 6.28, m); self.ph_freq = rng.uniform(0.6, 2.2, m)
        self.flash = _blob(7, (255, 255, 255))
        self.flash_core = _blob(2.2, (255, 255, 255))
        self.cpts = (cu_, cv_)
        self.blurred = None

    def frame(self, t, tr, energy, pulse, zoom=(1.0, 1.08), pan=(0.0, -0.02), dur=10.0, blurred=False):
        W, H, BW, BH = self.W, self.H, self.BW, self.BH
        u = min(max(tr / max(dur, 0.1), 0), 1)
        z = zoom[0] + (zoom[1] - zoom[0]) * (u * u * (3 - 2 * u))
        cw, ch = BW / 1.32 / z * 1.25, BH / 1.32 / z * 1.25
        cx = BW / 2 + pan[0] * BW * u; cy = BH / 2 + pan[1] * BH * u - BH * 0.03
        x0, y0 = cx - cw / 2, cy - ch / 2
        src = self.base if not blurred else self._blurred()
        k = 1.0 + 0.35 * pulse + 0.25 * energy
        arr = src + self.bloom * k
        # phone lights + flashes in base coords
        tw = np.sin(self.ph_phase + t * self.ph_freq * 2) * 0.5 + 0.5
        on = tw > 0.55
        xi = self.ph_x[on].astype(int); yi = self.ph_y[on].astype(int)
        b = ((tw[on] - 0.55) / 0.45)[:, None] * 255
        arr = arr.copy()
        # crowd (2x2 px) with a stadium wave travelling round the bowl, flashing the tricolour
        dphi = (((self.cang - (t * 1.1)) + math.pi) % (2 * math.pi) - math.pi)
        wave = np.exp(-(dphi ** 2) / 0.06)
        tri = np.array(TRICOLOUR, np.float32)
        band = np.clip(((dphi / 0.25) + 1.5).astype(int), 0, 2)  # saffron / white / green across the wave
        cc = self.ccol * (1 - 0.75 * wave[:, None]) + tri[band] * (wave[:, None] * 1.25)
        cy2 = self.cy_ - (wave * 2).astype(int)
        for dy in (0, 1):
            for dx in (0, 1):
                arr[cy2 + dy, self.cx_ + dx] = cc
        arr[yi, xi] += b
        rr = random.Random(int(t * 30) * 7 + 3)
        for _ in range(int(3 + 9 * energy + 10 * pulse)):
            cu_, cv_ = self.cpts
            j = rr.randrange(len(cu_))
            fx, fy = cu_[j] * BW, cv_[j] * BH
            _add(arr, self.flash * 0.9, fx, fy); _add(arr, self.flash_core, fx, fy)
        crop = np.clip(arr[int(max(0, y0)):int(min(BH, y0 + ch)), int(max(0, x0)):int(min(BW, x0 + cw))], 0, 255)
        img = Image.fromarray(crop.astype(np.uint8)).resize((W, H), Image.BILINEAR)
        return np.asarray(img, np.float32)

    def _blurred(self):
        if self.blurred is None:
            self.blurred = np.asarray(Image.fromarray(self.base.astype(np.uint8)).filter(ImageFilter.GaussianBlur(7)), np.float32) * 0.7
        return self.blurred


class Crowd:
    """Backlit fan silhouettes in rows; jump on beats, wave arms, tricolour flags, phone torches."""

    def __init__(self, W, H, seed=5, rows=((0.30, 30, 0.84, (48, 28, 36)), (0.42, 18, 0.96, (22, 12, 17)),
                                           (0.60, 10, 1.16, (6, 3, 5)))):
        self.W, self.H = W, H
        rr = random.Random(seed)
        self.rows = []
        for hs, n, base_y, col in rows:
            people = []
            for i in range(n):
                x = (i + 0.5 + rr.uniform(-0.35, 0.35)) / n * W * 1.08 - W * 0.04
                h = hs * H * rr.uniform(0.88, 1.12)
                mode = rr.choices(["v", "fist", "flag", "phone", "clap"], [0.36, 0.24, 0.13, 0.12, 0.15])[0]
                people.append(dict(x=x, h=h, by=base_y * H, mode=mode, ph=rr.uniform(0, 6.28),
                                   jump=rr.uniform(0.5, 1.0), lean=rr.uniform(-6, 6), side=rr.choice([-1, 1])))
            self.rows.append((col, people, hs))

    def draw(self, layer, t, beat_phase, energy, rim=(255, 120, 60)):
        d = ImageDraw.Draw(layer)
        for ri, (col, people, hs) in enumerate(self.rows):
            rim_k = (0.45, 0.75, 1.0)[min(ri, 2)] if len(self.rows) == 3 else (0.75, 1.0)[min(ri, 1)]
            rcol = tuple(int(c * rim_k) for c in rim)
            for p in people:
                self._person(layer, d, p, t, beat_phase, energy, col, rcol)

    def _person(self, layer, d, p, t, beat_phase, energy, col, rcol):
        h = p["h"]
        jmp = abs(math.sin(math.pi * ((beat_phase + p["ph"] / 6.28 * 0.3) % 1.0))) * h * 0.05 * p["jump"] * (0.6 + 0.6 * energy)
        bx, by = p["x"] + math.sin(t * 1.3 + p["ph"]) * h * 0.01, p["by"] - jmp
        neck = by - h * 0.60
        shw = h * 0.19
        hr = h * 0.085
        head = (bx + math.sin(t * 2 + p["ph"]) * h * 0.012, neck - hr * 1.15)
        sh_l, sh_r = (bx - shw, neck + h * 0.075), (bx + shw, neck + h * 0.075)
        wave = math.sin(t * 7 + p["ph"]) * 14
        pump = abs(math.sin(math.pi * beat_phase)) * 35
        arms = []
        for s, (sx, sy) in ((-1, sh_l), (1, sh_r)):
            m = p["mode"]
            if m == "v":
                a1 = -90 + s * (32 + wave); a2 = a1 + s * 8
            elif m == "fist":
                if s == p["side"]:
                    a1 = -90 + s * (8 + pump * 0.25); a2 = a1 - s * 5
                else:
                    a1 = 90 - s * 12; a2 = 90 - s * 30
            elif m in ("flag", "phone"):
                if s == p["side"]:
                    a1 = -90 + s * (14 + wave * 0.4); a2 = a1 - s * 8
                else:
                    a1 = 90 - s * 10; a2 = 90 - s * 35
            else:  # clap overhead
                c = abs(math.sin(math.pi * beat_phase * 2))
                a1 = -90 + s * (38 + 8 * c); a2 = -90 - s * (30 * c)
            a1, a2 = math.radians(a1), math.radians(a2)
            L1, L2 = h * 0.21, h * 0.20
            ex, ey = sx + math.cos(a1) * L1, sy + math.sin(a1) * L1
            hx, hy = ex + math.cos(a2) * L2, ey + math.sin(a2) * L2
            arms.append(((sx, sy), (ex, ey), (hx, hy), s))
        w1, w2, wh = h * 0.085, h * 0.062, h * 0.05

        def limb(p0, p1, wa, wb, off, grow, cc):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy) + 1e-6
            nx, ny = -dy / L, dx / L
            wa, wb = wa / 2 + grow, wb / 2 + grow
            d.polygon([(p0[0] + nx * wa, p0[1] + ny * wa + off), (p1[0] + nx * wb, p1[1] + ny * wb + off),
                       (p1[0] - nx * wb, p1[1] - ny * wb + off), (p0[0] - nx * wa, p0[1] - ny * wa + off)], fill=cc)

        for c, off, g in ((rcol, -max(1.0, h * 0.010), max(1.0, h * 0.006)), (col, 0, 0)):
            cc = tuple(int(v) for v in c) + (255,)
            # torso with rounded shoulders, continuing off the bottom of the frame
            d.polygon([(sh_l[0] - g, sh_l[1] + off), (bx - h * 0.05, neck + off - g), (bx + h * 0.05, neck + off - g),
                       (sh_r[0] + g, sh_r[1] + off), (bx + shw * 1.05 + g, by + h * 0.6), (bx - shw * 1.05 - g, by + h * 0.6)], fill=cc)
            for sx, sy in (sh_l, sh_r):
                r = h * 0.065 + g
                d.ellipse([sx - r, sy - r + off, sx + r, sy + r + off], fill=cc)
            # neck + head
            d.rectangle([bx - h * 0.04 - g, head[1] + off, bx + h * 0.04 + g, neck + h * 0.03 + off], fill=cc)
            d.ellipse([head[0] - hr * 0.86 - g, head[1] - hr - g + off, head[0] + hr * 0.86 + g, head[1] + hr * 1.05 + g + off], fill=cc)
            for (s0, e, hnd, s) in arms:
                limb(s0, e, w1, w2, off, g, cc)
                limb(e, hnd, w2, w2 * 0.85, off, g, cc)
                r = w2 * 0.55 + g
                d.ellipse([e[0] - r, e[1] - r + off, e[0] + r, e[1] + r + off], fill=cc)
                r = wh * 0.7 + g
                d.ellipse([hnd[0] - r, hnd[1] - r + off, hnd[0] + r, hnd[1] + r + off], fill=cc)
        for (s0, e, hnd, s) in arms:
            if s != p["side"]:
                continue
            if p["mode"] == "flag":
                self._flag(layer, d, hnd, h, t, p["ph"], s)
            elif p["mode"] == "phone":
                r = h * 0.2
                glow = Image.new("RGBA", (int(2 * r), int(2 * r)), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glow)
                for k in range(6, 0, -1):
                    gd.ellipse([r - r * k / 6, r - r * k / 6, r + r * k / 6, r + r * k / 6], fill=(255, 250, 235, 12 + 14 * (6 - k)))
                layer.alpha_composite(glow, (int(hnd[0] - r), int(hnd[1] - r - h * 0.04)))
                d.ellipse([hnd[0] - 2.5, hnd[1] - h * 0.04 - 2.5, hnd[0] + 2.5, hnd[1] - h * 0.04 + 2.5], fill=(255, 255, 255, 255))

    def _flag(self, layer, d, hand, h, t, ph, s):
        pole_top = (hand[0] + s * h * 0.02, hand[1] - h * 0.36)
        d.line([hand, pole_top], fill=(30, 22, 18, 255), width=max(2, int(h * 0.02)))
        fw, fh = h * 0.34, h * 0.22
        n = 14
        dirx = s
        for bi, col in enumerate(TRICOLOUR):
            top, bot = [], []
            for i in range(n + 1):
                x = i / n
                wv = math.sin(x * 5.5 - t * 9 + ph) * fh * 0.13 * x
                px = pole_top[0] + dirx * x * fw
                top.append((px, pole_top[1] + fh * bi / 3 + wv))
                bot.append((px, pole_top[1] + fh * (bi + 1) / 3 + wv))
            shade = [int(c * 0.85) for c in col]
            d.polygon(top + bot[::-1], fill=tuple(shade) + (255,))
        # chakra
        x = 0.5; wv = math.sin(x * 5.5 - t * 9 + ph) * fh * 0.13 * x
        cx, cy = pole_top[0] + dirx * x * fw, pole_top[1] + fh / 2 + wv
        r = fh * 0.12
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(20, 30, 140, 255), width=max(1, int(r * 0.3)))


def confetti(arr, t, tr, W, H, n=260, seed=9):
    rr = np.random.default_rng(seed)
    x0 = rr.uniform(0, W, n); sp = rr.uniform(60, 160, n) * H / 720
    sw = rr.uniform(10, 40, n); ph = rr.uniform(0, 6.28, n); st = rr.uniform(-3, 6, n)
    cols = np.array([TRICOLOUR[0], TRICOLOUR[1], TRICOLOUR[2], (255, 210, 80), (255, 80, 60)], np.float32)
    c = cols[rr.integers(0, len(cols), n)]
    dt = tr - st
    m = dt > 0
    y = -20 + sp * dt
    x = x0 + np.sin(ph + dt * 3) * sw
    m &= (y < H + 10)
    flick = (np.abs(np.sin(ph + t * 8)) * 0.7 + 0.3)
    for i in np.nonzero(m)[0]:
        xi, yi = int(x[i]), int(y[i])
        if 0 <= xi < W - 4 and 0 <= yi < H - 3:
            w = 2 + int(2 * flick[i])
            arr[yi:yi + 2, xi:xi + w] = c[i] * flick[i] + 40


class Fireworks:
    def __init__(self, W, H, start, end, beats, seed=21):
        self.W, self.H = W, H
        rr = random.Random(seed)
        bs = [b for b in beats if start + 0.3 <= b <= end - 0.6]
        self.bursts = []
        for i, b in enumerate(bs):
            if i % 2 and rr.random() < 0.6:
                continue
            for _ in range(1 if rr.random() < 0.7 else 2):
                col = rr.choice([(255, 160, 60), (255, 250, 230), (60, 220, 90), (255, 210, 80), (255, 70, 60), (120, 170, 255)])
                cx, cy = rr.uniform(0.12, 0.88) * W, rr.uniform(0.07, 0.34) * H
                n = rr.randint(80, 120)
                ang = [rr.uniform(0, 6.283) for _ in range(n)]
                spd = [rr.uniform(0.75, 1.0) * H * 0.62 for _ in range(n)]
                self.bursts.append(dict(t=b, x=cx, y=cy, col=col, ang=ang, spd=spd, launch=0.45))

    def draw(self, t):
        W, H = self.W, self.H
        lay = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(lay)
        for b in self.bursts:
            dt = t - b["t"]
            if -b["launch"] < dt < 0:  # rocket trail
                k = 1 + dt / b["launch"]
                y = H * 0.75 + (b["y"] - H * 0.75) * k
                d.line([(b["x"], y), (b["x"], y + 26)], fill=(255, 210, 150), width=2)
            if 0 <= dt < 1.8:
                fade = (1 - dt / 1.8) ** 1.6
                col = tuple(int(c * fade) for c in b["col"])
                drag = 2.2
                for a, s in zip(b["ang"], b["spd"]):
                    r1 = s * (1 - math.exp(-drag * dt)) / drag
                    r0 = s * (1 - math.exp(-drag * max(0, dt - 0.09))) / drag
                    g1, g0 = 60 * dt * dt, 60 * max(0, dt - 0.09) ** 2
                    d.line([(b["x"] + math.cos(a) * r0, b["y"] + math.sin(a) * r0 + g0),
                            (b["x"] + math.cos(a) * r1, b["y"] + math.sin(a) * r1 + g1)], fill=col, width=3)
                    if dt > 0.5 and random.Random(int(t * 30) + int(a * 1000)).random() < 0.25:  # crackle
                        px, py = b["x"] + math.cos(a) * r1, b["y"] + math.sin(a) * r1 + g1
                        d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 250, 230))
                if dt < 0.12:
                    r = 40 * (1 - dt / 0.12)
                    d.ellipse([b["x"] - r, b["y"] - r, b["x"] + r, b["y"] + r], fill=(255, 240, 220))
        core = np.asarray(lay, np.float32)
        glow = np.asarray(lay.filter(ImageFilter.GaussianBlur(7)), np.float32)
        wide = np.asarray(lay.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(6)).resize((W, H)), np.float32)
        return core * 1.3 + glow * 2.2 + wide * 3.0


class SceneLayer:
    def __init__(self, spec, W, H, plan, theme):
        self.s = spec; self.W, self.H = W, H
        self.start, self.end = spec["start"], spec["end"]
        self.type = spec["type"]
        self.fade = spec.get("fade", 0.45)
        self.beats = np.array(plan.get("beats", []), float)
        self.theme = theme
        self.stadium = Stadium(W, H, spec.get("seed", 11)) if self.type in ("stadium", "crowd", "finale") else None
        self.crowd = None
        if self.type == "crowd":
            self.crowd = Crowd(W, H, spec.get("seed", 5))
        elif self.type == "finale":
            self.crowd = Crowd(W, H, spec.get("seed", 7), rows=((0.40, 15, 1.0, (24, 12, 18)), (0.60, 9, 1.2, (6, 3, 5))))
            self.fw = Fireworks(W, H, self.start, self.end, plan.get("beats", []), spec.get("seed", 21))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = np.clip(1.12 - 0.45 * r ** 2, 0.3, 1)[..., None]

    def alpha(self, t):
        return max(0.0, min(1.0, (t - self.start) / self.fade, (self.end - t) / self.fade))

    def beat_phase(self, t):
        prev = self.beats[self.beats <= t]
        nxt = self.beats[self.beats > t]
        if not len(prev) or not len(nxt):
            return (t * 2.4) % 1.0
        return (t - prev[-1]) / max(0.05, nxt[0] - prev[-1])

    def render(self, t, energy, pulse):
        tr = t - self.start; dur = self.end - self.start
        if self.type == "stadium":
            arr = self.stadium.frame(t, tr, energy, pulse, zoom=(1.0, 1.12), pan=(0.03, -0.03), dur=dur)
        elif self.type == "crowd":
            arr = self.stadium.frame(t, tr, energy, pulse, zoom=(1.08, 1.12), pan=(-0.02, 0.0), dur=dur, blurred=True)
            # backlight from the floodlights above
            glow = np.zeros_like(arr)
            _add(glow, _blob(self.W * 0.45, (255, 200, 150)), self.W * 0.5, -self.H * 0.15)
            arr = arr * 0.75 + glow * (0.45 + 0.25 * pulse)
            lay = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
            self.crowd.draw(lay, t, self.beat_phase(t), energy)
            arr = np.asarray(lay.convert("RGB"), np.float32)
            confetti(arr, t, tr, self.W, self.H)
        else:  # finale
            arr = self.stadium.frame(t, tr, energy, pulse, zoom=(1.0, 1.06), pan=(0.0, -0.04), dur=dur)
            arr = arr + self.fw.draw(t)
            lay = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
            self.crowd.draw(lay, t, self.beat_phase(t), energy, rim=(255, 170, 110))
            arr = np.asarray(lay.convert("RGB"), np.float32)
            confetti(arr, t, tr, self.W, self.H, n=180, seed=13)
        # light theme grade so scenes sit with the rest of the video
        lum = arr.mean(-1, keepdims=True)
        arr = arr * 0.85 + lum * 0.15
        arr = arr * np.array([1.04, 0.95, 0.95], np.float32)
        return np.clip(arr * self.vig, 0, 255)


def build(plan, theme):
    W, H = plan["video"]["width"], plan["video"]["height"]
    return [SceneLayer(s, W, H, plan, theme) for s in plan.get("scenes", [])]


def composite(img, layers, t, energy, pulse):
    act = [s for s in layers if s.start <= t < s.end]
    if not act:
        return
    base = np.asarray(img.convert("RGB"), np.float32)
    for s in act:
        a = s.alpha(t)
        base = base * (1 - a) + s.render(t, energy, pulse) * a
    img.paste(Image.fromarray(base.astype(np.uint8)).convert("RGBA"))
