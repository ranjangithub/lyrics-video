#!/usr/bin/env python3
"""
props.py — illustrated objects (props) that fly in with the lyrics, with procedural fire.

Built-in props (drawn in code, no external art needed):
  bat   cricket bat, scorched willow with a red-striped grip
  ball  red leather cricket ball with stitched seam

Make the poster images (the props on fire, as stand-alone PNGs):
    python props.py --posters OUT_DIR

In a style plan, add:
  "props": [
    {"image": "bat",  "start": 17.28, "end": 18.10, "anim": "swing",    "fire": true},
    {"image": "ball", "start": 18.12, "end": 19.00, "anim": "fireball", "fire": true}
  ]
`image` may also be a path to any PNG with transparency (your own artwork or photo cut-out).
Anims: swing (bat-like arc from the right), fireball (rockets in from the top corner, grows),
slam (drops in big), float (fades in and drifts). Optional: "x", "y" (0..1 frame position of
the prop's centre), "scale" (1 = about 70% of frame height), "angle" (resting tilt, degrees).
"""
import math, os, random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "assets", "fonts")


def _noise(h, w, scale, rng, octaves=4, stretch=(1, 1)):
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for _ in range(octaves):
        sh = max(2, int(h / (scale * stretch[0])) + 2); sw = max(2, int(w / (scale * stretch[1])) + 2)
        small = (rng.random((sh, sw)) * 255).astype(np.uint8)
        out += amp * np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32) / 255
        tot += amp; amp *= 0.5; scale /= 2
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


# ------------------------------------------------------------------ bat
def make_bat(height=1400, seed=3, logo="प्रचंड"):
    """Vertical cricket bat, handle up, blade down. Returns RGBA."""
    rng = np.random.default_rng(seed)
    H = height; W = int(H * 0.15)
    S = 2  # supersample for smooth edges
    h, w = H * S, W * S
    cx = w / 2
    top = h * 0.012
    grip_end = h * 0.31          # handle ~31% of length
    shoulder_end = h * 0.36      # short sloping shoulders
    toe = h * 0.995
    handle_w = w * 0.30
    blade_w = w * 0.96
    corner = w * 0.10
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([cx - handle_w / 2, top, cx + handle_w / 2, shoulder_end], radius=handle_w * 0.35, fill=255)
    pts = []
    for i in range(31):  # concave shoulder curve, right side
        u = i / 30
        y = grip_end + (shoulder_end - grip_end) * u
        half = handle_w / 2 + (blade_w / 2 - handle_w / 2) * math.sin(u * math.pi / 2) ** 0.8
        pts.append((cx + half, y))
    pts += [(cx + blade_w / 2, toe - corner)]
    for i in range(11):
        a_ = math.pi / 2 * i / 10
        pts.append((cx + blade_w / 2 - corner + corner * math.cos(a_), toe - corner + corner * math.sin(a_)))
    for i in range(11):
        a_ = math.pi / 2 + math.pi / 2 * i / 10
        pts.append((cx - blade_w / 2 + corner + corner * math.cos(a_), toe - corner + corner * math.sin(a_)))
    pts += [(cx - blade_w / 2, toe - corner)]
    for i in range(30, -1, -1):
        u = i / 30
        y = grip_end + (shoulder_end - grip_end) * u
        half = handle_w / 2 + (blade_w / 2 - handle_w / 2) * math.sin(u * math.pi / 2) ** 0.8
        pts.append((cx - half, y))
    d.polygon(pts, fill=255)
    m = np.asarray(mask, np.float32) / 255

    # willow with long grain, rounded (cylindrical) shading across the face
    grain = _noise(h, w, w / 3, rng, 5, stretch=(18, 1))
    fine = _noise(h, w, w / 14, rng, 3, stretch=(40, 1))
    xs = (np.arange(w) - cx) / (blade_w / 2)
    face = np.clip(np.cos(np.clip(xs, -1, 1) * 1.25), 0.15, 1)[None, :]
    spine = np.exp(-(xs ** 2) / 0.05)[None, :]
    base = np.array([205, 150, 82], np.float32)
    light = np.array([245, 206, 140], np.float32)
    t = np.clip(0.3 + 0.45 * grain + 0.3 * fine + 0.2 * spine, 0, 1)
    col = base + (light - base) * t[..., None]
    col = col * (0.45 + 0.6 * face[..., None])
    # scorched toe and edges (it's a bat on fire)
    yy = np.linspace(0, 1, h)[:, None]
    burn = np.clip((yy - 0.72) * 3.0, 0, 1) * (0.55 + 0.45 * _noise(h, w, w / 3, rng, 4))
    burn = np.clip(burn + np.clip(np.abs(xs) - 0.82, 0, 1)[None, :] * 2.5 * (yy > 0.36), 0, 1)
    col = col * (1 - burn[..., None] * 0.8) + np.array([45, 18, 8], np.float32) * burn[..., None] * 0.8

    # grip: dark rubber with red spiral, cylindrical shading
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    in_grip = (Y > top + h * 0.03) & (Y < grip_end) & (np.abs(X - cx) < handle_w * 0.55)
    stripe = (((Y + (X - cx) * 0.6) / (h * 0.02)) % 1.0) < 0.3
    gcol = np.where(stripe[..., None], np.array([210, 24, 32], np.float32), np.array([26, 22, 24], np.float32))
    hx = np.clip((X - cx) / (handle_w / 2), -1, 1)
    gshade = np.clip(np.cos(hx * 1.3) * 1.15, 0.25, 1.15)
    col = np.where(in_grip[..., None], gcol * gshade[..., None], col)
    knob = (Y <= top + h * 0.03)
    col = np.where(knob[..., None], np.array([150, 110, 64], np.float32) * gshade[..., None], col)

    img = Image.fromarray(np.dstack([np.clip(col, 0, 255), m * 255]).astype(np.uint8), "RGBA")
    # logo across the top of the blade: "प्रचंड"
    try:
        dev = any("\u0900" <= ch <= "\u097F" for ch in logo)
        f = ImageFont.truetype(os.path.join(FONTS, "Teko-Devanagari-Bold.woff" if dev else "Anton-Latin.woff"),
                               int(blade_w * (0.42 if dev else 0.6 / max(1, len(logo) / 4))))
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(lay).text((cx, shoulder_end + h * 0.06), logo, font=f, fill=(200, 18, 24, 255), anchor="mm",
                                 stroke_width=max(2, int(blade_w * 0.015)), stroke_fill=(25, 5, 3, 255))
        la = np.asarray(lay, np.float32)
        la[..., 3] *= m
        img.alpha_composite(Image.fromarray(la.astype(np.uint8), "RGBA"))
    except Exception:
        pass
    a_ = img.split()[3]
    ol = Image.new("RGBA", img.size, (14, 5, 2, 0)); ol.putalpha(a_.filter(ImageFilter.MaxFilter(7)))
    ol.alpha_composite(img)
    return ol.resize((W, H), Image.LANCZOS)


# ------------------------------------------------------------------ ball
def make_ball(size=900, seed=5, spin=0.0):
    """Red leather cricket ball with a stitched seam. spin rotates the seam (radians)."""
    rng = np.random.default_rng(seed)
    S = 2; n = size * S; R = n * 0.46; c = n / 2
    Y, X = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (X - c) / R; y = (Y - c) / R
    r2 = x * x + y * y
    inside = r2 <= 1
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    diff = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1)
    rim = np.clip(1 - z, 0, 1) ** 2.5
    leather = _noise(n, n, n / 40, rng, 3)
    base = np.array([150, 14, 20], np.float32)
    col = base * (0.25 + 0.95 * diff[..., None]) * (0.9 + 0.2 * leather[..., None])
    # specular
    Hh = L + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
    spec = np.clip(x * Hh[0] + y * Hh[1] + z * Hh[2], 0, 1) ** 40
    col += np.array([255, 220, 210], np.float32) * spec[..., None] * 0.75
    col += np.array([255, 60, 20], np.float32) * rim[..., None] * 0.35
    img = Image.fromarray(np.dstack([np.clip(col, 0, 255), inside * 255]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)

    # seam: a great circle tilted toward the viewer
    tilt, yaw = 1.15, 0.35 + spin
    def rot(p):
        x_, y_, z_ = p
        # tilt around X then yaw around Z
        y2 = y_ * math.cos(tilt) - z_ * math.sin(tilt); z2 = y_ * math.sin(tilt) + z_ * math.cos(tilt)
        x3 = x_ * math.cos(yaw) - y2 * math.sin(yaw); y3 = x_ * math.sin(yaw) + y2 * math.cos(yaw)
        return x3, y3, z2
    pts = []
    for i in range(721):
        a = 2 * math.pi * i / 720
        pts.append(rot((math.cos(a), math.sin(a), 0.0)))
    def to2d(p): return (c + p[0] * R, c + p[1] * R)
    # seam band
    for i in range(720):
        p, q = pts[i], pts[i + 1]
        if p[2] > 0.02 and q[2] > 0.02:
            wdt = max(2, int(R * 0.045 * (0.35 + 0.65 * p[2])))
            d.line([to2d(p), to2d(q)], fill=(235, 225, 205, 255), width=wdt)
    # stitches either side of the seam
    for side in (-1, 1):
        for i in range(0, 720, 9):
            p, q = pts[i], pts[i + 1]
            if p[2] <= 0.25:
                continue
            tx, ty = q[0] - p[0], q[1] - p[1]
            nl = math.hypot(tx, ty) + 1e-9
            nx, ny = -ty / nl, tx / nl
            off = R * 0.05 * side * (0.3 + 0.7 * p[2])
            a0 = (c + (p[0] * R) + nx * off, c + (p[1] * R) + ny * off)
            a1 = (a0[0] + nx * side * R * 0.035 * (0.3 + 0.7 * p[2]), a0[1] + ny * side * R * 0.035 * (0.3 + 0.7 * p[2]))
            d.line([a0, a1], fill=(250, 245, 235, 255), width=max(2, int(R * 0.016 * (0.4 + 0.6 * p[2]))))
    # re-shade seam with the sphere lighting so it sits on the surface; clip anything off the sphere
    arr = np.asarray(img, np.float32)
    arr[..., 3] = np.where(inside, arr[..., 3], 0)
    shade = (0.35 + 0.75 * diff)[..., None]
    seam = (arr[..., 0] > 200) & inside
    arr[..., :3] = np.where(seam[..., None], arr[..., :3] * shade, arr[..., :3])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.LANCZOS)


def builtin(name, logo=None):
    if name == "bat":
        return make_bat(logo=logo) if logo else make_bat()
    if name == "ball":
        return make_ball()
    return Image.open(name).convert("RGBA")


# ------------------------------------------------------------------ fire
FIRE_RAMP = np.array([  # age 0 -> 1
    [255, 250, 220], [255, 214, 110], [255, 150, 40], [235, 70, 15], [150, 22, 8], [40, 8, 4]], np.float32)


def fire_color(u):
    u = min(max(u, 0.0), 0.999) * (len(FIRE_RAMP) - 1)
    i = int(u); f = u - i
    return FIRE_RAMP[i] * (1 - f) + FIRE_RAMP[i + 1] * f


def edge_points(img, n=400, seed=1):
    """Sample points on the opaque silhouette (sprite coords, centred)."""
    a = np.asarray(img.split()[3], np.uint8) > 128
    er = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)), np.uint8) > 128
    edge = np.argwhere(a & ~er)
    inner = np.argwhere(a)
    rng = np.random.default_rng(seed)
    e = edge[rng.integers(0, len(edge), int(n * 0.75))]
    i = inner[rng.integers(0, len(inner), n - len(e))]
    pts = np.vstack([e, i]).astype(np.float32)
    pts[:, 0] -= img.height / 2; pts[:, 1] -= img.width / 2
    return pts[:, ::-1]  # (x, y)


# ------------------------------------------------------------------ animation
def ease_out_back(x, s=1.8):
    x = min(max(x, 0), 1) - 1; return x * x * ((s + 1) * x + s) + 1


def ease_out_cubic(x):
    x = min(max(x, 0), 1); return 1 - (1 - x) ** 3


class Prop:
    def __init__(self, spec, W, H, cache):
        self.s = spec; self.W, self.H = W, H
        key = spec["image"]
        ck = (key, spec.get("logo"))
        if ck not in cache:
            cache[ck] = builtin(key, spec.get("logo"))
        src = cache[ck]
        scale = spec.get("scale", 1.0)
        target_h = H * (0.78 if key == "bat" else 0.42) * scale
        k = target_h / src.height
        self.img = src.resize((max(1, int(src.width * k)), max(1, int(src.height * k))), Image.LANCZOS)
        self.pts = edge_points(self.img, 500)
        self.start, self.end = spec["start"], spec["end"]
        self.anim = spec.get("anim", "float")
        self.fire = spec.get("fire", True)
        self.rng = random.Random(hash((key, self.start)) & 0xffffffff)
        self.spin = key == "ball"

    def pose(self, t):
        """(cx, cy, angle_deg, scale, alpha) at time t."""
        W, H = self.W, self.H
        tr = t - self.start; left = self.end - t
        x = self.s.get("x", 0.5) * W; y = self.s.get("y", 0.4) * H
        ang = self.s.get("angle", -32.0 if self.anim == "swing" else 0.0)
        sc, al = 1.0, 1.0
        if self.anim == "swing":
            e = ease_out_back(tr / 0.26, 1.4)
            a0 = ang + 125
            ang = a0 + (ang - a0) * e
            x = W * 1.25 + (x - W * 1.25) * ease_out_cubic(tr / 0.22)
            y = -H * 0.2 + (y + H * 0.2) * ease_out_cubic(tr / 0.22)
            ang += 2.5 * math.sin(tr * 7)
            if left < 0.16:  # follow-through swing out
                k = 1 - left / 0.16
                ang -= 70 * k * k; x -= W * 0.35 * k * k; al = 1 - k
        elif self.anim == "fireball":
            e = ease_out_cubic(tr / 0.28)
            sx, sy = W * 1.1, -H * 0.15
            x = sx + (x - sx) * e; y = sy + (y - sy) * e
            sc = 0.25 + 0.75 * e
            y += math.sin(tr * 6) * H * 0.008
            if left < 0.15:
                k = 1 - left / 0.15
                sc *= 1 + 0.9 * k; al = 1 - k
        elif self.anim == "slam":
            e = min(max(tr / 0.14, 0), 1)
            sc = 2.2 + (1 - 2.2) * e * e
            al = min(1, tr / 0.06)
            if left < 0.12: al *= left / 0.12
        else:  # float
            al = min(1, tr / 0.3) * min(1, left / 0.3)
            y -= tr * H * 0.02
        return x, y, ang, sc, max(0.0, min(1.0, al))

    def velocity(self, t, dt=1 / 60):
        a = self.pose(t - dt); b = self.pose(t)
        return (b[0] - a[0]) / dt, (b[1] - a[1]) / dt

    def transform(self, pts, pose):
        x, y, ang, sc, _ = pose
        r = math.radians(-ang)
        c, s = math.cos(r), math.sin(r)
        px = pts[:, 0] * sc; py = pts[:, 1] * sc
        return np.stack([x + px * c - py * s, y + px * s + py * c], 1)

    def draw_sprite(self, frame, t):
        x, y, ang, sc, al = self.pose(t)
        if al <= 0.01:
            return
        img = self.img
        if self.spin:
            img = img.rotate(-(t - self.start) * 420, resample=Image.BICUBIC)
        if abs(sc - 1) > 0.01:
            img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
        if abs(ang) > 0.1:
            img = img.rotate(ang, resample=Image.BICUBIC, expand=True)
        if al < 0.999:
            r, g, b, a = img.split(); img = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * al))))
        _paste(frame, img, int(x - img.width / 2), int(y - img.height / 2))

    def _particles(self):
        """Pre-compute every flame particle (birth pose, speed, wobble) once; frames are then stateless."""
        if getattr(self, "_P", None) is not None:
            return self._P
        rate = 2400.0
        n = max(1, int((self.end - self.start) * rate))
        rg = np.random.default_rng(int(self.start * 1000) + 17)
        b = self.start + np.arange(n) / rate
        sp = self.W / 1280
        idx = rg.integers(0, len(self.pts), n)
        ex = np.empty(n, np.float32); ey = np.empty(n, np.float32)
        vx = np.empty(n, np.float32); vy = np.empty(n, np.float32); al = np.empty(n, np.float32)
        sc = np.empty(n, np.float32)
        for i in range(n):
            pose = self.pose(b[i])
            p = self.transform(self.pts[idx[i]][None], pose)[0]
            ex[i], ey[i] = p
            vx[i], vy[i] = self.velocity(b[i])
            al[i] = pose[4]; sc[i] = pose[3]
        self._P = dict(
            b=b, life=rg.uniform(0.35, 0.95, n), ex=ex, ey=ey,
            ux=-vx * 0.3 + rg.normal(0, 55, n) * sp,
            uy=-vy * 0.3 - rg.uniform(200, 430, n) * sp * np.sqrt(np.maximum(sc, 0.2)),
            ph=rg.uniform(0, 6.28, n), fr=rg.uniform(5, 13, n), amp=rg.uniform(10, 34, n) * sp,
            w=rg.uniform(0.6, 1.4, n) * al * 0.45, sp=sp)
        return self._P

    def heat(self, t, grid, k):
        """Splat this prop's flame particles into a heat grid (grid scale k = grid px per frame px)."""
        if not self.fire:
            return
        P = self._particles()
        tt = t - P["b"]
        age = tt / P["life"]
        m = (tt >= 0) & (age < 1)
        if not m.any():
            return
        tt, age = tt[m], age[m]
        wob = np.sin(P["ph"][m] + tt * P["fr"][m]) * P["amp"][m] * np.minimum(tt * 4, 1)
        x = P["ex"][m] + P["ux"][m] * tt + wob
        y = P["ey"][m] + P["uy"][m] * tt - 140 * P["sp"] * tt * tt
        w = P["w"][m] * (1 - age) ** 1.4
        gh, gw = grid.shape
        # three taps along the path give flame tongues instead of dots
        for j, f in enumerate((0.0, 0.012, 0.024, 0.036, 0.048, 0.06)):
            xx = ((x - P["ux"][m] * f) * k).astype(np.int32)
            yy = ((y - P["uy"][m] * f) * k).astype(np.int32)
            ok = (xx >= 0) & (xx < gw) & (yy >= 0) & (yy < gh)
            np.add.at(grid, (yy[ok], xx[ok]), w[ok] * (1.0 - 0.12 * j))


def _paste(frame, spr, x, y):
    W, H = frame.size; w, h = spr.size
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    cx0, cy0 = max(0, -x), max(0, -y)
    cx1, cy1 = min(w, W - x), min(h, H - y)
    frame.alpha_composite(spr.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))


FIRE_STOPS = np.array([0.0, 0.18, 0.38, 0.58, 0.78, 1.0])
FIRE_COLS = np.array([[0, 0, 0], [110, 8, 0], [215, 45, 0], [255, 125, 15], [255, 205, 80], [255, 248, 215]], np.float32)


def fire_rgb(heat):
    h = np.clip(heat, 0, 1)
    return np.stack([np.interp(h, FIRE_STOPS, FIRE_COLS[:, c]) for c in range(3)], -1)


def draw_props(frame, props, t, gain=1.0):
    """Fire behind the object, the object, then a lighter fire pass in front so flames lick over it."""
    active = [p for p in props if p.start - 0.01 <= t <= p.end + 0.8]
    if not active:
        return
    from scipy.ndimage import gaussian_filter
    W, H = frame.size
    k = 0.5
    grid = np.zeros((int(H * k), int(W * k)), np.float32)
    for p in active:
        p.heat(t, grid, k)
    sp = W / 1280
    heat = gaussian_filter(grid, 2.4 * sp) * 0.7 + gaussian_filter(grid, 6.0 * sp) * 1.6
    heat = 1 - np.exp(-heat * 20.0 * gain)  # soft saturation
    fire = fire_rgb(heat)
    fire = np.asarray(Image.fromarray(fire.astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32)
    base = np.asarray(frame.convert("RGB"), np.float32)
    base = np.clip(base + fire, 0, 255)
    frame.paste(Image.fromarray(base.astype(np.uint8)).convert("RGBA"))
    for p in active:
        if p.start <= t <= p.end:
            p.draw_sprite(frame, t)
    base = np.asarray(frame.convert("RGB"), np.float32)
    base = 255 - (255 - base) * (255 - fire * 0.35) / 255  # screen
    frame.paste(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA"))


def make_posters(out_dir, W=1600, H=1600):
    """Stand-alone images: the flaming bat and the flaming ball on a dark grunge backdrop."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4)
    bg = _noise(H, W, W / 5, rng, 5)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = np.clip(1.1 - 0.6 * r ** 1.6, 0.1, 1)
    smoke = np.clip((_noise(H, W, W / 2.5, rng, 4) - 0.45) * 2.4, 0, 1) ** 1.5
    base = (np.array([10, 7, 7]) + np.array([48, 26, 22]) * bg[..., None] + np.array([150, 16, 20]) * smoke[..., None] * 0.45) * v[..., None]
    backdrop = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    out = []
    for name, spec in [("flaming_bat", {"image": "bat", "anim": "float", "x": 0.5, "y": 0.55, "angle": -32, "scale": 1.0}),
                       ("flaming_ball", {"image": "ball", "anim": "float", "x": 0.5, "y": 0.6, "scale": 1.15})]:
        frame = backdrop.copy()
        spec = dict(spec, start=0.0, end=10.0, fire=True)
        p = Prop(spec, W, H, {})
        p.spin = False
        p.pose = (lambda pp: (lambda t: (pp.s["x"] * W, pp.s["y"] * H, pp.s.get("angle", 0.0), 1.0, 1.0)))(p)
        p.velocity = lambda t: (0.0, 0.0)
        p.start, p.end = 0.0, 3.0
        draw_props(frame, [p], 2.9)
        path = os.path.join(out_dir, f"{name}.png")
        frame.convert("RGB").save(path)
        out.append(path)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--posters", help="write flaming_bat.png and flaming_ball.png here")
    ap.add_argument("--raw", help="write plain bat.png and ball.png (transparent) here")
    a = ap.parse_args()
    if a.raw:
        os.makedirs(a.raw, exist_ok=True)
        make_bat().save(os.path.join(a.raw, "bat.png")); make_ball().save(os.path.join(a.raw, "ball.png"))
        print(os.path.join(a.raw, "bat.png"), os.path.join(a.raw, "ball.png"))
    if a.posters:
        for p in make_posters(a.posters):
            print(p)
