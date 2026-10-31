#!/usr/bin/env python3
"""
cover.py — cover art / thumbnail in the video's style.

    python cover.py -o out/cover.png --photo PHOTO --focus 0.55 0.3 \
        --title "CRICKET" "HAI" "PRACHAND" --accent 2 --credit "ft. RAJESH RANJAN" \
        [--kicker "क्रिकेट है प्रचंड"] [--size 1280 720] [--no-bat] [--no-ball]

Layout: grunge smoke background, the photo graded into the theme on the right (feathered),
a big stacked title on the left (line --accent in red), flaming bat and ball, credit at the bottom.
Works for 16:9 thumbnails and square covers (in square, the photo goes behind the title at the top).
"""
import argparse, math, os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
import props as propslib  # noqa: E402


def graded_photo(path, W, H, focus, theme):
    img = Image.open(path).convert("RGB")
    k = max(W / img.width, H / img.height)
    img = img.resize((int(img.width * k + 0.5), int(img.height * k + 0.5)), Image.LANCZOS)
    x0 = int((img.width - W) * focus[0]); y0 = int((img.height - H) * focus[1])
    a = np.asarray(img.crop((x0, y0, x0 + W, y0 + H)), np.float32)
    lum = (a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11)[..., None]
    des = lum + (a - lum) * 0.45
    dark = np.array(theme["outline"], np.float32) * 1.6
    light = np.array([255, 220, 196], np.float32)
    tone = dark + (light - dark) * (lum / 255)
    a = des * 0.45 + tone * 0.55
    a = (a - 115) * 1.3 + 112 + np.array(theme["smoke"], np.float32) * 0.08
    return np.clip(a, 0, 255)


def static_prop(kind, W, H, x, y, angle, scale, logo=None):
    spec = {"image": kind, "start": 0.0, "end": 3.0, "anim": "float", "fire": True, "scale": scale}
    if logo:
        spec["logo"] = logo
    p = propslib.Prop(spec, W, H, {})
    p.spin = False
    p.pose = lambda t: (x * W, y * H, angle, 1.0, 1.0)
    p.velocity = lambda t: (0.0, 0.0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--photo")
    ap.add_argument("--focus", nargs=2, type=float, default=[0.5, 0.3])
    ap.add_argument("--title", nargs="+", required=True)
    ap.add_argument("--accent", type=int, default=-1, help="index of the title line drawn in red")
    ap.add_argument("--kicker", default="")
    ap.add_argument("--credit", default="")
    ap.add_argument("--credit-label", default="", help="small line above the credit, e.g. 'LYRICS & CONCEPT BY'")
    ap.add_argument("--size", nargs=2, type=int, default=[1280, 720])
    ap.add_argument("--theme", default="crimson-grunge")
    ap.add_argument("--logo", default="PRACHAND")
    ap.add_argument("--no-bat", action="store_true")
    ap.add_argument("--no-ball", action="store_true")
    a = ap.parse_args()

    W, H = a.size
    square = abs(W - H) < 10
    plan = {"video": {"width": W, "height": H, "fps": 30, "theme": a.theme, "font": "anton"}, "chunks": []}
    sc = render.Scene(plan, seed=3)
    th = sc.theme
    img = sc.background(4.0, 0.8, 0.2)
    base = np.asarray(img.convert("RGB"), np.float32)

    # photo, feathered into the background
    if a.photo:
        if square:
            pw, ph, px, py = W, int(H * 0.72), 0, 0
        else:
            pw, ph, px, py = int(W * 0.62), H, W - int(W * 0.62), 0
        photo = graded_photo(a.photo, pw, ph, a.focus, th)
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        if square:
            m = np.clip((ph - yy) / (ph * 0.42), 0, 1) ** 1.4
        else:
            m = np.clip(xx / (pw * 0.45), 0, 1) ** 1.3
        m = (m * 0.92)[..., None]
        reg = base[py:py + ph, px:px + pw]
        base[py:py + ph, px:px + pw] = reg * (1 - m) + photo * m
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    img.alpha_composite(sc.particles_layer(4.0, 0.9))

    # flaming bat and ball
    pr = []
    if square:
        if not a.no_bat:
            pr.append(static_prop("bat", W, H, 0.20, 0.36, -38, 0.85, a.logo))
        if not a.no_ball:
            pr.append(static_prop("ball", W, H, 0.88, 0.30, 0, 0.42))
    else:
        if not a.no_bat:
            pr.append(static_prop("bat", W, H, 0.53, 0.52, -28, 1.02, a.logo))
        if not a.no_ball:
            pr.append(static_prop("ball", W, H, 0.90, 0.80, 0, 0.40))
    if pr:
        propslib.draw_props(img, pr, 2.9, gain=1.0)

    # title block
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rng = np.random.default_rng(5)
    if square:
        cx, top, maxw = W * 0.5, H * 0.47, W * 0.9
    else:
        cx, top, maxw = W * 0.255, H * 0.12, W * 0.46
    y = top
    if a.kicker:
        ks = int(H * (0.05 if square else 0.065))
        kspr, _ = sc.word_sprite(a.kicker, ks, "gold", False, seed=7, clean=True)
        render.paste_sprite(layer, kspr, cx, y + ks * 0.5, 1.0, 1.0, 0, 0, rng, th)
        y += ks * 1.25
    for i, line in enumerate(a.title):
        size = int(H * (0.17 if square else 0.2))
        f = sc.font_for(line, size)
        while f.getlength(line) > maxw and size > 20:
            size = int(size * 0.94); f = sc.font_for(line, size)
        bb = f.getbbox(line)
        ink_h = bb[3] - bb[1]
        hero = i == a.accent or (a.accent < 0 and i == len(a.title) - 1)
        spr, pad = sc.word_sprite(line, size, "red" if hero else "white", hero, seed=31 + i)
        render.paste_sprite(layer, spr, cx, y + ink_h / 2 + (spr.size[1] / 2 - pad - (bb[1] + bb[3]) / 2) * 0, 1.0, 1.0, 0, 0, rng, th)
        y += ink_h * 1.12
    img.alpha_composite(layer)

    # credit strip
    if a.credit:
        cs = int(H * (0.05 if square else 0.06))
        band = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        by = H * (0.915 if square else 0.875)
        bw = W * (0.62 if square else 0.4)
        bcx = W * 0.5 if square else cx
        for k in range(40):
            al = int(150 * (1 - k / 40) ** 1.5)
            bd.rectangle([bcx - bw / 2 - k * 6, by - cs * 0.85, bcx + bw / 2 + k * 6, by + cs * 0.85], fill=(0, 0, 0, al // 6))
        ly = by - cs * (1.0 if a.credit_label else 0.95)
        bd.line([bcx - bw / 2, ly, bcx + bw / 2, ly], fill=tuple(th["red"]) + (230,), width=max(2, int(H * 0.004)))
        img.alpha_composite(band.filter(ImageFilter.GaussianBlur(3)))
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        if a.credit_label:
            ls = int(cs * 0.5)
            lspr, _ = sc.word_sprite(a.credit_label, ls, "white", False, seed=78, clean=True)
            render.paste_sprite(lay, lspr, bcx, by - cs * 0.42, 1.0, 1.0, 0, 0, rng, th)
        cspr, _ = sc.word_sprite(a.credit, cs, "gold", True, seed=77, clean=True)
        render.paste_sprite(lay, cspr, bcx, by + (cs * 0.42 if a.credit_label else cs * 0.1), 1.0, 1.0, 0, 0, rng, th)
        img.alpha_composite(lay)

    # final grain + vignette
    out = np.asarray(img.convert("RGB"), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    out *= np.clip(1.1 - 0.35 * r ** 2, 0.4, 1)[..., None]
    out += np.random.default_rng(1).normal(0, 4, out.shape[:2])[..., None]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
