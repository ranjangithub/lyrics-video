# style_plan.json reference

`make_plan.py` writes a working default. Edit it before rendering; this is where the video gets its drama.

## Top level

| field | meaning |
|---|---|
| `video.width`, `video.height`, `video.fps` | 1280×720@30 by default; 1920×1080 with `--hd` (about 2.2× slower to render) |
| `video.theme` | `crimson-grunge` (red/black, default), `midnight-steel` (blue/steel), `saffron-fire` (orange/gold) |
| `video.font` | `anton` (default, heavy condensed), `bebas`, `teko`. Devanagari text always uses Teko Devanagari Bold |
| `video.shake_on_hits` | small camera shake on strong drum hits |
| `video.flash_on_slam` | colour flash when a hero word slams in |
| `title_card` | `{text, sub, start, end}` shown during the intro, or `null` |
| `chunks` | the on-screen lyric pieces, in time order (below) |
| `beats`, `hits`, `energy`, `energy_fps`, `audio`, `duration` | copied from analysis — leave alone |

## Chunk

```json
{
  "text": "WAALE NAHIN",
  "start": 5.30, "end": 6.90,
  "anim": "slam",
  "emphasis": "hero",
  "color": "red",
  "size": 1.0,
  "words": [{"text": "WAALE", "start": 5.35}, {"text": "NAHIN", "start": 5.87}]
}
```

- `start` / `end` — seconds on screen. Chunks must not overlap. `end` = next chunk's `start` − 0.02 gives a hard cut; leaving a gap lets the background breathe.
- `words` — the words shown (already in display form). `text` is just a label. `words[].start` matters only for `stamp`.
- `emphasis` — `hero` fills ~88% of the width and up to ~42% of the height with a heavier glow; `normal` is ~27% height.
- `size` — multiplier on the height budget (e.g. 1.3 for a giant single word, 0.8 to calm a long phrase).
- `color` — `white` (pink-white distressed), `red` (theme accent), `gold`.
- `y` — vertical centre of the lyric, 0..1 (default 0.5). Use ~0.8 when a prop fills the upper frame.
- `over` — small clean caption above the main text (e.g. `"VS PAKISTAN  •  2012  •  ODI"`); `over_size` (default 0.16), `over_color`.
- `sub` — small clean caption under the main text (e.g. `"INTERNATIONAL RUNS"`); `sub_size` (fraction of the main font, default 0.24), `sub_color`.

### Animations (`anim`)

| anim | feel | best for |
|---|---|---|
| `slam` | drops from 2.3× scale and hits with a flash and camera shake | payoff words, hook, beat drops |
| `stamp` | each word slams in on its own sung timing | 3-word chunks, chants, counting |
| `punch` | pops up from small with overshoot | setup phrases |
| `glitch` | RGB split and sliced flicker, then settles | tension, "warning" lines, verses |
| `rise` | slides up and fades in | soft or reflective lines |
| `shake` | pops in and keeps trembling | anger, rage, chaos |
| `fade` | plain fade | quiet intros/outros |
| `count` | a number rolls up from 0 (keeps commas, decimals and suffixes like `254*`) | stats cards, scores, records |

**Stats intro recipe:** in an instrumental intro, make one chunk per stat on consecutive downbeats
(`beats.json` → `downbeats`): `{"text": "28,527", "anim": "count", "emphasis": "hero", "size": 1.55, "y": 0.42,
"sub": "INTERNATIONAL RUNS", "sub_size": 0.2, "color": "gold"}`. Lead with a name card (`glitch`, `sub: "BY THE NUMBERS"`),
end with the song title. For a "highlights" section (e.g. CHASE MASTER), slam a title card, then one card per
innings: `over` = opponent • year • format, main = score with `count`, `sub` = balls • target • result; give each
card 2–3 bars and put a matching photo behind it via `media` (lyric `y` ~0.65 so the face stays clear).
Only use figures the user gives you (or that you verify), and add up totals yourself.

**End-credits outro (when the user wants only lyrics on screen during the song):** move every stat, highlight
card, chart, caption and credit to after the song. Build the outro's audio from the song's own instrumental passages,
cut on downbeats (`beats.json` → `downbeats`) with 15 ms fades at each cut and a 2–3 s fade-out at the end; append it
to the song (ffmpeg `atrim` + `concat`), re-run `beats.py` on the combined file, point the plan's `audio`, `beats`,
`hits`, `energy` and `duration` at it, and shift the cards to the new times. Keep the artist credit small
(`size` ~0.4, `emphasis: "normal"`, role in `over`).

## Props (illustrations that fly in, optionally on fire)

```json
"props": [
  {"image": "bat",  "start": 17.28, "end": 18.12, "anim": "swing",    "fire": true, "y": 0.4, "angle": -55, "scale": 1.15},
  {"image": "ball", "start": 18.14, "end": 18.98, "anim": "fireball", "fire": true, "y": 0.37}
]
```

- `image` — built-in `bat` or `ball` (drawn in code), or a path to any transparent PNG (user artwork, a cut-out photo).
- `anim` — `swing` (arcs in from the right, follow-through exit), `fireball` (rockets in from the top corner, bursts toward the camera on exit), `slam`, `float`.
- `fire` — procedural flames that trail the motion and rise; `false` for a clean object.
- `x`, `y` (0..1 centre), `scale` (bat: 1 ≈ 78% of frame height; ball: 1 ≈ 42%), `angle` (resting tilt in degrees).
- Usually tie a prop to the chunk that names it (same start/end) and move that chunk's lyric to `y: 0.8`.
- `python scripts/props.py --posters DIR` writes stand-alone flaming bat/ball images.

## Charts (animated leaderboard bars)

```json
"charts": [{"start": 27.58, "end": 32.65, "title": "MOST HUNDREDS IN WINNING CHASES",
            "subtitle": "ODI  •  AS REPORTED SEPT 2026",
            "items": [{"label": "Virat Kohli", "value": 25, "highlight": true},
                      {"label": "Sachin Tendulkar", "value": 14}, {"label": "Rohit Sharma", "value": 14}]}]
```

Bars grow one after another, values count up, the `highlight` row is in the theme colour with a glow and a bright
leading edge. Give it 3 bars of music; 2–5 rows read best. `max` sets the scale (default: largest value).
Cite where/when the figures come from in `subtitle`.

## Scenes (generic cricket visuals drawn in code — fill empty stretches)

```json
"scenes": [{"type": "stadium", "start": 76.4, "end": 85.4},
           {"type": "crowd",   "start": 85.4, "end": 94.5, "seed": 5},
           {"type": "finale",  "start": 237.9, "end": 250.4}]
```

- `stadium` — night ground from the upper stands: floodlight towers, lit oval with pitch and LED boards, a packed bowl
  of fans with phone lights, camera flashes, and a crowd wave that flashes saffron/white/green as it rolls round.
- `crowd` — backlit fan silhouettes jumping on the beat, arms up, waving tricolours, phone torches, confetti.
- `finale` — stadium + fireworks bursting on the beat + front-row fans + confetti. Pair with a closing title chunk.
- Fully generic: no real people, logos or teams. `seed` varies the crowd; `fade` sets the crossfade (default 0.45 s).
- Find empty stretches by listing gaps between chunks/media/props/charts longer than ~2 s; instrumental breaks and
  long outros are the best places. Short gaps (2–3 s) suit a quick `crowd` cut.

## Shots (designed media — prefer these over plain `media` backdrops)

Plain photo/clip backdrops read as "showing pictures", and phone/TV footage looks amateur. `shots` make the media
part of the film:

- `cutout` — `scripts/cutout.py PHOTO -o cut/x.png` removes the background (ONNX matting; models in `models/`, from the
  rembg GitHub releases: `isnet-general-use.onnx` default, `u2net_human_seg.onnx` when isnet drops limbs). The person is
  then placed in the video's own world: red duotone, rim light, glow, slowly rotating light rays from behind the head
  (`head_x`/`head_y` locate it in the cut-out), slow push. `front: true` puts the person in front of the lyrics —
  only when the text stays readable (it usually doesn't over a centred head).
- `full` — full-frame cinematic: cover-cropped to 16:9 around `focus`, denoised + de-moiréd, auto-levelled (fixes washed
  TV recordings), graded, vignetted, 2.39-style letterbox (`letterbox: false` behind stats).
- `panels` — 2–3 slanted strips with red edges that slide in one after another: the designed way to use portrait
  phone clips. Never use blurred side-bars ("contain") — they scream phone video.
- Transitions `in`/`out`: whip, zoom, glitch, burn, reveal, rise, punch, slide, fade, cut. Enter on downbeats; vary them.
- **Opening rule:** never open on a photo. Open on the video's own world (fire props in the dark), then the first
  cut-out *rises out of the smoke*, then a cinematic full-frame clip, then panels.
- Lyrics over a shot sit at `y: 0.8` (size ≤ 0.9).

```json
{"kind": "cutout", "src": "cut/IMG_6390.png", "start": 3.878, "end": 7.268, "in": "reveal", "out": "zoom",
 "x": 0.56, "height": 0.98, "head_x": 0.62, "head_y": 0.25, "rays": true}
{"kind": "full", "src": "clip.mov", "start": 7.268, "end": 10.658, "clip_in": 0, "speed": 0.71,
 "focus": [0.5, 0.3], "in": "whip", "out": "whip"}
{"kind": "panels", "start": 10.658, "end": 14.048, "stagger": 0.32, "out": "glitch",
 "items": [{"src": "a.mov", "speed": 0.46}, {"src": "b.mov", "speed": 0.38}, {"src": "c.mov", "speed": 0.7}]}
```

## Media (the user's photos and video clips as backdrops)

```json
"media": [
  {"src": "/abs/path/roar.png", "start": 120.72, "end": 126.6, "focus": [0.55, 0.2], "zoom": [1.02, 1.16]},
  {"src": "/abs/path/clip.mov", "start": 191.95, "end": 203.55, "clip_in": 0.0, "dim": 0.7, "fade_out": 0},
  {"src": "/abs/path/clip.mov", "start": 203.55, "end": 205.85, "clip_in": 13.5, "fade_in": 0}
]
```

- Photos get a slow Ken Burns push (`zoom` from→to) around `focus` (x, y of the source, 0..1 — put it on the face).
- Videos play muted at normal speed from `clip_in`; split one clip into several entries to skip bad sections
  (phone-recorded screens often have blurry pans) and to sync a key moment (a six, a wicket) to a lyric word:
  `start = word_time − moment_in_clip + clip_in`.
- Everything is graded to the theme (desaturated, tinted, contrasty, vignetted) and dimmed (`dim`, default 0.62) so
  footage looks like one film and lyrics stay readable. `grade: "none"` keeps original colour.
- A cut-in fires a flash; `fade_in: 0` makes a hard cut (use between pieces of the same clip).
- Move lyrics that play over media to `y: 0.8` and shrink normal chunks (`size: 0.85`) so faces stay visible.
- Frames are cached in `media_cache/` next to the plan; rotation from phone videos is handled.
- `speed` — playback rate: `0.6` slow-motion for a shot you want to savour, `2`–`5` for a fast-forward montage
  (set `speed = source_seconds / slot_seconds` to fit a clip into a beat-length slot).
- `fit: "contain"` — show the whole frame (portrait phone clips, square photos) sharp in the middle with a blurred,
  darkened copy filling the sides, instead of cropping to 16:9. Use it whenever a crop would cut off the face.
  HEIC photos from iPhones are read directly.
- **Montage recipes:** a slow-motion "shots" montage in an instrumental break (one clip per bar, a one-word caption per
  clip at `y: 0.84` — CLASS, TIMING, POWER, SIX!); and a fast-forward finale (every clip at 2–5x, cut every half bar
  with `fade_in: 0.04` so each cut flashes, a couple of big stat `count` cards on top, then the `finale` scene + title).
- When a photo's face sits low in the frame, put the lyrics over it at `y: 0.2` instead of 0.8.

## Creative rules of thumb

1. **Hero words should be rare enough to hit.** Roughly a third of chunks. The hook and the last word of a powerful line earn it; filler words ("se", "the", "ko") never do.
2. **One or two words on screen.** Three only with `stamp`. Split long lines.
3. **Vary the animation** between neighbouring chunks; never three `slam`s in a row unless it's a deliberate chant.
4. **Keep the hook consistent.** Every repeat of the chorus line should look the same (same anim/colour), so it becomes the video's signature.
5. **Use `color: red` sparingly** — on the hook or the single most violent word.
6. **Quiet passages:** `rise`/`fade`, `normal` emphasis, maybe `size` 0.85.
7. **Title card** during an instrumental intro longer than ~2.5 s.
