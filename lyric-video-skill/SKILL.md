---
name: lyric-video
description: Make a dramatic kinetic-typography lyric video (MP4) from an MP3 plus pasted lyrics — big bold distressed words that slam, glitch and pop in time with the singing, over a grunge red/black animated background, with the user's photos and clips designed in. Use when the user uploads a song and wants a lyric video, lyric visualiser or animated lyrics, in Hindi, Hinglish, English or Devanagari.
---

# Lyric video

Turns **song audio + the lyrics the user pastes** into a cinematic lyric video:
one or two huge words at a time, timed to the vocals, with slams, glitches, camera shake,
embers and red smoke (other themes available).

Pipeline:

```
MP3 ─► separate.py ─► vocals.wav ─► ctc_align.py (+ lyrics.txt) ─► words.json (+ words.lrc)
        (fallback: align.py, no models)
MP3 ─────────────► beats.py ─► beats.json
words + beats ───► make_plan.py ─► style_plan.json ─► (Claude refines) ─► render.py ─► MP4
```

All scripts live in `scripts/`, fonts in `assets/fonts/` (SIL OFL). Work in a `work/` folder.

## 0. Setup (once per session)

```bash
pip install librosa soundfile scipy pillow numpy sherpa-onnx onnxruntime pillow-heif   # add --break-system-packages in sandboxes
which ffmpeg                                            # required
mkdir -p models
# vocal separation (~67 MB) and multilingual speech model for word timing (~290 MB), both from GitHub releases
curl -L -o models/UVR-MDX-NET-Voc_FT.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2 | tar xj -C models/
# person cut-outs for photos (~175 MB each; isnet is the default, u2net_human_seg a fallback for missed limbs)
curl -L -o models/isnet-general-use.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx
curl -L -o models/u2net_human_seg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx
```

If the downloads are blocked, skip them: the no-model `align.py` fallback still works (less precise).

## 1. Collect inputs

- The **audio** file (mp3/wav/m4a).
- The **lyrics**: pasted by the user, or embedded in the file. Suno and many MP3s carry them in a
  tag — check first: `ffprobe -v error -show_entries format_tags -of default=nw=1 SONG.mp3`.
  Save them to `work/lyrics.txt`, one sung line per line, in the script to show on screen
  (Devanagari, Hinglish/Latin, English). Drop `[Chorus]`/`(Music)`-style tags and stray `..` stretch marks.
  Repeated choruses must appear every time they are sung.
- If the lyrics already have LRC timestamps (`[00:12.40]line`), keep them — exact line starts.
- Use the file's title tag (or ask) for the intro card; default 720p, 1080p on request.

Only use lyrics the user provides or that ship inside their file. Don't look up lyrics of commercial songs.

## 2. Align lyrics to the audio

**Best path (word-level):** isolate the vocals, then force-align the lyrics with the speech model.

```bash
python scripts/separate.py SONG.mp3 -o work/vocals.wav            # ~1.1x song length on 2 cores, 10 s blocks
python scripts/ctc_align.py work/vocals.wav work/lyrics.txt --audio SONG.mp3 -o work/words.json
```

`ctc_align.py` matches Devanagari lyrics even when the model "hears" Hindi in Urdu or Gurmukhi script.
Memory: separation needs ~200 MB per second of block; keep `--block 10` (default) on 8 GB machines.

Check the result — it prints every line's start/end:
- Sung lines should last roughly the same each time they repeat (a chorus line ≈ same length every chorus).
- **Shouted chants / gang vocals** ("Cricket hai prachand!") are recognised poorly. A chant line that stretches over many
  seconds, or sits in an instrumental gap, is misplaced. Fix it by hand in `words.json`: find the vocal bursts with
  the vocal stem's energy (`align.vocal_score(load_audio('work/vocals.wav'), clean=True)` → frames at 86/s) and put
  each chant on its burst.
- `--skip-pen` lets the aligner drop lyric lines that aren't sung; leave it high (default 30) unless the lyrics
  clearly contain unsung lines, because low values make it skip hard-to-hear real lines.
- Long instrumental gaps (>8 s) can carry a small `glitch` title chunk so the screen isn't empty.

**Fallback (no models):**

```bash
python scripts/align.py SONG.mp3 work/lyrics.txt -o work/words.json [--vocals work/vocals.wav]
```

`auto` picks **lrc** (timestamps given) → **whisper** (stable-ts, if installed) → **vocal** (energy-based phrase fitting;
approximate, poor on dense mixes where singing never pauses). Show the user `work/words.lrc`; wrong lines can be fixed by
editing their `[mm:ss.xx]` and rerunning `align.py SONG.mp3 work/words.lrc -o work/words.json`.

## 3. Beats

```bash
python scripts/beats.py SONG.mp3 -o work/beats.json
```

## 4. Draft the style plan

```bash
python scripts/make_plan.py work/words.json work/beats.json -o work/style_plan.json --title "SONG TITLE" [--subtitle "ARTIST"] [--hd]
```

## 5. Direct it (this is Claude's job)

Read `references/style_plan.md`, then open `work/style_plan.json` and edit the chunks like a motion designer:

- **Re-chunk** so each chunk is a meaningful 1–2 word punch (split/merge `words`, keep their `start`s, keep chunks non-overlapping).
- **Choose hero words** by meaning: the defiant verb, the name, the hook. Demote filler.
- **Pick animations** to match the lyric's emotion (`slam` for impact, `glitch` for tension, `rise` for tender lines, `shake` for rage, `stamp` for chants).
- Make every repeat of the **hook** identical, and the one most violent word `color: "red"`.
- Choose a **theme** that fits the song (`crimson-grunge` for aggressive/defiant, `saffron-fire` for devotional or celebratory, `midnight-steel` for dark/cold).

- **Illustrate key nouns with props** when the user asks (or a word is a strong visual): add a `props` entry on the chunk
  that sings it — e.g. a flaming bat on "लम्बदण्ड" (bat) and a fireball on "गोलपिंड" (ball) — and drop that lyric to `y: 0.8`.
  Built-in: `bat`, `ball`; any transparent PNG the user supplies also works. See `references/style_plan.md`.

- **Credit the user** the way they ask (e.g. "LYRICS & CONCEPT BY …") on the cover and as an early title card
  (`over` = the role, main text = the name, gold).
- **User photos / clips** go in `media` (see `references/style_plan.md`): put each where the lyric talks about it
  (a match clip on the line naming that match, a roaring face on "roar"), sync the clip's key moment to the key word,
  skip unusable stretches by splitting the clip, and use the instrumental gaps for extra photos. Ask the user to
  only use footage they're allowed to publish if the video is going public.

Validate by eye with stills at hero moments (land ~0.2 s after a chunk's start):

```bash
python scripts/render.py work/style_plan.json --stills 12.3,31.0,58.4 -o work/stills
```

View the PNGs, fix anything cramped, too small, or badly split, and re-check.

## 6. Render

```bash
python scripts/render.py work/style_plan.json --range 30 45 -o out/preview.mp4   # optional quick preview
python scripts/render.py work/style_plan.json -o out/lyric_video.mp4
```

Speed on 2 CPU cores: about 1 minute per 25 s of song at 720p, roughly double at 1080p (a 4-minute song ≈ 10 min at 720p).
Run long renders in the background. Grain makes files big (~5 Mbps); before delivering, make a download-friendly copy:
`ffmpeg -i out/lyric_video.mp4 -vf hqdn3d=3:3:6:6 -c:v libx264 -preset faster -crf 30 -c:a aac -b:a 128k -movflags +faststart out/final.mp4` (~25 MB for 4 min). Then deliver the MP4 to the user, with one line on the alignment method
used and an offer to fix any line that's off.

## 7. Cover art and share preview

```bash
python scripts/cover.py -o out/cover_youtube.png --photo PHOTO --focus 0.6 0.15 \
  --title CRICKET HAI PRACHAND --accent 2 --kicker "क्रिकेट है प्रचंड" \
  --credit-label "LYRICS & CONCEPT BY" --credit "ARTIST NAME"
python scripts/cover.py -o out/cover_square.png --size 1600 1600 ...same options...
```

Messaging apps show a video's first frame as its preview, and the video fades in from black — so always put the
cover on the opening frames and embed it as the file's thumbnail:

```bash
ffmpeg -i out/final.mp4 -loop 1 -t 1.2 -i out/cover_youtube.png -i out/cover_youtube.png -filter_complex \
 "[1:v]scale=1280:720,format=rgba,fade=t=out:st=0.6:d=0.45:alpha=1[c];[0:v][c]overlay=eof_action=pass,format=yuv420p[v]" \
 -map "[v]" -map 0:a -map 2:v -c:v:0 libx264 -preset faster -crf 30 -c:a copy -c:v:1 mjpeg \
 -disposition:v:1 attached_pic -movflags +faststart out/final_with_cover.mp4
```

## Troubleshooting

- *Text too small:* chunk has too many letters — split it, or raise `size`.
- *Lines drift later in the song:* lyrics are missing a repeated chorus or include unsung lines; fix the text, or pin starts via LRC.
- *Mixed-script chunk renders boxes:* a chunk mixing Devanagari and Latin uses the Devanagari font; split them into separate chunks.
- *Whisper unavailable:* expected in restricted sandboxes; use separate.py + ctc_align.py, or LRC timing.
- *Separation killed (exit 137):* out of memory — lower `--block` to 6.
