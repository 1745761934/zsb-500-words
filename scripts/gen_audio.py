# -*- coding: utf-8 -*-
"""v6 audio build (2026-09-14 fix).

What changed vs v4 (gen_v4.py):
  * 44.1 kHz / 96 kbps instead of 24 kHz / 64 kbps  -> cleaner consonant onsets
    (Kokoro renders at 24 kHz; we resample up so the encoder uses MPEG-1 LIII
    instead of MPEG-2 LSF, which quantises speech much more coarsely).
  * ID3v2 header removed (ffmpeg writes one by default; some mobile decoders
    glitch on the first frames of short clips because of it).
Everything else is deliberately UNCHANGED from v4, because v4 already fixed the
real defect (start-of-word click) and its metrics verified clean:
  keep 30 ms pre-roll (preserves the natural attack) -> 12 ms fade-in -> 25 ms
  fade-out -> 60 ms tail pad -> frame-gated RMS norm to -20 dBFS -> alimiter.

Usage:  python work/gen_v6.py sample [n] | run [tier...]
"""
import os, sys, json, pathlib, re, subprocess, time
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import numpy as np
import onnxruntime as ort

_ORIG = ort.InferenceSession
def _patched(model_path, sess_options=None, providers=None, **kw):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return _ORIG(model_path, sess_options=so, providers=providers or ["CPUExecutionProvider"], **kw)
ort.InferenceSession = _patched

from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = str(ROOT / "work/kokoro/kokoro-v1.0.onnx")
VOICES = str(ROOT / "work/kokoro/voices-v1.0.bin")
VOICE = "af_heart"
SPEEDS = {"s": 0.53, "n": 0.64, "f": 0.80}     # same as v4 -> same speed/timbre as what is already online
OUT = ROOT / "work/site-v4/audio-v6"
FFM = r"C:\Users\17457\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
SR = 24000          # Kokoro native rate
OUT_SR = 44100      # encode rate
BITRATE = "96k"

# --- post processing (identical to v4 on purpose) --------------------------
def post(samples, sr=SR):
    x = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
    if x.size == 0:
        return x
    thr = 10 ** (-50.0 / 20.0)
    idx = np.where(np.abs(x) > thr)[0]
    if idx.size:
        a = max(0, idx[0] - int(0.030 * sr))       # 30 ms pre-roll keeps the natural attack
        b = min(x.size, idx[-1] + int(0.060 * sr))
        x = x[a:b]
    n = x.size
    k = int(0.012 * sr)                            # 12 ms fade in
    if k > 1 and n > 2 * k:
        x[:k] = x[:k] * np.linspace(0.0, 1.0, k, dtype=np.float32)
    k = int(0.025 * sr)                            # 25 ms fade out
    if k > 1 and n > 2 * k:
        x[-k:] = x[-k:] * np.linspace(1.0, 0.0, k, dtype=np.float32)
    x = np.concatenate([x, np.zeros(int(0.060 * sr), dtype=np.float32)])

    win, hop = int(0.025 * sr), int(0.010 * sr)
    if x.size >= win:
        fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop].astype(np.float64)
        s = np.sqrt((fr ** 2).mean(axis=1))
        keep = s > s.max() * 10 ** (-25.0 / 20.0)
        if keep.sum() < 2:
            keep = np.ones_like(s, bool)
        rms = float(np.sqrt((s[keep] ** 2).mean()))
    else:
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    x = x * np.float32((10 ** (-20.0 / 20.0)) / (rms or 1e-9))
    return np.clip(x, -1.0, 1.0).astype(np.float32)

def to_mp3(pcm_f32, dst):
    data = np.clip(pcm_f32, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16).tobytes()
    r = subprocess.run([FFM, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-af", "alimiter=limit=0.891:attack=5:release=60:level=disabled",
                        "-ar", str(OUT_SR),
                        "-b:a", BITRATE,
                        "-map_metadata", "-1", "-write_id3v2", "0", "-id3v2_version", "0",
                        str(dst)],
                       input=pcm, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 800

# --- job -------------------------------------------------------------------
_K = None
def init():
    global _K
    from kokoro_onnx import Kokoro
    _K = Kokoro(MODEL, VOICES)

def job(a):
    tier, name, text = a
    out = OUT / tier / name
    if out.exists() and out.stat().st_size > 800:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        samples, sr = _K.create(text, voice=VOICE, speed=SPEEDS[tier], lang="en-us")
        return 0 if to_mp3(post(samples, sr), out) else 1
    except Exception:
        return 1

# --- inputs ----------------------------------------------------------------
def load_jobs():
    sents = json.loads((ROOT / "work/tts_src/sentences.json").read_text(encoding="utf-8"))
    words = json.loads((ROOT / "work/tts_src/words.json").read_text(encoding="utf-8"))
    slug = lambda w: re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_") or "x"
    uniq = {}
    for w in words:
        uniq.setdefault(slug(w), w)
    jobs = []
    for tier in ("n", "s", "f"):
        for i, s in enumerate(sents, 1):
            jobs.append((tier, f"s{i:02d}.mp3", s))
        for sl, w in uniq.items():
            jobs.append((tier, f"w/{sl}.mp3", w))
    return jobs

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    jobs = load_jobs()
    if mode == "sample":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        sel = [j for j in jobs if j[0] == "n" and j[1].startswith("w/")][:n]
        sel += [j for j in jobs if j[0] == "n" and j[1].startswith("s")][:2]
        t = time.time()
        bad = 0
        with ProcessPoolExecutor(max_workers=4, initializer=init) as ex:
            for r in ex.map(job, sel, chunksize=1):
                bad += r
        print(f"sample: {len(sel)} files, failed={bad}, {time.time()-t:.1f}s", flush=True)
        return
    tiers = [t for t in sys.argv[2:] if t in SPEEDS] or ["n", "s", "f"]
    jobs = [j for j in jobs if j[0] in tiers]
    print(f"v6 jobs={len(jobs)} tiers={tiers}", flush=True)
    done = bad = 0
    t = time.time()
    with ProcessPoolExecutor(max_workers=6, initializer=init) as ex:
        for r in ex.map(job, jobs, chunksize=4):
            done += 1
            bad += r
            if done % 100 == 0:
                el = time.time() - t
                print(f"  ... {done}/{len(jobs)} failed={bad} {el:.0f}s (ETA {(el/done*(len(jobs)-done))/60:.1f}min)", flush=True)
    print(f"DONE done={done} failed={bad} in {time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main()
