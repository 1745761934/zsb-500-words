# -*- coding: utf-8 -*-
"""Patch render: word "active" + sentence 17 (new text) for all 3 tiers.
Reuses gen_v6.py's post-processing (30ms pre-roll / 12ms fade-in / 25ms fade-out /
gated RMS -20dBFS / alimiter / 44.1k 96k no-ID3) so the new clips match the rest.
"""
import os, sys, pathlib, subprocess
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import numpy as np
import onnxruntime as ort

_ORIG = ort.InferenceSession
def _patched(model_path, sess_options=None, providers=None, **kw):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1; so.log_severity_level = 3
    return _ORIG(model_path, sess_options=so, providers=providers or ["CPUExecutionProvider"], **kw)
ort.InferenceSession = _patched

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
from gen_v6 import post, to_mp3, SPEEDS, MODEL, VOICES, VOICE, OUT, SR, FFM

SENT17 = "To enjoy good health and stay healthy, you should stay active, exercise regularly, eat a balanced diet, and take medicine only on a doctor's advice; this can prevent disease and help patients recover faster."

def render():
    from kokoro_onnx import Kokoro
    k = Kokoro(MODEL, VOICES)
    jobs = []
    for tier in ("n", "s", "f"):
        jobs.append((tier, f"w/active.mp3", "active"))
        jobs.append((tier, f"s17.mp3", SENT17))
    bad = 0
    for tier, name, text in jobs:
        dst = OUT / tier / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        samples, sr = k.create(text, voice=VOICE, speed=SPEEDS[tier], lang="en-us")
        ok = to_mp3(post(samples, sr), dst)
        print(f"  {tier}/{name}: {'ok' if ok else 'FAILED'}", flush=True)
        if not ok: bad += 1
    print("bad =", bad)
    return bad

if __name__ == "__main__":
    sys.exit(render())
