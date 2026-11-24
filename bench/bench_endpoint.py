#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
断句阈值扫描台：调 rule1/rule2_min_trailing_silence，看同一段音频会被切成几段。

背景：人在句中换气、逗号处的停顿通常 0.4~0.9 秒。断句阈值（rule2）如果小于这个数，
一句话就会被从中间切开，翻译自然跟着割裂。这个脚本用来量出「既容忍句中停顿、
又能在真句尾及时断句」的那个值。

用法：
    ./venv/bin/python bench_endpoint.py             # 扫默认的一组阈值
    ./venv/bin/python bench_endpoint.py --wav x.wav
"""

import argparse
import os
import wave

import numpy as np
import sherpa_onnx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
BLOCK = int(SR * 0.1)

DEFAULT_MODEL = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
DEFAULT_WAV = "/tmp/pause_test.wav"

# 想要容忍的句中停顿上限 / 要能识别出的句尾停顿
INNER_PAUSE = 0.85     # 句中停顿最长这么多秒，不该断句
TAIL_PAUSE = 1.5       # 句尾停顿


def load_wav(path):
    with wave.open(path) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        idx = (np.arange(int(len(a) * SR / sr)) * sr / SR).astype(np.int64)
        a = a[np.clip(idx, 0, len(a) - 1)]
    return a


def build(md, rule1, rule2):
    common = dict(
        tokens=os.path.join(md, "tokens.txt"),
        num_threads=4, sample_rate=SR, feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=rule1,
        rule2_min_trailing_silence=rule2,
        decoding_method="greedy_search", provider="cpu",
    )
    files = dict(
        encoder=os.path.join(md, "encoder.int8.onnx"),
        decoder=os.path.join(md, "decoder.int8.onnx"),
        joiner=os.path.join(md, "joiner.int8.onnx"),
    )
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **common, **files, model_type="nemo_transducer")


def run(rec, audio):
    """喂完整段音频，返回 [(结束时刻秒, 这一段的文本), ...]"""
    s = rec.create_stream()
    pos = 0.0
    committed = ""
    segs = []

    # 跟 live_zh.py 保持一致：开头垫静音当预热
    s.accept_waveform(SR, np.zeros(int(SR * 0.8), dtype=np.float32))
    pos -= 0.8
    while rec.is_ready(s):
        rec.decode_stream(s)

    for i in range(0, len(audio), BLOCK):
        s.accept_waveform(SR, audio[i:i + BLOCK])
        pos += len(audio[i:i + BLOCK]) / SR
        while rec.is_ready(s):
            rec.decode_stream(s)

        if rec.is_endpoint(s):
            full = rec.get_result(s)
            seg = full[len(committed):].strip() if full.startswith(committed) else full.strip()
            if seg:
                segs.append((round(pos, 2), seg))
            committed = full

    # 收尾
    s.accept_waveform(SR, np.zeros(int(SR * 1.5), dtype=np.float32))
    while rec.is_ready(s):
        rec.decode_stream(s)
    full = rec.get_result(s)
    seg = full[len(committed):].strip() if full.startswith(committed) else full.strip()
    if seg:
        segs.append((round(pos, 2), seg))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=DEFAULT_WAV)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    md = args.model if os.path.isabs(args.model) else os.path.join(HERE, "models", args.model)
    if not os.path.isdir(md):
        raise SystemExit(f"[错误] 找不到模型目录 {md}")
    audio = load_wav(args.wav)
    print(f"音频: {args.wav}  {len(audio)/SR:.1f} 秒")
    print(f"目标: 句中停顿 ≤{INNER_PAUSE}s 不该断，句尾停顿 {TAIL_PAUSE}s 应该断")
    print()

    fixed_r1 = float(os.environ.get("FIX_RULE1", "0")) or None
    r2s = [float(x) for x in os.environ.get("R2S", "0.5,0.8,1.0,1.2,1.5,1.8").split(",")]
    for r2 in r2s:
        r1 = fixed_r1 if fixed_r1 else max(r2 + 0.4, 1.0)
        rec = build(md, round(r1, 1), r2)
        segs = run(rec, audio)
        print(f"rule1={r1:.1f}  rule2={r2:.1f}  → {len(segs)} 段")
        for t, txt in segs:
            print(f"      [{t:5.2f}s] {txt}")
        print()


if __name__ == "__main__":
    main()
