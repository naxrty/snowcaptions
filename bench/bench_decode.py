#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比三种流式断句策略在英文测试音频上的准确率（词错误率 WER）。"""
import os
import re
import sys
import wave

import numpy as np
import sherpa_onnx

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25")


def norm(s):
    s = s.upper()
    s = re.sub(r"[^A-Z0-9' ]+", " ", s)
    return [w for w in s.split() if w]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1)


def load(p):
    with wave.open(p) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def make(r1, r2):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{D}/tokens.txt", encoder=f"{D}/encoder.int8.onnx",
        decoder=f"{D}/decoder.int8.onnx", joiner=f"{D}/joiner.int8.onnx",
        model_type="nemo_transducer", num_threads=4, sample_rate=16000, feature_dim=80,
        enable_endpoint_detection=True, rule1_min_trailing_silence=r1,
        rule2_min_trailing_silence=r2, decoding_method="greedy_search", provider="cpu")


def run(rec, audio, strategy, flush=0.6, block=1600):
    """strategy: 'reset'（老办法）/ 'reset_flush'（重置+冲静音）/ 'delta'（不重置，取增量）"""
    s = rec.create_stream()
    segs, prev = [], ""

    def drain():
        while rec.is_ready(s):
            rec.decode_stream(s)

    for i in range(0, len(audio), block):
        s.accept_waveform(16000, audio[i:i + block])
        drain()
        if rec.is_endpoint(s):
            if strategy == "reset":
                t = rec.get_result(s).strip()
                if t:
                    segs.append(t)
                rec.reset(s)
                prev = ""
            elif strategy == "reset_flush":
                s.accept_waveform(16000, np.zeros(int(16000 * flush), dtype=np.float32))
                drain()
                t = rec.get_result(s).strip()
                if t:
                    segs.append(t)
                rec.reset(s)
                prev = ""
            else:  # delta：不重置，只取新增部分
                full = rec.get_result(s).strip()
                if full.startswith(prev) and len(full) > len(prev):
                    segs.append(full[len(prev):].strip())
                    prev = full
                elif full != prev:
                    prev = full

    # 收尾：灌静音把前瞻缓冲冲干净
    s.accept_waveform(16000, np.zeros(int(16000 * (1.0 + flush)), dtype=np.float32))
    drain()
    full = rec.get_result(s).strip()
    if strategy == "delta":
        if full.startswith(prev) and len(full) > len(prev):
            segs.append(full[len(prev):].strip())
        elif full and full != prev:
            segs.append(full[len(prev):].strip() if full.startswith(prev) else full)
    else:
        if full:
            segs.append(full)
    return [x for x in segs if x]


def main():
    refs = {}
    with open(f"{D}/test_wavs/trans.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]

    combos = [("reset", 1.0, 0.5), ("reset_flush", 1.0, 0.5), ("delta", 1.0, 0.5),
              ("reset_flush", 2.0, 1.2), ("delta", 2.0, 1.2)]
    print(f"{'策略':<14}{'rule1/2':<10}{'WAV':<9}{'WER':>7}   输出")
    print("-" * 100)
    for strat, r1, r2 in combos:
        rec = make(r1, r2)
        for name in ("0.wav", "1.wav", "8k.wav"):
            audio = load(f"{D}/test_wavs/{name}")
            segs = run(rec, audio, strat)
            hyp = " ".join(segs)
            w = wer(refs[name], hyp)
            print(f"{strat:<14}{f'{r1}/{r2}':<10}{name:<9}{w*100:6.1f}%   {hyp[:110]}")
        print()


if __name__ == "__main__":
    main()
