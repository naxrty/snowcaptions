#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两遍架构对比评测 —— 同一段音频，第一遍（流式）vs 各候选第二遍（精修）。

## 为什么必须做这个对比

只看精修模型的绝对成绩没有意义。必须和「不改动时的第一遍」放在一起比，
才知道精修到底是净收益还是净损失。

实测反例（已经踩到）：
  - 某句里第一遍把「再」听成「在」，精修改对了  → 净收益
  - 另一句里第一遍「KPI」是对的，精修听成「k」 → 净损失

所以结论不能靠感觉，只能靠同一批音频上的逐句对比。

用法：
    ./venv/bin/python bench_twopass.py                          # 全部
    ./venv/bin/python bench_twopass.py --only noise2,qiqiu1     # 指定几段
    ./venv/bin/python bench_twopass.py --refine qwen3_asr       # 只比某个精修模型
"""
import argparse
import os
import re
import subprocess
import sys
import time
import wave

import numpy as np
import sherpa_onnx

SR = 16000
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
REFINE_DIRS = [os.path.join(HERE, "models_refine"), MODELS_DIR]

STREAM_ZH = "sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30"
QWEN_DIR = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"

# 候选精修模型：(标签, 目录名, 加载方式)
REFINERS = [
    ("funasr_nano", "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17", "sense_voice"),
    ("qwen3_asr", QWEN_DIR, "qwen3_asr"),
]

WARMUP = 0.8
FLUSH = 1.5
BLOCK_SEC = 0.1
COMMIT_DELAY = 0.3
MIN_COMMIT = 4
LEAD_PUNCT = ".,!?;:、。，！？；：…·\"'“”‘’（）()[]【】《》<>-—– \t"

PUNCT = set("，。、！？；：（）〈〉《》【】…—·,.;:!?()[]{}<>-~\"'")
PUNCT |= set("\u201c\u201d\u2018\u2019")


def strip_punct(s):
    return "".join(ch for ch in s if ch not in PUNCT and not ch.isspace())


def cer(ref, hyp):
    ref, hyp = strip_punct(ref), strip_punct(hyp)
    if not ref:
        return 0.0, 0, 0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    d = prev[-1]
    return d / len(ref), d, len(ref)


def _read_16k(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError("not 16k mono")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def read_wav(path):
    """读 16k 单声道；采样率不对就用 ffmpeg 转一份临时文件（44.1k 的样本很常见）"""
    try:
        return _read_16k(path), False
    except ValueError:
        tmp = os.path.join("/tmp", "asr16k_" + os.path.basename(path))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", tmp],
                       check=True)
        return _read_16k(tmp), True


def _p(d, *a):
    return os.path.join(d, *a)


def build_stream_model():
    md = os.path.join(MODELS_DIR, STREAM_ZH)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=_p(md, "tokens.txt"),
        encoder=_p(md, "encoder.int8.onnx"),
        decoder=_p(md, "decoder.onnx"),
        joiner=_p(md, "joiner.int8.onnx"),
        num_threads=4, sample_rate=SR, feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.9, rule2_min_trailing_silence=1.5,
        rule3_min_utterance_length=99999.0,
        decoding_method="greedy_search", provider="cpu")


def build_refiner(name, dirname, kind, threads=4):
    for d0 in REFINE_DIRS:
        md = os.path.join(d0, dirname)
        if not os.path.isdir(md):
            continue
        if kind == "sense_voice" and os.path.exists(_p(md, "model.int8.onnx")):
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=_p(md, "model.int8.onnx"), tokens=_p(md, "tokens.txt"),
                num_threads=threads, sample_rate=SR, use_itn=True)
        if kind == "qwen3_asr" and all(os.path.exists(_p(md, f)) for f in
                                       ("conv_frontend.onnx", "encoder.int8.onnx",
                                        "decoder.int8.onnx")):
            tok = _p(md, "tokenizer")
            if not os.path.isdir(tok):
                tok = _p(md, "tokenizer.json")
            return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=_p(md, "conv_frontend.onnx"),
                encoder=_p(md, "encoder.int8.onnx"),
                decoder=_p(md, "decoder.int8.onnx"),
                tokenizer=tok, num_threads=threads, sample_rate=SR)
    return None


def run_stream(rec, samples):
    """完整复刻 live_zh.py 的第一遍逻辑（不重置 + 取增量）。"""
    stream = rec.create_stream()
    stream.accept_waveform(SR, np.zeros(int(SR * WARMUP), dtype=np.float32))
    while rec.is_ready(stream):
        rec.decode_stream(stream)

    committed, out, n_seg = "", [], 0
    pos, pending = 0.0, None
    step = int(SR * BLOCK_SEC)

    def tail_of(full):
        nonlocal committed
        if full.startswith(committed):
            raw = full[len(committed):]
        else:
            k, m = 0, min(len(committed), len(full))
            while k < m and committed[k] == full[k]:
                k += 1
            committed = full[:k]
            raw = full[k:]
        return raw.lstrip().lstrip(LEAD_PUNCT).lstrip()

    for i in range(0, len(samples), step):
        blk = samples[i:i + step]
        stream.accept_waveform(SR, blk)
        pos += len(blk) / SR
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        full = rec.get_result(stream)
        tail = tail_of(full)
        if rec.is_endpoint(stream):
            if pending is None:
                pending = pos
        else:
            pending = None
        if pending is not None and pos - pending >= COMMIT_DELAY:
            pending = None
            if len(tail.strip()) >= MIN_COMMIT:
                committed = full
                out.append(tail.strip())
                n_seg += 1

    stream.accept_waveform(SR, np.zeros(int(SR * FLUSH), dtype=np.float32))
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    final = tail_of(rec.get_result(stream)).strip()
    if final:
        out.append(final)
        n_seg += 1
    return "".join(out), n_seg


def run_refine(rec, samples):
    s = rec.create_stream()
    s.accept_waveform(SR, samples)
    rec.decode_stream(s)
    return s.result.text.strip()


def load_refs(path):
    refs = {}
    if not os.path.exists(path):
        return refs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+\.wav)\s+(.*)$", line)
            if m:
                refs[os.path.splitext(m.group(1))[0]] = m.group(2).strip()
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="音频目录（默认 Qwen3-ASR 自带 test_wavs）")
    ap.add_argument("--only", help="只测这几个（逗号分隔，不带 .wav）")
    ap.add_argument("--refine", help="只比指定的精修模型标签")
    ap.add_argument("--stream-only", action="store_true")
    args = ap.parse_args()

    testdir = args.dir or os.path.join(MODELS_DIR, QWEN_DIR, "test_wavs")
    if not os.path.isdir(testdir):
        print(f"[错误] 找不到音频目录 {testdir}")
        return 1

    refs = load_refs(os.path.join(testdir, "transcript.txt"))
    wavs = sorted(f for f in os.listdir(testdir) if f.endswith(".wav"))
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        wavs = [w for w in wavs if os.path.splitext(w)[0] in keep]

    print("=" * 78)
    print("两遍架构对比：第一遍（流式 xlarge）  vs  第二遍（各候选精修模型）")
    print("=" * 78)

    t0 = time.time()
    srec = build_stream_model()
    print(f"第一遍模型加载：{time.time()-t0:.1f}s")

    rrecs = []
    if not args.stream_only:
        for name, dirname, kind in REFINERS:
            if args.refine and name != args.refine:
                continue
            t0 = time.time()
            r = build_refiner(name, dirname, kind)
            if r is None:
                print(f"第二遍[{name}]：模型不可用，跳过")
                continue
            print(f"第二遍[{name}] 加载：{time.time()-t0:.2f}s")
            rrecs.append((name, r))
    print()

    acc = {"stream": [0, 0]}
    for name, _ in rrecs:
        acc[name] = [0, 0]

    for w in wavs:
        name = os.path.splitext(w)[0]
        path = os.path.join(testdir, w)
        try:
            samples, converted = read_wav(path)
        except Exception as e:
            print(f"### {name}  [跳过] {e}")
            continue
        dur = len(samples) / SR
        ref = refs.get(name)

        s_txt, n_seg = run_stream(srec, samples)
        print(f"### {name}   （{dur:.1f}s{'，已重采样' if converted else ''}"
              f"，第一遍切 {n_seg} 段）")
        if ref:
            print(f"  参考  : {ref}")
        print(f"  第一遍: {s_txt}")
        if ref:
            r, d, n = cer(ref, s_txt)
            acc["stream"][0] += d
            acc["stream"][1] += n
            print(f"          → 错 {d}/{n}（{r*100:.1f}%）")

        for label, rrec in rrecs:
            t0 = time.time()
            r_txt = run_refine(rrec, samples)
            cost = time.time() - t0
            print(f"  {label}: {r_txt}   [{cost:.2f}s RTF {cost/dur:.3f}]")
            if ref:
                r, d, n = cer(ref, r_txt)
                acc[label][0] += d
                acc[label][1] += n
                print(f"          → 错 {d}/{n}（{r*100:.1f}%）")
        print()

    print("=" * 78)
    if acc["stream"][1]:
        sd, sr_ = acc["stream"]
        print(f"第一遍    ：错 {sd}/{sr_} = {sd/sr_*100:.1f}%")
        for label, _ in rrecs:
            d, n = acc[label]
            if n:
                delta = sd - d
                verdict = "更准" if delta > 0 else ("更差" if delta < 0 else "持平")
                print(f"{label:<10}：错 {d}/{n} = {d/n*100:.1f}%   "
                      f"净变化 {delta:+d} 字（{verdict}）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
