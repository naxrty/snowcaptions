#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精修模型评测台 —— 给「两遍架构」挑第二遍用哪个模型。

为什么需要它：换精修模型前必须拿数字说话（耗时 / 准确率），
否则只能凭感觉，而感觉在小样本上极不可靠（技能里已有教训）。

用法：
    ./venv/bin/python bench_refine.py                 # 测所有已下载的模型
    ./venv/bin/python bench_refine.py --model funasr_nano
    ./venv/bin/python bench_refine.py --wav /tmp/asrtest/s1.wav

输出：每个模型 × 每段音频的 识别结果 / 耗时 / RTF / 字错误率(CER)
"""
import argparse
import os
import sys
import time
import wave

import numpy as np
import sherpa_onnx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 精修模型放在 models_refine/（与流式模型隔离），老位置 models/ 也兜底找一下
REFINE_DIRS = [os.path.join(HERE, "models_refine"), os.path.join(HERE, "models")]
SAMPLE_RATE = 16000

# 每个模型：目录名、加载方式、说明
CANDIDATES = [
    ("funasr_nano",
     "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17",
     "sense_voice",
     "Fun-ASR-Nano 264MB · 噪声鲁棒 · 中英混说"),
    ("qwen3_asr",
     "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
     "qwen3_asr",
     "Qwen3-ASR 0.6B 982MB · 精度最高 · LLM架构"),
]

# 测试音频 + 参考文本（TTS 合成，已知答案）
TESTS = [
    ("s1", "这个项目得三天才能完成，我们需要再确认一下。"),
    ("s2", "张明和李晓东明天去深圳出差。"),
    ("s3", "今天天气不错，我们一起去公园散步吧。"),
    ("s4", "如果看左边的图表，可以看到通货膨胀在春季开始下降。"),
    ("s5", "我们的KPI达成情况不太理想，需要重新评估一下。"),
]


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, w.getframerate()
        assert w.getnchannels() == 1, w.getnchannels()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_recognizer(kind, model_dir, threads=4):
    """按类型构造离线识别器。文件缺失时抛 FileNotFoundError，由调用方跳过。"""
    p = lambda *a: os.path.join(model_dir, *a)

    if kind == "sense_voice":
        for f in ("model.int8.onnx", "tokens.txt"):
            if not os.path.exists(p(f)):
                raise FileNotFoundError(f)
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=p("model.int8.onnx"), tokens=p("tokens.txt"),
            num_threads=threads, sample_rate=SAMPLE_RATE,
            use_itn=True)

    if kind == "qwen3_asr":
        need = ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"]
        for f in need:
            if not os.path.exists(p(f)):
                raise FileNotFoundError(f)
        # tokenizer 在不同导出里可能是目录、也可能是单个 json
        tok = p("tokenizer")
        if not os.path.isdir(tok):
            for cand in ("tokenizer.json", "tokenizer_config.json"):
                if os.path.exists(p(cand)):
                    tok = p(cand)
                    break
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=p("conv_frontend.onnx"),
            encoder=p("encoder.int8.onnx"),
            decoder=p("decoder.int8.onnx"),
            tokenizer=tok,
            num_threads=threads, sample_rate=SAMPLE_RATE)

    raise ValueError(kind)


PUNCT = set("，。、！？；：（）〈〉《》【】…—·,.;:!?()[]{}<>-~\"'")
PUNCT |= set("\u201c\u201d\u2018\u2019")


def strip_punct(s):
    """只留实义字。
    标点有没有属于「风格差异」——两个模型都不产标点，拿它算错字会污染结论。
    实测教训：第一版口径把标点算进错字，Fun-ASR-Nano 的 CER 从 ~1% 被抬到 11.7%。
    """
    return "".join(ch for ch in s if ch not in PUNCT and not ch.isspace())


def cer(ref, hyp):
    """字错误率 = 编辑距离 / 参考长度（已排除标点，中文按字计）"""
    ref = strip_punct(ref)
    hyp = strip_punct(hyp)
    if not ref:
        return 0.0, 0, len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    d = prev[-1]
    return d / len(ref), d, len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="只测指定模型（funasr_nano / qwen3_asr）")
    ap.add_argument("--wav", help="只测单个 wav 文件")
    ap.add_argument("--wavdir", default="/tmp/asrtest", help="测试音频目录")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    targets = [c for c in CANDIDATES if not args.model or c[0] == args.model]
    if not targets:
        print(f"[错误] 未知模型 {args.model}，可选：{[c[0] for c in CANDIDATES]}")
        return 1

    # 组装测试用例
    if args.wav:
        tests = [(os.path.splitext(os.path.basename(args.wav))[0], None)]
        wavs = {tests[0][0]: args.wav}
    else:
        tests = TESTS
        wavs = {n: os.path.join(args.wavdir, n + ".wav") for n, _ in TESTS}

    print("=" * 74)
    print("精修模型评测（只读：只是加载模型跑音频，不改任何文件）")
    print("=" * 74)

    for key, dirname, kind, desc in targets:
        model_dir = next((os.path.join(d, dirname) for d in REFINE_DIRS
                          if os.path.isdir(os.path.join(d, dirname))), None)
        print(f"\n### {key}  ——  {desc}")
        print(f"    目录：{model_dir}")
        if model_dir is None:
            print("    [跳过] 两个模型目录里都找不到它")
            continue
        try:
            t0 = time.time()
            rec = build_recognizer(kind, model_dir, args.threads)
            load_s = time.time() - t0
        except FileNotFoundError as e:
            print(f"    [跳过] 模型文件不齐（缺 {e}），可能还没下载完")
            continue
        except Exception as e:
            print(f"    [跳过] 加载失败：{type(e).__name__}: {e}")
            continue

        print(f"    加载耗时：{load_s:.2f}s")
        print(f"    {'音频':<6}{'耗时':>8}{'音频长':>9}{'RTF':>8}   识别结果")

        total_d, total_r = 0, 0
        for name, ref in tests:
            path = wavs[name]
            if not os.path.exists(path):
                print(f"    {name:<6}  (缺文件 {path})")
                continue
            samples = read_wav(path)
            dur = len(samples) / SAMPLE_RATE

            s = rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, samples)

            t0 = time.time()
            rec.decode_stream(s)
            cost = time.time() - t0
            hyp = s.result.text.strip()

            rtf = cost / dur if dur else 0
            line = f"    {name:<6}{cost:>7.2f}s{dur:>8.2f}s{rtf:>8.3f}   {hyp}"
            if ref:
                rate, d, r = cer(ref, hyp)
                total_d += d
                total_r += r
                line += f"\n           {'':>6}参考：{ref}\n           {'':>6}错 {d}/{r} 字，CER {rate*100:.1f}%"
            print(line)

        if total_r:
            print(f"    {'':<6}—— 总体 CER：{total_d}/{total_r} = {total_d/total_r*100:.1f}%")

    print("\n" + "=" * 74)
    print("判定标准：RTF < 0.3 且单句耗时 < 0.5s → 可做同步精修（不卡顿）")
    print("          RTF ≥ 0.3 → 考虑异步精修，否则定稿时会顶住音频队列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
