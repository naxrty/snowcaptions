#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型发现、挑选与加载。"""
import os
import sys

import sherpa_onnx

import config
from config import MODELS_DIR, NEVER_TRIGGER_SEC, PREFERRED, SAMPLE_RATE, THREADS

# --------------------------------------------------------------------------
# 模型选择与加载
# --------------------------------------------------------------------------

def has(d, *names):
    return next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))), None)


def detect_kind(md):
    """返回 ('transducer', 文件字典) / ('ctc', 文件字典, 构造器名) / None"""
    if not os.path.exists(os.path.join(md, "tokens.txt")):
        return None
    enc = has(md, "encoder.int8.onnx", "encoder.onnx")
    dec = has(md, "decoder.int8.onnx", "decoder.onnx")
    joi = has(md, "joiner.int8.onnx", "joiner.onnx")
    if enc and dec and joi:
        return ("transducer", {"encoder": enc, "decoder": dec, "joiner": joi})
    m = has(md, "model.int8.onnx", "model.onnx")
    if m:
        return ("ctc", {"model": m})
    return None


def detect_lang(md):
    """从词表判断模型说的是什么语言：zh / en / mixed"""
    tk = os.path.join(md, "tokens.txt")
    if not os.path.exists(tk):
        return "en"
    cjk = latin = 0
    with open(tk, encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.rstrip("\n").rsplit(" ", 1)[0]
            if any("\u4e00" <= c <= "\u9fff" for c in tok):
                cjk += 1
            elif tok.isascii() and tok.isalpha():
                latin += 1
    if cjk >= 100 and cjk > latin:
        return "zh"
    if latin >= 50 and cjk < 50:
        return "en"
    return "mixed"


def list_models():
    rows = []
    if not os.path.isdir(MODELS_DIR):
        return rows
    for name in sorted(os.listdir(MODELS_DIR)):
        p = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(p):
            continue
        kind = detect_kind(p)
        rows.append((name, kind[0] if kind else "❌ 不可用", detect_lang(p) if kind else "-"))
    return rows


def pick_model_dir(lang):
    """决定用哪个模型文件夹。lang 取 auto / zh / en。"""
    env = os.environ.get("SHERPA_MODEL", "").strip()
    if env:
        p = os.path.join(MODELS_DIR, env)
        if not os.path.isdir(p):
            sys.exit(f"[错误] 找不到模型目录：{p}")
        if not detect_kind(p):
            sys.exit(f"[错误] {p} 里没有可用的模型文件（可能下载不完整）")
        return p

    names = [n for n, _, _ in list_models()]
    langs = ["en", "zh"] if lang == "auto" else [lang]
    for L in langs:
        for want in PREFERRED[L] + names:      # 先按优先级，再按目录扫描兜底
            p = os.path.join(MODELS_DIR, want)
            if os.path.isdir(p) and detect_kind(p) and detect_lang(p) == L:
                return p
    sys.exit(
        f"[错误] 在 {MODELS_DIR} 里找不到{'可用的 ' + lang + ' 模型' if lang != 'auto' else '可用模型'}\n"
        f"        当前 models/ 内容：{names}"
    )


def transducer_model_type(md):
    """NeMo 系的模型要显式告诉 sherpa-onnx，否则解不出来。"""
    env = os.environ.get("SHERPA_MODEL_TYPE", "").strip()
    if env:
        return env
    n = os.path.basename(md).lower()
    if "nemo" in n or "nemotron" in n or "parakeet" in n or "conformer" in n:
        return "nemo_transducer"
    return ""          # 空串 = zipformer（默认）


def build_recognizer(md, hotwords=""):
    """构建流式识别器。hotwords 为逗号分隔的热词串（可为空）。

    热词只在 modified_beam_search 解码下生效（sherpa-onnx 限制），
    所以有热词时切 beam_search + 提供 hotwords_file/hotwords_score；
    没热词时保持 greedy_search（更快、也够准）。
    """
    info = detect_kind(md)
    if not info:
        sys.exit(f"[错误] {md} 里没有可用的模型文件（可能下载不完整）")
    kind, files = info
    common = dict(
        tokens=os.path.join(md, "tokens.txt"),
        num_threads=THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=config.RULE1_SILENCE,
        rule2_min_trailing_silence=config.RULE2_SILENCE,
        # 关掉「utterance 够长就随时断句」这条规则（默认 20 秒，见文件头部说明）。
        # 不关的话，连续用满 20 秒之后就会开始吐 2~4 个词的碎片。
        rule3_min_utterance_length=NEVER_TRIGGER_SEC,
        decoding_method="greedy_search",
        provider="cpu",
    )

    # 热词处理：只有 transducer 支持热词；有热词才切 beam_search
    hw = os.path.join(md, "hotwords.txt")
    hw_active = False
    if hotwords and kind == "transducer":
        # 写一份热词文件到模型目录（一行一个词），beam_search 需要它
        try:
            words = [w.strip() for w in hotwords.replace("，", ",").split(",")
                     if w.strip()]
            if words:
                with open(hw, "w", encoding="utf-8") as f:
                    f.write("\n".join(words) + "\n")
                common["decoding_method"] = "modified_beam_search"
                common["hotwords_file"] = hw
                common["hotwords_score"] = 1.5   # 官方推荐值附近，偏置强度
                hw_active = True
        except Exception as e:
            print(f"    热词写入失败，忽略热词：{type(e).__name__}: {e}")

    if kind == "transducer":
        mt = transducer_model_type(md)
        extra = {"model_type": mt} if mt else {}
        if mt:
            print(f"  类型: transducer / {mt}")
        else:
            print("  类型: transducer（编码器+解码器+联合网络）")
        for k, v in files.items():
            print(f"    {k:<8}: {os.path.basename(v)}")
        if hw_active:
            print(f"    热词表 : hotwords.txt（{hotwords}，已切 beam_search）")
        return sherpa_onnx.OnlineRecognizer.from_transducer(**common, **files, **extra)
    print("  类型: CTC（单模型）")
    print(f"    model   : {os.path.basename(files['model'])}")
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(**common, **files)
