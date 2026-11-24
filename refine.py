#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二遍精修 —— 「两遍架构」里的第二遍。

## 为什么需要它

第一遍用的流式模型（zipformer）为了低延迟，内置的语言模型非常小：
它只能判断「这个音最像哪个字」，没法判断「这句话该用哪个字」。
所以会出现同音字混淆（的/得/地、在/再）、人名和专有词按发音硬拼。

离线大模型（Qwen3-ASR / Fun-ASR-Nano）能看完整句子再决定用哪个字，
准确率明显更高——但慢，做不到边听边出。

## 所以做法是分工

  第一遍（流式，现有模型）：你的字 0.1~0.3 秒就滚出来   → 负责「快」
  第二遍（离线，本模块）  ：一句话说完后重听那一小段     → 负责「准」

只有「一句话说完」之后才对那一小段音频重跑，实时预览完全不受影响。

## 设计取舍

- 模型**延迟加载**：第一次真正要精修时才加载，不拖慢启动
- 精修失败一律**静默降级**回第一遍的结果，绝不吞掉用户的话
- 整个模块可以被关掉（enabled=False），行为与改造前完全一致
"""
import os
import sys

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000

# 候选精修模型：按优先级排列（越靠前越先被选中）
# 目录名必须在 models_refine/（或 models/）下真实存在才会被启用。
#
# ⚠️ 顺序是**两组实测**定的，两次结论不同，别只看了其中一个就重排：
#
#   场景一「按句精修」（真实使用方式，句子 3~7 秒）
#      Qwen3-ASR     CER  0.0%（5 句全对，含 KPI）+ 自带标点，单句 0.35~0.49s
#      Fun-ASR-Nano  CER  3.2%（把 KPI 听成 k），单句 0.04~0.07s，无标点
#      → Qwen3-ASR 胜。这也是真正的使用场景（精修是按句做的）。
#
#   场景二「整段长音频」（bench_twopass.py，最长 51 秒）
#      Fun-ASR-Nano  CER 18.7%，RTF 0.011，5 段全胜
#      Qwen3-ASR     CER 45.3%，51 秒那段直接输出乱码
#      → Fun-ASR-Nano 胜。但这段长音频**只有在整段喂进去时才会出现**：
#        Qwen3-ASR 的 max_total_len=512 会把超长音频截断。
#
# 所以默认选 Qwen3-ASR，并配一道长度保险（见 MAX_REFINE_SEC）：
# 句子太长时直接放弃精修、沿用第一遍结果，绝不吐乱码。
CANDIDATES = [
    ("qwen3_asr", "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
     "Qwen3-ASR 0.6B（短句全对 + 自带标点，单句约 0.4s）"),
    ("funasr_nano", "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17",
     "Fun-ASR-Nano 264MB（快 10 倍、噪声更鲁棒，但不产标点）"),
    ("sense_voice", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
     "SenseVoice（轻量兜底）"),
]

# 单句音频超过这个秒数就放弃精修，直接用第一遍的结果。
# 依据：Qwen3-ASR 自带 max_total_len=512 的上下文上限，20 秒还能正常处理，
# 51 秒就会把音频截断、输出乱码（实测）。留足余量取 25 秒。
MAX_REFINE_SEC = 25.0


def _p(d, *a):
    return os.path.join(d, *a)


def _ready(kind, d):
    """模型文件是否齐全（缺一个都不算可用）"""
    # Fun-ASR-Nano 是 SenseVoice 形状的导出（单文件 model.onnx + tokens.txt），
    # 两者文件构成相同，必须一起判断——漏了它会被静默跳过，然后错选下一个模型。
    if kind in ("sense_voice", "funasr_nano"):
        return all(os.path.exists(_p(d, f)) for f in ("model.int8.onnx", "tokens.txt")) or \
               all(os.path.exists(_p(d, f)) for f in ("model.onnx", "tokens.txt"))
    if kind == "qwen3_asr":
        return all(os.path.exists(_p(d, f)) for f in
                   ("conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"))
    return False


def find_model(models_dirs, prefer=None):
    """在给定目录下找可用的精修模型。models_dirs 可传单个目录或目录列表。

    返回 (kind, dir, desc) 或 None。
    """
    if isinstance(models_dirs, str):
        models_dirs = [models_dirs]
    order = CANDIDATES
    if prefer:
        order = [c for c in CANDIDATES if c[0] == prefer] + \
                [c for c in CANDIDATES if c[0] != prefer]
    for kind, name, desc in order:
        for base in models_dirs:
            d = os.path.join(base, name)
            if os.path.isdir(d) and _ready(kind, d):
                return kind, d, desc
    return None


def _build(kind, d, threads=4, language="", hotwords=""):
    if kind in ("sense_voice", "funasr_nano"):
        model = _p(d, "model.int8.onnx")
        if not os.path.exists(model):
            model = _p(d, "model.onnx")
        kw = dict(model=model, tokens=_p(d, "tokens.txt"),
                  num_threads=threads, sample_rate=SAMPLE_RATE, use_itn=True)
        if language:
            kw["language"] = language
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kw)

    if kind == "qwen3_asr":
        tok = _p(d, "tokenizer")
        if not os.path.isdir(tok):
            tok = _p(d, "tokenizer.json")
        kw = dict(conv_frontend=_p(d, "conv_frontend.onnx"),
                  encoder=_p(d, "encoder.int8.onnx"),
                  decoder=_p(d, "decoder.int8.onnx"),
                  tokenizer=tok,
                  num_threads=threads, sample_rate=SAMPLE_RATE)
        if hotwords:
            kw["hotwords"] = hotwords
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(**kw)

    raise ValueError(kind)


class Refiner:
    """把「重听一遍」这件事封装起来。

    refine(samples) 返回更准的文本；任何异常都返回 None，让调用方沿用第一遍结果。
    """

    def __init__(self, models_dir, prefer=None, threads=4, language="",
                 hotwords="", debug=False):
        self.debug = debug
        self.language = language
        self.hotwords = hotwords
        self.threads = threads
        self._rec = None
        self._load_err = None
        self.kind = self.model_dir = self.desc = None

        found = find_model(models_dir, prefer)
        if found:
            self.kind, self.model_dir, self.desc = found
        self.enabled = found is not None

    @property
    def ready(self):
        """模型是否已经加载进内存"""
        return self._rec is not None

    def load(self):
        """预加载模型。启动时调用，避免第一句定稿时才卡一下（加载约 0.4 秒）。"""
        self._ensure()

    def _ensure(self):
        if self._rec is not None or not self.enabled:
            return
        self._rec = _build(self.kind, self.model_dir,
                           threads=self.threads, language=self.language,
                           hotwords=self.hotwords)
        if self.debug:
            print(f"[精修] 模型已加载：{self.desc}", file=sys.stderr)

    def refine(self, samples):
        """samples: float32 单声道 16kHz。返回精修文本或 None（None = 沿用第一遍结果）。"""
        if not self.enabled:
            return None
        if samples is None or len(samples) < int(SAMPLE_RATE * 0.2):
            return None            # 太短，不值得精修，也可能触发模型怪异输出
        dur = len(samples) / SAMPLE_RATE
        if dur > MAX_REFINE_SEC:
            # 太长：可能触发模型上下文截断（见 MAX_REFINE_SEC 的说明），
            # 宁可不精修，也不能吐乱码。
            if self.debug:
                print(f"[精修] 音频 {dur:.1f}s 超过 {MAX_REFINE_SEC}s，跳过，沿用第一遍",
                      file=sys.stderr)
            return None
        try:
            self._ensure()
            s = self._rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
            self._rec.decode_stream(s)
            text = (s.result.text or "").strip()
            return text or None
        except Exception as e:
            if self.debug:
                print(f"[精修] 失败，沿用第一遍结果：{type(e).__name__}: {e}",
                      file=sys.stderr)
            return None

    def describe(self):
        if not self.enabled:
            return "未启用（models/ 下没有可用的精修模型）"
        state = "已加载" if self.ready else "待加载"
        return f"{self.desc} [{state}]"
