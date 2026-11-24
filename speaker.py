#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""说话人识别（speaker diarization）：给每句定稿标上「说话人 N」。

原理（流式、实时、纯 CPU）：
    用 sherpa-onnx 的 SpeakerEmbeddingExtractor 给一段音频算一个「声纹」
    （192 维向量，L2 归一化后的余弦相似度表示两个人声音有多像），再用
    SpeakerEmbeddingManager 维护「已经出现过的说话人」列表。

    每句话定稿时，拿这一句的音频算声纹 → 跟已注册的说话人比对：
      * 最像的那个相似度 ≥ 阈值 → 判为那个人
      * 都不够像（或还没有任何人）→ 注册成新说话人「说话人 N」

和身份识别（认出"具体是谁"）的区别：这里不认名字，只保证「同一个人
前后得到同一个编号」，用于把对话里不同的人分开标注。

模型：3D-Speaker 的 CAM++（中文 16k，约 27MB，Apache-2.0），放 models_speaker/。
"""
import os

import numpy as np

from config import HERE, SAMPLE_RATE

SPEAKER_MODEL = os.path.join(HERE, "models_speaker",
                             "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

# 相似度阈值：超过才算「同一个人」。CAM++ 的 embedding 是余弦相似度，
# 静音/噪声也可能给出 0.6~0.7 的分数，所以阈值不能太低，否则不同人会混。
# 实测（同一段静音 0.999、随机噪声 0.685），取 0.55 偏保守，宁可偶尔多
# 切一个编号，也不要把两个人并成一个人。
SIM_THRESHOLD = 0.55

# 音频太短时声纹不可靠（几个音节没特征），直接跳过、不给编号。
MIN_AUDIO_SEC = 0.6


class SpeakerTagger:
    """说话人标注器。惰性加载模型，句边界时调用 identify()。"""

    def __init__(self, model_path=SPEAKER_MODEL, threshold=SIM_THRESHOLD,
                 min_audio_sec=MIN_AUDIO_SEC, threads=2):
        self.model_path = model_path
        self.threshold = threshold
        self.min_audio_sec = min_audio_sec
        self.threads = threads
        self._extractor = None
        self._manager = None
        self._enabled = None      # None=未探测；True/False=加载结果
        self._load_error = ""

    # ---------------- 加载 ----------------

    @property
    def enabled(self):
        """模型是否可用。惰性探测一次，之后缓存。"""
        if self._enabled is None:
            self._try_load()
        return self._enabled is True

    def _try_load(self):
        try:
            import sherpa_onnx
            if not os.path.exists(self.model_path):
                self._enabled = False
                self._load_error = f"声纹模型不存在：{self.model_path}"
                return
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model_path, num_threads=self.threads, provider="cpu")
            ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self._extractor = ex
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(ex.dim)
            self._enabled = True
        except Exception as e:
            self._enabled = False
            self._load_error = f"{type(e).__name__}: {e}"

    def describe(self):
        if not self.enabled:
            return f"未启用（{self._load_error}）"
        return f"CAM++ 声纹（{self._extractor.dim} 维，阈值 {self.threshold}）"

    # ---------------- 核心 ----------------

    def _embed(self, samples):
        """一段 float32 音频 → 192 维声纹向量（L2 归一化）。失败返回 None。"""
        if self._extractor is None or samples is None or len(samples) == 0:
            return None
        try:
            s = self._extractor.create_stream()
            s.accept_waveform(SAMPLE_RATE, samples.astype(np.float32))
            s.input_finished()
            if not self._extractor.is_ready(s):
                return None
            v = self._extractor.compute(s)
            return np.asarray(v, dtype=np.float32)
        except Exception:
            return None

    def identify(self, samples):
        """给一段音频标说话人。返回编号字符串 "spk1"/"spk2"/...，识别不了返回 None。

        完全非阻塞式设计：任何异常都返回 None（调用方静默跳过），
        绝不影响转写主链路。
        """
        if not self.enabled or self._manager is None:
            return None
        if samples is None or len(samples) < self.min_audio_sec * SAMPLE_RATE:
            return None
        v = self._embed(samples)
        if v is None:
            return None
        try:
            name = self._manager.search(v, self.threshold)
            if name:
                # 名字形如 "spk1"/"spk2"，摘出数字返回
                return name
            # 新人：注册
            idx = self._manager.num_speakers + 1
            name = f"spk{idx}"
            self._manager.add(name, v)
            return name
        except Exception:
            return None

    def reset(self):
        """清空已注册说话人（比如换了一批人 / 重新开始）。"""
        if self._manager is not None:
            try:
                for spk in list(self._manager.all_speakers):
                    self._manager.remove(spk)
            except Exception:
                pass
