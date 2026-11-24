#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地英译中（完全离线）。

用的是 Argos Translate 的语言包（CTranslate2 格式 + SentencePiece 分词），
但直接调用底层的 CTranslate2，不经过 Argos 那层外壳。原因：
  1. Argos 外壳自带的断句器（stanza）启动时要联网下资源，离线直接报错；
  2. 我们上游的语音识别已经按句子断好了，不需要再断一次；
  3. 少一堆依赖（torch / spacy / stanza 加起来近 600MB，全是白装的）。

模型文件位置：~/.local/share/argos-translate/packages/translate-en_zh-1_9/
  model/model.bin            真正的翻译模型（CTranslate2）
  sentencepiece.model        分词器
"""

import json
import os
import re

import ctranslate2
import sentencepiece as spm

# 分词器用 "▁" 表示词界，中文里不该出现，会串到译文里（比如 "▁所以在得出结论之前"）
_WORD_BOUNDARY = "\u2581"
_CJK = r"\u3000-\u303f\u4e00-\u9fff\uff00-\uffef"


def _clean(text):
    """把词界符还原成空格，再去掉夹在汉字之间的多余空格。"""
    text = text.replace(_WORD_BOUNDARY, " ")
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    return text.strip()

PKG_DIR = os.path.expanduser(
    "~/.local/share/argos-translate/packages/translate-en_zh-1_9"
)


class Translator:
    """英译中。首次调用会慢一点（加载模型），之后每句只要几毫秒。"""

    def __init__(self, pkg_dir=PKG_DIR, threads=4, beam_size=4):
        if not os.path.isdir(pkg_dir):
            raise FileNotFoundError(
                f"找不到翻译语言包：{pkg_dir}\n"
                f"（应该有个 translate-en_zh-1_9 目录，重装：pip install argostranslate 后下 en->zh 包）"
            )
        self.pkg_dir = pkg_dir
        meta_path = os.path.join(pkg_dir, "metadata.json")
        self.target_prefix = ""
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.target_prefix = json.load(f).get("target_prefix", "") or ""

        self.sp = spm.SentencePieceProcessor(
            model_file=os.path.join(pkg_dir, "sentencepiece.model")
        )
        self.ct = ctranslate2.Translator(
            os.path.join(pkg_dir, "model"),
            device="cpu",
            compute_type="auto",
            inter_threads=1,
            intra_threads=threads,
        )
        self.beam_size = beam_size

    def translate(self, text):
        """一句英文 → 一句中文。空串原样返回。"""
        text = (text or "").strip()
        if not text:
            return ""
        tokens = self.sp.encode(text, out_type=str)
        if not tokens:
            return ""
        prefix = [[self.target_prefix]] if self.target_prefix else None
        results = self.ct.translate_batch(
            [tokens],
            target_prefix=prefix,
            replace_unknowns=True,
            beam_size=self.beam_size,
            num_hypotheses=1,
            length_penalty=0.2,
            return_scores=True,
        )
        out = self.sp.decode(results[0].hypotheses[0])
        if self.target_prefix and out.startswith(self.target_prefix):
            out = out[len(self.target_prefix):]
        return _clean(out)


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    tr = Translator()
    print(f"模型加载耗时 {time.time() - t0:.2f} 秒\n")

    samples = sys.argv[1:] or [
        "The Fed's decision to hold rates steady reflects growing concerns about the labor market.",
        "Nobody expected the situation to turn around so quickly.",
        "So what I want to talk about today is why this matters for ordinary people.",
    ]
    for s in samples:
        t0 = time.time()
        out = tr.translate(s)
        print(f"[{(time.time() - t0) * 1000:.0f} ms] {s}")
        print(f"        -> {out}\n")
