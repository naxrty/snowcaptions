#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频来源与音频缓冲切片。

  * mic_blocks / wav_blocks：统一产出 0.1 秒一块的 float32 音频
  * take_segment / trim_buf：第二遍精修要重听那一段音频，所以留一份
"""
import sys
import wave

import numpy as np

from config import BLOCK_SEC, SAMPLE_RATE

# --------------------------------------------------------------------------
# 第二遍精修用的音频缓冲
#
# 为什么要缓冲：流式模型是「喂进去就丢」，不保留音频。而精修要拿这段音频
# 重跑一遍，所以必须自己留一份。只留「上次定稿之后」的那一小段，
# 定稿一次清一次，长会话也不会把内存撑大。
# --------------------------------------------------------------------------

def take_segment(buf, start, end):
    """从缓冲里取出 [start, end] 这段音频（按累计时长切，秒）。

    buf 的元素是 (该块结束时刻, 样本)，块之间首尾相接。
    """
    if end <= start:
        return None
    parts = []
    for endpos, ch in buf:
        st = endpos - len(ch) / SAMPLE_RATE
        if endpos <= start:
            continue
        if st >= end:
            break
        a = max(0, int(round((start - st) * SAMPLE_RATE)))
        b = len(ch) - max(0, int(round((endpos - end) * SAMPLE_RATE)))
        if b > a:
            parts.append(ch[a:b])
    if not parts:
        return None
    return np.concatenate(parts)


def trim_buf(buf, keep_from):
    """丢掉整块都在 keep_from 之前的音频（已经精修过，没用了）"""
    while buf and buf[0][0] <= keep_from:
        buf.pop(0)

def mic_blocks(audio_q):
    """麦克风队列 → 音频块。注意：必须用生成器，
    不能写 iter(audio_q.get, None)——那样会拿 numpy 数组跟 None 做 == 比较，直接报
    'truth value of an array ... is ambiguous'。"""
    while True:
        yield audio_q.get()


def wav_blocks(path):
    """把 wav 文件按 0.1 秒切块产出，格式不对就顺手转一下。"""
    with wave.open(path) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit("[错误] 只支持 16bit PCM 的 wav")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:                      # 简单线性重采样到 16k
        idx = (np.arange(int(len(a) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE).astype(np.int64)
        a = a[np.clip(idx, 0, len(a) - 1)]
    n = int(SAMPLE_RATE * BLOCK_SEC)
    for i in range(0, len(a), n):
        yield a[i:i + n]


# --------------------------------------------------------------------------
