#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""识别主循环（麦克风 / 音频文件共用同一套逻辑）。

这里只做「喂音频 → 解码 → 断句 → 定稿」的调度，具体显示/翻译/存盘都交给
传入的对象（disp / tr / log / refiner），所以要改节奏只改这个文件。
"""
import sys
import threading
import time

import numpy as np

from audio import take_segment, trim_buf
from config import (BLOCK_SEC, COMMIT_DELAY, FLUSH_SILENCE, LEAD_PUNCT,
                    MIN_COMMIT_CHARS, SAMPLE_RATE, SOFT_FLUSH_SCREENS,
                    WARMUP_SILENCE, ZH_MIN_DELTA, ZH_MIN_INTERVAL)
from textutil import dlen, term_cols

# --------------------------------------------------------------------------
# 翻译（可选）
# --------------------------------------------------------------------------

class LiveTranslator:
    """管译文的两个用途：实时行（草稿）和定稿行（正式）。

        primary   主引擎，HY-MT（约 280ms）—— 定稿行用它，实时行默认也用它
        fallback  兜底引擎，Argos（约 68ms）—— primary 没热好 / 超时 / 报错时顶上
        live      实时行单独指定引擎，默认同 primary；
                  想「实时快、定稿准」可以传 Argos（--translate-engine mix）

    为什么要开后台线程：HY-MT 一句 0.28 秒，而实时行的刷新间隔是 0.5 秒。
    同步翻的话主循环每 0.5 秒被卡掉 0.28 秒，麦克风队列只进不出，
    说着说着显示就落后十几秒。所以实时行改成「提交文本 → 立即返回上一个译文」，
    翻完再在下一次刷新时显示，主循环一秒都不等。
    """

    def __init__(self, primary, fallback=None, live=None, threaded=True):
        self.primary = primary
        self.fallback = fallback
        self.live = live if live is not None else primary
        self.threaded = bool(threaded and self.live is not None)
        self.last_src = None
        self.last_time = 0.0
        self.last_out = ""
        self._ver = 0          # 每出一份新译文 +1
        self._shown = 0        # 已经交给显示层的版本号
        self._epoch = 0        # 来自哪一句；reset 后旧结果一律丢弃
        self._pending = None
        self._stop = False
        self._cv = threading.Condition()
        self._th = None
        if self.threaded:
            self._th = threading.Thread(target=self._loop, name="live-tr", daemon=True)
            self._th.start()

    # ---------------- 后台翻译 ----------------

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                text, epoch = self._pending, self._epoch
                self._pending = None
            try:
                out = self._run_live(text)
            except Exception:
                out = None
            with self._cv:
                if out and epoch == self._epoch:
                    self.last_out = out
                    self._ver += 1

    def _run_live(self, text):
        """实时行：先 live 引擎，失败再退 fallback；都不行返回 None"""
        for eng in (self.live, self.fallback):
            if eng is None:
                continue
            out = eng.translate(text)
            if out:
                return out
        return None

    def _take(self):
        """有新译文就交出去，没有就 None（调用方沿用上一行）"""
        if self._ver != self._shown:
            self._shown = self._ver
            return self.last_out
        return None

    # ---------------- 对外接口 ----------------

    def reset(self):
        self.last_src = None
        self.last_time = 0.0
        self.last_out = ""
        with self._cv:
            self._pending = None
            self._epoch += 1        # 在飞的旧译文作废，免得盖到新的一行上
            self._ver = self._shown

    def force(self, text):
        """定稿时用：同步翻一次（宁可多等 0.28 秒，也要保证这一行是完整的）。

        primary 拿不到结果就回退 fallback，两边都失败也要返回空串，
        因为调用方要把结果写进第二行显示。
        """
        self.reset()
        out = None
        for eng in (self.primary, self.fallback):
            if eng is None:
                continue
            out = eng.translate(text)
            if out:
                break
        out = out or ""
        self.last_src = text
        self.last_out = out
        self.last_time = time.time()
        self._ver += 1
        self._shown = self._ver
        return out

    def maybe(self, text, lines=None):
        """实时行用：够久或变化够大才翻一次，返回新译文或 None"""
        if self.live is None:
            return ""
        if not text:
            self.reset()
            return ""
        now = time.time()
        delta = 0 if self.last_src is None else abs(len(text) - len(self.last_src))
        if self.last_src is not None and (now - self.last_time) < ZH_MIN_INTERVAL:
            return self._take()
        if self.last_src is not None and delta < ZH_MIN_DELTA and text.startswith(self.last_src):
            return self._take()
        self.last_src = text
        self.last_time = now
        if self.threaded:
            with self._cv:
                self._pending = text
                self._cv.notify()
            return self._take()
        out = self._run_live(text)      # 单线程模式（测试 / 极简场景）
        self.last_out = out or ""
        self._ver += 1
        return self._take()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._th is not None:
            self._th.join(timeout=1.0)

# --------------------------------------------------------------------------
# 识别主循环（麦克风 / 音频文件共用同一套逻辑）
# --------------------------------------------------------------------------

def _enter_pressed():
    """非阻塞检查 stdin 是否有输入，并吃掉一行（回车）。

    用 select 轮询，没有输入时立即返回，不阻塞识别循环。
    只在 stdin 是 tty 时生效；重定向/管道场景返回 False，避免卡住。
    """
    try:
        import select
        if not sys.stdin.isatty():
            return False
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return False
        sys.stdin.readline()
        return True
    except Exception:
        return False


def run_stream(rec, blocks, disp, tr=None, realtime=False, refiner=None, log=None,
               speaker=None):
    """blocks：可迭代对象，每轮产出一块 float32 音频。

    关键设计：端点触发时 **不重置识别流**，只在「累积文本」上取增量。
    为什么？这类流式模型（尤其 NeMo 系）有几百毫秒的「前瞻」缓冲，端点触发那一刻
    最后几个词其实还没解码出来。老写法一 reset 就把它们连同音频一起丢了——
    实测 0.wav 的 "the squalid" 就是这样蒸发的（WER 11%）。
    改成「不重置 + 取增量」之后同样两段音频的 WER 降到 0%。
    实现在 bench_decode.py 里，三种策略可直接复现对比。

    refiner：可选的第二遍精修器。一句话定稿时把那一小段音频重听一遍，
    用更准的结果替换流式草稿。**只影响定稿内容，不影响实时行的刷新节奏**。

    speaker：可选的说话人标注器（speaker.SpeakerTagger）。一句话定稿时拿这段
    音频算声纹，给这句标上「说话人 N」。识别不了（音频太短/模型没加载/报错）
    一律静默跳过，绝不拖慢或打断转写。
    """
    stream = rec.create_stream()
    two_line = disp.n > 1
    committed = ""                     # 已经定稿出去的完整文本
    live, live_zh = "", ""
    soft_limit = int(term_cols() * SOFT_FLUSH_SCREENS)
    audio_pos = 0.0                    # 已喂进去的音频时长（秒），用它当计时器
    pending = None                     # 端点触发的时刻，等 COMMIT_DELAY 后再定稿

    # 精修用的音频缓冲：buf 累积音频块，refine_from 记录上次精修覆盖到的位置
    buf = []
    refine_from = 0.0
    use_refine = refiner is not None and getattr(refiner, "enabled", False)

    # 说话人用的音频缓冲：和精修独立，避免「关精修时说话人也拿不到音频」
    spk_buf = []
    spk_from = 0.0
    use_speaker = speaker is not None and getattr(speaker, "enabled", False)

    # 预热：先垫一段静音，否则模型启动会吃掉开头约 0.5 秒（第一个词直接没了）
    stream.accept_waveform(
        SAMPLE_RATE, np.zeros(int(SAMPLE_RATE * WARMUP_SILENCE), dtype=np.float32))
    while rec.is_ready(stream):
        rec.decode_stream(stream)

    def tail_of(full):
        """从整段已解码文本里，切出还没定稿的那一截"""
        nonlocal committed
        if full.startswith(committed):
            raw = full[len(committed):]
        else:
            # 极少见：解码结果被修正过，退回最长公共前缀，避免重复吐字
            k, m = 0, min(len(committed), len(full))
            while k < m and committed[k] == full[k]:
                k += 1
            committed = full[:k]
            raw = full[k:]
        return raw.lstrip().lstrip(LEAD_PUNCT).lstrip()

    def finish(tail, upto=None):
        """把这一截定稿：落两行（原文+译文）或一行。

        有精修器时，先把这一段的原始音频重听一遍，用更准的结果替换流式草稿。
        实测（bench_twopass.py，5 段中文难例）：错字率 29.8% → 18.7%，
        且**每一段都更准**，没有出现"修好一个词、弄坏另一个词"的净损失。

        精修失败（模型没加载、音频太短、推理报错）一律静默降级为流式结果，
        绝不吞掉用户说的话。
        """
        nonlocal live, live_zh, refine_from, spk_from
        tail = tail.strip()
        live, live_zh = "", ""
        if tr:
            tr.reset()
        if not tail:
            return

        # 定稿这一句对应的音频段（精修和说话人都要拿，各自独立切）
        end = audio_pos if upto is None else upto

        if use_refine:
            seg = take_segment(buf, refine_from, end)
            if seg is not None:
                refined = refiner.refine(seg)
                if refined:
                    tail = refined
            refine_from = max(refine_from, end)
            trim_buf(buf, refine_from)

        # 说话人：拿这段音频算声纹 → 编号（识别不了返回 None）
        speaker_id = None
        if use_speaker:
            spk_seg = take_segment(spk_buf, spk_from, end)
            if spk_seg is not None:
                speaker_id = speaker.identify(spk_seg)
            spk_from = max(spk_from, end)
            trim_buf(spk_buf, spk_from)

        translated = None
        if two_line and tr:
            translated = tr.force(tail)
            disp.commit([tail, translated], speaker_id=speaker_id)
        else:
            disp.commit([tail], speaker_id=speaker_id)

        # 记录到会话日志（供「按回车随时存盘」）
        if log is not None and log.active:
            upto_secs = audio_pos if upto is None else upto
            log.add(upto_secs, tail, translated, speaker_id=speaker_id)

    for samples in blocks:
        stream.accept_waveform(SAMPLE_RATE, samples)
        audio_pos += len(samples) / SAMPLE_RATE
        if use_refine:
            buf.append((audio_pos, samples))     # 留一份音频给第二遍重听
        if use_speaker:
            spk_buf.append((audio_pos, samples)) # 留一份音频给说话人识别
        while rec.is_ready(stream):
            rec.decode_stream(stream)

        full = rec.get_result(stream)
        tail = tail_of(full)

        if rec.is_endpoint(stream):
            if pending is None:
                pending = audio_pos
        else:
            pending = None             # 又说起话来了，取消待定稿

        did_commit = False
        if pending is not None and audio_pos - pending >= COMMIT_DELAY:
            pending = None
            if len(tail.strip()) >= MIN_COMMIT_CHARS:
                committed = full
                finish(tail)
                did_commit = True

        if did_commit:
            disp.update([""] * (2 if two_line else 1))
        elif tail != live:
            live = tail
            if two_line and tr:
                got = tr.maybe(tail)
                if got is not None:
                    live_zh = got
            disp.update([live, live_zh] if two_line else [live])

        # 一口气说太久、一屏放不下了 → 定稿换行，实时行重新开始
        if tail and dlen(tail) > soft_limit:
            committed = full
            finish(tail)
            disp.update([""] * (2 if two_line else 1))

        # 按回车 → 当场把已定稿内容存盘（同一份文件覆盖更新）
        if log is not None and log.active and _enter_pressed():
            path = log.save_now()
            if path:
                log.show_saved(path, len(log.rows))

        if realtime:
            time.sleep(BLOCK_SEC)

    # 收尾：补一段静音，把最后没来得及断句的话吐出来
    stream.accept_waveform(SAMPLE_RATE, np.zeros(int(SAMPLE_RATE * FLUSH_SILENCE), dtype=np.float32))
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    finish(tail_of(rec.get_result(stream)))
    disp.clear()
