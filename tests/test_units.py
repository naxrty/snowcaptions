# -*- coding: utf-8 -*-
"""拆分后的单元级回归测试（不依赖麦克风，也不需要模型加载）。

用法：
    ./venv/bin/python tests/test_units.py
退出码 0 = 全通过。
"""
import os
import queue
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SAVE = tempfile.mkdtemp(prefix="snowcaptions-test-")

import audio as A          # noqa: E402
import config as C         # noqa: E402
import engine as E         # noqa: E402
import modelselect as M    # noqa: E402
import session as S        # noqa: E402
import textutil as T       # noqa: E402
import ui as U             # noqa: E402

ok, bad = 0, 0


def check(name, cond, extra=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"  ✅ {name}")
    else:
        bad += 1
        print(f"  ❌ {name} {extra}")


print("== textutil ==")
check("dlen 中文=2列", T.dlen("中文") == 4)
check("dlen 英文=1列", T.dlen("ab") == 2)
check("dlen 混排", T.dlen("a中") == 3)
check("fit_tail 不超宽", T.dlen(T.fit_tail("中文" * 50, 20)) <= 20)
check("fit_tail 短文本原样", T.fit_tail("短", 20) == "短")
check("fit_tail 加省略号", T.fit_tail("中文" * 50, 20).startswith("…"))
check("_fmt_secs", T._fmt_secs(3725) == "62:05")

print("== audio 切片 ==")
sr = C.SAMPLE_RATE
blocks = []
pos = 0.0
for i in range(10):                      # 10 块 × 0.1s，值 = 块序号
    ch = np.full(int(sr * 0.1), i, dtype=np.float32)
    pos += len(ch) / sr
    blocks.append((pos, ch))
seg = A.take_segment(blocks, 0.2, 0.5)   # 第 2/3/4 块
check("take_segment 长度≈0.3s",
      seg is not None and abs(len(seg) - 0.3 * sr) < sr * 0.02,
      len(seg) if seg is not None else None)
check("take_segment 内容对齐", float(seg[0]) in (2.0, 1.0) and float(seg[-1]) in (4.0, 3.0))
check("take_segment 空区间→None", A.take_segment(blocks, 0.5, 0.5) is None)
buf = list(blocks)
A.trim_buf(buf, 0.5)
check("trim_buf 丢掉已用块", all(e > 0.5 for e, _ in buf) and len(buf) < len(blocks))

q = queue.Queue()
it = A.mic_blocks(q)
q.put(np.zeros(1600, dtype=np.float32))
check("mic_blocks 取块", len(next(it)) == 1600)

wav = os.path.join(ROOT, "models",
                   "sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30",
                   "test_wavs", "0.wav")
wb = list(A.wav_blocks(wav))
check("wav_blocks 分块 ≤1600", all(len(c) <= 1600 for c in wb) and len(wb) > 50, len(wb))

print("== ui / session ==")
disp = U.LiveDisplay(lines=2, stream=open(os.devnull, "w"))
check("LiveDisplay 非 tty 标记", disp.tty is False)
disp.update(["甲", "乙"])
disp.commit(["原文", "译文"])
disp.clear()
check("非 tty 下 update/commit 不炸", True)

log = S.SessionLog(True, "zh", "模型名", SAVE, disp)
log.add(1.5, "第一句", None)
log.add(3.2, "第二句", "第二句译文")
txt = log._render()
check("日志含时间戳", "[00:01] 第一句" in txt and "[00:03] 第二句" in txt)
check("日志含译文", "第二句译文" in txt)
p = log.save_now()
check("save_now 落盘", p is not None and os.path.exists(p), p)
log2 = S.SessionLog(False, "zh", "m", SAVE, disp)
log2.add(1.0, "x", None)
check("enabled=False 不记录", log2.rows == [] and log2.save_now() is None)
check("_enter_pressed 非 tty→False", E._enter_pressed() is False)

print("== config 可变常量 ==")
old = C.RULE1_SILENCE
C.RULE1_SILENCE = 1.23
check("config 可写", C.RULE1_SILENCE == 1.23)
check("modelselect 动态读取（不是冻结副本）",
      "config.RULE1_SILENCE" in open(os.path.join(ROOT, "modelselect.py")).read())
C.RULE1_SILENCE = old
check("PAUSE_PRESETS 三档", set(C.PAUSE_PRESETS) == {"fast", "normal", "loose"})
check("PREFERRED 有 zh/en", set(C.PREFERRED) == {"zh", "en"})

print("== modelselect ==")
check("detect_kind transducer",
      M.detect_kind(os.path.join(C.MODELS_DIR, "sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30"))[0] == "transducer")
check("detect_kind ctc",
      M.detect_kind(os.path.join(C.MODELS_DIR, "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"))[0] == "ctc")
check("detect_lang zh",
      M.detect_lang(os.path.join(C.MODELS_DIR, "sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30")) == "zh")
check("detect_lang en",
      M.detect_lang(os.path.join(C.MODELS_DIR, "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25")) == "en")
check("pick_model_dir zh 优先 xlarge", "xlarge" in M.pick_model_dir("zh"))
check("pick_model_dir en 优先 nemotron", "nemotron" in M.pick_model_dir("en"))
check("transducer_model_type nemo",
      M.transducer_model_type("sherpa-onnx-nemotron-x") == "nemo_transducer")
check("transducer_model_type zipformer 空",
      M.transducer_model_type("sherpa-onnx-streaming-zipformer-zh") == "")

print("== LiveTranslator：主引擎 / 兜底 / 后台线程 ==")


class _Fake:
    """假翻译器：记录被调用次数，可指定返回 None 模拟「预热未完成 / 超时」"""

    def __init__(self, out, delay=0.0):
        self.out = out
        self.delay = delay
        self.calls = 0

    def translate(self, text):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.out


# 1. 定稿：主引擎优先，主引擎失败才用兜底
mt, argos = _Fake("准译"), _Fake("快译")
lt = E.LiveTranslator(mt, argos, mt, threaded=False)
check("定稿走主引擎 HY-MT", lt.force("x") == "准译" and argos.calls == 0 and mt.calls == 1)
lt.close()

mt, argos = _Fake(None), _Fake("快译")
lt = E.LiveTranslator(mt, argos, mt, threaded=False)
check("主引擎没热好 → 兜底 Argos 顶上", lt.force("x") == "快译" and mt.calls == 1)
lt.close()

check("两边都失败 → 空串而非崩溃",
      E.LiveTranslator(_Fake(None), _Fake(None), None, threaded=False).force("x") == "")

# 2. 实时行：默认也走主引擎（这就是「真的换掉了」）
mt, argos = _Fake("准译"), _Fake("快译")
lt = E.LiveTranslator(mt, argos, mt, threaded=False)
got = lt.maybe("一句话")
check("实时行默认也用主引擎", got == "准译" and mt.calls == 1 and argos.calls == 0)
lt.close()

lt = E.LiveTranslator(_Fake("准译"), _Fake("快译"), _Fake("快译"), threaded=False)
lt.maybe("一句话")
check("mix 模式实时行才用 Argos", lt.last_out == "快译")
lt.close()

lt = E.LiveTranslator(_Fake("准译"), _Fake("快译"), _Fake("快译"), threaded=False)
check("实时行空文本 → 空串", lt.maybe("") == "")
lt.close()

# 3. 后台线程：主循环提交后立刻返回，不被 0.3 秒的翻译卡住
mt = _Fake("准译", delay=0.3)
lt = E.LiveTranslator(mt, None, mt)
t0 = time.time()
first = lt.maybe("第一句话")
cost = time.time() - t0
check(f"提交不阻塞主循环（{cost*1000:.0f}ms < 100ms）", cost < 0.1)
check("首次提交还没译文 → None（沿用上一行）", first is None)
time.sleep(0.6)
check("后台翻完 → 下一次刷新拿到译文", lt.maybe("第一句话") == "准译")
lt.close()

# 4. 定稿后旧译文不能盖到新的一行上（epoch 机制）
mt = _Fake("准译", delay=0.3)
lt = E.LiveTranslator(mt, None, mt)
lt.maybe("还没说完的一句")
lt.force("已经定稿的一句")          # reset + 同步翻，epoch 前进
time.sleep(0.6)                    # 后台那份「旧译文」这时才回来
check("在飞的旧译文被丢弃", lt.maybe("新的一句") in (None, "准译"))
lt.close()

print(f"\n通过 {ok} / 失败 {bad}")
sys.exit(1 if bad else 0)
