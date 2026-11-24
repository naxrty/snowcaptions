#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局常量与路径（所有「实测调出来的」数字都集中在这里）。

注意：RULE1_SILENCE / RULE2_SILENCE 会被 --pause 在运行时改写。
所以其它模块必须 `import config` 之后用 config.RULE1_SILENCE 访问，
**不能** `from config import RULE1_SILENCE` —— 那样拿到的是一份冻结的副本，
--pause 就不生效了。其它常量不会被改写，可以放心 from-import。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
# 第二遍精修模型单独放一个目录。为什么不能混在 models/ 里：Fun-ASR-Nano 的
# 目录里有 tokens.txt + model.int8.onnx，detect_kind 会把它认成一个可用的
# **流式 CTC 模型**，于是 pick_model_dir 的自动选择就可能挑中它当第一遍用，
# 直接报错或输出乱码。分开放最干净，--list 的输出也保持可读。
REFINE_DIRS = [os.path.join(HERE, "models_refine"), MODELS_DIR]

SAMPLE_RATE = 16000
BLOCK_SEC = 0.1          # 每次处理 0.1 秒音频，越小越实时
THREADS = 4

# 断句规则（秒）：停顿多久算一句话说完。
#
# 2026-09-18 实测（见 bench_endpoint.py）：人说话时句中的换气、逗号停顿通常在
# 0.6~1.1 秒。原来设 1.0/0.5 太小了——一句话说到一半换口气就被判定成"说完了"，
# 于是半句被切出去单独翻译，译文自然割裂。
#
# 扫参结果（一段含 0.74s 和 1.06s 句中停顿、1.7s 句尾停顿的音频）：
#     rule1 ≤ 1.7  → 被切成 4~5 段（错）
#     rule1 ≥ 1.8  → 正好 3 段（对）
# 所以取 1.9 留一点余量（真人停顿不如合成音频规整）。
# 注意：实际起决定作用的是 rule1，rule2 在这个模型上几乎不起作用（扫参 0.8~1.8
# 结果一字不变），但两个一起给更保险。
RULE1_SILENCE, RULE2_SILENCE = 1.9, 1.5

# 断句松紧档位：(rule1, rule2)
#   fast   跟读、连续说话——切得勤，反应快，但句中断顿容易割裂
#   normal 默认，容忍到约 1.1 秒的句中停顿
#   loose  讲座、朗读、边说边想——句子长、停顿多时用
PAUSE_PRESETS = {
    "fast": (1.2, 0.8),
    "normal": (1.9, 1.5),
    "loose": (2.6, 2.2),
}

# 一句话连续说太久（实时行已远超一屏）时自动定稿换行，避免无限增长
SOFT_FLUSH_SCREENS = 2.0

# 收尾时补的静音长度（秒）：把模型前瞻缓冲里剩的话冲出来
FLUSH_SILENCE = 1.5

# 开头先垫一段静音（秒）。实测这个模型刚启动时会吃掉开头约 0.5 秒的音频——
# 同一句 "Good evening everyone…" 直接喂进去，开头那个 "Good" 就没了；前面垫 0.5 秒
# 静音就完整了。所以要在正式音频之前先喂一段静音当预热，否则用户开口第一个词会丢。
WARMUP_SILENCE = 0.8

# 转录记录：所有定稿句子统一存到一个目录。默认放在脚本同级 transcripts/ 下。
SAVE_DIR = os.path.join(HERE, "transcripts")

# 端点触发后，再等这么久（秒）才真正定稿。
# 这段时间里如果又检测到人声（端点解除），就取消定稿、继续往下累积——
# 所以它同时也是第二道「误切保护」，能再吸收一点额外的停顿。
# 因为 rule1 已经包含了 1.9 秒的静音等待，这里不宜再叠加太久，
# 否则「说完一句话 → 看到译文」的总延迟会超过 2.5 秒，实时感就没了。
COMMIT_DELAY = 0.3

# 从下一句开头摘掉的「上一句漏下来的标点」。
# 实测（见 bench_decode.py / 手测时间线）：这句模型的句末标点要等到**下一句开始说话**
# 时才吐出来——端点触发后 2.2 秒才出现，等多久都等不到。所以只能在边界上处理：
# 把漏到下一句开头的 ". The question is..." 摘成 "The question is..."。
# 代价是丢掉一个句号，对「看懂大意」没有影响。
LEAD_PUNCT = ".,!?;:、。，！？；：…·\"'“”‘’（）()[]【】《》<>-—– \t"

# 定稿的最短长度（字符）。太短的碎片不单独成句——音频开头/中间的静音会触发端点，
# 把 "God" 这种一个词切出来，脱离上下文翻译出来就是「天呐」这种错译。
# 不达标的碎片留在实时行里跟后面的词合并，下次端点再一起定稿。
MIN_COMMIT_CHARS = 4

# 端点规则 3 的 utterance 长度阈值——**必须设成极大值，等于关掉这条规则**。
#
# 背景：sherpa-onnx 的端点检测有三条规则，其中 rule3 是「utterance 已经够长
# （默认 20 秒）时，只要有一点点静音就断句」。而我们为了解决「端点重置吞词」
# 的问题，选择了**全程不 reset 识别流**，于是 utterance 长度会一路累加，
# 用满 20 秒就会踩中这条规则。
#
# 症状非常有辨识度：**前面几句是完整的长句，然后越用越碎，最后每段只剩
# 2~4 个词**（用户遇到的正是这个）。实测 35 秒纯连续朗读——音频里一处静音
# 都没有——前 30 秒正常，之后碎成 28 段。
#
# 注意：不要试图靠「定稿后 reset 识别流」来绕开。那样确实能让碎句变少
# （28 段 → 5 段），但会把模型前瞻缓冲里还没解码出来的词一起丢掉
# （实测 "your brain never GETS time to relax" 的 "gets" 消失）。
# 直接把这条规则关掉才是对的：既不碎句，也不丢词。
NEVER_TRIGGER_SEC = 99999.0

# 实时译文的最小刷新间隔（秒）与最小文本变化量（字符）——
# 每冒一个字就重翻一次太浪费，攒一攒再翻，肉眼看不出差别
ZH_MIN_INTERVAL = 0.5
ZH_MIN_DELTA = 4

# --------------------------------------------------------------------------
# 第二遍翻译（定稿行专用）：HY-MT1.5-1.8B（GGUF + llama-server）
#
# 为什么是「两遍」：实时行要 0.05 秒级的反馈，只有 Argos（82MB / ~68ms）扛得住；
# 而定稿行可以等 0.3 秒，那就用翻译专用大模型换质量。
#
# 实测（2026-11-24，24 句难句，见 /tmp/mt_test.py）：
#     Argos        完全正确 25% / 可用 54% / 明显错误 46%   平均 68ms
#     HY-MT 1.8B   完全正确 79% / 可用 92% / 明显错误  8%   平均 278ms
# Argos 错在「字面直译」（kicked the can→踢罐子、numbers→号码）；
# HY-MT 只剩法律/专业搭配和极少数习语会错，可用术语表补。
# --------------------------------------------------------------------------
MT_DIR = os.path.join(HERE, "models_mt")     # 放 *.gguf（不入 git，约 1.1G）
MT_GGUF = ""                 # 留空 = 自动挑 models_mt/ 下的 *.gguf（优先 Q4_K_M）
MT_PORT = 8097               # llama-server 只监听 127.0.0.1，端口被占用时自动往后找
MT_THREADS = 4
MT_CTX = 2048                # 一句字幕用不完，给 2048 足够
MT_TIMEOUT = 3.0             # 单句超时（秒）。超时/报错一律回退 Argos，绝不卡住定稿
MT_TERMS = os.path.join(MT_DIR, "terms.txt")   # 术语表（可选），每行「原文=>译文」

MT_PROMPT = ("将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n"
             "{src}")
# 有术语表时用这个模板（模型官方推荐的「术语干预」写法）
MT_TERMS_PROMPT = ("参考下面的翻译：\n{terms}\n\n"
                   "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n"
                   "{src}")
# 官方推荐采样参数。temperature 别调 0：翻译要的是稳定，不是发挥
MT_SAMPLING = dict(temperature=0.7, top_k=20, top_p=0.6, repeat_penalty=1.05)

# 自动挑选模型时的优先级（越靠前越优先）
PREFERRED = {
    "en": [
        "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    ],
    "zh": [
        "sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30",
        "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01",
    ],
}
