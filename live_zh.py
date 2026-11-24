#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地实时语音转写（+ 可选翻译），完全离线，不联网。

这个文件只负责「命令行入口」：解析参数、挑模型、装配各部件、启动主循环。
真正的逻辑分在几个模块里：
    config.py       全局常量（断句阈值、路径、模型优先级）
    modelselect.py  模型发现 / 挑选 / 加载
    audio.py        麦克风与 wav 音源、精修用的音频切片
    engine.py       识别主循环（断句、定稿、调度）
    ui.py           终端实时显示
    session.py      会话记录与存盘
    textutil.py     纯文本工具（可单测）
    refine.py       第二遍精修（离线大模型重听）
    translator.py   英译中（CTranslate2）

用法：
    双击「实时转写.command」（中文）或「英文同传.command」（英文 + 翻译）
    或在终端里：
        ./venv/bin/python live_zh.py                    # 自动挑模型
        ./venv/bin/python live_zh.py --lang en          # 强制用英文模型
        ./venv/bin/python live_zh.py --list             # 列出本机所有模型
        ./venv/bin/python live_zh.py --wav a.wav        # 拿音频文件当音源（测试）
        ./venv/bin/python live_zh.py --wav a.wav --realtime   # 按真实速度播放
        ./venv/bin/python live_zh.py --no-translate     # 关掉翻译

换模型：SHERPA_MODEL=模型目录名 ./venv/bin/python live_zh.py
"""

import argparse
import os
import queue
import time

import sounddevice as sd

import config
from audio import mic_blocks, wav_blocks
from config import (BLOCK_SEC, MODELS_DIR, PAUSE_PRESETS, REFINE_DIRS,
                    SAMPLE_RATE, SAVE_DIR, THREADS)
from engine import LiveTranslator, run_stream
from modelselect import build_recognizer, detect_lang, list_models, pick_model_dir
from session import SessionLog
from ui import LiveDisplay

# --------------------------------------------------------------------------
# 命令行入口
#
# 职责只有：解析参数 → 挑模型 → 装配各部件 → 启动主循环。
# 想改识别行为请看 engine.py / config.py，别在这里加逻辑。
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 翻译引擎装配
#
# 默认（auto）：HY-MT 1.8B 全程接管 —— 实时行和定稿行都是它，Argos 只在
# HY-MT 还没热好 / 超时 / 报错时顶上当兜底。实测（24 句难句）完全正确率
# Argos 25% → HY-MT 79%，可用率 54% → 92%。
# 实时行的调用放在后台线程（见 engine.LiveTranslator），所以 0.28 秒的
# 单句耗时不会卡住采音，麦克风不会越说越滞后。
# --------------------------------------------------------------------------

def _load_hotwords(spec):
    """把 --hotwords 的输入归一成逗号分隔的字符串。

    支持两种形式：
      1. 直接逗号串："KPI,张明,深圳"
      2. 文件路径：文件内容一行一个词（自动合并成逗号串）
    返回空串表示没有热词。
    """
    if not spec:
        return ""
    # 如果是一个存在的文件路径，读文件内容（一行一个词）
    if os.path.exists(spec):
        try:
            with open(spec, encoding="utf-8") as f:
                words = [ln.strip() for ln in f if ln.strip()]
            return ",".join(words)
        except Exception as e:
            print(f"  热词文件读取失败，忽略：{type(e).__name__}: {e}")
            return ""
    # 否则按逗号串处理（兼容中文逗号）
    words = [w.strip() for w in spec.replace("，", ",").split(",") if w.strip()]
    return ",".join(words)


def _build_translator(args, keep_alive):
    mode = args.translate_engine          # auto / hymt / argos / mix
    fast = slow = None
    want_argos = mode in ("auto", "argos", "mix")
    want_mt = mode in ("auto", "hymt", "mix")

    if want_argos:
        try:
            from translator import PKG_DIR, Translator
            fast = Translator(threads=THREADS)
            print(f"  翻译: Argos 已加载（兜底，本地离线）\n    {PKG_DIR}")
        except Exception as e:
            print(f"  翻译: Argos 不可用 —— {type(e).__name__}: {e}")

    if want_mt:
        try:
            from mtengine import MTTranslator
            mt = MTTranslator(gguf=args.mt_model or config.MT_GGUF,
                              model_dir=config.MT_DIR, port=config.MT_PORT,
                              threads=config.MT_THREADS, timeout=config.MT_TIMEOUT,
                              ctx=config.MT_CTX, terms=config.MT_TERMS,
                              prompt=config.MT_PROMPT,
                              terms_prompt=config.MT_TERMS_PROMPT,
                              sampling=config.MT_SAMPLING)
            if mt.available:
                mt.start_async()          # 后台拉服务，不挡启动
                slow = mt
                keep_alive.append(mt)
                print(f"  翻译: HY-MT 已挂载（主力，后台预热）")
                print(f"    {os.path.basename(mt.gguf)}  (端口 {mt.want_port})")
            else:
                print(f"  翻译: HY-MT 未启用 —— {mt.why}")
        except Exception as e:
            print(f"  翻译: HY-MT 启用失败 —— {type(e).__name__}: {e}")

    if fast is None and slow is None:
        print("  翻译: 两个引擎都不可用，改为只转写")
        return None

    if mode == "mix" and fast is not None and slow is not None:
        print("  分工: 实时行 Argos（快 0.07s）/ 定稿行 HY-MT（准）")
        return LiveTranslator(slow, fast, fast)

    if slow is not None:
        # auto / hymt：全程 HY-MT，Argos 只当兜底
        if fast is not None:
            print("  分工: 全程 HY-MT（实时行后台线程翻，Argos 兜底）")
        else:
            print("  分工: 全程 HY-MT（无兜底）")
        return LiveTranslator(slow, fast, slow)

    print("  分工: 只有 Argos 可用，全程走旧引擎")
    return LiveTranslator(fast, None, fast, threaded=False)


def main():
    ap = argparse.ArgumentParser(description="本地实时语音转写 / 翻译（离线）")
    ap.add_argument("--wav", help="不接麦克风，改用一个 wav 文件当音源（测试用）")
    ap.add_argument("--realtime", action="store_true", help="配合 --wav：按真实速度播放")
    ap.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                    help="强制使用某语种的模型（默认自动）")
    ap.add_argument("--model", help="直接指定 models/ 下的模型目录名")
    ap.add_argument("--translate", dest="translate", action="store_true", default=None,
                    help="强制开启翻译")
    ap.add_argument("--no-translate", dest="translate", action="store_false",
                    help="强制关闭翻译")
    ap.add_argument("--translate-engine", choices=["auto", "hymt", "argos", "mix"],
                    default="auto",
                    help="翻译引擎：auto=全程 HY-MT，Argos 兜底（默认）/ hymt=只用 HY-MT"
                         "/ argos=只用旧引擎（行为与改造前完全一致）"
                         "/ mix=实时行 Argos + 定稿行 HY-MT")
    ap.add_argument("--mt-model", default="",
                    help="指定 models_mt/ 下的 gguf 文件名或完整路径（默认自动挑 Q4_K_M）")
    ap.add_argument("--pause", choices=["fast", "normal", "loose"],
                    help="断句松紧：fast 切得勤 / normal 默认 / loose 句子长（讲座、朗读）")
    ap.add_argument("--list", action="store_true", help="列出本机所有模型后退出")
    ap.add_argument("--refine", dest="refine", action="store_true", default=True,
                    help="第二遍精修（默认开启）：定稿时用离线大模型重听一遍，更准")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="关闭第二遍精修，回到纯流式（行为与本改动前完全一致）")
    ap.add_argument("--refine-model",
                    choices=["funasr_nano", "qwen3_asr", "sense_voice"],
                    help="指定第二遍用哪个精修模型（默认自动挑）")
    ap.add_argument("--hotwords", default="",
                    help="热词表：逗号分隔的专有名词/人名/术语（如 \"KPI,张明,深圳\"），"
                         "可读文件路径（一行一个词）。用于精修和第一遍识别纠偏")
    ap.add_argument("--speaker", dest="speaker", action="store_true", default=None,
                    help="开启说话人识别：每句话标上「说话人 N」（区分不同人）")
    ap.add_argument("--no-speaker", dest="speaker", action="store_false",
                    help="关闭说话人识别（默认）")
    ap.add_argument("--save-dir", default=SAVE_DIR,
                    help="转录记录保存目录（默认 scripts 同级的 transcripts/）")
    ap.add_argument("--no-save", action="store_true",
                    help="完全不记录转写内容（行为和以前一样）")
    ap.add_argument("--ui", choices=["terminal", "socket"], default="terminal",
                    help="显示层：terminal=终端实时区（默认）/ socket=推给 Swift 菜单栏 App")
    ap.add_argument("--ui-port", type=int, default=7711,
                    help="socket 显示层的端口（默认 7711，与 Swift 端约定）")
    args = ap.parse_args()

    if args.list:
        print(f"模型目录：{MODELS_DIR}\n")
        print(f"{'模型':<62} {'类型':<12} {'语种'}")
        print("-" * 88)
        for name, kind, lang in list_models():
            print(f"{name:<62} {kind:<12} {lang}")
        return

    if args.model:
        os.environ["SHERPA_MODEL"] = args.model

    if args.pause:
        config.RULE1_SILENCE, config.RULE2_SILENCE = PAUSE_PRESETS[args.pause]

    # 热词：支持直接逗号串，或给一个文件路径（一行一个词）
    hotwords = _load_hotwords(args.hotwords)

    md = pick_model_dir(args.lang)
    src_lang = detect_lang(md)
    print("正在加载模型…")
    print(f"  目录: {os.path.basename(md)}")
    print(f"  语种: {src_lang}")
    print(f"  断句: 静音 {config.RULE1_SILENCE}s 算一句说完"
          f"（{'默认' if not args.pause else args.pause}）")
    if hotwords:
        print(f"  热词: {hotwords}")
    rec = build_recognizer(md, hotwords=hotwords)

    # 第二遍精修：定稿时把那一小段音频交给离线大模型重听，纠正流式模型的同音字错误。
    # 实测（bench_twopass.py，5 段中文难例）：错字率 29.8% → 18.7%，且每段都更准。
    # 它只影响「定稿」的内容，实时行的刷新节奏完全不变。
    refiner = None
    if args.refine:
        try:
            from refine import Refiner
            refiner = Refiner(REFINE_DIRS, prefer=args.refine_model, threads=THREADS,
                              hotwords=hotwords)
            if refiner.enabled:
                t0 = time.time()
                refiner.load()          # 启动时预加载，免得第一句定稿才卡一下
                print(f"  精修: {refiner.describe()}（加载 {time.time()-t0:.2f}s）")
            else:
                print("  精修: 未启用（models_refine/ 下没有可用的精修模型）")
        except Exception as e:
            print(f"  精修: 启用失败，仅用第一遍 —— {type(e).__name__}: {e}")
            refiner = None

    # 说话人识别：可选。每句定稿时拿这段音频算声纹，标上「说话人 N」。
    # 默认关闭；开启需要 models_speaker/ 下有 CAM++ 声纹模型（约 27MB）。
    speaker = None
    if args.speaker:
        try:
            from speaker import SpeakerTagger
            spk = SpeakerTagger(threads=THREADS)
            if spk.enabled:
                speaker = spk
                print(f"  说话人: {spk.describe()}")
            else:
                print(f"  说话人: {spk.describe()}")
        except Exception as e:
            print(f"  说话人: 启用失败 —— {type(e).__name__}: {e}")
            speaker = None
    else:
        print("  说话人: 未启用")

    # 决定要不要翻译：英文源默认翻，中文源默认不翻
    do_tr = args.translate if args.translate is not None else (src_lang == "en")
    tr = None
    keep_alive = []          # 需要退出时收拾的引擎（HY-MT 的 llama-server 子进程）
    if do_tr:
        tr = _build_translator(args, keep_alive)
    else:
        print("  翻译: 未启用")
    if not args.wav:
        # 打印实际用的是哪个麦克风。戴耳机时容易搞不清系统切到了哪个输入设备，
        # 有问题一眼就能看出来（是不是误用了耳机麦或外接屏的麦）。
        try:
            dev = sd.query_devices(kind="input")
            print(f"  麦克风: {dev['name']}（{int(dev['default_samplerate'])}Hz）")
        except Exception as e:
            print(f"  麦克风: 读取失败 —— {e}")
    print("  加载完成")

    if args.ui == "socket":
        from socket_ui import SocketDisplay
        disp = SocketDisplay(port=args.ui_port, n=2 if tr else 1)
    else:
        from socket_ui import SocketDisplay   # 始终导入，供退出分支 isinstance 判断用
        disp = LiveDisplay(lines=2 if tr else 1)

    # 会话记录：默认开启。按回车随时存盘；--no-save 则完全关闭。
    log = None if args.no_save else SessionLog(
        True, src_lang, os.path.basename(md), args.save_dir, disp)

    print()
    print("=" * 56)
    if args.wav:
        print(f"  音源: {args.wav}")
    else:
        print("  开始说话吧（说完停顿一下会自动断句）")
    if log is not None:
        print("  随时按回车 = 保存当前转写记录")
    print("  按 Ctrl+C 退出")
    print("=" * 56)
    print()

    try:
        if args.wav:
            run_stream(rec, wav_blocks(args.wav), disp, tr=tr, realtime=args.realtime,
                       refiner=refiner, log=log, speaker=speaker)
            return

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(indata[:, 0].copy())

        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_SEC),
            callback=callback,
        ):
            run_stream(rec, mic_blocks(audio_q), disp, tr=tr, refiner=refiner,
                       log=log, speaker=speaker)
    except KeyboardInterrupt:
        disp.clear()
        print("\n已退出。")
        if log is not None:
            if isinstance(disp, SocketDisplay):
                # socket 模式：Swift 端点停止，这里直接落盘并回传路径，不交互询问
                path = log.save_now()
                if path:
                    disp.notify("saved", path)
            else:
                log.prompt_on_exit()
    except Exception as e:
        disp.clear()
        print(f"\n[出错] {type(e).__name__}: {e}")
        if "PaErrorCode" in str(e) or "device" in str(e).lower():
            print("提示：可能是麦克风权限没给。打开 系统设置 → 隐私与安全性 → 麦克风，勾选你用来运行的程序。")
        raise
    finally:
        # 通知 Swift 端停止（关闭窗口前给它一个收尾信号）
        if isinstance(disp, SocketDisplay):
            disp.notify("stopped")
            disp.close()
        # 后台翻译线程 + HY-MT 的 llama-server（我们的子进程）都要带走，
        # 否则会留僵尸进程占内存
        if tr is not None:
            tr.close()
        for w in keep_alive:
            w.stop()


if __name__ == "__main__":
    main()
