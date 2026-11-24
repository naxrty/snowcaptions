#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能基线：把「有多快」变成能比较的数字。

为什么需要它：改异步精修、改队列之前，得先知道现在到底多慢。
没有这个数字，改完只能说"感觉好点"，说不清是不是真变了。

测四件事：
  1. RTF          解码耗时 ÷ 音频时长（<1 才追得上实时，越接近 1 越没余量）
  2. 定稿延迟     最后一块有声音频被喂进去 → 整句真正定稿的墙钟时间
  3. 启动耗时     流式模型 / 精修模型各自加载多久
  4. 峰值内存     进程 RSS 峰值

## 关于「定稿延迟」的一个坑（2026-11-24 实测）

模型自带的 test_wavs 说完话只剩 0.3 秒静音，而端点检测要求 1.9 秒静音
（rule1），所以那些音频**从来没有触发过端点**——它们只在音频结束时被
「收尾冲缓冲」定稿。拿它们测出来的延迟（0.6 秒左右）是"尾部静音 + 冲缓冲"
的时间，不是用户真正感受到的等待。

所以本脚本会把每句定稿分成两类，并分开统计：
  * 端点定稿（句中 > 1.9s 静音触发）  → 这才是真实的「说完话到看见整句」
  * 收尾定稿（音频播完才吐出来）      → 只出现在文件结尾，单独列出、不计入均值

另外默认会用 --wav 拼一段「0.wav + 2.5s 静音 + 1.wav + 2.5s 静音」的合成长
音频，专门用来触发两次真正的端点定稿。

## 两个延迟指标：音频域 vs 墙钟

  * **音频域**：从说话结束到定稿，还需要多少秒**音频**。这个数与回放快慢、
    机器快慢无关，纯看逻辑（rule1 + COMMIT_DELAY + 精修是否阻塞）。
    改代码时看这个——它稳定、可复现（实测 2.2s 量级）。
  * **墙钟**：用户真实体感秒数。它会被实时回放的节拍误差放大：
    run_stream() 里是 `time.sleep(BLOCK_SEC)`，没扣掉解码耗时，所以
    --wav --realtime 回放会慢 10% 左右（实测 RTF≈0.11 → 慢 11%），
    墙钟数字相应偏大（约 2.9s）。麦克风模式下没有这个问题，
    但队列一旦堆积，墙钟就会再涨。

两个都记，别拿错那个当成绩单。

## 会记录当时用的参数

BLOCK_SEC / THREADS / RULE1 / RULE2 / COMMIT_DELAY / 模型目录 / 精修模型，
不记这些的话，以后一调参基线就不可比了。

用法：
    ./venv/bin/python bench/baseline.py                  # 跑一次，存到 bench/baseline/
    ./venv/bin/python bench/baseline.py --quick          # 只测 RTF，跳过实时回放
    ./venv/bin/python bench/baseline.py --compare bench/baseline/latest.json
"""
import argparse
import datetime
import json
import os
import platform
import resource
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import audio as A          # noqa: E402
import config as C         # noqa: E402
import engine as E         # noqa: E402
import modelselect as M    # noqa: E402

OUT_DIR = os.environ.get("BASELINE_OUT") or os.path.join(ROOT, "bench", "baseline")

# 判断「这块音频里有说话」的能量阈值（float32 满量程 1.0）。
# 测试音频是干净录音，0.01 足够把静音和说话分开；用真人麦克风录音时可以再调。
SILENCE_RMS = 0.01

# 合成长音频里每句话后面垫多长静音（必须 > rule1，否则端点不触发）
PAD_SILENCE = 2.5

# 判定「这句是句中端点定稿」而不是「结尾收尾定稿」的余量（秒）
ENDPOINT_MARGIN = 0.05


def peak_rss_mb():
    """进程 RSS 峰值（MB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KB。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024)


def audio_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def measure_rtf(rec, path):
    """不做任何等待，尽快把整段音频解码完，测「解码耗时 ÷ 音频时长」。

    这是硬指标：RTF ≥ 1 时，无论上层代码怎么写，实时都追不上。
    """
    stream = rec.create_stream()
    t0 = time.perf_counter()
    for blk in A.wav_blocks(path):
        stream.accept_waveform(C.SAMPLE_RATE, blk)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    # 收尾：补静音把前瞻缓冲冲出来，这部分耗时要算进去
    stream.accept_waveform(C.SAMPLE_RATE,
                           np.zeros(int(C.SAMPLE_RATE * C.FLUSH_SILENCE), dtype=np.float32))
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    elapsed = time.perf_counter() - t0
    dur = audio_duration(path)
    return {
        "audio_s": round(dur, 3),
        "decode_s": round(elapsed, 3),
        "rtf": round(elapsed / dur, 4),
        "realtime_margin": round(dur / elapsed, 2) if elapsed else None,
        "text": rec.get_result(stream).strip(),
    }


class Recorder:
    """假的显示层：不画屏，只记录「这句什么时候定稿、定稿时音频喂到哪了」。"""

    n = 1

    def __init__(self, probe):
        self.probe = probe
        self.rows = []

    def update(self, rows):
        pass

    def clear(self):
        pass

    def _erase(self):
        pass

    def commit(self, rows):
        now = time.perf_counter()
        last_loud = self.probe.get("last_loud")
        last_loud_pos = self.probe.get("last_loud_pos")
        pos = self.probe.get("pos", 0.0)
        self.rows.append({
            "text": rows[0] if rows else "",
            "audio_pos": round(pos, 3),
            # 墙钟延迟：用户体感（含回放/队列节拍误差）
            "latency": None if last_loud is None else round(now - last_loud, 3),
            # 音频域延迟：还需要多少秒音频才定稿（与节拍无关，改逻辑时看这个）
            "audio_latency": None if last_loud_pos is None else round(pos - last_loud_pos, 3),
        })


def wav_blocks_paced(path, probe):
    """产出音频块，顺手记录累计音频位置和「最后一块有声」的时刻。"""
    for blk in A.wav_blocks(path):
        rms = float(np.sqrt(np.mean(blk.astype(np.float64) ** 2)))
        probe["pos"] += len(blk) / C.SAMPLE_RATE
        if rms >= SILENCE_RMS:
            probe["last_loud"] = time.perf_counter()
            probe["last_loud_pos"] = probe["pos"]
        yield blk


def silence_blocks(seconds, probe):
    """产出静音块（按 0.1 秒一块，保证实时节奏正确）。"""
    n = int(seconds / C.BLOCK_SEC)
    for _ in range(n):
        probe["pos"] += C.BLOCK_SEC
        yield np.zeros(int(C.SAMPLE_RATE * C.BLOCK_SEC), dtype=np.float32)


def concat_blocks(paths, pad_secs, probe):
    """把多段 wav 用长静音拼起来：让每句话后面都有足够静音去触发端点。

    这是测「真实定稿延迟」的关键：单文件 test_wavs 尾部静音不够长，
    端点根本不触发，只能测到收尾定稿。
    """
    for p in paths:
        for blk in wav_blocks_paced(p, probe):
            yield blk
        for blk in silence_blocks(pad_secs, probe):
            yield blk


def measure_latency(rec, make_blocks, total_dur, refiner=None):
    """实时回放一遍完整链路，按「端点定稿 / 收尾定稿」分类统计延迟。

    make_blocks(probe) 必须用**同一个** probe，否则打点对不上。
    """
    probe = {"pos": 0.0, "last_loud": None, "last_loud_pos": None}
    disp = Recorder(probe)
    t0 = time.perf_counter()
    E.run_stream(rec, make_blocks(probe), disp, tr=None, realtime=True,
                 refiner=refiner, log=None)
    wall = time.perf_counter() - t0

    endpoint, flush = [], []
    for r in disp.rows:
        if r["latency"] is None:
            continue
        # 定稿发生在音频还没播完时 → 是端点触发；否则是结尾冲缓冲
        if r["audio_pos"] < total_dur - ENDPOINT_MARGIN:
            endpoint.append(r)
        else:
            flush.append(r)

    def stat(xs, fld):
        vals = [r[fld] for r in xs if r.get(fld) is not None]
        if not vals:
            return None, None, 0
        return round(sum(vals) / len(vals), 3), round(max(vals), 3), len(vals)

    m, mx, n = stat(endpoint, "latency")
    am, amx, _ = stat(endpoint, "audio_latency")
    return {
        "audio_s": round(total_dur, 3),
        "wall_s": round(wall, 3),
        "sentences": len(disp.rows),
        "endpoint": {"count": n,
                     "latency_mean_s": m, "latency_max_s": mx,
                     "audio_mean_s": am, "audio_max_s": amx,
                     "samples": [r["latency"] for r in endpoint],
                     "audio_samples": [r["audio_latency"] for r in endpoint]},
        "flush": {"count": len(flush),
                  "latency_s": [r["latency"] for r in flush]},
        "texts": [r["text"] for r in disp.rows],
    }


def main():
    ap = argparse.ArgumentParser(description="性能基线")
    ap.add_argument("--quick", action="store_true", help="只测 RTF，跳过实时回放")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="用哪个语言的流式模型")
    ap.add_argument("--files", nargs="*", help="额外要测的 wav（默认用模型自带 test_wavs）")
    ap.add_argument("--pad", type=float, default=PAD_SILENCE,
                    help=f"合成长音频里每句话后垫的静音秒数（默认 {PAD_SILENCE}，必须 > rule1）")
    ap.add_argument("--compare", help="跟一份已有基线 json 对比")
    args = ap.parse_args()

    md = M.pick_model_dir(args.lang)
    tv = os.path.join(md, "test_wavs")
    wavs = args.files or [os.path.join(tv, "0.wav"),
                          os.path.join(tv, "1.wav") if os.path.exists(os.path.join(tv, "1.wav")) else None]
    wavs = [os.path.join(ROOT, w) if not os.path.isabs(w) else w for w in wavs if w]

    result = {
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "params": {
            "model": os.path.basename(md),
            "block_sec": C.BLOCK_SEC,
            "threads": C.THREADS,
            "rule1_silence": C.RULE1_SILENCE,
            "rule2_silence": C.RULE2_SILENCE,
            "commit_delay": C.COMMIT_DELAY,
            "warmup_silence": C.WARMUP_SILENCE,
            "flush_silence": C.FLUSH_SILENCE,
            "min_commit_chars": C.MIN_COMMIT_CHARS,
            "pad_silence": args.pad,
        },
        "files": {},
    }

    print(f"模型：{os.path.basename(md)}")
    t0 = time.perf_counter()
    rec = M.build_recognizer(md)
    result["params"]["startup_stream_s"] = round(time.perf_counter() - t0, 2)
    print(f"  流式模型加载 {result['params']['startup_stream_s']}s"
          f" · 峰值内存 {peak_rss_mb()}MB\n")

    refiner = None
    try:
        from refine import Refiner
        r = Refiner(C.REFINE_DIRS, threads=C.THREADS)
        if r.enabled:
            t0 = time.perf_counter()
            r.load()
            refiner = r
            result["params"]["refiner"] = r.describe()
            result["params"]["startup_refiner_s"] = round(time.perf_counter() - t0, 2)
            print(f"  精修模型加载 {result['params']['startup_refiner_s']}s"
                  f" · 峰值内存 {peak_rss_mb()}MB\n")
    except Exception as e:
        result["params"]["refiner"] = f"启用失败：{type(e).__name__}: {e}"

    # ---- 1. 单文件：测 RTF（快）----
    for path in wavs:
        key = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)}"
        print(f"== {os.path.basename(path)} ==")
        item = {"rtf_stream": measure_rtf(rec, path)}
        print(f"  RTF（纯解码）: {item['rtf_stream']['rtf']:.3f}"
              f"  （实时余量 {item['rtf_stream']['realtime_margin']}×）")
        result["files"][key] = item
        print()

    # ---- 2. 合成长音频：测真实端点定稿延迟（必须实时回放）----
    if not args.quick:
        total = sum(audio_duration(p) for p in wavs) + args.pad * len(wavs)
        key = "endpoint-test"
        print(f"== 端点定稿测试（{len(wavs)} 句 + 每句垫 {args.pad}s 静音，"
              f"共 {total:.1f}s，实时回放）==")
        item = result["files"].setdefault(key, {})
        item["latency_stream"] = measure_latency(
            rec, lambda pr: concat_blocks(wavs, args.pad, pr), total, refiner=None)
        e = item["latency_stream"]["endpoint"]
        if e["count"]:
            print(f"  仅流式 : 端点定稿 {e['count']} 句，"
                  f"音频域 {e['audio_mean_s']}s / 墙钟 {e['latency_mean_s']}s"
                  f"（最大 {e['latency_max_s']}s）")
            m_stream = e["audio_mean_s"]
        else:
            print("  ⚠️ 没有检测到端点定稿——pad 静音可能不够长（必须 > rule1）")
            m_stream = None

        if refiner is not None:
            item["latency_refine"] = measure_latency(
                rec, lambda pr: concat_blocks(wavs, args.pad, pr), total, refiner=refiner)
            e2 = item["latency_refine"]["endpoint"]
            if e2["count"]:
                print(f"  +精修   : 端点定稿 {e2['count']} 句，"
                      f"音频域 {e2['audio_mean_s']}s / 墙钟 {e2['latency_mean_s']}s"
                      f"（最大 {e2['latency_max_s']}s）")
                if m_stream:
                    item["refine_cost_s"] = round(e2["audio_mean_s"] - m_stream, 3)
                    item["refine_cost_wall_s"] = round(
                        e2["latency_mean_s"] - e["latency_mean_s"], 3)
                    print(f"  精修额外等待: 墙钟 {item['refine_cost_wall_s']:+.3f}s"
                          f"（音频域不变：精修不吃音频，只占墙钟）")
                    print(f"    → 异步化后这 {item['refine_cost_wall_s']:.2f}s"
                          f" 可以从「说完话到看见整句」里去掉")
        print()

    result["host"]["peak_rss_mb"] = peak_rss_mb()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(OUT_DIR, f"baseline-{stamp}.json")
    for p in (path, os.path.join(OUT_DIR, "latest.json")):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已保存基线 → {os.path.relpath(path, ROOT)}")
    print(f"  峰值内存 {result['host']['peak_rss_mb']}MB"
          f" · 流式加载 {result['params']['startup_stream_s']}s"
          f" · 精修加载 {result['params'].get('startup_refiner_s', '-')}s")

    if args.compare:
        compare(args.compare, path)


def compare(old_path, new_path):
    """打印两份基线的对比表（同口径才对得上）。"""
    old = json.load(open(old_path, encoding="utf-8"))
    new = json.load(open(new_path, encoding="utf-8"))
    key = "endpoint-test"

    def get(d, *keys):
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    def rtf_avg(doc):
        vals = [f["rtf_stream"]["rtf"] for f in doc.get("files", {}).values() if "rtf_stream" in f]
        return round(sum(vals) / len(vals), 4) if vals else None

    rows = [
        ("RTF（平均）", rtf_avg(old), rtf_avg(new)),
        ("端点定稿 音频域 仅流式",
         get(old, "files", key, "latency_stream", "endpoint", "audio_mean_s"),
         get(new, "files", key, "latency_stream", "endpoint", "audio_mean_s")),
        ("端点定稿 音频域 +精修",
         get(old, "files", key, "latency_refine", "endpoint", "audio_mean_s"),
         get(new, "files", key, "latency_refine", "endpoint", "audio_mean_s")),
        ("端点定稿 墙钟 仅流式",
         get(old, "files", key, "latency_stream", "endpoint", "latency_mean_s"),
         get(new, "files", key, "latency_stream", "endpoint", "latency_mean_s")),
        ("端点定稿 墙钟 +精修",
         get(old, "files", key, "latency_refine", "endpoint", "latency_mean_s"),
         get(new, "files", key, "latency_refine", "endpoint", "latency_mean_s")),
        ("精修额外等待（音频域）",
         get(old, "files", key, "refine_cost_s"),
         get(new, "files", key, "refine_cost_s")),
        ("峰值内存 MB", get(old, "host", "peak_rss_mb"), get(new, "host", "peak_rss_mb")),
        ("流式加载 s", get(old, "params", "startup_stream_s"),
         get(new, "params", "startup_stream_s")),
    ]
    print(f"\n对比：{os.path.basename(old_path)} → {os.path.basename(new_path)}")
    print(f"{'指标':<28}{'旧':>10}{'新':>10}{'变化':>12}")
    print("-" * 62)
    for label, o, n in rows:
        if o is None and n is None:
            continue
        if isinstance(o, (int, float)) and isinstance(n, (int, float)) and o:
            print(f"{label:<28}{o:>10.3f}{n:>10.3f}{(n - o) / abs(o) * 100:>+11.1f}%")
        else:
            print(f"{label:<28}{str(o):>10}{str(n):>10}{'-':>12}")


if __name__ == "__main__":
    main()
