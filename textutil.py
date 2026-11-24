#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯文本工具：终端列宽、字符宽度、时间格式。

都是无副作用的纯函数，最适合写单测。
"""
import shutil
import unicodedata

def term_cols():
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def dlen(s):
    """字符串占几个显示列（中日韩全角字符算 2 列）"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def fit_tail(text, cols):
    """只保留末尾能塞进 cols 列的部分，前面用省略号"""
    budget = max(cols - 2, 8)
    out, used = [], 0
    for ch in reversed(text):
        w = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + w > budget:
            break
        out.append(ch)
        used += w
    out.reverse()
    tail = "".join(out)
    return ("…" + tail) if len(tail) < len(text) else text

def _fmt_secs(secs):
    secs = int(secs)
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"


def _fmt_stamp(secs):
    secs = int(secs)
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"
