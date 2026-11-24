#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终端显示层：固定高度的实时区 + 向上滚动的定稿区。"""
import sys

from textutil import dlen, fit_tail, term_cols

LABEL_EN = "EN  "      # 4 列宽
LABEL_ZH = "中  "      # 中(2列) + 2 空格 = 4 列宽，跟上面刚好对齐


class LiveDisplay:
    """固定高度的实时区（1 行或 2 行）+ 向上滚动的定稿区。

    输出被重定向到文件（非 tty）时，只输出定稿内容，不做光标控制。

    关于换行：这里一律用 "\\r\\n" 而不是 "\\n"。原因是在真实终端里 "\\n" 能回行首，
    靠的是终端驱动的 ONLCR 后处理；一旦这个开关被关掉（或者换到别的终端模拟器），
    只写 "\\n" 就会导致光标停在上一次的列位上——屏幕上表现为每一行都比上一行多缩进
    几个空格，越滚越歪。显式写 "\\r" 就没这个隐患。
    """

    def __init__(self, lines=1, stream=sys.stdout):
        self.out = stream
        self.n = lines
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self.drawn = 0        # 屏幕上当前实时区实际占了几行

    def _erase(self):
        """擦掉实时区，并把光标留在实时区第一行的行首"""
        if not self.tty or self.drawn == 0:
            return
        self.out.write("\r\033[2K")
        for _ in range(self.drawn - 1):
            self.out.write("\033[1A\r\033[2K")
        self.drawn = 0

    def update(self, rows):
        """重画实时区。rows：每一行的文本（长度应为 self.n）"""
        if not self.tty:
            return
        cols = term_cols()
        self._erase()
        for i, row in enumerate(rows):
            line = fit_tail(row, cols - 1) if row else ""
            self.out.write(line)
            if i < len(rows) - 1:
                self.out.write("\r\n")
        self.out.flush()
        self.drawn = len(rows)

    def clear(self):
        self._erase()
        if self.tty:
            self.out.flush()

    def commit(self, rows, speaker_id=None):
        """一句话定稿：擦掉实时区，把整句（含译文）落到上方滚动区"""
        self._erase()
        spk = ""
        if speaker_id:
            n = "".join(ch for ch in str(speaker_id) if ch.isdigit()) or str(speaker_id)
            spk = f"👤{n} "
        for i, row in enumerate(rows):
            if not row or not row.strip():
                continue
            tag = (LABEL_EN if i == 0 else LABEL_ZH) if len(rows) > 1 else ""
            # 说话人标签只加在原文行（第一行）上
            prefix = spk if i == 0 else ""
            self.out.write(tag + prefix + row.strip() + "\r\n")
        self.out.flush()
