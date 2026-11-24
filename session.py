#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""会话记录：攒定稿句子，按回车随时存盘。"""
import datetime
import os

from textutil import _fmt_secs, _fmt_stamp


def _spk_num(speaker_id):
    """把 'spk1'/'spk2' 转成 '1'/'2'；拿不到数字就原样返回。"""
    s = str(speaker_id)
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or s


class SessionLog:
    """把定稿内容攒起来，按回车随时存盘。

    设计（针对「直接叉窗口」的使用习惯）：
      * 每句定稿立即追加到内存，不落盘、不卡识别。
      * 用户随时按一下回车 → 当场写文件（同一份文件，按一次覆盖更新一次）。
      * 一直不按、直接叉窗口 → 什么都不留（叉窗口是强杀，没机会抢救）。
      * Ctrl+C 退出时若有内容且还没存过，会问一次：回车存 / 其他键不存。
    """

    def __init__(self, enabled, src_lang, model_name, save_dir, disp):
        self.enabled = enabled
        self.src_lang = src_lang
        self.model_name = model_name
        self.save_dir = save_dir
        self.disp = disp
        self.rows = []              # [(秒数, 原文, 译文或 None)]
        self.started = datetime.datetime.now()
        self.saved = False          # 至少成功存过一次
        self._final_path = None
        self._final_name = None

    @property
    def active(self):
        return self.enabled

    def add(self, secs, text, translated, speaker_id=None):
        """定稿一句就追加一条。text 已 strip；translated 可为 None；speaker_id 可选。"""
        if not self.enabled or not text:
            return
        self.rows.append((secs, text, translated, speaker_id))

    def _render(self):
        hdr = self.started.strftime("%Y-%m-%d %H:%M")
        lang_label = "中文转写" if self.src_lang == "zh" else "英文转写 + 中文翻译"
        total = max((r[0] for r in self.rows), default=0)
        lines = [
            f"# {hdr} {lang_label}",
            f"模型: {self.model_name}",
            f"时长 {_fmt_secs(total)} · 共 {len(self.rows)} 句",
            "",
        ]
        for secs, text, translated, speaker_id in self.rows:
            spk = f"[说话人 {_spk_num(speaker_id)}] " if speaker_id else ""
            lines.append(f"- [{_fmt_stamp(secs)}] {spk}{text}")
            if translated:
                lines.append(f"      {translated}")
        return "\n".join(lines) + "\n"

    def _target(self):
        """正式文件名：日期_时间_语种.md。同一分钟内多次保存会覆盖同一份。"""
        if self._final_name is None:
            self._final_name = self.started.strftime("%Y-%m-%d_%H%M") + \
                ("_中文" if self.src_lang == "zh" else "_英译中") + ".md"
            self._final_path = os.path.join(self.save_dir, self._final_name)
        return self._final_path

    def save_now(self):
        """当场写盘。返回保存路径，失败返回 None。"""
        if not self.enabled or not self.rows:
            return None
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            path = self._target()
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._render())
            self.saved = True
            return path
        except Exception as e:
            self.disp._erase()
            print(f"[保存失败] {type(e).__name__}: {e}")
            self.disp.flush if hasattr(self.disp, "flush") else None
            return None

    def show_saved(self, path, count):
        """在定稿区下打印一行保存提示（不影响实时区）。"""
        self.disp._erase()
        print(f"  已保存 {count} 句 → {path}")
        self.disp.flush if hasattr(self.disp, "flush") else None

    def prompt_on_exit(self):
        """Ctrl+C 退出时：有内容且从没存过才问。回车存、其他不存。返回是否已保存。"""
        if not self.enabled or not self.rows:
            return False
        if self.saved:
            return True
        try:
            ans = input("  本次共 %d 句，回车保存 / 其他键不保存：" % len(self.rows))
        except (EOFError, KeyboardInterrupt):
            return False
        if ans.strip() == "":
            path = self.save_now()
            if path:
                print(f"  已保存 → {path}")
                return True
        return False
