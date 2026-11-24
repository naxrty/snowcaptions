#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Socket 显示层：把实时字幕/定稿推给本机 Swift 菜单栏 App。

和 ui.LiveDisplay 实现同一套接口（update / commit / clear），只是不写终端，
改成通过本地 TCP socket 把内容发出去。Swift 端负责监听并渲染到液态玻璃小窗。

协议（JSON 行，每行一条消息，以 \\n 结尾）：
    {"type":"update","rows":["原文","译文"]}  实时区刷新
    {"type":"commit","rows":["原文","译文"]}  一句话定稿
    {"type":"clear"}                          清空实时区
    {"type":"status","state":"recording|saved","path":"..."}  状态通知

为什么用 TCP 而不是 Unix socket：macOS 上 Python 和 Swift 都原生支持，
端口固定（默认 7711），实现最简单、排障最直观。Swift 端是服务端，
Python 启动时去连；连不上就静默降级为「不输出」，绝不因此拖垮识别。
"""
import json
import socket
import threading
import time


class SocketDisplay:
    """把字幕推给本机 Swift App。连接失败时静默降级（相当于空显示）。"""

    def __init__(self, host="127.0.0.1", port=7711, n=2):
        self.n = n
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()
        self._connected = False
        self._stop = False
        # 后台线程持续尝试连接（Swift App 可能晚于 Python 启动）
        self._th = threading.Thread(target=self._connect_loop, name="socket-ui",
                                    daemon=True)
        self._th.start()

    # ---------------- 连接管理 ----------------

    def _connect_loop(self):
        while not self._stop:
            if not self._connected:
                self._try_connect()
            time.sleep(1.0)

    def _try_connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            s.settimeout(None)
            with self._lock:
                self._sock = s
                self._connected = True
            self._send({"type": "hello"})
            return True
        except Exception:
            return False

    def _send(self, obj):
        """发一条 JSON。任何失败都静默处理，并标记断线等待重连。"""
        with self._lock:
            s = self._sock
        if s is None:
            return
        try:
            data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            s.sendall(data)
        except Exception:
            with self._lock:
                self._connected = False
                self._sock = None
            try:
                s.close()
            except Exception:
                pass

    def _reset_connection(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
            self._sock = None
            self._connected = False

    # ---------------- 与 LiveDisplay 对齐的接口 ----------------

    def update(self, rows):
        """刷新实时区。rows 为 [原文] 或 [原文, 译文]。"""
        self._send({"type": "update", "rows": list(rows)})

    def commit(self, rows, speaker_id=None):
        """一句话定稿。把整句（含译文）交给 Swift 端落到滚动区。"""
        clean = [r.strip() for r in rows if r and r.strip()]
        if not clean:
            return
        msg = {"type": "commit", "rows": clean}
        if speaker_id:
            msg["speaker"] = speaker_id
        self._send(msg)

    def clear(self):
        """清空实时区。"""
        self._send({"type": "clear"})

    def notify(self, state, path=None):
        """发状态通知（如保存完成、开始录音）。"""
        msg = {"type": "status", "state": state}
        if path:
            msg["path"] = path
        self._send(msg)

    # session.py 会安全探测这两个方法，提供出来避免 AttributeError
    def _erase(self):
        pass

    def flush(self):
        pass

    def close(self):
        self._stop = True
        self._reset_connection()
