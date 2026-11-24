#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二遍翻译引擎：HY-MT1.5-1.8B（GGUF）+ llama-server，全程离线。

## 为什么单独一个模块

`translator.py` 一 import 就要求 ctranslate2 装好；而这个模块**只用标准库**
（subprocess + urllib），模型或 llama-server 缺失时只是「静默退化」，
不会拖累整个程序启动。两个引擎的依赖互不牵连，谁坏了都不影响另一个。

## 分工

整个翻译都归它：实时行（后台线程翻，不挡采音）和定稿行（同步翻，保证完整）。
Argos 只剩兜底角色——它没热好、超时、报错的时候顶上，见 engine.LiveTranslator。

## 设计约定（与 refine.py 保持一致）

- **后台预热**：启动时开线程拉服务，不挡主进程；没热好之前 translate() 返回 None，
  调用方自动回退 Argos，用户感知不到
- **任何失败都返回 None**：没模型 / 找不到 llama-server / 端口全占 / 超时 / 返回乱码，
  一律交给调用方沿用第一遍结果，绝不吐错误给用户
- **可以被整体关掉**（available=False），行为与改造前完全一致
- 服务是本进程的子进程，退出时一定会被收拾干净（atexit + stop() 幂等）
"""
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# llama-server 的常见落点（PATH 里找不到时按顺序试）
_BIN_CANDIDATES = [
    "llama-server",
    "/opt/homebrew/bin/llama-server",
    "/usr/local/bin/llama-server",
]


def find_binary():
    for c in _BIN_CANDIDATES:
        p = shutil.which(c) if os.sep not in c else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def find_gguf(model_dir, prefer=""):
    """在 models_mt/ 里挑一个 gguf。prefer 可以是文件名、完整路径或空。"""
    if prefer:
        if os.path.isfile(prefer):
            return prefer
        p = os.path.join(model_dir, prefer)
        if os.path.isfile(p):
            return p
        return None
    if not os.path.isdir(model_dir):
        return None
    cands = [os.path.join(model_dir, n) for n in sorted(os.listdir(model_dir))
             if n.lower().endswith(".gguf")]
    if not cands:
        return None
    # 优先 Q4_K_M（体积/质量最平衡），其次任意
    cands.sort(key=lambda p: (0 if "q4_k_m" in os.path.basename(p).lower() else 1,
                              os.path.getsize(p)))
    return cands[0]


def _port_free(port):
    s = socket.socket()
    s.settimeout(0.2)
    try:
        return s.connect_ex(("127.0.0.1", port)) != 0
    finally:
        s.close()


def load_terms(path):
    """读术语表：每行「原文=>译文」或「原文 <TAB> 译文」，# 开头为注释。"""
    if not path or not os.path.exists(path):
        return ""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=>" in line:
                    src, dst = line.split("=>", 1)
                elif "\t" in line:
                    src, dst = line.split("\t", 1)
                else:
                    continue
                src, dst = src.strip(), dst.strip()
                if src and dst:
                    rows.append(f"{src} 翻译成 {dst}")
    except Exception:
        return ""
    return "\n".join(rows)


class MTTranslator:
    """英译中的「慢但准」引擎。接口与 translator.Translator 一致：translate(text) -> str。

    失败时返回 None（**不是空串**），这样调用方能区分「翻出来是空的」和「没翻成」。
    """

    def __init__(self, gguf="", model_dir="", port=8097, threads=4, timeout=3.0,
                 ctx=2048, terms="", prompt="{src}", terms_prompt=None,
                 sampling=None, debug=False):
        self.debug = debug
        self.model_dir = model_dir
        self.want_port = int(port or 8097)
        self.port = self.want_port
        self.threads = int(threads or 4)
        self.timeout = float(timeout or 3.0)
        self.ctx = int(ctx or 2048)
        self.prompt = prompt
        self.terms_prompt = terms_prompt or prompt
        self.sampling = dict(sampling or {})
        self.terms_block = load_terms(terms)

        self.bin = find_binary()
        self.gguf = find_gguf(model_dir, gguf) if self.bin else None
        self._proc = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._warned = False

        if not self.bin:
            self.why = "没找到 llama-server（brew install llama.cpp）"
        elif not self.gguf:
            self.why = (f"找不到指定的翻译模型：{gguf}" if gguf
                        else f"{model_dir} 下没有 .gguf 模型")
        else:
            self.why = ""
        self.available = not self.why
        atexit.register(self.stop)

    # ---------------- 服务生命周期 ----------------

    def start_async(self):
        """后台拉服务。已启动/不可用则什么都不做。"""
        if not self.available or self._proc is not None:
            return
        threading.Thread(target=self._boot, name="mt-boot", daemon=True).start()

    def _boot(self):
        with self._lock:
            if self._proc is not None:
                return
            port = self._pick_port()
            if port is None:
                self.why = f"{self.want_port}~{self.want_port + 5} 端口都被占用"
                self.available = False
                return
            cmd = [self.bin, "-m", self.gguf, "-c", str(self.ctx),
                   "-t", str(self.threads), "--host", "127.0.0.1", "--port", str(port)]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
            except Exception as e:
                self.why = f"llama-server 启动失败：{type(e).__name__}: {e}"
                self.available = False
                self._proc = None
                return
            self.port = port
        # 等就绪（1.8B Q4 在 M 系列上约 1~2 秒）
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._proc.poll() is not None:      # 进程自己退了
                self.why = "llama-server 启动后立即退出"
                self.available = False
                self._proc = None
                return
            if self._healthy():
                self._ready.set()
                if self.debug:
                    print(f"[MT] 就绪：端口 {self.port}", file=sys.stderr)
                return
            time.sleep(0.15)
        self.why = "等待 llama-server 就绪超时（30s）"
        self.available = False
        self.stop()

    def _pick_port(self):
        for p in range(self.want_port, self.want_port + 6):
            if _port_free(p):
                return p
        return None

    def _healthy(self):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1.0) as r:
                return r.status == 200
        except Exception:
            return False

    @property
    def ready(self):
        return self._ready.is_set() and self._proc is not None and self._proc.poll() is None

    def stop(self):
        """停服务。幂等，可以在 finally / atexit 里随便调。"""
        p, self._proc = self._proc, None
        self._ready.clear()
        if p is None:
            return
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    # ---------------- 翻译 ----------------

    def _build_prompt(self, text):
        # 用 replace 而不是 format：原文里可能出现 {}，format 会直接抛异常
        if self.terms_block:
            return (self.terms_prompt
                    .replace("{terms}", self.terms_block)
                    .replace("{src}", text))
        return self.prompt.replace("{src}", text)

    def translate(self, text):
        text = (text or "").strip()
        if not text or not self.available:
            return None
        if not self.ready:
            return None                  # 还没热好 → 让调用方回退 Argos
        payload = {
            "messages": [{"role": "user", "content": self._build_prompt(text)}],
            "max_tokens": max(64, min(512, len(text) * 4)),
            "stream": False,
        }
        payload.update(self.sampling)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
            out = (d["choices"][0]["message"]["content"] or "").strip()
            return _clean(out)
        except Exception as e:
            if self.debug and not self._warned:
                self._warned = True
                print(f"[MT] 翻译失败，本轮起回退 Argos：{type(e).__name__}: {e}",
                      file=sys.stderr)
            return None

    def describe(self):
        if not self.available:
            return f"未启用（{self.why}）"
        state = "已就绪" if self.ready else "预热中"
        return f"{os.path.basename(self.gguf)} [{state} pid={getattr(self._proc, 'pid', '-')}]"


def _clean(out):
    """清掉偶发的引号包裹和「译文：」前缀。"""
    out = out.strip()
    for pre in ("译文：", "译文:", "翻译：", "翻译:"):
        if out.startswith(pre):
            out = out[len(pre):].strip()
    if len(out) > 1 and out[0] in "「“\"'《" and out[-1] in "」”\"'》":
        out = out[1:-1].strip()
    return out


if __name__ == "__main__":
    # 自测：起服务 → 翻几句 → 关掉
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    t0 = time.time()
    m = MTTranslator(gguf=config.MT_GGUF, model_dir=config.MT_DIR, port=config.MT_PORT,
                     threads=config.MT_THREADS, timeout=config.MT_TIMEOUT,
                     ctx=config.MT_CTX, terms=config.MT_TERMS,
                     prompt=config.MT_PROMPT, terms_prompt=config.MT_TERMS_PROMPT,
                     sampling=config.MT_SAMPLING, debug=True)
    print("引擎:", m.describe())
    if not m.available:
        sys.exit(1)
    m.start_async()
    while not m.ready and time.time() - t0 < 40:
        time.sleep(0.2)
    print(f"就绪耗时 {time.time() - t0:.2f}s\n")
    for s in (sys.argv[1:] or [
        "The committee kicked the can down the road again.",
        "Revenue growth was flat year over year, largely due to FX headwinds.",
        "Sovereignty, in the Westphalian sense, was never as absolute as the textbooks suggest.",
    ]):
        t0 = time.time()
        out = m.translate(s)
        print(f"[{(time.time()-t0)*1000:.0f}ms] {s}\n        -> {out}\n")
    m.stop()
    print("服务已关闭")
