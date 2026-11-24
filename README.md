# snowcaptions · 本地实时字幕

完全**离线**的实时语音转写与中文字幕工具。基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)，
两遍架构：流式模型负责「快」，离线大模型负责「准」。可选英文→中文翻译与说话人区分。

- 🌐 纯本地，音频不出本机，无网络、无账号
- ⚡ 实时流式转写（中文 / 英文）
- 🔁 第二遍精修，大幅降低错字率
- 🈯 英文 → 中文字幕（HY-MT 大模型 + Argos 兜底）
- 👤 说话人识别（区分不同人，标「说话人 1/2」）
- 🖥️ 菜单栏原生 App（macOS 液态玻璃小窗）+ 终端两种界面

> 平台：macOS（菜单栏 App 为 Swift/AppKit）。Python 引擎本身跨平台，但菜单栏界面
> 仅在 macOS 上可用。

---

## 快速上手

### 1. 装依赖

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. 下载模型

模型不进 git（体积大、是第三方发布的权重）。跑一次下载脚本：

```bash
./download_models.sh            # 必需：流式转写 + 精修 + 声纹（约 3GB）
./download_models.sh --all      # 额外下翻译大模型 HY-MT（约 +1.1GB）
./download_models.sh --zh-only  # 只下中文（最省）
```

> 国内网络慢可换镜像：`SHERPA_MODELS_BASE=你的镜像前缀 ./download_models.sh`。

模型来源均为官方（[k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)，
翻译模型为腾讯 [HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF)）。

### 3. 跑起来

**终端版**（最简单，先验证能跑）：

```bash
./venv/bin/python live_zh.py                  # 自动挑模型，直接开始转写
./venv/bin/python live_zh.py --lang zh        # 强制中文
./venv/bin/python live_zh.py --lang en        # 强制英文（默认带中文字幕）
./venv/bin/python live_zh.py --list           # 列出本机所有可用模型
```

**菜单栏 App 版**（推荐日常用）：

```bash
cd SubtitleMenuBar && ./build.sh              # 编译 Swift App（纯 swiftc，无需 Xcode）
open build/SubtitleMenuBar.app                # 或双开「菜单栏字幕.command」
```

点菜单栏雪花图标 → 弹窗里点「开始」即可。设置面板可调：语种（中文转写 / 英文转写 /
英文转译成中文）、第二遍精修、热词表、说话人识别。点「停止」自动保存到 `transcripts/`。

> App 需要找到 Python 引擎目录。默认会探测 `~/Desktop/GitHub库/snowcaptions`、
> `~/Desktop/snowcaptions`、`~/snowcaptions`、`~/Downloads/snowcaptions`；放在别处时，
> 写一行路径到 `~/.subtitle-menubar-engine` 即可。

### 常用命令行参数

```bash
./venv/bin/python live_zh.py --wav a.wav               # 用音频文件当音源（调试）
./venv/bin/python live_zh.py --wav a.wav --realtime    # 按真实速度回放
./venv/bin/python live_zh.py --pause fast              # 断句松紧：fast / normal / loose
./venv/bin/python live_zh.py --no-refine               # 关掉第二遍精修
./venv/bin/python live_zh.py --no-save                 # 不记录转写内容
./venv/bin/python live_zh.py --speaker                 # 开启说话人识别
./venv/bin/python live_zh.py --hotwords "KPI,张明"     # 热词纠偏（专名/人名/术语）
./venv/bin/python live_zh.py --translate-engine argos  # 翻译退回轻量引擎
```

会话中随时**按回车**保存当前转写记录到 `transcripts/`；`Ctrl+C` 退出。

---

## 架构

```
live_zh.py       入口：解析参数、挑模型、装配部件、启动主循环
config.py        全局常量（断句阈值、路径、模型优先级）——调参先看这里
modelselect.py   模型发现 / 挑选 / 加载（transducer 与 ctc 两种）
audio.py         麦克风与 wav 音源 → 0.1 秒音频块；音频切片
engine.py        识别主循环：喂音频 → 解码 → 断句 → 定稿 → 调度显示/翻译/存盘
ui.py            终端显示层（实时区 + 滚动定稿区）
session.py       会话记录与存盘
textutil.py      纯文本工具（显示列宽、时间格式）
refine.py        第二遍精修（离线大模型重听一句话的音频）
translator.py    Argos / CTranslate2 翻译兜底（不联网）
mtengine.py      HY-MT1.5-1.8B + llama-server 翻译（主力，不联网）
speaker.py       说话人识别（声纹提取 + 聚类）
socket_ui.py     TCP 显示层（把字幕推给菜单栏 App）

SubtitleMenuBar/ 菜单栏 App（Swift/AppKit，纯 swiftc 编译，无第三方依赖）
bench/           对照实验脚本
tests/           单元回归测试
```

模型目录（均不入 git）：

```
models/          流式转写模型
models_refine/   精修模型（Qwen3-ASR）
models_mt/       翻译模型（HY-MT GGUF）
models_speaker/  声纹模型（CAM++）
transcripts/     转写记录输出
```

---

## 翻译引擎：HY-MT 主用，Argos 兜底

英文→中文翻译分两层，互相独立，一个挂了另一个照常用：

| 场景 | 引擎 | 延迟 | 说明 |
|---|---|---|---|
| 实时行（还在说话） | HY-MT 1.8B | ~280ms | 后台线程翻，主循环一秒不等 |
| 定稿行（一句话说完） | HY-MT 1.8B | ~280ms | 同步翻，保证完整 |
| 任一环节失败 / 没热好 | Argos | ~68ms | 兜底，不吐错误 |

HY-MT 需要 `llama-server`（llama.cpp 的二进制，macOS 可 `brew install llama.cpp`）和
`models_mt/` 下的 GGUF。缺失时自动回退 Argos（`ctranslate2` + Argos 语言包），仅在启动时
打印一行回退原因。要高质量中译，建议装齐：

```bash
brew install llama.cpp
./download_models.sh --all
```

---

## 开发

改断句/性能参数前先读 `config.py` 里每个常量旁边的注释——那里记录了「为什么是这个数」的
实测结论，别重踩一遍坑。改完跑回归：

```bash
./venv/bin/python tests/test_units.py     # 单元测试
./venv/bin/python live_zh.py --list       # 确认模型就位
```

性能基线：`./venv/bin/python bench/baseline.py`（测 RTF、定稿延迟、启动耗时、峰值内存）。

## 许可证

代码以 [MIT](./LICENSE) 开源。模型权重归属各自发布方（多为 Apache-2.0），
使用时请遵守对应模型的许可证。
