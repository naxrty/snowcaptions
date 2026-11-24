# 第三方许可声明

本项目代码以 [MIT](./LICENSE) 开源。项目**依赖**以下第三方软件与模型，
其版权归各自作者所有，使用时请遵守各自的许可证。

## 运行时依赖（Python 包）

| 包 | 用途 | 许可证 |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 语音识别推理引擎 | Apache-2.0 |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | 麦克风音频采集 | MIT |
| [numpy](https://numpy.org) | 数值计算 | BSD-3-Clause |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | 翻译兜底引擎底层 | MIT |
| [sentencepiece](https://github.com/google/sentencepiece) | 翻译分词 | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)（llama-server 二进制） | 翻译大模型推理 | MIT |

## 模型权重（由 `download_models.sh` 从官方源下载，不随仓库分发）

| 模型 | 用途 | 来源 | 许可证 |
|---|---|---|---|
| sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30 | 中文流式转写 | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)（源自 [icefall](https://github.com/k2-fsa/icefall)） | Apache-2.0 |
| sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 | 英文流式转写 | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)（源自 NVIDIA [Nemotron](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)） | NVIDIA Open Model License |
| sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 | 第二遍精修 | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)（源自 [Qwen](https://github.com/QwenLM/Qwen3)） | Apache-2.0 |
| 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx | 说话人识别（声纹） | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)（源自 [3D-Speaker](https://github.com/modelscope/3D-Speaker)） | Apache-2.0 |
| HY-MT1.5-1.8B-Q4_K_M.gguf | 英文→中文翻译 | [腾讯 HY-MT1.5-1.8B-GGUF](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF) | 见模型发布页 |

> 说明：上述模型**不在本仓库内**，需用户自行从官方源下载。表中许可证以各模型
> 发布页为准，可能随版本更新；如与官方声明冲突，以官方为准。

## 免责声明

本项目是学习与个人使用性质的工具。模型权重的授权与合规责任由各模型发布方承担；
使用者应自行确认其在自身使用场景（含商用）下的合法性。
