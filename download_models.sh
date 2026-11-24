#!/bin/bash
# 下载所有模型到正确位置。
#
# 模型都不进 git（太大，且是第三方发布的权重），首次 clone 后跑一次本脚本即可。
# 全部来自官方源，可断点续传（curl -C -）。模型许可证见各模型发布页，多为 Apache-2.0。
#
# 用法：
#   ./download_models.sh            # 下载「必需」模型（流式转写 + 精修 + 声纹）
#   ./download_models.sh --all      # 额外下载可选的翻译大模型（约 1.1GB）
#   ./download_models.sh --zh-only  # 只下中文，不碰英文/翻译/声纹
#
# 环境变量：
#   SHERPA_MODELS_BASE  覆盖模型下载基地址（默认 GitHub，国内可换镜像）

set -euo pipefail

cd "$(dirname "$0")"

BASE="${SHERPA_MODELS_BASE:-https://github.com/k2-fsa/sherpa-onnx/releases/download}"
ASR="$BASE/asr-models"
SPK="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"

MODE="${1:---required}"

# 各模型：名称 | 归档名 | 目标目录（相对本项目根）
# 中文流式（第一遍，主力）
ZH="sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30"
# 英文流式（第一遍，主力）
EN="sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
# 第二遍精修（Qwen3-ASR）
REFINE="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
# 声纹（说话人识别，单文件 .onnx）
SPK_MODEL="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

# 翻译大模型 GGUF（可选，约 1.1GB，腾讯 HY-MT1.5-1.8B）
MT_URL="https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/resolve/main/HY-MT1.5-1.8B-Q4_K_M.gguf"

say()  { printf '\033[1;34m[下载]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[提示]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[失败]\033[0m %s\n' "$*" >&2; exit 1; }

# 下载并解压一个 tarball 到指定目标目录
fetch_tar() {
    local name="$1" url="$2" dest_dir="$3"
    if [ -d "$dest_dir/$name" ]; then
        say "已存在，跳过：$name"
        return
    fi
    local tarball="/tmp/$name.tar.bz2"
    say "下载 $name ..."
    curl -L --fail --retry 3 -C - -o "$tarball" "$url/$name.tar.bz2" || die "下载失败：$name"
    mkdir -p "$dest_dir"
    say "解压 $name ..."
    tar xjf "$tarball" -C "$dest_dir"
    rm -f "$tarball"
}

# 单文件下载（声纹 / GGUF）
fetch_file() {
    local name="$1" url="$2" dest="$3"
    if [ -s "$dest" ]; then
        say "已存在，跳过：$name"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    say "下载 $name ..."
    curl -L --fail --retry 3 -C - -o "$dest" "$url" || die "下载失败：$name"
}

case "$MODE" in
  --zh-only)
    fetch_tar "$ZH" "$ASR" "models"
    ;;
  --required|--all)
    fetch_tar "$ZH"     "$ASR" "models"
    fetch_tar "$EN"     "$ASR" "models"
    fetch_tar "$REFINE" "$ASR" "models_refine"
    fetch_file "$SPK_MODEL" "$SPK/$SPK_MODEL" "models_speaker/$SPK_MODEL"
    ;;
  *)
    die "未知参数：$MODE（支持 --required / --all / --zh-only）"
    ;;
esac

# 可选：翻译大模型（关掉翻译或只用 Argos 兜底也能跑，需要高质量中译再下这个）
if [ "$MODE" = "--all" ]; then
    fetch_file "HY-MT1.5-1.8B-Q4_K_M.gguf" "$MT_URL" "models_mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
fi

say "完成。用 ./venv/bin/python live_zh.py --list 检查模型是否就位。"
