#!/bin/bash
cd "$(dirname "$0")"
clear
echo ""
echo "  ┌────────────────────────────────────────────┐"
echo "  │ 英文实时转写 + 中文字幕 · 离线             │"
echo "  └────────────────────────────────────────────┘"
echo ""
./venv/bin/python live_zh.py --lang en
echo ""
echo "已停止。按任意键关闭此窗口。"
read -n 1
