#!/bin/bash
cd "$(dirname "$0")"
clear
echo ""
echo "  ┌────────────────────────────────────────────┐"
echo "  │ 本地实时中文转写 · 完全离线                │"
echo "  └────────────────────────────────────────────┘"
echo ""
./venv/bin/python live_zh.py --lang zh
echo ""
echo "已停止。按任意键关闭此窗口。"
read -n 1
