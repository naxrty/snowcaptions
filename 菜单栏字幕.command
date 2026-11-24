#!/bin/bash
# 一键启动：菜单栏字幕 App + Python 引擎
# 双击「菜单栏字幕.command」即可
cd "$(dirname "$0")"

# 优先用 ~/Applications 里已部署的版本（麦克风授权绑定这个路径），
# 不存在再退回 build/ 编译产物。
APP="$HOME/Applications/SubtitleMenuBar.app"
BUILD="$PWD/SubtitleMenuBar/build/SubtitleMenuBar.app"

echo "实时字幕 · 菜单栏版"
echo ""

# 1. 确保 App 存在：优先已部署版，否则用 build 产物，再否则现场编译
if [ ! -f "$APP/Contents/MacOS/SubtitleMenuBar" ] && [ ! -f "$BUILD/Contents/MacOS/SubtitleMenuBar" ]; then
    echo "首次使用，先编译 Swift App…"
    (cd SubtitleMenuBar && ./build.sh) || { echo "编译失败"; read -n 1; exit 1; }
fi
if [ -f "$APP/Contents/MacOS/SubtitleMenuBar" ]; then
    RUN="$APP"
else
    RUN="$BUILD"
fi

# 2. 启动菜单栏 App（如果没在跑）
if ! pgrep -f "SubtitleMenuBar.app/Contents/MacOS/SubtitleMenuBar" >/dev/null 2>&1; then
    open "$RUN"
    echo "✓ 菜单栏图标已启动（看屏幕右上角 ❄️ 雪花）"
    sleep 1
else
    echo "✓ 菜单栏图标已在运行"
fi

echo "✓ 点菜单栏 ❄️ 雪花图标 → 弹窗里点「开始」即可转写"
echo "✓ 点「停止」结束并自动保存到 transcripts/"
echo ""
echo "本窗口可关闭，识别在后台进行。"
