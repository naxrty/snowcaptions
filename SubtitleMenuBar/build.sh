#!/bin/bash
# 编译并打包 SubtitleMenuBar.app（纯 swiftc，无需 Xcode）
set -e
cd "$(dirname "$0")"

SRC="SubtitleMenuBar.swift"
APP="build/SubtitleMenuBar.app"
BIN="$APP/Contents/MacOS/SubtitleMenuBar"
SDK="$(xcrun --show-sdk-path)"

echo "==> 清理旧构建"
rm -rf build
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "==> 编译 Swift（target macOS 13，玻璃效果在 26+ 运行时自动启用）"
swiftc "$SRC" \
  -o "$BIN" \
  -sdk "$SDK" \
  -target arm64-apple-macosx13.0 \
  -framework Cocoa \
  -O

echo "==> 写入 Info.plist"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>SubtitleMenuBar</string>
	<key>CFBundleIdentifier</key>
	<string>com.local.subtitlemenubar</string>
	<key>CFBundleName</key>
	<string>Subtitle Menu Bar</string>
	<key>CFBundleDisplayName</key>
	<string>实时字幕</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>13.0</string>
	<key>LSUIElement</key>
	<true/>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSMicrophoneUsageDescription</key>
	<string>需要麦克风权限来进行语音转写。</string>
	<key>NSHumanReadableCopyright</key>
	<string>本地自建，MIT</string>
</dict>
</plist>
PLIST

echo "==> 签名（ad-hoc，稳定 identifier，让麦克风授权能长期记住）"
# 用固定 identifier 做 ad-hoc 签名。虽然重编译后代码目录哈希仍会变，
# 但固定 identifier + 固定安装路径（~/Applications）能最大限度让 TCC 复用授权。
codesign --force --sign - --identifier "com.local.subtitlemenubar" "$APP"

echo "==> 完成"
echo "    App 位置: $(cd "$APP" && pwd)"
echo "    试运行:  open \"$(cd "$APP" && pwd)\""
echo "    注意: 授权麦克风后，尽量别再移动/重编译此 App，否则系统可能重新询问。"
