//  SubtitleMenuBar.swift — 本地实时字幕 · 菜单栏 App（纯 AppKit，无第三方依赖）
//
//  架构：本 App 是「显示层」，Python 引擎（live_zh.py --ui socket）是「识别层」。
//  两者通过本机 TCP（127.0.0.1:7711）通信：
//     Python → Swift：  JSON 行 {"type":"update"/"commit"/"clear"/"hello"/"status"...}
//     Swift  → Python：  这里不直接回推，只负责「按设置启动引擎 + 显示字幕 + 点停止收尾」。
//
//  设置面板：语种、断句松紧、翻译、精修 四项，持久化到引擎目录下 settings.json。
//  点「开始」时把这些设置翻译成 live_zh.py 的命令行参数传给引擎。
//
//  参照用户机器上已有的「DeepSeek 余额」菜单栏工具骨架：
//     NSStatusBar.statusItem + NSPopover + LSUIElement=true。
//  字幕窗用 macOS 26+ 的液态玻璃（NSGlassEffectView），旧系统回退系统材质。

import Cocoa

// MARK: - 常量

let kSocketPort: UInt16 = 7711
let kPopoverWidth: CGFloat = 460
let kPopoverHeight: CGFloat = 240

// Python 引擎目录：动态定位，不硬编码用户名/绝对路径（便于开源分享）。
// 查找顺序：
//   1) ~/.subtitle-menubar-engine 里写明的路径（用户可手动指定）；
//   2) 一组常见候选路径里，第一个存在 live_zh.py 的；
//   3) 都找不到就返回空串（此时「开始」会提示找不到引擎）。
func enginePath() -> String {
    // 1) 用户显式配置
    let cfg = (NSHomeDirectory() as NSString).appendingPathComponent(".subtitle-menubar-engine")
    if let content = try? String(contentsOfFile: cfg, encoding: .utf8) {
        let p = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !p.isEmpty && FileManager.default.fileExists(atPath: p + "/live_zh.py") {
            return p
        }
    }
    // 2) 常见候选路径
    let home = NSHomeDirectory()
    let candidates = [
        home + "/Desktop/GitHub库/snowcaptions",
        home + "/Desktop/snowcaptions",
        home + "/snowcaptions",
        home + "/Downloads/snowcaptions",
    ]
    for c in candidates where FileManager.default.fileExists(atPath: c + "/live_zh.py") {
        return c
    }
    // 3) 找不到
    return ""
}
func settingsPath() -> String { enginePath() + "/settings.json" }
func hotwordsPath() -> String { enginePath() + "/hotwords.txt" }

// 简易日志（调试用，写 /tmp，不影响正常使用）
func dlog(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    if let d = line.data(using: .utf8) {
        if let fh = FileHandle(forWritingAtPath: "/tmp/subtitle-menubar.log") {
            fh.seekToEndOfFile()
            fh.write(d)
            fh.closeFile()
        } else {
            try? line.write(toFile: "/tmp/subtitle-menubar.log", atomically: true, encoding: .utf8)
        }
    }
}

// MARK: - 设置模型

struct Settings: Codable {
    var mode: String = "zh"          // zh / en / en2zh（三种模式）
    var model: String = ""           // 选中的模型目录名（空 = 用该语种默认）
    var pause: String = "normal"     // fast / normal / loose
    var refine: Bool = true          // 第二遍精修
    var hotwords: String = ""        // 热词，逗号分隔（存 hotwords.txt 一份，一行一个词）
    var speaker: Bool = false        // 说话人识别（区分不同人，标「说话人 N」）
}

// 三种模式 → 语种 + 是否翻译
let kModes: [(key: String, title: String, lang: String, translate: Bool)] = [
    ("zh",   "中文转写",      "zh", false),
    ("en",   "英文转写",      "en", false),
    ("en2zh","英文转译成中文","en", true),
]

// 本机可用流式模型（中文 / 英文）。value 是 models/ 下的目录名。
let kZhModels: [(dir: String, title: String)] = [
    ("sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30", "中文·xlarge（最准）"),
    ("sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05", "中文·X-ASR 480ms（快·带标点）"),
]
let kEnModels: [(dir: String, title: String)] = [
    ("sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25", "英文·nemotron 0.6B"),
]

// MARK: - 通信：收 Python 的字幕

final class SubtitleServer: NSObject {
    var onMessage: (([String: Any]) -> Void)?
    private var listener: SocketListener?
    private var clients: [FileHandle] = []

    func start() {
        let l = SocketListener(port: kSocketPort) { [weak self] fh in
            self?.addClient(fh)
        }
        listener = l
        l.start()
    }

    private func addClient(_ fh: FileHandle) {
        clients.append(fh)
        NotificationCenter.default.addObserver(
            forName: .NSFileHandleDataAvailable, object: fh, queue: .main
        ) { [weak self] _ in
            self?.readData(fh)
        }
        fh.waitForDataInBackgroundAndNotify()
    }

    private func readData(_ fh: FileHandle) {
        let data = fh.availableData
        if data.count == 0 {   // 对端断开
            clients.removeAll { $0 === fh }
            return
        }
        if let str = String(data: data, encoding: .utf8) {
            for line in str.components(separatedBy: "\n") {
                let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !t.isEmpty, let d = t.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
                else { continue }
                onMessage?(obj)
            }
        }
        fh.waitForDataInBackgroundAndNotify()
    }
}

// 极简 TCP 监听（AppKit 里没有现成的流式 server，用 C socket 包一层）
final class SocketListener {
    private let port: UInt16
    private let onAccept: (FileHandle) -> Void
    private var sock: Int32 = -1
    private var source: DispatchSourceRead?

    init(port: UInt16, onAccept: @escaping (FileHandle) -> Void) {
        self.port = port
        self.onAccept = onAccept
    }

    func start() {
        sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return }
        var yes: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bindR = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindR == 0 else { close(sock); return }
        guard listen(sock, 5) == 0 else { close(sock); return }

        source = DispatchSource.makeReadSource(fileDescriptor: sock, queue: .main)
        source?.setEventHandler { [weak self] in self?.acceptLoop() }
        source?.resume()
    }

    private func acceptLoop() {
        var addr = sockaddr()
        var len = socklen_t(MemoryLayout<sockaddr>.size)
        let fd = accept(sock, &addr, &len)
        guard fd >= 0 else { return }
        let fh = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        onAccept(fh)
    }
}

// MARK: - 菜单栏 + 字幕窗

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover?
    private var server = SubtitleServer()
    private var settingsWindow: NSWindow?
    private var engineProcess: Process?

    // 字幕内容
    private var liveRows: [String] = ["", ""]      // 实时区 [原文, 译文]
    private var committed: [(String, String, String)] = [] // 定稿区 [(原文, 译文, 说话人)]
    private let maxCommitted = 6                   // 滚动区最多保留几句

    // UI 控件
    private var liveLabel: NSTextField!
    private var liveZhLabel: NSTextField!
    private var historyLabel: NSTextField!
    private var statusLabel: NSTextField!
    private var startBtn: NSButton!
    private var stopBtn: NSButton!
    private var engineRunning = false
    // 设置
    private var settings = Settings()

    // 设置面板控件
    private var modeSeg: NSSegmentedControl!
    private var modelPopup: NSPopUpButton!
    private var refineCheck: NSButton!
    private var speakerCheck: NSButton!
    private var hotwordsField: NSTextField!

    func applicationDidFinishLaunching(_ notification: Notification) {
        dlog("app launched")
        loadSettings()
        buildStatusItem()
        server.onMessage = { [weak self] msg in self?.handle(msg) }
        server.start()
        dlog("server started on \(kSocketPort)")
    }

    // MARK: 设置读写

    private func loadSettings() {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: settingsPath())),
              let s = try? JSONDecoder().decode(Settings.self, from: data) else {
            saveSettings()   // 首次运行写入默认值
            return
        }
        settings = s
    }

    private func saveSettings() {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted]
        if let data = try? enc.encode(settings) {
            try? data.write(to: URL(fileURLWithPath: settingsPath()))
        }
    }

    // MARK: 菜单栏图标

    private func buildStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.autosaveName = "SubtitleMenuBarStatusItem"
        if let b = statusItem.button {
            // 白描风雪花 template 图标（纯黑轮廓，系统自动适配深/浅色）
            if let img = snowmanIcon() {
                img.isTemplate = true
                b.image = img
                b.imagePosition = .imageOnly
            } else {
                b.title = "❄️"
            }
            b.target = self
            b.action = #selector(statusClicked(_:))
            b.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
    }

    // 用 CoreGraphics 画一个白描风雪花（六角对称线条）
    private func snowmanIcon() -> NSImage? {
        let size = NSSize(width: 20, height: 20)
        let img = NSImage(size: size)
        img.lockFocus()
        defer { img.unlockFocus() }

        NSColor.black.setStroke()
        let lineWidth: CGFloat = 1.4
        let cx: CGFloat = 10
        let cy: CGFloat = 10
        let r: CGFloat = 8.2

        // 6 条主轴（0°, 60°, 120°, ...），从中心向外
        // 每条轴上：主杆 + 两侧分支（V 字形）
        for i in 0..<6 {
            let angle = CGFloat(i) * .pi / 3.0
            let dx = cos(angle)
            let dy = sin(angle)

            // 主杆：从中心到外缘
            let main = NSBezierPath()
            main.move(to: NSPoint(x: cx, y: cy))
            main.line(to: NSPoint(x: cx + dx * r, y: cy + dy * r))
            main.lineWidth = lineWidth
            main.stroke()

            // 两侧分支：在 55% 处，向两边 ±30° 分叉，长度约 2.4
            let branchAngle = .pi / 6.0   // 30°
            let br = r * 0.55
            let branchLen: CGFloat = 2.4
            for sign in [CGFloat(1), CGFloat(-1)] {
                let a = angle + sign * branchAngle
                let bx = cos(a)
                let by = sin(a)
                let branch = NSBezierPath()
                branch.move(to: NSPoint(x: cx + dx * br, y: cy + dy * br))
                branch.line(to: NSPoint(x: cx + dx * br + bx * branchLen,
                                        y: cy + dy * br + by * branchLen))
                branch.lineWidth = lineWidth
                branch.stroke()
            }
        }

        // 中心小圆点
        NSColor.black.setFill()
        let centerDot = NSBezierPath(ovalIn: NSRect(x: cx - 1.0, y: cy - 1.0, width: 2.0, height: 2.0))
        centerDot.fill()

        return img
    }

    @objc private func statusClicked(_ sender: NSStatusBarButton) {
        let ev = NSApp.currentEvent
        if ev?.type == .rightMouseUp {
            showMenu()
        } else {
            togglePopover()
        }
    }

    // MARK: 弹窗（字幕窗）

    private func togglePopover() {
        if let p = popover, p.isShown {
            p.performClose(nil)
        } else {
            ensurePopover()
            popover?.show(relativeTo: statusItem.button!.bounds,
                          of: statusItem.button!, preferredEdge: .minY)
        }
    }

    private func ensurePopover() {
        if popover != nil { return }
        let p = NSPopover()
        p.behavior = .transient
        let vc = NSViewController()
        vc.view = buildContentView()
        p.contentViewController = vc
        popover = p
    }

    private func buildContentView() -> NSView {
        let root = NSView(frame: NSRect(x: 0, y: 0, width: kPopoverWidth, height: kPopoverHeight))

        // 顶部状态行
        statusLabel = NSTextField(labelWithString: "未启动")
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        // 实时字幕（原文，大字）——加大字号+加粗+纯白，提升可读性
        liveLabel = NSTextField(labelWithString: " ")
        liveLabel.font = .systemFont(ofSize: 22, weight: .bold)
        liveLabel.textColor = .white
        liveLabel.maximumNumberOfLines = 1
        liveLabel.lineBreakMode = .byTruncatingTail
        liveLabel.translatesAutoresizingMaskIntoConstraints = false

        // 实时译文
        liveZhLabel = NSTextField(labelWithString: " ")
        liveZhLabel.font = .systemFont(ofSize: 17, weight: .semibold)
        liveZhLabel.textColor = .white.withAlphaComponent(0.85)
        liveZhLabel.maximumNumberOfLines = 1
        liveZhLabel.lineBreakMode = .byTruncatingTail
        liveZhLabel.translatesAutoresizingMaskIntoConstraints = false

        // 定稿历史
        historyLabel = NSTextField(labelWithString: "")
        historyLabel.font = .systemFont(ofSize: 16, weight: .medium)
        historyLabel.textColor = .white.withAlphaComponent(0.78)
        historyLabel.maximumNumberOfLines = 6
        historyLabel.lineBreakMode = .byTruncatingTail
        historyLabel.translatesAutoresizingMaskIntoConstraints = false

        // 按钮行（固定左下角）
        startBtn = NSButton(title: "开始", target: self, action: #selector(startEngine))
        startBtn.bezelStyle = .rounded
        stopBtn = NSButton(title: "停止", target: self, action: #selector(stopEngine))
        stopBtn.bezelStyle = .rounded
        let settingsBtn = NSButton(title: "设置", target: self, action: #selector(openSettings))
        settingsBtn.bezelStyle = .rounded
        let btns = NSStackView(views: [startBtn, stopBtn, settingsBtn])
        btns.orientation = .horizontal
        btns.spacing = 8
        btns.translatesAutoresizingMaskIntoConstraints = false

        // 字幕内容容器（状态 + 原文 + 译文 + 历史）
        let content = NSStackView(views: [statusLabel, liveLabel, liveZhLabel, historyLabel])
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 6
        content.translatesAutoresizingMaskIntoConstraints = false

        // 液态玻璃容器（macOS 26+），旧系统用深色材质（保证白色字幕清晰）
        let container: NSView
        if #available(macOS 26, *) {
            let glass = NSGlassEffectView()
            glass.contentView = content
            glass.translatesAutoresizingMaskIntoConstraints = false
            container = glass
        } else {
            let vis = NSVisualEffectView()
            vis.material = .hudWindow
            vis.blendingMode = .behindWindow
            vis.state = .active
            vis.appearance = NSAppearance(named: .darkAqua)
            vis.addSubview(content)
            container = vis
        }

        root.addSubview(container)
        root.addSubview(btns)
        NSLayoutConstraint.activate([
            container.topAnchor.constraint(equalTo: root.topAnchor),
            container.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            container.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            container.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            content.topAnchor.constraint(equalTo: root.topAnchor, constant: 12),
            content.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: 14),
            content.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -14),
            // 按钮固定左下角
            btns.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: 14),
            btns.bottomAnchor.constraint(equalTo: root.bottomAnchor, constant: -10),
        ])
        if #available(macOS 26, *) {
            // NSGlassEffectView 的 contentView 自动铺满
        } else {
            content.translatesAutoresizingMaskIntoConstraints = true
        }
        return root
    }

    // MARK: 处理 Python 消息

    private func handle(_ msg: [String: Any]) {
        guard let type = msg["type"] as? String else { return }
        dlog("recv type=\(type)")
        switch type {
        case "hello":
            engineRunning = true
            statusLabel?.stringValue = "● 引擎已连接"
            statusLabel?.textColor = .systemGreen
        case "update":
            if let rows = msg["rows"] as? [String] {
                liveRows = rows.count >= 2 ? [rows[0], rows[1]] : [rows.first ?? "", ""]
                liveLabel?.stringValue = liveRows[0].isEmpty ? " " : liveRows[0]
                liveZhLabel?.stringValue = liveRows[1].isEmpty ? " " : liveRows[1]
            }
        case "commit":
            if let rows = msg["rows"] as? [String], let first = rows.first, !first.isEmpty {
                let zh = rows.count > 1 ? rows[1] : ""
                let spk = msg["speaker"] as? String ?? ""
                committed.append((first, zh, spk))
                if committed.count > maxCommitted { committed.removeFirst(committed.count - maxCommitted) }
                renderHistory()
                liveRows = ["", ""]
                liveLabel?.stringValue = " "
                liveZhLabel?.stringValue = " "
            }
        case "clear":
            liveRows = ["", ""]
            liveLabel?.stringValue = " "
            liveZhLabel?.stringValue = " "
        case "status":
            if let state = msg["state"] as? String {
                if state == "saved", let path = msg["path"] as? String {
                    statusLabel?.stringValue = "已保存：\(path)"
                    statusLabel?.textColor = .secondaryLabelColor
                    notifySaved(path: path)
                } else if state == "stopped" {
                    engineRunning = false
                    // 若刚保存过，状态栏保留保存路径；否则显示"已停止"
                    if statusLabel?.stringValue.hasPrefix("已保存") != true {
                        statusLabel?.stringValue = "已停止"
                    }
                    statusLabel?.textColor = .secondaryLabelColor
                    refreshButtons()
                }
            }
        default:
            break
        }
    }

    // 保存后弹系统通知，让用户明确知道文件存到哪了
    private func notifySaved(path: String) {
        let note = NSUserNotification()
        note.title = "字幕已保存"
        note.informativeText = path
        note.soundName = nil
        NSUserNotificationCenter.default.deliver(note)
    }

    private func renderHistory() {
        let lines = committed.map { en, zh, spk in
            let num = spk.filter { $0.isNumber }
            let tag = num.isEmpty ? "" : "👤\(num) "
            let head = tag + en
            return zh.isEmpty ? head : "\(head)\n　↳ \(zh)"
        }
        historyLabel?.stringValue = lines.joined(separator: "\n")
    }

    private func refreshButtons() {
        startBtn?.isEnabled = !engineRunning
        stopBtn?.isEnabled = engineRunning
    }

    // MARK: 按钮动作

    @objc private func startEngine() {
        guard !engineRunning else { return }

        // 引擎目录找不到就提示，不硬启动
        let ep = enginePath()
        guard !ep.isEmpty else {
            statusLabel?.stringValue = "找不到引擎目录（live_zh.py）"
            statusLabel?.textColor = .systemRed
            dlog("startEngine: enginePath() 为空，无法启动")
            return
        }

        // 用设置拼出命令行参数
        let mode = kModes.first { $0.key == settings.mode } ?? kModes[0]
        var args = ["--lang", mode.lang,
                    "--pause", settings.pause,
                    "--ui", "socket",
                    "--ui-port", "\(kSocketPort)"]
        // 指定具体模型（非空才传）
        if !settings.model.isEmpty {
            args.append("--model")
            args.append(settings.model)
        }
        if mode.translate {
            args.append("--translate")
        } else {
            args.append("--no-translate")
        }
        if settings.refine {
            args.append("--refine")
        } else {
            args.append("--no-refine")
        }
        if settings.speaker {
            args.append("--speaker")
        } else {
            args.append("--no-speaker")
        }
        // 热词：有就传 hotwords.txt 文件路径（一行一个词）
        let hotwords = settings.hotwords
            .replacingOccurrences(of: "，", with: ",")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !hotwords.isEmpty {
            args.append("--hotwords")
            args.append(hotwordsPath())
        }

        let cmd = (["cd \(enginePath()) && ./venv/bin/python live_zh.py"] + args)
            .joined(separator: " ")
        dlog("launch engine: \(cmd)")

        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", cmd]
        task.currentDirectoryURL = URL(fileURLWithPath: enginePath())
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            // 让引擎成为独立进程组 leader，停止时能对整个组发 SIGINT
            _ = setpgid(task.processIdentifier, task.processIdentifier)
            engineProcess = task
        } catch {
            statusLabel?.stringValue = "启动失败：\(error.localizedDescription)"
            statusLabel?.textColor = .systemRed
            return
        }
        engineRunning = true
        statusLabel?.stringValue = "启动中…"
        statusLabel?.textColor = .secondaryLabelColor
        refreshButtons()
    }

    @objc private func stopEngine() {
        // 结束当前引擎：对整个进程组发 SIGINT（等价 Ctrl+C，触发 Python 优雅保存 +
        // 清理 llama-server），再 SIGTERM 兜底强杀。
        if let p = engineProcess, p.isRunning {
            let pgid = getpgid(p.processIdentifier)
            let int = Process()
            int.launchPath = "/bin/kill"
            int.arguments = ["-INT", "-\(pgid)"]   // 负号 = 进程组
            try? int.run()
            int.waitUntilExit()

            // 给 Python 一点时间优雅收尾（保存 + 关 llama-server），再兜底 SIGTERM
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                if p.isRunning {
                    p.terminate()
                }
            }
        }
        engineProcess = nil
        engineRunning = false
        statusLabel?.stringValue = "已停止"
        statusLabel?.textColor = .secondaryLabelColor
        refreshButtons()
    }

    // MARK: 设置面板

    @objc private func openSettings() {
        if settingsWindow == nil {
            settingsWindow = buildSettingsWindow()
        }
        settingsWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func buildSettingsWindow() -> NSWindow {
        let win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 380, height: 465),
                           styleMask: [.titled, .closable],
                           backing: .buffered, defer: false)
        win.title = "字幕设置"
        win.isReleasedWhenClosed = false
        win.center()

        let content = NSView(frame: win.contentRect(forFrameRect: win.frame))

        // 模式（三段滑块）
        let modeTitle = NSTextField(labelWithString: "模式")
        modeTitle.font = .systemFont(ofSize: 13, weight: .semibold)
        modeSeg = NSSegmentedControl(labels: kModes.map { $0.title },
                                     trackingMode: .selectOne,
                                     target: self,
                                     action: #selector(modeChanged))
        modeSeg.selectedSegment = max(0, kModes.firstIndex { $0.key == settings.mode } ?? 0)

        // 模型（下拉，选项跟随模式过滤）
        let modelTitle = NSTextField(labelWithString: "模型")
        modelTitle.font = .systemFont(ofSize: 13, weight: .semibold)
        modelPopup = NSPopUpButton(frame: .zero, pullsDown: false)
        modelPopup.target = self
        modelPopup.action = #selector(modelChanged)

        // 精修开关
        refineCheck = NSButton(checkboxWithTitle: "第二遍精修（更准，稍慢）", target: nil, action: nil)
        refineCheck.state = settings.refine ? .on : .off

        // 说话人识别开关
        speakerCheck = NSButton(checkboxWithTitle: "说话人识别（区分不同人，标「说话人 1/2」）", target: nil, action: nil)
        speakerCheck.state = settings.speaker ? .on : .off

        // 热词
        let hotwordsTitle = NSTextField(labelWithString: "热词（逗号分隔，专名/人名/术语）")
        hotwordsTitle.font = .systemFont(ofSize: 13, weight: .semibold)
        hotwordsField = NSTextField(frame: .zero)
        hotwordsField.placeholderString = "如：KPI,张明,深圳,古希腊"
        hotwordsField.stringValue = settings.hotwords
        hotwordsField.font = .systemFont(ofSize: 13)
        let hotwordsHint = NSTextField(labelWithString: "转写总听错的专有名词加进来，能纠正识别错误。留空 = 不用。")
        hotwordsHint.font = .systemFont(ofSize: 11)
        hotwordsHint.textColor = .tertiaryLabelColor

        // 保存按钮
        let saveBtn = NSButton(title: "保存", target: self, action: #selector(saveSettingsTapped))
        saveBtn.bezelStyle = .rounded
        saveBtn.keyEquivalent = "\r"

        // 说明
        let hint = NSTextField(labelWithString: "保存后，下次点「开始」生效。")
        hint.font = .systemFont(ofSize: 11)
        hint.textColor = .tertiaryLabelColor

        let stack = NSStackView(views: [
            modeTitle, modeSeg,
            modelTitle, modelPopup,
            refineCheck, speakerCheck,
            hotwordsTitle, hotwordsField, hotwordsHint,
            saveBtn, hint
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false

        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 20),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            modeSeg.widthAnchor.constraint(equalTo: stack.widthAnchor),
            modelPopup.widthAnchor.constraint(equalTo: stack.widthAnchor),
            hotwordsField.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])

        win.contentView = content

        // 初始填充模型下拉（跟随当前模式）
        reloadModelPopup(preserveSelection: true)
        return win
    }

    // 根据当前模式，刷新模型下拉的选项
    private func reloadModelPopup(preserveSelection: Bool) {
        let mode = kModes[modeSeg?.selectedSegment ?? 0]
        let list = mode.lang == "en" ? kEnModels : kZhModels
        let prevDir = settings.model
        modelPopup.removeAllItems()
        modelPopup.addItems(withTitles: list.map { $0.title })

        // 选中：优先保留 settings.model（如果它在该语种的列表里），否则选第一个
        var target = 0
        if let idx = list.firstIndex(where: { $0.dir == prevDir }) {
            target = idx
        }
        modelPopup.selectItem(at: target)
        settings.model = list[target].dir
    }

    @objc private func modeChanged() {
        reloadModelPopup(preserveSelection: false)
    }

    @objc private func modelChanged() {
        let mode = kModes[modeSeg.selectedSegment]
        let list = mode.lang == "en" ? kEnModels : kZhModels
        let idx = modelPopup.indexOfSelectedItem
        if idx >= 0 && idx < list.count {
            settings.model = list[idx].dir
        }
    }

    @objc private func saveSettingsTapped() {
        let mode = kModes[modeSeg.selectedSegment]
        settings.mode = mode.key
        // 确保 model 已经是最新（modelChanged 或 reloadModelPopup 已同步）
        let list = mode.lang == "en" ? kEnModels : kZhModels
        if let idx = list.firstIndex(where: { $0.dir == settings.model }) {
            settings.model = list[idx].dir
        } else {
            settings.model = list.first?.dir ?? ""
        }
        settings.refine = refineCheck.state == .on
        settings.speaker = speakerCheck.state == .on
        settings.hotwords = hotwordsField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        saveSettings()
        saveHotwordsFile()
        settingsWindow?.orderOut(nil)
        dlog("settings saved: mode=\(settings.mode) model=\(settings.model) refine=\(settings.refine) speaker=\(settings.speaker) hotwords=[\(settings.hotwords)]")
    }

    // 热词额外写一份 hotwords.txt（一行一个词），供引擎 --hotwords 文件模式读取
    private func saveHotwordsFile() {
        let words = settings.hotwords
            .replacingOccurrences(of: "，", with: ",")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let content = words.joined(separator: "\n") + (words.isEmpty ? "" : "\n")
        try? content.write(toFile: hotwordsPath(), atomically: true, encoding: .utf8)
    }

    // MARK: 右键菜单

    private func showMenu() {
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "开始", action: #selector(startEngine), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "停止", action: #selector(stopEngine), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "设置…", action: #selector(openSettings), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "退出", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    func applicationWillTerminate(_ notification: Notification) {
        // 退出时优雅停止引擎（SIGINT 触发保存），不留孤儿进程
        if let p = engineProcess, p.isRunning {
            let pgid = getpgid(p.processIdentifier)
            let int = Process()
            int.launchPath = "/bin/kill"
            int.arguments = ["-INT", "-\(pgid)"]
            try? int.run()
            int.waitUntilExit()
            p.terminate()
        }
    }
}

// MARK: - 入口

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)   // 菜单栏 App，不占 Dock
app.run()
