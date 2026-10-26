// 仅供在浏览器中预览界面：没有宿主程序时模拟一次运行中的会话。
// 数值取自 2026-10-26 在 RTX 4050 Laptop 上的实测（nr_detect，YOLO26n FP16，WGC 窗口采集，测试源 144 FPS）；
// 类别名来自参考模型的导出契约。真实程序中这些数据全部由宿主推送。
window.NR_PREVIEW = (emit) => {
  const classes = {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"};
  const model = {
    path: 'D:\Myproject\neural-reflex\models\reference\yolo26n-640-nms-fp16.onnx',
    name: 'yolo26n-640-nms-fp16', sizeBytes: 5079119, inputSize: 640, precision: 'FP16',
    head: 'one-to-many (needs NMS)', decoder: 'yolo-anchors', keypoints: 0, classes, engineCached: true,
  };
  const windows = [
    { hwnd: '1835362', title: 'NR Test Source - 144.0 FPS - 帧 8642', exe: 'nr_testsource.exe', pid: 41208, width: 1280, height: 720, minimized: false, icon: '' },
    { hwnd: '657958', title: 'performance-design.md - neural-reflex', exe: 'WindowsTerminal.exe', pid: 18844, width: 1704, height: 1043, minimized: false, icon: '' },
    { hwnd: '400812', title: 'docs', exe: 'explorer.exe', pid: 9124, width: 1180, height: 760, minimized: false, icon: '' },
    { hwnd: '1180244', title: 'TensorRT for RTX 文档 - Microsoft Edge', exe: 'msedge.exe', pid: 22716, width: 1536, height: 980, minimized: true, icon: '' },
  ];
  const monitors = [{ index: 0, width: 2560, height: 1600, refreshHz: 165, primary: true }];
  const settings = { conf: 0.25, iou: 0.7, mode: 'wgc', overlay: true, cudaGraph: true, gpuRealtime: false, publishResults: false };
  let target = { kind: 'window', ...windows[0], icon: 'img/mark.svg', alive: true };
  let phase = 'running';

  // 确定性的小幅波动，使每次预览一致
  let seed = 20261026;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const jitter = (v, r) => v * (1 + (rand() - 0.5) * r);
  const second = (t) => {
    const p50 = jitter(1.51, 0.06), p95 = jitter(2.65, 0.10);
    return { t, dps: jitter(145.0, 0.01), ops: jitter(164.0, 0.01), p50, p95, p99: p95 * jitter(1.2, 0.06),
             pre: jitter(0.015, 0.2), infer: jitter(1.17, 0.04), post: jitter(0.035, 0.05), boxes: jitter(3.8, 0.05), overflow: 0 };
  };
  const seconds = Array.from({ length: 60 }, (_, i) => second(i + 1));
  seconds[37] = { ...seconds[37], p95: 4.12, p99: 6.31 };  // 一次后台任务造成的尖峰
  const detections = Math.round(seconds.reduce((s, x) => s + x.dps, 0));
  console.assert(seconds.length === 60, '示例会话应有 60 秒');

  const stats = (secs) => ({
    type: 'stats', phase, elapsed: secs.at(-1).t, prepareMs: 442, adapter: 'NVIDIA GeForce RTX 4050 Laptop GPU',
    inputSize: 640, numClasses: 80, detections, overflowFrames: 0,
    latency: { n: detections, p50: 1.515, p95: 2.695, p99: 3.185, max: 13.648 },
    pre: { n: detections, p50: 0.015, p95: 0.055, p99: 0.255, max: 0.585 },
    infer: { n: detections, p50: 1.175, p95: 2.155, p99: 2.365, max: 9.051 },
    post: { n: detections, p50: 0.035, p95: 0.035, p99: 0.035, max: 0.219 },
    vramUsed: 207 * 1048576, vramBudget: 5152 * 1048576, seconds: secs,
    classes: [{ id: 0, name: 'person', count: 3 }, { id: 5, name: 'bus', count: 1 }],
  });
  const stateMsg = () => ({ type: 'state', phase, error: '', target, model, settings, gpu: 'NVIDIA GeForce RTX 4050 Laptop GPU' });

  // 插件：两个示例插件运行中（其中一个报告降级），另一个声明了宿主不支持的协议主版本
  const t0 = Date.parse('2026-10-26T14:02:10+08:00');
  const alertManifest = {"description": "指定类别的目标连续出现一段时间后发出提示，并可把告警记录到 CSV。", "settings": [{"key": "classes", "type": "string", "label": "告警类别", "description": "类别名称，多个用逗号分隔", "default": "person", "live": true}, {"key": "minScore", "type": "number", "label": "最低置信度", "min": 0.05, "max": 0.95, "step": 0.05, "default": 0.5, "live": true}, {"key": "minCount", "type": "number", "label": "最少目标数", "description": "同一帧中达到这个数量才算出现", "min": 1, "max": 50, "step": 1, "default": 1, "live": true}, {"key": "holdFrames", "type": "number", "label": "持续帧数", "description": "连续出现这么多帧才告警", "min": 1, "max": 1440, "step": 1, "default": 30, "live": true}, {"key": "cooldown", "type": "number", "label": "告警间隔（秒）", "min": 1, "max": 600, "step": 1, "default": 10, "live": true}, {"key": "sound", "type": "enum", "label": "提示音", "options": [{"value": "none", "label": "无"}, {"value": "beep", "label": "系统提示音"}], "default": "none", "live": true}, {"key": "record", "type": "boolean", "label": "记录到 CSV", "description": "写入插件数据目录下的 alerts.csv", "default": true, "live": true}], "actions": [{"id": "test", "label": "测试提示", "description": "按当前设置发出一次提示（含提示音）"}, {"id": "reset", "label": "清零计数", "description": "告警次数归零", "confirm": "确认清零？"}]};
  const recorderManifest = {"description": "按顺序把每帧检测结果写入 JSON Lines 文件，每个会话一个文件；来不及读取而丢失的帧以缺口记录写明。", "settings": [{"key": "classes", "type": "string", "label": "记录类别", "description": "类别名称，多个用逗号分隔；留空记录全部类别", "default": "", "live": true}, {"key": "minScore", "type": "number", "label": "最低置信度", "min": 0, "max": 0.95, "step": 0.05, "default": 0, "live": true}, {"key": "keypoints", "type": "boolean", "label": "记录关键点", "description": "姿态模型的关键点一并写入", "default": true, "live": true}, {"key": "skipEmpty", "type": "boolean", "label": "跳过没有目标的帧", "description": "按类别与置信度筛选后没有目标的帧不写入（缺口记录不受影响）", "default": false, "live": true}, {"key": "maxFileMB", "type": "number", "label": "单个文件上限（MB）", "description": "超过后在同一会话内新建文件", "min": 10, "max": 4000, "step": 10, "default": 500, "live": true}], "actions": [{"id": "rotate", "label": "新建文件", "description": "结束当前文件，之后的帧写入新文件"}, {"id": "open", "label": "打开记录目录"}, {"id": "clear", "label": "删除旧记录", "description": "删除当前文件之外的全部记录", "confirm": "确认删除？"}]};
  const plugins = {
    type: 'plugins', api: '1.1', features: ['control', 'results.ring'], hostVersion: '0.1.0',
    userDir: 'C:/Users/me/AppData/Local/NeuralReflex/plugins', resultsMapping: 'Local/NeuralReflex.Results.v1',
    items: [
      {
        id: 'target-alert', name: '目标告警', version: '0.2.0', description: alertManifest.description,
        source: 'bundled', dir: 'D:/Myproject/neural-reflex/plugins/target-alert', api: 1, compatible: true, command: 'python',
        features: [], optionalFeatures: ['control'], activeFeatures: ['control'], control: true, restartPolicy: 'always',
        actions: alertManifest.actions, enabled: true, state: 'running', error: '', pid: 30412, restarts: 0,
        startedAt: t0, readyMs: 142, status: '监视 person · 模型 yolo26n-640-nms-fp16', health: { state: 'ok', text: '' },
        metrics: [
          { key: 'alerts', label: '告警次数', value: 6, unit: '' }, { key: 'nr.fps', label: 'nr.fps', value: 144.9, unit: '' },
          { key: 'nr.latency', label: 'nr.latency', value: 0.047, unit: '' }, { key: 'nr.process', label: 'nr.process', value: 0.112, unit: '' },
          { key: 'nr.skipped', label: 'nr.skipped', value: 0, unit: '' }, { key: 'nr.lost', label: 'nr.lost', value: 0, unit: '' },
          { key: 'streak', label: '当前连续帧', value: 8642, unit: '' },
        ],
        resources: { cpu: 1.6, memoryMB: 12.4, processes: 2 },
        settingsSchema: alertManifest.settings,
        settings: { classes: 'person', minScore: 0.5, minCount: 1, holdFrames: 30, cooldown: 10, sound: 'none', record: true },
        logs: [
          { t: t0, level: 'info', text: '已启动，进程号 30412，等待就绪' },
          { t: t0 + 142, level: 'info', text: '已就绪，启动用时 142 ms' },
          { t: t0 + 150, level: 'info', text: '监视 person，置信度 ≥ 0.50，至少 1 个，连续 30 帧，间隔 10 s' },
          { t: t0 + 4300, level: 'info', text: '提示：检测到 person × 3（最高置信度 0.86）' },
          { t: t0 + 14500, level: 'info', text: '提示：检测到 person × 3（最高置信度 0.85）' },
        ],
      },
      {
        id: 'result-recorder', name: '结果记录', version: '0.1.0', description: recorderManifest.description,
        source: 'bundled', dir: 'D:/Myproject/neural-reflex/plugins/result-recorder', api: 1, compatible: true, command: 'python',
        features: ['control', 'results.ring'], optionalFeatures: [], activeFeatures: ['control', 'results.ring'], control: true,
        restartPolicy: 'always', actions: recorderManifest.actions, enabled: true, state: 'running', error: '', pid: 30988, restarts: 0,
        startedAt: t0, readyMs: 128, status: '写入 20261026-140212-1792994532000-001.jsonl',
        health: { state: 'degraded', text: '最近 10 秒有 37 帧来不及读取' },
        metrics: [
          { key: 'recorded', label: '已记录帧', value: 18211, unit: '' }, { key: 'rate', label: '写入速率', value: 141.2, unit: '帧/秒' },
          { key: 'lost', label: '缺口帧数', value: 37, unit: '' }, { key: 'size', label: '当前文件', value: 21.6, unit: 'MB' },
          { key: 'nr.fps', label: 'nr.fps', value: 141.2, unit: '' }, { key: 'nr.latency', label: 'nr.latency', value: 0.31, unit: '' },
          { key: 'nr.process', label: 'nr.process', value: 0.62, unit: '' }, { key: 'nr.skipped', label: 'nr.skipped', value: 0, unit: '' },
          { key: 'nr.lost', label: 'nr.lost', value: 37, unit: '' },
        ],
        resources: { cpu: 3.1, memoryMB: 14.8, processes: 2 },
        settingsSchema: recorderManifest.settings,
        settings: { classes: '', minScore: 0, keypoints: true, skipEmpty: false, maxFileMB: 500 },
        logs: [
          { t: t0, level: 'info', text: '已启动，进程号 30988，等待就绪' },
          { t: t0 + 128, level: 'info', text: '已就绪，启动用时 128 ms' },
          { t: t0 + 2100, level: 'info', text: '开始记录 20261026-140212-1792994532000-001.jsonl' },
          { t: t0 + 9300, level: 'warning', text: '健康状态：最近 10 秒有 37 帧来不及读取' },
        ],
      },
      {
        id: 'cloud-export', name: '云端导出', version: '2.0.0', description: '把每帧结果上传到远程服务。',
        source: 'user', dir: 'C:/Users/me/AppData/Local/NeuralReflex/plugins/cloud-export', api: 2, compatible: false, command: 'cloud-export.exe',
        features: [], optionalFeatures: [], activeFeatures: [], control: false, restartPolicy: 'always', actions: [],
        enabled: false, state: 'incompatible', error: '插件需要协议主版本 2，宿主协议版本为 1.1', pid: 0, restarts: 0, status: '',
        health: null, resources: null, metrics: [], settingsSchema: [], settings: {}, logs: [],
      },
    ],
  };

  return {
    handle(msg) {
      if (msg.cmd === 'hello') {
        emit(stateMsg());
        emit({ type: 'targets', windows, monitors });
        emit({ type: 'models', items: [
          { path: model.path, name: model.name, sizeBytes: model.sizeBytes, precision: 'FP16', inputSize: 640, classCount: 80, engineCached: true, supported: true, decoder: 'yolo-anchors' },
          { path: model.path.replace('fp16', 'fp32'), name: 'yolo26n-640-nms-fp32', sizeBytes: 9930306, precision: 'FP32', inputSize: 640, classCount: 80, engineCached: true, supported: true, decoder: 'yolo-anchors' },
          { path: model.path.replace('nms-fp16', 'e2e-fp32'), name: 'yolo26n-640-e2e-fp32', sizeBytes: 9942036, precision: 'FP32', inputSize: 640, classCount: 80, engineCached: false, supported: true, decoder: 'yolo-end2end' },
        ] });
        emit(stats(seconds));
        emit(plugins);
      } else if (msg.cmd === 'setPluginEnabled') {
        const p = plugins.items.find((x) => x.id === msg.id);
        if (p && p.compatible) Object.assign(p, msg.enabled ? { enabled: true, state: 'running', pid: 30412 } : { enabled: false, state: 'stopped', pid: 0 });
        emit(plugins);
      } else if (msg.cmd === 'setPluginSettings' || msg.cmd === 'restartPlugin') {
        const p = plugins.items.find((x) => x.id === msg.id);
        if (p && msg.settings) p.settings = msg.settings;
        if (p && p.enabled) { p.restarts += 1; p.logs.push({ t: Date.now(), level: 'info', text: '已停止，退出代码 0' }, { t: Date.now(), level: 'info', text: '已启动，进程号 30412' }); }
        emit(plugins);
      } else if (msg.cmd === 'pluginAction') {
        const p = plugins.items.find((x) => x.id === msg.id);
        const a = p && p.actions.find((x) => x.id === msg.action);
        if (a) emit({ type: 'toast', level: 'info', text: `${p.name}：${a.label}：已完成` });
      } else if (msg.cmd === 'exportDiagnostics') {
        emit({ type: 'toast', level: 'info', text: '已导出诊断：C:/Users/me/AppData/Local/NeuralReflex/diagnostics/diag-20261026-140530.json' });
      } else if (msg.cmd === 'rescanPlugins') {
        emit(plugins);
      } else if (msg.cmd === 'refreshTargets') {
        emit({ type: 'targets', windows, monitors });
      } else if (msg.cmd === 'setSettings') {
        Object.assign(settings, msg.settings);
        emit(stateMsg());
      } else if (msg.cmd === 'stop') {
        phase = 'idle';
        emit(stateMsg());
      } else if (msg.cmd === 'start') {
        phase = 'running';
        emit(stateMsg());
      } else if (msg.cmd === 'selectWindow') {
        const w = windows.find((x) => x.hwnd === msg.hwnd);
        if (w) target = { kind: 'window', ...w, alive: true };
        emit(stateMsg());
      } else if (msg.cmd === 'selectMonitor') {
        target = { kind: 'monitor', ...monitors[msg.index] };
        emit(stateMsg());
      }
    },
  };
};
