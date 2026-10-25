// 仅供在浏览器中预览界面：没有宿主程序时模拟一次运行中的会话。
// 数值取自 2026-10-26 在 RTX 4050 Laptop 上的实测（nr_detect，YOLO26n FP16，WGC 窗口采集，测试源 144 FPS）；
// 类别名来自参考模型的导出契约。真实程序中这些数据全部由宿主推送。
window.NR_PREVIEW = (emit) => {
  const classes = {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"};
  const model = {
    path: 'D:\Myproject\neural-reflex\models\reference\yolo26n-640-nms-fp16.onnx',
    name: 'yolo26n-640-nms-fp16', sizeBytes: 5079119, inputSize: 640, precision: 'FP16',
    head: 'one-to-many (needs NMS)', classes, engineCached: true,
  };
  const windows = [
    { hwnd: '1835362', title: 'NR Test Source - 144.0 FPS - 帧 8642', exe: 'nr_testsource.exe', pid: 41208, width: 1280, height: 720, minimized: false, icon: '' },
    { hwnd: '657958', title: 'performance-design.md - neural-reflex', exe: 'WindowsTerminal.exe', pid: 18844, width: 1704, height: 1043, minimized: false, icon: '' },
    { hwnd: '400812', title: 'docs', exe: 'explorer.exe', pid: 9124, width: 1180, height: 760, minimized: false, icon: '' },
    { hwnd: '1180244', title: 'TensorRT for RTX 文档 - Microsoft Edge', exe: 'msedge.exe', pid: 22716, width: 1536, height: 980, minimized: true, icon: '' },
  ];
  const monitors = [{ index: 0, width: 2560, height: 1600, refreshHz: 165, primary: true }];
  const settings = { conf: 0.25, iou: 0.7, mode: 'wgc', overlay: true, cudaGraph: true };
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

  return {
    handle(msg) {
      if (msg.cmd === 'hello') {
        emit(stateMsg());
        emit({ type: 'targets', windows, monitors });
        emit({ type: 'models', items: [
          { path: model.path, name: model.name, sizeBytes: model.sizeBytes, precision: 'FP16', inputSize: 640, classCount: 80, engineCached: true, supported: true },
          { path: model.path.replace('fp16', 'fp32'), name: 'yolo26n-640-nms-fp32', sizeBytes: 9930306, precision: 'FP32', inputSize: 640, classCount: 80, engineCached: true, supported: true },
          { path: model.path.replace('nms-fp16', 'e2e-fp32'), name: 'yolo26n-640-e2e-fp32', sizeBytes: 9942036, precision: 'FP32', inputSize: 640, classCount: 80, engineCached: false, supported: false },
        ] });
        emit(stats(seconds));
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
