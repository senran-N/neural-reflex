// Neural Reflex 界面：状态全部来自宿主程序（WebView2 消息），界面只负责呈现和发送命令。
// 在普通浏览器中打开时没有宿主，改用 sample-data.js 的示例会话，供设计检查使用。
(() => {
  'use strict';

  const FRAME_HZ = 144;
  const FRAME_MS = 1000 / FRAME_HZ;  // 6.94 ms
  const HISTORY = 60;

  const $ = (id) => document.getElementById(id);
  const nf = (d) => new Intl.NumberFormat('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d });
  const ms = (v, d = 2) => (v == null || !isFinite(v) ? '—' : nf(d).format(v));
  const int = (v) => new Intl.NumberFormat('zh-CN').format(Math.round(v));
  const mb = (bytes) => int(bytes / 1048576);
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const withUnit = (value, unit) => `${value}<span class="unit">${unit}</span>`;

  const state = {
    phase: 'idle', error: '', target: { kind: 'none' }, model: null, gpu: '',
    settings: { conf: 0.25, iou: 0.7, mode: 'wgc', overlay: true, cudaGraph: true, gpuRealtime: false, publishResults: false },
    windows: [], monitors: [], models: [], stats: null, history: [],
  };

  // ---------- 与宿主通信 ----------
  const host = window.chrome && window.chrome.webview;
  let preview = null;
  const send = (cmd, extra = {}) => {
    const msg = { cmd, ...extra };
    if (host) host.postMessage(JSON.stringify(msg));
    else if (preview) preview.handle(msg);
  };

  function onMessage(msg) {
    switch (msg.type) {
      case 'state':
        Object.assign(state, { phase: msg.phase, error: msg.error, target: msg.target, model: msg.model, settings: msg.settings, gpu: msg.gpu });
        break;
      case 'targets': state.windows = msg.windows; state.monitors = msg.monitors; break;
      case 'models': state.models = msg.items; break;
      case 'statsReset': state.history = []; state.stats = null; break;
      case 'stats':
        state.stats = msg;
        state.phase = msg.phase;
        state.history.push(...msg.seconds);
        if (state.history.length > HISTORY) state.history.splice(0, state.history.length - HISTORY);
        break;
      case 'toast': toast(msg.text); break;
    }
    render();
  }

  // ---------- 路由：每个导航项是一个独立视图 ----------
  const views = ['overview', 'targets', 'models', 'settings'];
  function route() {
    const name = (location.hash.match(/^#\/(\w+)/) || [])[1];
    const view = views.includes(name) ? name : 'overview';
    views.forEach((v) => { $(`view-${v}`).hidden = v !== view; });
    document.querySelectorAll('.rail a[data-view]').forEach((a) => {
      if (a.dataset.view === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (view === 'targets') send('refreshTargets');
    render();
  }

  // ---------- 概览 ----------
  const phaseText = { idle: '已停止', preparing: '准备中', running: '运行中', failed: '失败' };
  const modeText = (m, kind) => (m === 'dd' ? 'DXGI 桌面复制' : kind === 'monitor' ? 'WGC 显示器' : 'WGC 窗口');

  function targetName() {
    const t = state.target;
    if (t.kind === 'window') return t.title;
    if (t.kind === 'monitor') return `显示器 ${t.index + 1}${t.primary ? '（主）' : ''}`;
    return '未选择采集目标';
  }

  function renderOverview() {
    const t = state.target, st = state.stats, running = state.phase === 'running';
    $('ov-title').textContent = targetName();
    const icon = $('ov-icon');
    icon.hidden = !(t.kind === 'window' && t.icon);
    if (!icon.hidden) icon.src = t.icon;

    // 标题下一句：本屏的发现，相对数字都写明基准
    let finding;
    if (state.phase === 'failed') finding = `会话失败：${esc(state.error)}`;
    else if (state.phase === 'preparing') finding = '正在准备设备、引擎与采集。首次使用一个模型时需要编译引擎，约需数十秒。';
    else if (running && st && st.latency.n) {
      const share = st.latency.p95 / FRAME_MS * 100;
      finding = `P95 <strong>${ms(st.latency.p95)} ms</strong>，占 ${FRAME_HZ} Hz 帧预算（${ms(FRAME_MS)} ms）的 ${nf(0).format(share)}%；有效检测 ${nf(1).format(lastSecond()?.dps ?? 0)} 次/秒。`;
    } else if (t.kind === 'none') finding = '选择一个窗口或显示器，再开始检测。';
    else if (!state.model) finding = '尚未加载模型。';
    else finding = `就绪：${esc(state.model.name)} · ${modeText(state.settings.mode, t.kind)}`;
    $('ov-finding').innerHTML = finding;

    const pill = $('ov-phase');
    pill.className = `pill ${state.phase}`;
    pill.textContent = phaseText[state.phase] || state.phase;
    const toggle = $('ov-toggle');
    const active = state.phase === 'running' || state.phase === 'preparing';
    toggle.textContent = active ? '停止' : '开始检测';
    toggle.disabled = !active && (t.kind === 'none' || !state.model);

    // 指标
    const last = lastSecond();
    const has = st && st.latency.n > 0;
    $('k-p95').innerHTML = has ? withUnit(ms(st.latency.p95), 'ms') : '—';
    $('k-p50').innerHTML = has ? withUnit(ms(st.latency.p50), 'ms') : '—';
    $('k-p99').innerHTML = has ? withUnit(ms(st.latency.p99), 'ms') : '—';
    $('k-dps').innerHTML = last ? withUnit(nf(1).format(last.dps), '次/秒') : '—';
    $('k-total').innerHTML = has ? withUnit(int(st.detections), '次') : '—';
    $('lat-window').textContent = state.history.length ? `最近 ${state.history.length} 秒，每秒 P95` : '每秒 P95';
    renderLatencyChart();
    renderBudget();
    renderTargetCard();
    renderModelCard();
    renderClasses();
    renderGpu();
  }

  const lastSecond = () => state.history[state.history.length - 1];
  const clock = (t) => { const r = Math.round(t); return `${Math.floor(r / 60)}:${String(r % 60).padStart(2, '0')}`; };  // taste-skill-ignore: 时钟格式

  function renderLatencyChart() {
    const el = $('lat-chart');
    const h = state.history;
    if (h.length < 2) {
      el.innerHTML = `<div class="chart-empty">${state.phase === 'preparing' ? '正在准备…' : '开始检测后，每秒记录一个 P95。'}</div>`;
      return;
    }
    Charts.line(el, {
      labels: h.map((s) => clock(s.t)),
      series: [{ name: 'P95', values: h.map((s) => s.p95) }],
      reference: { value: FRAME_MS, label: `${FRAME_HZ} Hz 帧预算 ${ms(FRAME_MS)} ms` },
      format: (v) => `${ms(v)} ms`,
      tickFormat: (v) => nf(0).format(v),
      xAt: h.map((s, i) => i).filter((i) => Math.round(h[i].t) % 10 === 0),
      height: 210,
    });
  }

  function renderBudget() {
    const st = state.stats;
    const segs = { pre: 0, infer: 0, post: 0, host: 0 };
    let total = 0, p95 = null;
    if (st && st.latency.n) {
      segs.pre = st.pre.p50; segs.infer = st.infer.p50; segs.post = st.post.p50;
      total = st.latency.p50;
      segs.host = Math.max(0, total - segs.pre - segs.infer - segs.post);
      p95 = st.latency.p95;
    }
    const scale = (v) => Math.min(100, (v / FRAME_MS) * 100);
    let left = 0;
    for (const key of ['pre', 'infer', 'post', 'host']) {
      const seg = document.querySelector(`#budget-track .seg.${key}`);
      seg.style.left = `${scale(left)}%`;
      seg.style.width = `${scale(segs[key])}%`;
      left += segs[key];
    }
    const tick = document.querySelector('#budget-track .tick');
    tick.hidden = p95 == null;
    if (p95 != null) {
      tick.style.left = `calc(${scale(p95)}% - 1px)`;
      tick.style.background = p95 > FRAME_MS ? 'var(--negative)' : 'var(--text)';
    }
    const remaining = FRAME_MS - total;
    $('budget-left').innerHTML = total ? withUnit((remaining < 0 ? '−' : '') + ms(Math.abs(remaining)), 'ms') : '—';
    $('budget-left-label').textContent = remaining < 0 ? '超出帧预算（按 P50）' : `余量（按 P50；竖线为 P95）`;
    $('budget-mid').textContent = `${ms(FRAME_MS / 2)} ms`;
    $('budget-end').textContent = `${ms(FRAME_MS)} ms`;

    const rows = [
      ['pre', '预处理', segs.pre, '#d8f7e6'], ['infer', '推理', segs.infer, 'var(--accent)'],
      ['post', '后处理与回读', segs.post, '#9ee9c0'], ['host', '复制、同步与调度', segs.host, '#4a4a4e'],
    ];
    const pct = (v) => (total ? `${nf(1).format((v / FRAME_MS) * 100)}%` : '—');
    $('stage-rows').innerHTML = rows.map(([, name, v, colour]) =>
      `<tr><td><span class="key" style="background:${colour}"></span>${name}</td><td class="num">${total ? ms(v, 3) : '—'}</td><td class="num">${pct(v)}</td></tr>`).join('') +
      `<tr class="total"><td>合计（P50）</td><td class="num">${total ? ms(total, 3) : '—'}</td><td class="num">${pct(total)}</td></tr>`;
  }

  function renderTargetCard() {
    const t = state.target;
    const obj = $('tg-object');
    const kv = [];
    if (t.kind === 'window') {
      obj.innerHTML = `${t.icon ? `<img src="${esc(t.icon)}" alt="">` : '<span class="ph" aria-hidden="true">—</span>'}<div style="min-width:0"><div class="name">${esc(t.title)}</div><div class="sub">${esc(t.exe)}</div></div>`;
      kv.push(['PID', int(t.pid)], ['客户区', `${int(t.width)} × ${int(t.height)}`], ['采集', modeText(state.settings.mode, t.kind)],
              ['状态', t.alive === false ? '窗口已关闭' : '可采集']);
    } else if (t.kind === 'monitor') {
      obj.innerHTML = `<span class="ph" aria-hidden="true">▭</span><div style="min-width:0"><div class="name">${esc(targetName())}</div><div class="sub">整个画面</div></div>`;
      kv.push(['分辨率', `${int(t.width)} × ${int(t.height)}`], ['刷新率', `${int(t.refreshHz)} Hz`], ['采集', modeText(state.settings.mode, t.kind)],
              ['主显示器', t.primary ? '是' : '否']);
    } else {
      obj.innerHTML = '<span class="ph" aria-hidden="true">—</span><div style="min-width:0"><div class="name">未选择</div><div class="sub">窗口或整个显示器</div></div>';
      kv.push(['PID', '—'], ['客户区', '—'], ['采集', '—'], ['状态', '—']);
    }
    $('tg-kv').innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
  }

  // 解码器说明（扩展机制方案 4.4）；清单未声明解码器时退回导出脚本记录的检测头说明
  function headLabel(m, short) {
    switch (m.decoder) {
      case 'yolo-anchors': return short ? '一对多 + GPU NMS' : '一对多检测头 · GPU NMS';
      case 'yolo-end2end': return short ? '端到端（无 NMS）' : '端到端检测头 · 无 NMS';
      case 'yolo-pose-anchors': return `姿态 · ${m.keypoints} 个关键点 · GPU NMS`;
      default: return m.head && m.head.includes('NMS') ? (short ? '一对多 + GPU NMS' : '一对多检测头 · GPU NMS') : m.head || '—';
    }
  }

  function renderModelCard() {
    const m = state.model;
    $('md-name').textContent = m ? m.name : '未加载';
    $('md-sub').textContent = m ? headLabel(m, false) : '—';
    const classes = m ? Object.keys(m.classes || {}).length : 0;
    const kv = m
      ? [['精度', m.precision || '—'], ['输入', m.inputSize ? `${m.inputSize} × ${m.inputSize}` : '—'], ['类别', classes ? `${classes} 类` : '—'],
         ['引擎', m.engineCached ? '已缓存' : '首次运行时编译']]
      : [['精度', '—'], ['输入', '—'], ['类别', '—'], ['引擎', '—']];
    $('md-kv').innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
  }

  function renderClasses() {
    const list = (state.stats && state.stats.classes) || [];
    const sorted = [...list].sort((a, b) => b.count - a.count || a.id - b.id);
    const shown = sorted.slice(0, 5);
    const boxes = list.reduce((s, c) => s + c.count, 0);
    $('cls-total').textContent = state.stats ? `${list.length} 类` : '—';
    $('cls-sum').textContent = state.stats ? `${boxes} 个框` : '—';
    const top = shown.length ? shown[0].count : 1;
    const rows = shown.map((c) => `<li><span>${esc(c.name)}</span><span class="num">${c.count}</span><span class="bar" style="width:${(c.count / top) * 100}%"></span></li>`);
    if (!shown.length) rows.push(`<li class="blank"><span>${state.phase === 'running' ? '这一帧没有检出目标' : '开始检测后显示每类的框数'}</span><span></span></li>`);
    if (sorted.length > 5) rows[4] = `<li class="blank"><span>其余 ${sorted.length - 4} 类</span><span class="num">${sorted.slice(4).reduce((s, c) => s + c.count, 0)}</span></li>`;
    while (rows.length < 5) rows.push('<li class="blank"><span>—</span><span></span></li>');
    $('cls-list').innerHTML = rows.join('');
  }

  function renderGpu() {
    const st = state.stats;
    $('gpu-name').textContent = (st && st.adapter) || state.gpu || '—';
    const used = st ? st.vramUsed : 0, budget = st ? st.vramBudget : 0;
    $('gpu-vram').innerHTML = budget ? withUnit(mb(used), 'MB') : '—';
    $('gpu-budget').innerHTML = budget ? withUnit(mb(budget), 'MB') : '—';
    $('gpu-meter').style.width = budget ? `${Math.min(100, (used / budget) * 100)}%` : '0%';
    $('gpu-share').textContent = budget ? `${nf(1).format((used / budget) * 100)}%` : '—';
    const last = lastSecond();
    $('gpu-overlay').innerHTML = !state.settings.overlay ? '已关闭' : last ? withUnit(nf(0).format(last.ops), '次/秒') : '已开启';
  }

  // ---------- 采集目标 ----------
  function renderTargets() {
    const q = $('tg-search').value.trim().toLowerCase();
    const t = state.target;
    const rows = state.windows.filter((w) => !q || w.title.toLowerCase().includes(q) || w.exe.toLowerCase().includes(q));
    $('tg-count').textContent = `${state.windows.length} 个窗口 · ${state.monitors.length} 个显示器${t.kind !== 'none' ? ` · 当前：${targetName()}` : ''}`;
    $('win-rows').innerHTML = rows.length ? rows.map((w) => {
      const selected = t.kind === 'window' && t.hwnd === w.hwnd;
      return `<tr data-hwnd="${esc(w.hwnd)}" aria-selected="${selected}">
        <td class="name"><span class="cell-object">${w.icon ? `<img src="${esc(w.icon)}" alt="">` : '<span class="ph"></span>'}<span>${esc(w.title)}</span></span></td>
        <td class="muted">${esc(w.exe)}</td>
        <td class="num muted hide-sm">${int(w.pid)}</td>
        <td class="num">${w.minimized ? '<span class="muted">已最小化</span>' : `${int(w.width)} × ${int(w.height)}`}</td>
        <td class="num">${selected ? '<span class="pill running">当前</span>' : `<button class="btn text" type="button" data-select-window="${esc(w.hwnd)}">选择</button>`}</td></tr>`;
    }).join('') : `<tr><td colspan="5" class="muted">${q ? '没有匹配的窗口' : '没有可采集的窗口'}</td></tr>`;
    $('mon-rows').innerHTML = state.monitors.map((m) => {
      const selected = t.kind === 'monitor' && t.index === m.index;
      return `<tr aria-selected="${selected}"><td>显示器 ${m.index + 1}${m.primary ? '（主）' : ''}</td><td class="num">${int(m.width)} × ${int(m.height)}</td><td class="num">${int(m.refreshHz)} Hz</td>
        <td class="num">${selected ? '<span class="pill running">当前</span>' : `<button class="btn text" type="button" data-select-monitor="${m.index}">选择</button>`}</td></tr>`;
    }).join('');
  }

  // ---------- 模型 ----------
  function renderModels() {
    const m = state.model;
    $('md-h1-sub').textContent = m ? `当前：${m.name} · ${m.precision || '精度未知'} · ${state.models.length} 个可用` : `${state.models.length} 个可用模型`;
    $('mdc-path').textContent = m ? m.path : '—';
    $('mdc-in').textContent = m && m.inputSize ? `1 × 3 × ${m.inputSize} × ${m.inputSize}` : '—';
    $('mdc-head').textContent = m ? headLabel(m, true) : '—';
    $('mdc-engine').textContent = m ? (m.engineCached ? '已缓存' : '首次运行时编译') : '—';
    const classes = m ? Object.entries(m.classes || {}).sort((a, b) => a[0] - b[0]) : [];
    $('mdc-classes-title').textContent = classes.length ? `类别（${classes.length}）` : '类别';
    $('mdc-classes').innerHTML = classes.length
      ? classes.map(([id, name]) => `<li><span class="id">${id}</span><span>${esc(name)}</span></li>`).join('')
      : '<li class="muted">没有找到类别名称（模型旁缺少导出时生成的 .json 契约文件）</li>';
    $('mdl-count').textContent = `${state.models.length} 个`;
    // 同名模型在名称下注明所在目录
    const nameCount = {};
    state.models.forEach((x) => { nameCount[x.name] = (nameCount[x.name] || 0) + 1; });
    const dirOf = (path) => path.replace(/[\/][^\/]*$/, '');
    const where = (x) => (nameCount[x.name] > 1 ? `<div class="muted" style="font-size:12px;line-height:16px;overflow:hidden;text-overflow:ellipsis">${esc(dirOf(x.path))}</div>` : '');
    $('mdl-rows').innerHTML = state.models.map((x) => {
      const selected = m && x.path === m.path;
      if (x.supported === false) {
        return `<tr><td class="name"><span>${esc(x.name)}</span><div class="muted" style="font-size:12px;line-height:16px">${esc(x.unsupported || '不支持')}</div></td>
        <td class="num muted">${esc(x.precision || '—')}</td><td class="num muted">${nf(1).format(x.sizeBytes / 1048576)}<span class="unit">MB</span></td></tr>`;
      }
      return `<tr data-model="${esc(x.path)}" aria-selected="${selected}" style="cursor:pointer"><td class="name"><span class="cell-object"><span>${esc(x.name)}</span></span>${where(x)}</td>
        <td class="num">${esc(x.precision || '—')}</td><td class="num">${nf(1).format(x.sizeBytes / 1048576)}<span class="unit">MB</span></td></tr>`;
    }).join('') || '<tr><td colspan="3" class="muted">models 目录中没有 .onnx 文件</td></tr>';
  }

  // ---------- 设置 ----------
  function renderSettings() {
    const s = state.settings;
    if (document.activeElement !== $('st-conf')) $('st-conf').value = s.conf;
    if (document.activeElement !== $('st-iou')) $('st-iou').value = s.iou;
    $('st-conf-v').textContent = nf(2).format($('st-conf').value);
    $('st-iou-v').textContent = nf(2).format($('st-iou').value);
    document.querySelectorAll('.segmented button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === s.mode)));
    $('st-overlay').setAttribute('aria-checked', String(s.overlay));
    $('st-graph').setAttribute('aria-checked', String(s.cudaGraph));
    $('st-realtime').setAttribute('aria-checked', String(!!s.gpuRealtime));
    $('st-publish').setAttribute('aria-checked', String(!!s.publishResults));
    $('st-sub').textContent = `置信度 ${nf(2).format(s.conf)} · IoU ${nf(2).format(s.iou)} · ${s.mode === 'dd' ? '桌面复制' : 'WGC'} · 覆盖层${s.overlay ? '开' : '关'} · CUDA Graph ${s.cudaGraph ? '开' : '关'}${s.gpuRealtime ? ' · GPU 实时优先级' : ''}${s.publishResults ? ' · 结果共享' : ''}`;
  }

  let savedTimer = 0;
  function saveSettings(patch) {
    Object.assign(state.settings, patch);
    send('setSettings', { settings: state.settings });
    $('st-saved').classList.add('show');
    clearTimeout(savedTimer);
    savedTimer = setTimeout(() => $('st-saved').classList.remove('show'), 1600);
    render();
  }

  // ---------- 提示 ----------
  let toastTimer = 0;
  function toast(text) {
    $('toast-text').textContent = text;
    $('toast').classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('show'), 6000);
  }

  // 统计每秒更新多次，只重绘可见的视图
  function render() {
    if (!$('view-overview').hidden) renderOverview();
    if (!$('view-targets').hidden) renderTargets();
    if (!$('view-models').hidden) renderModels();
    if (!$('view-settings').hidden) renderSettings();
  }

  // ---------- 事件 ----------
  $('ov-toggle').addEventListener('click', () => send(state.phase === 'running' || state.phase === 'preparing' ? 'stop' : 'start'));
  $('tg-refresh').addEventListener('click', () => send('refreshTargets'));
  $('tg-search').addEventListener('input', renderTargets);
  $('win-rows').addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-hwnd]');
    if (row) send('selectWindow', { hwnd: row.dataset.hwnd });
  });
  $('mon-rows').addEventListener('click', (e) => {
    const b = e.target.closest('[data-select-monitor]');
    if (b) send('selectMonitor', { index: Number(b.dataset.selectMonitor) });
  });
  $('md-browse').addEventListener('click', () => send('browseModel'));
  $('mdl-rows').addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-model]');
    if (row) send('selectModel', { path: row.dataset.model });
  });
  $('st-conf').addEventListener('input', () => { $('st-conf-v').textContent = nf(2).format($('st-conf').value); });
  $('st-iou').addEventListener('input', () => { $('st-iou-v').textContent = nf(2).format($('st-iou').value); });
  $('st-conf').addEventListener('change', () => saveSettings({ conf: Number($('st-conf').value) }));
  $('st-iou').addEventListener('change', () => saveSettings({ iou: Number($('st-iou').value) }));
  document.querySelectorAll('.segmented button').forEach((b) => b.addEventListener('click', () => saveSettings({ mode: b.dataset.mode })));
  $('st-overlay').addEventListener('click', () => saveSettings({ overlay: !state.settings.overlay }));
  $('st-graph').addEventListener('click', () => saveSettings({ cudaGraph: !state.settings.cudaGraph }));
  $('st-realtime').addEventListener('click', () => saveSettings({ gpuRealtime: !state.settings.gpuRealtime }));
  $('st-publish').addEventListener('click', () => saveSettings({ publishResults: !state.settings.publishResults }));
  window.addEventListener('hashchange', route);
  window.addEventListener('resize', () => { if (!$('view-overview').hidden) renderLatencyChart(); });

  // ---------- 启动 ----------
  if (host) {
    host.addEventListener('message', (e) => onMessage(e.data));
    route();
    send('hello');
  } else {
    const s = document.createElement('script');
    s.src = 'sample-data.js';
    s.onload = () => { preview = window.NR_PREVIEW(onMessage); route(); send('hello'); };
    document.body.appendChild(s);
  }
})();
