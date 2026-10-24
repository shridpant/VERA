const $ = id => document.getElementById(id);
const status = $('status');
let selectedFile = null;
let uploadedPath = '';
let previewURL = '';
let selectedStages = [];
let activeJob = '';
let sessionToken = '';
let serverDefaults = {};
// Every change of input bumps this, so a late lucky or upload reply cannot replace a newer choice.
let selection = 0;

function say(message, error = false) {
  status.textContent = message;
  status.classList.toggle('error', error);
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const artifactURL = (job, path) => '/api/artifact?job=' + encodeURIComponent(job) + '&file=' + encodeURIComponent(path);
const gigabytes = bytes => (bytes / 1e9).toFixed(1) + ' GB';
const joinPath = (base, path) => (base ? base + '/' + path : path);
const labelText = label => (label === null ? 'unparsed' : label === undefined ? '…' : String(label));

function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgElement(name, attributes, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- Video input ---------- */

function setFile(file) {
  if (!file) return;
  selection += 1;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $('preview').src = previewURL;
  $('preview').classList.remove('hidden');
  $('chosen').textContent = file.name;
  $('path').value = '';
  uploadedPath = '';
  selectedFile = file;
}

const drop = $('drop');
drop.addEventListener('click', event => {
  if (event.target.tagName !== 'VIDEO') $('file').click();
});
drop.addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    $('file').click();
  }
});
$('file').addEventListener('change', event => setFile(event.target.files[0]));
drop.addEventListener('dragover', event => {
  event.preventDefault();
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', event => {
  event.preventDefault();
  drop.classList.remove('over');
  setFile(event.dataTransfer.files[0]);
});
$('path').addEventListener('input', () => {
  selection += 1;
  if ($('path').value.trim()) {
    selectedFile = null;
    uploadedPath = '';
    $('chosen').textContent = 'Using the path below';
  }
});

$('lucky').addEventListener('click', async () => {
  const button = $('lucky');
  button.disabled = true;
  const ticket = ++selection;
  say('Choosing a random sample…');
  button.classList.add('rolling');
  const faces = '⚀⚁⚂⚃⚄⚅';
  const tumble = setInterval(() => { $('die').textContent = faces[Math.floor(Math.random() * 6)]; }, 90);
  try {
    const response = await fetch('/api/lucky?target=' + encodeURIComponent(currentTarget()));
    const choice = await response.json();
    if (!response.ok) throw new Error(choice.error);
    if (ticket !== selection) return;
    selectedFile = null;
    uploadedPath = '';
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = '';
    $('preview').classList.add('hidden');
    $('path').value = choice.path;
    $('anchor').value = choice.anchor_seconds;
    $('feature').value = choice.feature_file || '';
    if (choice.annotation_file) $('annotation').value = choice.annotation_file;
    $('chosen').textContent = 'Random sample: ' + choice.name;
    say('Random sample selected. Review the settings, then run.');
  } catch (error) {
    say(error.message, true);
  } finally {
    clearInterval(tumble);
    button.disabled = false;
    button.classList.remove('rolling');
  }
});

async function upload(file) {
  if (file.size > 200 * 1024 * 1024) throw new Error('Video exceeds the 200 MB upload limit');
  say('Uploading ' + file.name + '…');
  const response = await fetch('/api/upload', {
    method: 'POST',
    headers: {'X-Filename': file.name, 'Content-Type': 'application/octet-stream', 'X-VERA-Token': sessionToken},
    body: file
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Upload failed');
  return result.path;
}

/* ---------- Simple and advanced settings ---------- */

let advanced = false;
let scope = 'window';
try { advanced = localStorage.getItem('vera-lab-advanced') === '1'; } catch (error) {}

// Pipeline stages each run target executes; the diagram dims the rest.
const TARGET_STAGES = {
  window: ['input', 'preprocess', 'infer'],
  preprocess: ['input', 'preprocess'],
  segments: ['input', 'preprocess', 'infer'],
  refine: ['input', 'preprocess', 'infer', 'refine'],
  evaluate: ['input', 'preprocess', 'infer', 'refine', 'evaluate'],
  full: ['input', 'preprocess', 'infer', 'refine', 'evaluate'],
};

const currentTarget = () => (advanced ? $('target').value : scope);
const boxesWanted = () => (advanced ? $('boxesOn').checked : true);

function updateTargetFields() {
  const target = currentTarget();
  $('anchorField').classList.toggle('hidden', target !== 'window');
  $('featureField').classList.toggle('hidden', !['refine', 'evaluate'].includes(target));
  $('annotationField').classList.toggle('hidden', target !== 'evaluate');
  $('run').textContent = target === 'window' ? 'Run one window' : target === 'full' ? 'Run the whole video' : 'Run through ' + target;
  $('windowsNode').textContent = target === 'window' ? 'One window' : 'All windows';
  const stages = TARGET_STAGES[target].concat(boxesWanted() && TARGET_STAGES[target].includes('infer') ? ['boxes'] : []);
  for (const node of document.querySelectorAll('.pnode')) {
    const on = stages.includes(node.dataset.stage);
    node.classList.toggle('on', on);
    node.classList.toggle('off', !on);
  }
  for (const tag of document.querySelectorAll('.knob[data-only]')) tag.classList.toggle('hidden', tag.dataset.only !== target);
  $('pipelineNote').textContent = target === 'full'
    ? 'Whole video: every window, then refinement and metrics when the lab has this video\'s feature file and annotation; otherwise it stops after the labels and boxes.'
    : 'Highlighted stages run. The model never receives the frames between the 8 images, or audio.';
}

function setMode(isAdvanced) {
  advanced = isAdvanced;
  document.body.classList.toggle('advanced', advanced);
  $('simpleMode').setAttribute('aria-pressed', String(!advanced));
  $('advancedMode').setAttribute('aria-pressed', String(advanced));
  try { localStorage.setItem('vera-lab-advanced', advanced ? '1' : '0'); } catch (error) {}
  updateTargetFields();
}
$('simpleMode').addEventListener('click', () => setMode(false));
$('advancedMode').addEventListener('click', () => setMode(true));
for (const button of document.querySelectorAll('[data-scope]')) {
  button.addEventListener('click', () => {
    scope = button.dataset.scope;
    for (const other of document.querySelectorAll('[data-scope]')) other.setAttribute('aria-pressed', String(other === button));
    updateTargetFields();
  });
}
$('target').addEventListener('change', updateTargetFields);
$('boxesOn').addEventListener('change', updateTargetFields);
$('device').addEventListener('change', () => {
  $('dtype').value = {auto: 'auto', mps: 'float16', cpu: 'float32', cuda: 'bfloat16'}[$('device').value];
});

for (const node of document.querySelectorAll('.pnode')) {
  node.addEventListener('click', () => {
    const control = $(node.dataset.focus);
    if (!control) return;
    if (control.closest('.adv') && !advanced) setMode(true);
    const field = control.closest('.field') || control;
    if (field.classList.contains('hidden') && TARGET_STAGES[node.dataset.stage]) {
      $('target').value = node.dataset.stage;
      updateTargetFields();
    }
    if (control.tagName === 'DETAILS') control.open = true;
    field.scrollIntoView({behavior: 'smooth', block: 'center'});
    control.focus({preventScroll: true});
    field.classList.remove('flash');
    void field.offsetWidth;
    field.classList.add('flash');
  });
}

// VERA's original settings: released questions, pinned InternVL2-8B, device-matched precision.
function resetKnobs() {
  selection += 1;
  selectedFile = null;
  uploadedPath = '';
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = '';
  $('preview').classList.add('hidden');
  $('file').value = '';
  $('chosen').textContent = 'No video selected';
  for (const [id, value] of Object.entries({
    target: 'window', anchor: '5', device: 'auto', dtype: 'auto', path: '', questions: '', feature: '', boxPhrase: '',
    model: serverDefaults.model_path || 'OpenGVLab/InternVL2-8B', annotation: serverDefaults.annotation_file || ''})) {
    $(id).value = value;
  }
  $('download').checked = false;
  $('boxesOn').checked = true;
  scope = 'window';
  for (const button of document.querySelectorAll('[data-scope]')) button.setAttribute('aria-pressed', String(button.dataset.scope === 'window'));
  updateTargetFields();
  refreshPrompt();
}
$('reset').addEventListener('click', () => {
  resetKnobs();
  say("Reset to VERA's original settings. Choose a video to run.");
});

/* ---------- Exact prompt ---------- */

let promptTimer = null;
async function refreshPrompt() {
  const questions = advanced ? $('questions').value : '';
  try {
    const response = await fetch('/api/prompt?questions=' + encodeURIComponent(questions));
    const rendered = await response.json();
    if (!response.ok) throw new Error(rendered.error);
    $('promptText').textContent = rendered.prompt;
    const generation = rendered.generation;
    $('promptMeta').textContent = '· ' + rendered.prompt.length + ' characters · ' + rendered.questions +
      ' · ' + (generation.do_sample ? 'sampling' : 'greedy') + ', up to ' + generation.max_new_tokens + ' new tokens';
    $('namePrompt').textContent = rendered.name_prompt;
    $('groundPrompt').textContent = rendered.ground_prompt.replace('{}', '<phrase>');
  } catch (error) {
    $('promptText').textContent = 'Could not render the prompt: ' + error.message;
  }
}
$('questions').addEventListener('input', () => {
  clearTimeout(promptTimer);
  promptTimer = setTimeout(refreshPrompt, 350);
});

/* ---------- Server defaults, warm model and usage ---------- */

function describeWarmModel(warm) {
  $('warm').textContent = warm?.loaded
    ? 'Model loaded on ' + warm.device + (warm.dtype ? ' (' + warm.dtype + ')' : '') + ': the next window skips loading.'
    : 'No model loaded: the first window loads the checkpoint.';
}

function meter(label, fraction, text) {
  const box = element('div', {class: 'meter'});
  box.append(element('span', {}, label), element('b', {}, text));
  if (fraction !== null) {
    const bar = element('i');
    bar.style.setProperty('--fill', (Math.min(1, Math.max(0, fraction)) * 100).toFixed(1) + '%');
    box.append(bar);
  }
  return box;
}

// Where the lab server runs, which process holds the model, and what it and the GPU use.
async function refreshUsage() {
  try {
    const response = await fetch('/api/usage');
    if (!response.ok) return;
    const usage = await response.json();
    $('usageHead').replaceChildren('Lab server ', element('b', {}, usage.host),
      usage.role ? ' · ' + usage.role : ' · no model loaded', usage.system === 'Darwin' ? ' · this is a Mac' : '');
    const meters = [];
    for (const gpu of usage.gpus) {
      const memory = gpu.unified_memory ? 'GPU-allocated unified memory' : 'GPU memory';
      meters.push(meter(gpu.name + ' busy', gpu.utilization_percent / 100, gpu.utilization_percent.toFixed(0) + ' %'));
      if (gpu.memory_used_bytes !== null) {
        meters.push(meter(memory, gpu.memory_used_bytes / gpu.memory_total_bytes,
          gigabytes(gpu.memory_used_bytes) + ' / ' + gigabytes(gpu.memory_total_bytes)));
      }
      if (gpu.process_memory_bytes) meters.push(meter('of which this model', null, gigabytes(gpu.process_memory_bytes)));
    }
    if (usage.process) {
      meters.push(meter('Model process CPU', null, usage.process.cpu_percent.toFixed(0) + ' %'));
      meters.push(meter('Model process memory', null, gigabytes(usage.process.memory_bytes)));
    }
    if (!usage.gpus.length) meters.push(meter('GPU', null, 'none detected; CPU only'));
    $('meters').replaceChildren(...meters);
    $('usage').classList.remove('hidden');
  } catch (error) {
    // Usage is informational; a missed sample is simply retried.
  }
}
refreshUsage();
setInterval(() => { if (document.visibilityState === 'visible') refreshUsage(); }, 4000);

/* ---------- Live view: one entry per window, filled from timeline events ---------- */

const live = {
  job: '', after: {run: 0, manual: 0}, windows: [], byKey: new Map(), current: -1, fps: null, frames: null,
  total: 0, events: 0, dirty: false, follow: true, playing: null, picked: new Set(), scores: null, boxing: false,
};

function resetLive(job) {
  clearInterval(live.playing);
  Object.assign(live, {job, after: {run: 0, manual: 0}, windows: [], byKey: new Map(), current: -1, fps: null,
    frames: null, total: 0, events: 0, dirty: true, playing: null, picked: new Set(), scores: null, boxing: false,
    scorePath: null, single: false});
  $('playWindows').textContent = '▶ Play';
  $('eventLog').replaceChildren();
  $('scoreTrack').classList.add('hidden');
  for (const node of document.querySelectorAll('.pnode')) {
    node.classList.remove('now');
    node.querySelector('.live').textContent = '';
  }
}

function markNode(name, text) {
  for (const node of document.querySelectorAll('.pnode')) node.classList.toggle('now', node.dataset.node === name);
  const node = document.querySelector(`.pnode[data-node="${name}"]`);
  if (node && text !== undefined) node.querySelector('.live').textContent = text;
}

function windowFor(event) {
  const key = event.anchor + ':' + (event.condition || 'baseline');
  let row = live.byKey.get(key);
  if (!row) {
    row = {anchor: event.anchor, condition: event.condition || 'baseline', images: [], frame_indices: [], boxes: [],
           raw_response: '', strict_label: undefined, partial: false};
    live.byKey.set(key, row);
    live.windows.push(row);
  }
  return row;
}

function touched(row) {
  live.dirty = true;
  if (live.follow) live.current = live.windows.indexOf(row);
}

function logEvent(event) {
  const time = new Date(event.t * 1000).toLocaleTimeString();
  const details = Object.entries(event).filter(([key]) => !['t', 'event', 'images', 'text'].includes(key))
    .map(([key, value]) => key + '=' + (typeof value === 'object' ? JSON.stringify(value) : value)).join(' ');
  const text = event.text ? ' text=' + JSON.stringify(event.text.length > 160 ? event.text.slice(0, 160) + '…' : event.text) : '';
  $('eventLog').append(element('li', {}, time + '  ' + event.event + '  ' + details + text));
}

function applyEvent(event, base) {
  live.events += 1;
  if (event.event !== 'generating') logEvent(event);
  const where = event.anchor !== undefined ? ' · anchor ' + event.anchor : '';
  switch (event.event) {
    case 'run':
      live.single = event.stages.includes('window');
      markNode('video', 'starting');
      break;
    case 'video':
      live.total = event.windows;
      live.fps = event.fps;
      live.frames = event.frames;
      markNode('video', event.frames + ' frames · ' + event.fps.toFixed(2) + ' fps');
      live.dirty = true;
      break;
    case 'model_loading':
      markNode('model', 'loading the checkpoint');
      break;
    case 'model_ready':
      markNode('model', 'ready on ' + event.device + ' ' + (event.dtype || '') + (event.flash_attention ? ' · FlashAttention' : '') +
        ' · loaded in ' + event.seconds.toFixed(1) + ' s');
      break;
    case 'model_reused':
      markNode('model', 'already loaded on ' + event.device);
      break;
    case 'prompt':
      markNode('prompt', event.text.length + ' characters, every window');
      break;
    case 'model_input':
      $('modelInputText').textContent = event.text;
      $('modelInputPanel').classList.remove('hidden');
      break;
    case 'window': {
      const row = windowFor(event);
      Object.assign(row, {index: event.index, start: event.start, end: event.end, frame_indices: event.frame_indices,
                          images: event.images.map(path => joinPath(base, path))});
      live.total = event.total;
      markNode('images', 'window ' + (event.index + 1) + ' of ' + event.total);
      touched(row);
      break;
    }
    case 'generating': {
      const row = windowFor(event);
      Object.assign(row, {raw_response: event.text, partial: true});
      markNode('model', 'generating' + where + ' · ' + event.tokens + ' tokens so far');
      touched(row);
      break;
    }
    case 'answer': {
      const row = windowFor(event);
      Object.assign(row, {raw_response: event.text, partial: false, strict_label: event.label, seconds: event.seconds});
      document.querySelector('.pnode[data-node="model"] .live').textContent = 'answered in ' + event.seconds.toFixed(1) + ' s' + where;
      markNode('answer', 'label ' + labelText(event.label) + where);
      touched(row);
      break;
    }
    case 'naming': {
      const row = windowFor(event);
      row.boxes.push({mode: 'auto', phrase: event.phrase, naming: event, frames: []});
      markNode('boxes', 'named “' + event.phrase + '”' + where);
      touched(row);
      break;
    }
    case 'box': {
      const row = windowFor(event);
      const mode = event.mode || 'auto';
      let record = row.boxes.findLast(item => item.mode === mode && item.phrase === event.phrase);
      if (!record) {
        record = {mode, phrase: event.phrase, frames: []};
        row.boxes.push(record);
      }
      record.frames = record.frames.filter(frame => frame.slot !== event.slot).concat([event]);
      markNode('boxes', 'image ' + (event.slot + 1) + ': ' + event.status + where);
      touched(row);
      break;
    }
    case 'manual_boxes':
      markNode('boxes', 'your phrase “' + event.phrase + '” on ' + event.frames + ' frames');
      break;
    case 'refine':
      markNode('refine', 'scene, temporal and frame scores saved');
      live.scorePath = event.arrays.find(path => path.endsWith('frame_scores.npy'));
      if (live.scorePath) live.scorePath = joinPath(base, live.scorePath);
      break;
    case 'metrics':
      markNode('metrics', 'ROC-AUC ' + event.frame_roc_auc.toFixed(4) + ' · AP ' + event.frame_average_precision.toFixed(4));
      break;
    case 'error':
      say('The model process reported: ' + event.message, true);
      break;
    case 'done':
      for (const node of document.querySelectorAll('.pnode')) node.classList.remove('now');
      break;
    default:
      break;
  }
}

async function pullTimeline(job, skip = '') {
  const query = '?job=' + encodeURIComponent(job) + '&after_run=' + live.after.run + '&after_manual=' + live.after.manual +
    (skip ? '&skip=' + skip : '');
  const response = await fetch('/api/timeline' + query);
  if (!response.ok) return;
  const timeline = await response.json();
  for (const name of ['run', 'manual']) {
    for (const event of timeline.events[name]) applyEvent(event, timeline.base[name]);
    live.after[name] = timeline.next[name];
  }
  $('eventCount').textContent = '(' + $('eventLog').children.length + ')';
  if (live.dirty) renderLive();
}

// After a run, the saved files are authoritative; they replace what the events built.
async function loadWindows(job) {
  const response = await fetch('/api/windows?job=' + encodeURIComponent(job));
  const saved = await response.json();
  if (!response.ok || !saved.windows.length) return saved;
  const anchor = live.windows[live.current]?.anchor;
  live.windows = saved.windows;
  live.byKey = new Map(saved.windows.map(row => [row.anchor + ':' + row.condition, row]));
  live.fps = saved.fps || live.fps;
  live.single = saved.mode === 'window';
  if (!live.single) live.total = saved.windows.length;
  const kept = live.windows.findIndex(row => row.anchor === anchor);
  const firstAlert = live.windows.findIndex(row => row.strict_label === 1);
  live.current = kept >= 0 ? kept : (firstAlert >= 0 ? firstAlert : 0);
  live.dirty = true;
  renderLive();
  return saved;
}

function renderStrip() {
  const strip = $('windowStrip');
  const cells = [];
  // A one-window run shows that window only; a video run shows every window, reached or not.
  const count = live.single ? live.windows.length : Math.max(live.total, live.windows.length);
  for (let index = 0; index < count; index += 1) {
    const row = live.windows[index];
    const cell = element('button', {type: 'button'});
    if (!row) {
      cell.className = 'pending';
      cell.disabled = true;
      cell.title = 'Not reached yet';
    } else {
      cell.className = row.strict_label === 1 ? 'pos' : row.strict_label === null ? 'bad' : row.strict_label === 0 ? '' : 'pending';
      if (row.boxes.length) cell.classList.add('boxed');
      if (index === live.current) cell.classList.add('sel');
      cell.title = 'Anchor ' + row.anchor + ' · label ' + labelText(row.strict_label) + (row.boxes.length ? ' · boxed' : '');
      cell.setAttribute('aria-label', 'Window ' + (index + 1) + ', label ' + labelText(row.strict_label));
      cell.addEventListener('click', () => {
        live.follow = false;
        $('followLive').checked = false;
        live.current = index;
        renderLive();
      });
    }
    cells.push(cell);
  }
  strip.replaceChildren(...cells);
}

function boxOverlay(row, slot) {
  const svg = svgElement('svg', {viewBox: '0 0 1000 1000', preserveAspectRatio: 'none', 'aria-hidden': 'true'});
  for (const record of row.boxes) {
    const frame = record.frames.find(item => item.slot === slot);
    if (!frame) continue;
    for (const [index, box] of frame.boxes.entries()) {
      const [x1, y1, x2, y2] = box;
      svg.append(svgElement('rect', {x: x1, y: y1, width: x2 - x1, height: y2 - y1,
        class: 'box ' + record.mode + (frame.status === 'full_frame' && index === 0 ? ' whole' : '')}));
    }
  }
  return svg;
}

function frameBadges(row, slot) {
  const badges = element('div', {class: 'badges'});
  for (const record of row.boxes) {
    const frame = record.frames.find(item => item.slot === slot);
    if (!frame) continue;
    const text = frame.status === 'box' ? 'box ' + Math.round(frame.area_fraction * 100) + ' % of image'
      : frame.status === 'full_frame' ? 'whole image: not a location' : frame.status === 'no_box' ? 'no box' : 'unreadable box';
    badges.append(element('span', {class: 'badge ' + record.mode + ' ' + frame.status}, (record.mode === 'manual' ? 'you: ' : '') + text));
  }
  return badges;
}

function renderFrames(row) {
  const fps = live.fps;
  const figures = row.images.map((path, slot) => {
    const key = row.anchor + ':' + slot;
    const figure = element('figure', {class: 'frame-fig' + (live.picked.has(key) ? ' picked' : '')});
    const holder = element('div', {class: 'frame-img'});
    const image = element('img', {src: artifactURL(live.job, path), alt: 'Model input image ' + (slot + 1) + ' of window ' + row.anchor});
    holder.append(image, boxOverlay(row, slot));
    const frame = row.frame_indices[slot];
    const caption = element('figcaption', {}, 'Image ' + (slot + 1) + ' · frame ' + frame + (fps ? ' · ' + (frame / fps).toFixed(2) + ' s' : ''));
    figure.append(holder, caption, frameBadges(row, slot));
    figure.addEventListener('click', () => {
      if (!$('manualMode').checked) return;
      if (live.picked.has(key)) live.picked.delete(key);
      else live.picked.add(key);
      figure.classList.toggle('picked');
      updatePicked();
    });
    return figure;
  });
  $('frames').replaceChildren(...figures);
}

function renderBoxDetails(row) {
  const rows = [];
  if (!row.boxes.length) {
    $('boxSummary').textContent = row.strict_label === 1
      ? 'This window is labelled 1; its boxes appear here as each call returns.'
      : 'No boxes: only windows labelled 1 are boxed automatically. Pick frames below to ask for boxes yourself.';
  } else {
    $('boxSummary').textContent = '';
  }
  for (const record of row.boxes) {
    const block = element('div', {class: 'boxrecord ' + record.mode});
    const source = record.mode === 'manual' ? 'Your phrase' : record.naming?.source === 'config' ? 'Configured phrase' : 'Named by the model';
    block.append(element('p', {}, source + ': “' + (record.phrase || '—') + '”'));
    if (record.naming?.prompt) {
      block.append(element('pre', {class: 'tagged'}, 'Asked (same conversation as the window): ' + record.naming.prompt +
        '\nReplied: ' + record.naming.raw_text));
    }
    for (const frame of [...record.frames].sort((a, b) => a.slot - b.slot)) {
      block.append(element('pre', {class: 'tagged'}, 'Image ' + (frame.slot + 1) + ' · ' + frame.status +
        (frame.area_fraction ? ' · ' + Math.round(frame.area_fraction * 100) + ' % of image' : '') +
        ' · ' + (frame.seconds ?? 0).toFixed(2) + ' s\n' + frame.tagged_text));
    }
    rows.push(block);
  }
  $('boxRows').replaceChildren(...rows);
}

function renderLive() {
  live.dirty = false;
  $('live').classList.remove('hidden');
  renderStrip();
  if (live.current < 0 && live.windows.length) live.current = 0;
  const row = live.windows[live.current];
  if (!row) {
    $('windowMeta').textContent = 'Waiting for the first window…';
    return;
  }
  const seconds = anchor => (live.fps ? (anchor / live.fps).toFixed(1) + ' s' : '');
  const position = live.single
    ? 'One window' + (row.index !== undefined ? ' (number ' + (row.index + 1) + ' of ' + live.total + ' in this video)' : '')
    : 'Window ' + (live.current + 1) + ' of ' + Math.max(live.total, live.windows.length);
  $('windowMeta').textContent = position +
    ' · anchor frame ' + row.anchor + ' (' + seconds(row.anchor) + ')' +
    (row.start !== undefined ? ' · frames ' + Math.round(row.start) + '–' + Math.round(row.end) : '') +
    ' · label ' + labelText(row.strict_label) + (row.seconds ? ' · ' + row.seconds.toFixed(1) + ' s to answer' : '');
  renderFrames(row);
  $('answerText').textContent = (row.raw_response || '…waiting for the model') + (row.partial ? '\n▌ generating' : '');
  renderBoxDetails(row);
  drawScoreCursor();
}

/* ---------- Timelapse player ---------- */

function step(delta) {
  if (!live.windows.length) return;
  live.current = Math.min(live.windows.length - 1, Math.max(0, live.current + delta));
  renderLive();
}
$('prevWindow').addEventListener('click', () => { live.follow = false; $('followLive').checked = false; step(-1); });
$('nextWindow').addEventListener('click', () => { live.follow = false; $('followLive').checked = false; step(1); });
$('followLive').addEventListener('change', () => {
  live.follow = $('followLive').checked;
  if (live.follow && live.windows.length) {
    live.current = live.windows.length - 1;
    renderLive();
  }
});
function stopPlaying() {
  clearInterval(live.playing);
  live.playing = null;
  $('playWindows').textContent = '▶ Play';
}
$('playWindows').addEventListener('click', () => {
  if (live.playing) return stopPlaying();
  live.follow = false;
  $('followLive').checked = false;
  if (live.current >= live.windows.length - 1) live.current = -1;
  $('playWindows').textContent = '⏸ Pause';
  live.playing = setInterval(() => {
    if (live.current >= live.windows.length - 1) return stopPlaying();
    step(1);
  }, Number($('speed').value));
});
$('speed').addEventListener('change', () => {
  if (live.playing) {
    stopPlaying();
    $('playWindows').click();
  }
});

// The refined frame-score timeline, with a cursor at the window on screen.
async function loadScoreTrack(job) {
  if (!live.scorePath) return;
  const response = await fetch('/api/array?job=' + encodeURIComponent(job) + '&file=' + encodeURIComponent(live.scorePath));
  if (!response.ok) return;
  live.scores = await response.json();
  drawScoreCursor();
}
function drawScoreCursor() {
  const track = $('scoreTrack');
  if (!live.scores) return track.classList.add('hidden');
  const data = live.scores;
  const range = Math.max(0.000001, data.maximum - data.minimum);
  const svg = svgElement('svg', {viewBox: '0 0 1000 60', preserveAspectRatio: 'none', role: 'img',
    'aria-label': 'Final frame scores over the video, with the shown window marked'});
  svg.append(svgElement('polyline', {fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke',
    points: data.values.map((value, index) => (index * 1000 / Math.max(1, data.values.length - 1)).toFixed(1) + ',' +
      (56 - (value - data.minimum) * 52 / range).toFixed(1)).join(' ')}));
  const row = live.windows[live.current];
  if (row) {
    const x = (row.anchor / data.length) * 1000;
    svg.append(svgElement('line', {x1: x, x2: x, y1: 0, y2: 60, class: 'cursor', 'vector-effect': 'non-scaling-stroke'}));
  }
  track.replaceChildren(element('span', {class: 'hint'}, 'Final frame scores (' + data.length + ' frames, ' +
    data.minimum.toFixed(2) + '–' + data.maximum.toFixed(2) + ')'), svg);
  track.classList.remove('hidden');
}

/* ---------- Manual boxes ---------- */

function updatePicked() {
  const count = live.picked.size;
  $('boxSelected').textContent = 'Box ' + count + ' frame' + (count === 1 ? '' : 's');
  $('boxSelected').disabled = !count || live.boxing;
}
$('manualMode').addEventListener('change', () => {
  document.body.classList.toggle('picking', $('manualMode').checked);
  if (!$('manualMode').checked) live.picked.clear();
  updatePicked();
  if (live.windows.length) renderLive();
});
$('manualPhrase').addEventListener('input', updatePicked);
$('boxSelected').addEventListener('click', async () => {
  const phrase = $('manualPhrase').value.trim() || live.windows[live.current]?.boxes.find(record => record.phrase)?.phrase || '';
  if (!phrase) return ($('manualState').textContent = 'Type what to box first.');
  const frames = [...live.picked].map(key => ({anchor: Number(key.split(':')[0]), slot: Number(key.split(':')[1])}));
  live.boxing = true;
  updatePicked();
  $('manualState').textContent = 'Boxing “' + phrase + '” on ' + frames.length + ' frames; the first request loads the model if needed.';
  try {
    const response = await fetch('/api/boxes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-VERA-Token': sessionToken},
      body: JSON.stringify({job: live.job, phrase, frames})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Boxes could not start');
    let saved;
    do {
      await sleep(700);
      await pullTimeline(live.job);
      saved = await (await fetch('/api/windows?job=' + encodeURIComponent(live.job))).json();
    } while (saved.boxing);
    live.picked.clear();
    await loadWindows(live.job);
    $('manualState').textContent = saved.boxing_failed ? 'Boxing failed; see manual_console.log in the stage files.' :
      'Done. Your boxes are drawn in orange and saved in manual_boxes.jsonl.';
    refreshStages(live.job);
  } catch (error) {
    $('manualState').textContent = error.message;
  } finally {
    live.boxing = false;
    updatePicked();
  }
});

/* ---------- Stage files ---------- */

async function previewArtifact(file) {
  const box = $('artifactPreview');
  box.replaceChildren();
  if (file.kind === 'directory') {
    async function listPage(offset) {
      const query = '?job=' + encodeURIComponent(activeJob) + '&dir=' + encodeURIComponent(file.path) + '&offset=' + offset;
      const response = await fetch('/api/list' + query);
      const listing = await response.json();
      if (!response.ok) throw new Error(listing.error);
      box.replaceChildren(element('p', {}, listing.directory + ' · ' + listing.total + ' entries · showing ' +
        (listing.offset + 1) + '–' + Math.min(listing.offset + 50, listing.total)));
      for (const item of listing.items) {
        const button = element('button', {class: 'btn', type: 'button'}, (item.kind === 'directory' ? '▸ ' : '') + item.path.split('/').at(-1));
        button.addEventListener('click', () => previewArtifact(item));
        box.append(button);
      }
      if (listing.next_offset !== null) {
        const next = element('button', {class: 'btn', type: 'button'}, 'Next 50 →');
        next.addEventListener('click', () => listPage(listing.next_offset));
        box.append(next);
      }
    }
    try { await listPage(0); } catch (error) { box.textContent = error.message; }
    return;
  }
  const query = '?job=' + encodeURIComponent(activeJob) + '&file=' + encodeURIComponent(file.path);
  try {
    if (file.kind === 'npy') {
      const response = await fetch('/api/array' + query);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      box.append(element('p', {class: 'stage-meta'}, file.path + ' · ' + data.length + ' values · displayed every ' + data.stride +
        ' · range ' + data.minimum.toFixed(3) + ' to ' + data.maximum.toFixed(3)), plot(data, file.path));
      return;
    }
    if (file.kind === 'png' || file.kind === 'jpg') {
      const image = element('img', {src: '/api/artifact' + query, alt: file.path});
      image.style.maxWidth = '100%';
      box.append(image);
      return;
    }
    const response = await fetch('/api/artifact' + query);
    if (!response.ok) throw new Error('Artifact could not be read');
    const text = await response.text();
    box.append(element('pre', {}, text.slice(0, 12000) + (text.length > 12000 ? '\n…preview truncated; download the full artifact.' : '')));
  } catch (error) {
    box.textContent = error.message;
  }
}

// Score array as a line with its index range on x and its value range on y.
function plot(data, label) {
  const [left, right, top, bottom] = [44, 590, 12, 170];
  const range = Math.max(0.000001, data.maximum - data.minimum);
  const svg = svgElement('svg', {viewBox: '0 0 600 200', role: 'img', 'aria-label': label + ' score plot'});
  svg.append(svgElement('line', {class: 'axis', x1: left, y1: bottom, x2: right, y2: bottom}),
             svgElement('line', {class: 'axis', x1: left, y1: top, x2: left, y2: bottom}),
             svgElement('text', {x: left - 6, y: top + 4, 'text-anchor': 'end'}, data.maximum.toFixed(2)),
             svgElement('text', {x: left - 6, y: bottom, 'text-anchor': 'end'}, data.minimum.toFixed(2)),
             svgElement('text', {x: left, y: 190}, '0'),
             svgElement('text', {x: right, y: 190, 'text-anchor': 'end'}, String(data.length - 1)),
             svgElement('text', {x: (left + right) / 2, y: 190, 'text-anchor': 'middle'}, 'index (frame or window)'));
  svg.append(svgElement('polyline', {
    points: data.values.map((value, index) =>
      (left + index * (right - left) / Math.max(1, data.values.length - 1)).toFixed(1) + ',' +
      (bottom - (value - data.minimum) * (bottom - top) / range).toFixed(1)).join(' '),
    fill: 'none', stroke: 'currentColor', 'stroke-width': 2}));
  return svg;
}

function renderStage() {
  const stage = selectedStages.find(item => item.stage === $('stageSelect').value);
  const files = $('stageFiles');
  files.replaceChildren();
  $('artifactPreview').replaceChildren();
  if (!stage) return;
  for (const artifact of stage.artifacts) {
    const button = element('button', {class: 'btn', type: 'button'}, artifact.path.split('/').at(-1) +
      (artifact.kind === 'directory' ? ' (' + artifact.files + ' files)' : ''));
    button.addEventListener('click', () => previewArtifact(artifact));
    files.append(button);
    if (artifact.kind !== 'directory') {
      const link = element('a', {class: 'btn link ghost', href: artifactURL(activeJob, artifact.path)});
      link.download = artifact.path.split('/').at(-1);
      link.textContent = '↓ ' + link.download;
      files.append(link);
    }
  }
  if (stage.artifacts.length) previewArtifact(stage.artifacts[0]);
  else files.textContent = 'No artifact from this stage.';
}

function renderStages(job, response) {
  activeJob = job;
  selectedStages = response.stages || [];
  $('results').classList.remove('hidden');
  $('scratchPath').textContent = 'Scratch directory: ' + response.scratch_directory +
    '. The next run clears this location. Save any files you want to keep.';
  const select = $('stageSelect');
  const keep = select.value;
  select.replaceChildren(...selectedStages.map(stage => {
    const option = element('option', {value: stage.stage}, stage.stage.replaceAll('_', ' ') + ' (' + stage.artifacts.length + ')');
    return option;
  }));
  select.value = selectedStages.some(stage => stage.stage === keep) ? keep
    : (selectedStages.findLast(item => item.artifacts.length)?.stage || selectedStages[0]?.stage || '');
  select.onchange = renderStage;
  renderStage();
}

async function refreshStages(job) {
  const response = await fetch('/api/status?job=' + encodeURIComponent(job));
  if (response.ok) renderStages(job, await response.json());
}

/* ---------- Following a run ---------- */

async function follow(job, finished = false) {
  if (live.job !== job) resetLive(job);
  $('live').classList.remove('hidden');
  let result;
  let failures = 0;
  while (true) {
    try {
      const response = await fetch('/api/status?job=' + encodeURIComponent(job));
      result = await response.json();
      if (!response.ok) throw Object.assign(new Error(result.error || 'Could not read the run'), {final: true});
      failures = 0;
    } catch (error) {
      if (error.final) {
        $('run').disabled = false;
        return say(error.message, true);
      }
      // A dropped connection or tunnel hiccup is not a failed run: keep following the job.
      failures += 1;
      say('Lost contact with the lab server; retrying (' + failures + ')…', true);
      await sleep(Math.min(30000, 1000 * 2 ** failures));
      continue;
    }
    await pullTimeline(job, result.state === 'running' ? '' : 'generating');
    if (result.state !== 'running') break;
    const progress = result.progress;
    say(progress?.total_windows
      ? 'VERA is running: ' + (progress.completed_requests || 0) + ' of ' + progress.total_windows + ' windows · ' + (progress.stage || 'starting') + '.'
      : 'VERA is running. Every step appears below as it happens.');
    $('liveState').textContent = 'live';
    await sleep(800);
  }
  $('liveState').textContent = result.state === 'complete' ? 'finished' : 'failed';
  await loadWindows(job);
  await loadScoreTrack(job);
  renderStages(job, result);
  $('run').disabled = false;
  fetch('/api/defaults').then(reply => reply.json()).then(defaults => describeWarmModel(defaults.warm_model)).catch(() => {});
  if (result.state === 'failed') {
    say('Run failed. The recorded error is in the event log and the stage files.', true);
    $('answerText').textContent = result.log || 'No log available';
    return;
  }
  say(finished ? 'Showing the last run. The next run clears its scratch directory.' : 'Run complete. Step through the windows, or press Play.');
}

$('run').addEventListener('click', async () => {
  const button = $('run');
  button.disabled = true;
  stopPlaying();
  try {
    if (selectedFile && !uploadedPath) {
      const ticket = selection;
      const stored = await upload(selectedFile);
      if (ticket !== selection) throw new Error('The selected video changed during upload. Press Run again.');
      uploadedPath = stored;
    }
    const video = uploadedPath || $('path').value.trim();
    if (!video) throw new Error('Drop a video, roll the die, or enter a video path.');
    // Simple mode sends VERA's original settings; only Advanced sends the other knobs.
    const request = advanced ? {
      video, target_stage: $('target').value, anchor_seconds: Number($('anchor').value),
      feature_file: $('feature').value.trim(), annotation_file: $('annotation').value.trim(),
      questions: $('questions').value, model_path: $('model').value.trim(), device: $('device').value,
      dtype: $('dtype').value, allow_download: $('download').checked, boxes: $('boxesOn').checked,
      box_phrase: $('boxPhrase').value.trim()
    } : {video, target_stage: scope, anchor_seconds: Number($('anchor').value), boxes: true};
    say('Starting VERA…');
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-VERA-Token': sessionToken},
      body: JSON.stringify(request)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The run could not start');
    uploadedPath = '';
    live.follow = true;
    $('followLive').checked = true;
    $('results').classList.add('hidden');
    $('modelInputPanel').classList.add('hidden');
    resetLive(result.job);
    $('live').scrollIntoView({behavior: 'smooth'});
    await follow(result.job);
  } catch (error) {
    button.disabled = false;
    say(error.message, true);
  }
});

/* ---------- Start ---------- */

setMode(advanced);
updatePicked();
refreshPrompt();
fetch('/api/defaults').then(response => response.json()).then(defaults => {
  serverDefaults = defaults;
  sessionToken = defaults.session_token;
  $('model').value = defaults.model_path;
  if (defaults.annotation_file) $('annotation').value = defaults.annotation_file;
  describeWarmModel(defaults.warm_model);
  if (defaults.missing_dependencies.length) {
    say('This Python is missing ' + defaults.missing_dependencies.join(', ') +
        '. Start the lab from the provisioned .venv (Mac) or the Oz VERA environment; see DEMO.md.', true);
  } else {
    say('Ready. Each run clears and reuses ' + defaults.scratch_directory);
  }
  if (defaults.current_job) {
    $('run').disabled = true;
    live.follow = false;
    $('followLive').checked = false;
    follow(defaults.current_job, true);
  }
}).catch(error => say('Could not check server environment: ' + error.message, true));
