const $ = id => document.getElementById(id);
const status = $('status');
let selectedFile = null;
let uploadedPath = '';
let previewURL = '';
let selectedStages = [];
let activeJob = '';
let sessionToken = '';
// Every change of input bumps this, so a late lucky or upload reply cannot replace a newer choice.
let selection = 0;

function say(message, error = false) {
  status.textContent = message;
  status.classList.toggle('error', error);
}

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
$('device').addEventListener('change', () => {
  const device = $('device').value;
  $('dtype').value = {auto: 'auto', mps: 'float16', cpu: 'float32', cuda: 'bfloat16'}[device];
});

function describeWarmModel(warm) {
  $('warm').textContent = warm?.loaded
    ? 'Model loaded on ' + warm.device + (warm.dtype ? ' (' + warm.dtype + ')' : '') + ': the next window skips loading.'
    : 'No model loaded: the first window loads the checkpoint.';
}

const gigabytes = bytes => (bytes / 1e9).toFixed(1) + ' GB';

function meter(label, fraction, text) {
  const box = document.createElement('div');
  box.className = 'meter';
  const name = document.createElement('span');
  name.textContent = label;
  const value = document.createElement('b');
  value.textContent = text;
  box.append(name, value);
  if (fraction !== null) {
    const bar = document.createElement('i');
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
    const head = $('usageHead');
    head.replaceChildren('Lab server ', Object.assign(document.createElement('b'), {textContent: usage.host}),
      usage.role ? ' · ' + usage.role : ' · no model loaded',
      usage.system === 'Darwin' ? ' · this is a Mac' : '');
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

// Pipeline stages each run target executes; the diagram dims the rest.
const TARGET_STAGES = {
  window: ['input', 'preprocess', 'infer'],
  preprocess: ['input', 'preprocess'],
  segments: ['input', 'preprocess', 'infer'],
  refine: ['input', 'preprocess', 'infer', 'refine'],
  evaluate: ['input', 'preprocess', 'infer', 'refine', 'evaluate'],
};

function updateTargetFields() {
  const target = $('target').value;
  $('anchor').disabled = target !== 'window';
  $('featureField').classList.toggle('hidden', !['refine', 'evaluate'].includes(target));
  $('annotationField').classList.toggle('hidden', target !== 'evaluate');
  $('run').textContent = target === 'window' ? 'Run one window' : 'Run through ' + target;
  $('windowsNode').textContent = target === 'window' ? 'One window' : 'All windows';
  for (const node of document.querySelectorAll('.pnode')) {
    const on = TARGET_STAGES[target].includes(node.dataset.stage);
    node.classList.toggle('on', on);
    node.classList.toggle('off', !on);
  }
}
$('target').addEventListener('change', updateTargetFields);
updateTargetFields();

for (const node of document.querySelectorAll('.pnode')) {
  node.addEventListener('click', () => {
    const control = $(node.dataset.focus);
    const field = control.closest('.field') || control;
    if (field.classList.contains('hidden')) {
      $('target').value = node.dataset.stage;
      updateTargetFields();
    }
    field.scrollIntoView({behavior: 'smooth', block: 'center'});
    control.focus({preventScroll: true});
    field.classList.remove('flash');
    void field.offsetWidth;
    field.classList.add('flash');
  });
}

fetch('/api/defaults').then(response => response.json()).then(defaults => {
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
    poll(defaults.current_job);
  }
}).catch(error => say('Could not check server environment: ' + error.message, true));

$('lucky').addEventListener('click', async () => {
  const button = $('lucky');
  button.disabled = true;
  const ticket = ++selection;
  say('Choosing a random sample…');
  button.classList.add('rolling');
  const faces = '⚀⚁⚂⚃⚄⚅';
  const tumble = setInterval(() => { $('die').textContent = faces[Math.floor(Math.random() * 6)]; }, 90);
  try {
    const response = await fetch('/api/lucky?target=' + encodeURIComponent($('target').value));
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
    say('Random sample selected. Review the controls, then run the model.');
  } catch (error) {
    say(error.message, true);
  } finally {
    clearInterval(tumble);
    button.disabled = false;
    button.classList.remove('rolling');
  }
});

async function upload(file) {
  if (file.size > 200 * 1024 * 1024) {
    throw new Error('Video exceeds the 200 MB upload limit');
  }
  say('Uploading ' + file.name + '…');
  const response = await fetch('/api/upload', {
    method: 'POST',
    headers: {'X-Filename': file.name, 'Content-Type': 'application/octet-stream',
              'X-VERA-Token': sessionToken},
    body: file
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Upload failed');
  return result.path;
}

function show(job, result) {
  $('resultTitle').textContent = 'One-window output';
  $('results').classList.remove('hidden');
  // Results saved before the warm model existed only have a total time.
  const timing = result.inference_seconds === undefined
    ? result.elapsed_seconds.toFixed(1) + ' s'
    : (result.model_reused ? 'model already loaded' : 'model load ' + result.model_load_seconds.toFixed(1) + ' s') +
      ' · inference ' + result.inference_seconds.toFixed(1) + ' s';
  $('resultMeta').textContent =
    'Anchor frame ' + result.anchor_frame +
    ' · requested ' + result.requested_anchor_seconds + ' s' +
    ' · actual FPS ' + result.actual_fps.toFixed(2) +
    ' · ' + result.device + (result.dtype ? ' ' + result.dtype : '') +
    ' · ' + timing + '. One local window only.';
  const frames = $('realFrames');
  frames.replaceChildren();
  result.sampled_frame_indices.forEach((frame, index) => {
    const box = document.createElement('div');
    const image = document.createElement('img');
    image.src = '/api/image?job=' + encodeURIComponent(job) + '&slot=' + index;
    image.alt = 'Sampled input frame ' + (index + 1);
    const label = document.createElement('span');
    label.textContent = 'Frame ' + frame + ' · ' +
      result.sampled_times_seconds[index].toFixed(2) + ' s';
    box.append(image, label);
    frames.append(box);
  });
  $('realResult').textContent =
    'Parsed label: ' + (result.parsed_label === null ? 'invalid' : result.parsed_label) +
    '\n\n' + result.raw_response;
  $('responseHeading').classList.remove('hidden');
  $('realResult').classList.remove('hidden');
  $('results').scrollIntoView({behavior: 'smooth'});
  say('Real model call complete. Inspect the exact eight images below.');
}

async function previewArtifact(file) {
  const box = $('artifactPreview');
  box.replaceChildren();
  if (file.kind === 'directory') {
    async function listPage(offset) {
      const query = '?job=' + encodeURIComponent(activeJob) + '&dir=' +
        encodeURIComponent(file.path) + '&offset=' + offset;
      const response = await fetch('/api/list' + query);
      const listing = await response.json();
      if (!response.ok) throw new Error(listing.error);
      box.replaceChildren();
      const note = document.createElement('p');
      note.textContent = listing.directory + ' · ' + listing.total + ' entries · showing ' +
        (listing.offset + 1) + '–' + Math.min(listing.offset + 50, listing.total);
      box.append(note);
      for (const item of listing.items) {
        const button = document.createElement('button');
        button.className = 'btn';
        button.type = 'button';
        button.textContent = (item.kind === 'directory' ? '▸ ' : '') + item.path.split('/').at(-1);
        button.addEventListener('click', () => previewArtifact(item));
        box.append(button);
      }
      if (listing.next_offset !== null) {
        const next = document.createElement('button');
        next.className = 'btn';
        next.textContent = 'Next 50 →';
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
      const note = document.createElement('p');
      note.className = 'stage-meta';
      note.textContent = file.path + ' · ' + data.length + ' values · displayed every ' + data.stride +
        ' · range ' + data.minimum.toFixed(3) + ' to ' + data.maximum.toFixed(3);
      box.append(note, plot(data, file.path));
      return;
    }
    if (file.kind === 'png' || file.kind === 'jpg') {
      const image = document.createElement('img');
      image.src = '/api/artifact' + query;
      image.alt = file.path;
      image.style.maxWidth = '100%';
      box.append(image);
      return;
    }
    const response = await fetch('/api/artifact' + query);
    if (!response.ok) throw new Error('Artifact could not be read');
    const text = await response.text();
    const preview = document.createElement('pre');
    preview.textContent = text.slice(0, 12000) + (text.length > 12000 ? '\n…preview truncated; download the full artifact.' : '');
    box.append(preview);
  } catch (error) {
    box.textContent = error.message;
  }
}

function svgElement(name, attributes, text) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
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

function showWindow(job, windows, index) {
  const item = windows.windows[index];
  for (const [position, cell] of [...$('windowStrip').children].entries()) {
    cell.classList.toggle('sel', position === index);
  }
  const frames = $('realFrames');
  frames.replaceChildren();
  item.frame_indices.forEach((frame, slot) => {
    const box = document.createElement('div');
    const image = document.createElement('img');
    image.src = '/api/artifact?job=' + encodeURIComponent(job) + '&file=' +
      encodeURIComponent(windows.inputs + '/' + item.anchor + '/' + item.condition + '/' + String(slot).padStart(2, '0') + '.png');
    image.alt = 'Window ' + item.anchor + ' input ' + (slot + 1);
    const label = document.createElement('span');
    label.textContent = 'Frame ' + frame + ' · ' + (frame / windows.fps).toFixed(2) + ' s';
    box.append(image, label);
    frames.append(box);
  });
  $('windowNote').textContent = 'Window ' + (index + 1) + ' of ' + windows.windows.length + ' · anchor frame ' +
    item.anchor + ' (' + (item.anchor / windows.fps).toFixed(1) + ' s) · parsed label ' +
    (item.strict_label === null ? 'invalid' : item.strict_label) + ' · ' + item.seconds.toFixed(1) + ' s inference.';
  $('responseHeading').textContent = 'Generated response for this window';
  $('responseHeading').classList.remove('hidden');
  $('realResult').textContent = item.raw_response;
  $('realResult').classList.remove('hidden');
}

async function showWindows(job) {
  const response = await fetch('/api/windows?job=' + encodeURIComponent(job));
  const windows = await response.json();
  if (!response.ok || !windows.windows.length) return;
  const strip = $('windowStrip');
  strip.replaceChildren();
  windows.windows.forEach((item, index) => {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = item.strict_label === 1 ? 'pos' : (item.strict_label === null ? 'bad' : '');
    cell.title = 'Anchor ' + item.anchor + ' · label ' + (item.strict_label ?? 'invalid');
    cell.setAttribute('aria-label', 'Window ' + (index + 1) + ', label ' + (item.strict_label ?? 'invalid'));
    cell.addEventListener('click', () => showWindow(job, windows, index));
    strip.append(cell);
  });
  strip.classList.remove('hidden');
  $('windowNote').classList.remove('hidden');
  const alerts = windows.windows.filter(item => item.strict_label === 1).length;
  $('resultMeta').textContent = windows.windows.length + ' windows · ' + alerts +
    ' labelled 1 (red) · select a window to see the eight images and the text it produced.';
  const first = windows.windows.findIndex(item => item.strict_label === 1);
  showWindow(job, windows, first >= 0 ? first : 0);
}

function renderStage() {
  const stage = selectedStages.find(item => item.stage === $('stageSelect').value);
  const files = $('stageFiles');
  files.replaceChildren();
  $('artifactPreview').replaceChildren();
  if (!stage) return;
  for (const artifact of stage.artifacts) {
    const button = document.createElement('button');
    button.className = 'btn';
    button.type = 'button';
    button.textContent = artifact.path.split('/').at(-1) +
      (artifact.kind === 'directory' ? ' (' + artifact.files + ' files)' : '');
    button.addEventListener('click', () => previewArtifact(artifact));
    files.append(button);
    if (artifact.kind !== 'directory') {
      const link = document.createElement('a');
      link.className = 'btn link ghost';
      link.href = '/api/artifact?job=' + encodeURIComponent(activeJob) +
        '&file=' + encodeURIComponent(artifact.path);
      link.download = artifact.path.split('/').at(-1);
      link.textContent = '↓ ' + link.download;
      files.append(link);
    }
  }
  if (stage.artifacts.length) previewArtifact(stage.artifacts[0]);
  else files.textContent = 'No artifact from this stage yet.';
}

function renderStages(job, response) {
  activeJob = job;
  selectedStages = response.stages || [];
  $('scratchPath').textContent = 'Scratch directory: ' + response.scratch_directory +
    '. The next run clears this location. Save any artifacts you want to keep.';
  const select = $('stageSelect');
  select.replaceChildren();
  for (const stage of selectedStages) {
    const option = document.createElement('option');
    option.value = stage.stage;
    option.textContent = stage.stage.replaceAll('_', ' ') + ' (' + stage.artifacts.length + ')';
    select.append(option);
  }
  select.value = selectedStages.findLast(item => item.artifacts.length)?.stage || selectedStages[0]?.stage || '';
  select.onchange = renderStage;
  renderStage();
}

async function poll(job, failures = 0) {
  let response;
  try {
    response = await fetch('/api/status?job=' + encodeURIComponent(job));
  } catch (error) {
    // A dropped connection or tunnel hiccup is not a failed run: keep following the job.
    say('Lost contact with the lab server; retrying (' + (failures + 1) + ')…', true);
    setTimeout(() => poll(job, failures + 1), Math.min(30000, 1700 * 2 ** failures));
    return;
  }
  try {
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not read model status');
    if (result.state === 'running') {
      const progress = result.progress;
      if (progress?.total_windows) {
        say('VERA is running: ' + (progress.completed_requests || 0) + '/' +
          progress.total_windows + ' windows · ' + (progress.stage || 'starting') + '.');
      } else {
        say('VERA is running. A full video may take several minutes.');
      }
      setTimeout(() => poll(job), 1700);
      return;
    }
    $('results').classList.remove('hidden');
    renderStages(job, result);
    fetch('/api/defaults').then(reply => reply.json()).then(defaults => describeWarmModel(defaults.warm_model))
      .catch(() => {});
    if (result.state === 'failed') {
      $('run').disabled = false;
      say('Run failed. Available earlier stages are shown below.', true);
      $('resultTitle').textContent = 'Partial run output';
      $('realFrames').replaceChildren();
      $('resultMeta').textContent = 'The model or pipeline failed. The recorded error follows.';
      $('realResult').textContent = result.log || 'No log available';
      $('responseHeading').textContent = 'Error log';
      $('responseHeading').classList.remove('hidden');
      $('realResult').classList.remove('hidden');
      return;
    }
    if (result.mode === 'window') {
      if (!result.result) throw new Error('The model finished without a result file');
      $('responseHeading').textContent = 'Generated response';
      show(job, result.result);
      $('run').disabled = false;
    } else {
      $('resultTitle').textContent = 'Full-video stage output';
      $('resultMeta').textContent = 'The selected target stage completed. Choose a stage and an artifact to inspect it.';
      $('realFrames').replaceChildren();
      $('responseHeading').classList.add('hidden');
      $('realResult').classList.add('hidden');
      await showWindows(job);
      $('run').disabled = false;
      $('results').scrollIntoView({behavior: 'smooth'});
      say('Run complete. The next run will clear the scratch directory.');
    }
  } catch (error) {
    $('run').disabled = false;
    say(error.message, true);
  }
}

$('run').addEventListener('click', async () => {
  const button = $('run');
  button.disabled = true;
  $('results').classList.add('hidden');
  $('windowStrip').classList.add('hidden');
  $('windowNote').classList.add('hidden');
  try {
    if (selectedFile && !uploadedPath) {
      const ticket = selection;
      const stored = await upload(selectedFile);
      if (ticket !== selection) throw new Error('The selected video changed during upload. Press Run again.');
      uploadedPath = stored;
    }
    const video = uploadedPath || $('path').value.trim();
    if (!video) throw new Error('Drop a video, choose one, or enter a video path.');
    const request = {
      video,
      target_stage: $('target').value,
      anchor_seconds: Number($('anchor').value),
      feature_file: $('feature').value.trim(),
      annotation_file: $('annotation').value.trim(),
      questions: $('questions').value,
      model_path: $('model').value.trim(),
      device: $('device').value,
      dtype: $('dtype').value,
      allow_download: $('download').checked
    };
    say('Starting VERA…');
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-VERA-Token': sessionToken},
      body: JSON.stringify(request)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The run could not start');
    uploadedPath = '';
    await poll(result.job);
  } catch (error) {
    button.disabled = false;
    say(error.message, true);
  }
});
