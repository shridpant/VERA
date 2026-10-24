const $ = id => document.getElementById(id);
const status = $('status');
let selectedFile = null;
let uploadedPath = '';
let previewURL = '';
let selectedStages = [];
let activeJob = '';
let sessionToken = '';

function say(message, error = false) {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function setFile(file) {
  if (!file) return;
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
  if ($('path').value.trim()) {
    selectedFile = null;
    uploadedPath = '';
    $('chosen').textContent = 'Using the path below';
  }
});
$('device').addEventListener('change', () => {
  const device = $('device').value;
  if (device === 'mps') $('dtype').value = 'float16';
  else if (device === 'cpu') $('dtype').value = 'float32';
  else if (device === 'cuda') $('dtype').value = 'bfloat16';
});

function updateTargetFields() {
  const target = $('target').value;
  $('anchor').disabled = target !== 'window';
  $('featureField').classList.toggle('hidden', !['refine', 'evaluate'].includes(target));
  $('annotationField').classList.toggle('hidden', target !== 'evaluate');
  $('run').textContent = target === 'window' ? 'Run one window' : 'Run through ' + target;
}
$('target').addEventListener('change', updateTargetFields);
updateTargetFields();

fetch('/api/defaults').then(response => response.json()).then(defaults => {
  sessionToken = defaults.session_token;
  $('model').value = defaults.model_path;
  if (defaults.annotation_file) $('annotation').value = defaults.annotation_file;
  if (defaults.missing_dependencies.length) {
    say('This Python is missing ' + defaults.missing_dependencies.join(', ') +
        '. Use the Oz VERA environment for real model runs; see DEMO.md.', true);
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
  say('Choosing a random sample…');
  try {
    const response = await fetch('/api/lucky?target=' + encodeURIComponent($('target').value));
    const choice = await response.json();
    if (!response.ok) throw new Error(choice.error);
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
    button.disabled = false;
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
  $('resultMeta').textContent =
    'Anchor frame ' + result.anchor_frame +
    ' · requested ' + result.requested_anchor_seconds + ' s' +
    ' · actual FPS ' + result.actual_fps.toFixed(2) +
    ' · ' + result.elapsed_seconds.toFixed(1) + ' s' +
    ' · ' + result.device + '. One local window only.';
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
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 600 180');
      svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', file.path + ' score plot');
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      const range = Math.max(0.000001, data.maximum - data.minimum);
      line.setAttribute('points', data.values.map((value, index) =>
        (index * 600 / Math.max(1, data.values.length - 1)).toFixed(1) + ',' +
        (170 - (value - data.minimum) * 155 / range).toFixed(1)).join(' '));
      line.setAttribute('fill', 'none');
      line.setAttribute('stroke', 'currentColor');
      line.setAttribute('stroke-width', '2');
      svg.append(line);
      box.append(note, svg);
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

async function poll(job) {
  try {
    const response = await fetch('/api/status?job=' + encodeURIComponent(job));
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
    $('run').disabled = false;
    $('results').classList.remove('hidden');
    renderStages(job, result);
    if (result.state === 'failed') {
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
    } else {
      $('resultTitle').textContent = 'Full-video stage output';
      $('resultMeta').textContent = 'The selected target stage completed. Choose a stage and an artifact to inspect it.';
      $('realFrames').replaceChildren();
      $('responseHeading').classList.add('hidden');
      $('realResult').classList.add('hidden');
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
  try {
    if (selectedFile && !uploadedPath) uploadedPath = await upload(selectedFile);
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
