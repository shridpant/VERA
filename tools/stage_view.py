"""Inventory the existing VERA outputs without changing their contents."""
from pathlib import Path


def _entry(root, path):
    if path.is_file():
        return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                'kind': path.suffix.lower().lstrip('.') or 'text'}
    if path.is_dir():
        return {'path': str(path.relative_to(root)), 'kind': 'directory',
                'files': sum(1 for file in path.rglob('*') if file.is_file())}
    return None


def stage_manifest(folder, mode, video_id):
    """Return ordered stages and every named core artifact already on disk."""
    root = Path(folder)
    if mode == 'window':
        groups = [
            ('00_request', ['lab_job.json', 'config.json', 'console.log']),
            ('01_sampled_input', ['probe/sample_manifest.json', 'probe/frames']),
            ('02_model_prompt', ['probe/prompt.txt', 'probe/model_input.txt']),
            ('03_local_prediction', ['probe/result.json']),
            ('04_anomaly_boxes', ['probe/boxes.json', 'manual_boxes.jsonl', 'manual_console.log']),
            ('05_timeline', ['probe/timeline.jsonl', 'manual_timeline.jsonl']),
        ]
    else:
        output = Path('pipeline/run 1/output')
        video = output / video_id
        groups = [
            ('00_request', ['lab_job.json', 'config.json', 'registry.json', 'console.log',
                            'pipeline/run 1/command.json', output / 'invocation.json',
                            output / 'dataset_manifest.json', output / 'source.json',
                            output / 'source_snapshot', output / 'environment.txt']),
            ('01_preprocess', [video / 'video.json', video / 'coverage.json',
                               video / 'frames']),
            ('02_model_prompt', [output / 'prompt.txt', output / 'model_input.txt',
                                 output / 'model_identity.json', output / 'model_progress.jsonl']),
            ('03_window_predictions', [video / 'raw_responses.jsonl',
                                       video / 'segment_scores.json', video / 'inputs',
                                       video / 'probe_comparison.json']),
            ('04_anomaly_boxes', [video / 'boxes.jsonl', 'manual_boxes.jsonl', 'manual_console.log']),
            ('05_scene_refinement', [video / 'feature_identity.json', video / 'scene.npy']),
            ('06_temporal_smoothing', [video / 'temporal.npy']),
            ('07_frame_timeline', [video / 'frame_scores.npy']),
            ('08_evaluation', [output / 'metrics.json']),
            ('09_diagnostics', [output / 'summary.json', output / 'status.json', output / 'timeline.jsonl',
                                'manual_timeline.jsonl', output / 'inference.log', output / 'diagnostic.log',
                                output / 'error.txt']),
        ]
    return [{'stage': name, 'artifacts': [entry for item in items
             if (entry := _entry(root, root / item)) is not None]}
            for name, items in groups]
