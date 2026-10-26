"""ImageBind segment features for VERA's refinement, by LAVAD's procedure.

VERA's README says its per-segment features come from LAVAD (Zanella et al., CVPR 2024) with
ImageBind. LAVAD (github.com/lucazanella/lavad @ 1ad46c66, src/models/video_text_score_refiner.py
and its libs/ImageBind/imagebind/data.py) computes, for each anchor c in range(0, n, 16):
  1. the decoded frames range(max(c - F//2, 0), min(c + F//2, n)), with F = int(T * fps) = 300;
  2. N = 10 of them, np.linspace(0, t - 1, N, dtype=int) (all of them when t <= N);
  3. pytorchvideo FrameVideo over those frame files (its default 30 FPS),
     ConstantClipsPerVideoSampler(clip_duration=2, clips_per_video=5), UniformTemporalSubsample(2),
     /255, ShortSideScale(224), NormalizeVideo(CLIP mean/std) and SpatialCrop(224, 3): 15 clips;
  4. the ImageBind-Huge vision embedding of the 15 clips (the model averages over clips), FP32.
This module re-implements that path on the official ImageBind package at 80d07458, whose model
files equal LAVAD's vendored copy byte for byte. No LAVAD code is copied. Frames are the
runner's (OpenCV decode, default JPEG), as LAVAD's extract_frames.py writes them.
"""
import json
from pathlib import Path
import numpy as np

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def anchors(n_frames, stride=16):
    return list(range(0, n_frames, stride))


def clip_frame_indices(anchor, n_frames, fps=30.0, clip_seconds=10.0, num_samples=10):
    frames_per_clip = int(clip_seconds * fps)
    clip = list(range(max(anchor - frames_per_clip // 2, 0), min(anchor + frames_per_clip // 2, n_frames)))
    if num_samples >= len(clip):
        return clip
    picks = np.clip(np.linspace(0, len(clip) - 1, num_samples, dtype=int), 0, len(clip) - 1)
    return [clip[i] for i in picks]


def clip_tensor(frame_paths):
    """(15, 3, 2, 224, 224): five temporal clips x three spatial crops of the frame files."""
    import torch
    from imagebind.data import SpatialCrop, get_clip_timepoints
    from pytorchvideo import transforms as pv_transforms
    from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
    from pytorchvideo.data.frame_video import FrameVideo
    from torchvision import transforms
    from torchvision.transforms._transforms_video import NormalizeVideo
    transform = transforms.Compose([pv_transforms.ShortSideScale(224), NormalizeVideo(mean=CLIP_MEAN, std=CLIP_STD)])
    sampler = ConstantClipsPerVideoSampler(clip_duration=2, clips_per_video=5)
    subsample = pv_transforms.UniformTemporalSubsample(num_samples=2)
    video = FrameVideo.from_frame_paths([str(p) for p in frame_paths])
    clips = []
    for start, end in get_clip_timepoints(sampler, video.duration):
        clip = video.get_clip(start, end)
        if clip is None:
            raise ValueError('No clip found')
        clips.append(subsample(clip['video']) / 255.0)
    clips = SpatialCrop(224, num_crops=3)([transform(c) for c in clips])
    return torch.stack(clips, dim=0)


class Extractor:
    def __init__(self, weights, device='cuda'):
        import torch
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.torch, self.vision, self.device = torch, ModalityType.VISION, device
        model = imagebind_model.imagebind_huge(pretrained=False)
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        self.model = model.eval().to(device)

    def embed(self, clip_batches):
        with self.torch.no_grad():
            out = self.model({self.vision: self.torch.stack(clip_batches).to(self.device)})
        return out[self.vision].float().cpu().numpy()


def extract_video(extractor, frames_dir, n_frames, settings, stride=16):
    """One float32 row per anchor, in anchor order; also returns the frame indices of each clip."""
    rows, used = [], []
    batch = []
    for anchor in anchors(n_frames, stride):
        indices = clip_frame_indices(anchor, n_frames, settings['fps'], settings['clip_seconds'], settings['num_samples'])
        used.append(indices)
        batch.append(clip_tensor([Path(frames_dir) / f'{i:06d}.jpg' for i in indices]))
        if len(batch) == settings['batch_size']:
            rows.append(extractor.embed(batch))
            batch = []
    if batch:
        rows.append(extractor.embed(batch))
    return np.concatenate(rows).astype(np.float32), used


def compare(ours, released):
    """Row-wise cosine similarity and shape agreement against a released feature file."""
    if ours.shape != released.shape:
        return {'shape_ours': list(ours.shape), 'shape_released': list(released.shape), 'same_shape': False}
    a = ours / np.linalg.norm(ours, axis=1, keepdims=True)
    b = released / np.linalg.norm(released, axis=1, keepdims=True)
    cos = (a * b).sum(axis=1)
    return {'same_shape': True, 'shape': list(ours.shape), 'cosine_min': float(cos.min()), 'cosine_median': float(np.median(cos)),
            'cosine_mean': float(cos.mean()), 'released_norm_median': float(np.median(np.linalg.norm(released, axis=1))),
            'ours_norm_median': float(np.median(np.linalg.norm(ours, axis=1))), 'released_dtype': str(released.dtype)}


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=1) + '\n')
