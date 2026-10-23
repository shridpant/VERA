"""Parameterize compute_auc.py's numerical operations; reject undefined short-input cases."""
import numpy as np


def refine(records, features, length, stride, cfg):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != len(records):
        raise ValueError('Feature rows must match ALL ordered anchors; features are not automatically extracted')
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features,axis=1)==0):
        raise ValueError('Features must be finite and nonzero')
    n = len(records)
    top = int(cfg['neighbor_fraction'] * n)
    if top < 1 or n < cfg['smoothing_size']:
        raise ValueError('Too few windows for faithful upstream refinement; refuses upstream -0/neighborhood and convolution-length edge cases')
    labels = [r['strict_label'] for r in records]
    if any(x not in (0,1) for x in labels):
        raise ValueError('Unparseable responses cannot be silently refined into labels')
    normalized = features / np.linalg.norm(features,axis=1,keepdims=True)
    similarity = normalized @ normalized.T
    neighbors = np.argsort(similarity,axis=1)[:,-top:]
    scene=[]
    for i,row in enumerate(neighbors):
        x=similarity[i,row]*cfg['similarity_temperature']
        weights=np.exp(x-np.max(x)); weights/=weights.sum()
        scene.append(np.dot(weights,np.asarray(labels)[row]))
    size,sigma=cfg['smoothing_size'],cfg['smoothing_sigma']
    kernel=np.exp(-np.linspace(-size//2,size//2,size)**2/(2*sigma**2));kernel/=kernel.sum()
    temporal=np.convolve(np.array(scene),kernel,mode='same')
    frames=[]
    for i,r in enumerate(records):
        count=stride if i!=n-1 else int(r['end']-r['anchor'])
        frames.extend([round(temporal[i],cfg['round_decimals'])]*count)
    frames=np.asarray(frames)
    if len(frames)!=length:raise ValueError('Frame expansion length mismatch')
    if cfg['position_weighting']:
        sigma=int(length/2)
        if sigma==0:raise ValueError('Video too short for positional Gaussian')
        frames*=np.exp(-.5*((np.arange(length)-int(length/2))/sigma)**2)
    return dict(scene=np.asarray(scene),temporal=temporal,frame_scores=frames)


def ground_truth(entry,length):
    values=entry['temporal_label']
    if len(values)%2:raise ValueError('Intervals must be start/end pairs')
    gt=np.zeros(length)
    for a,b in zip(values[::2],values[1::2]):
        if a==-1 and b==-1:continue
        if type(a) is not int or type(b) is not int or a<0 or b<=a or a>=length:
            raise ValueError('Ground truth uses zero-based frame intervals [start,end)')
        end=min(b,length);gt[a:end]=1
    return gt
