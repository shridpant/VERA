"""FIT-only 24-video outcome/grounding feasibility panel; no STOP-dependent rewards."""
import json
import math
import signal
import time
from pathlib import Path
from .data import balanced_order, load_split, write_json
from .model import NativeModel
from .rewards import score, VERSION
from .runtime import append, provenance, resource_metrics, save_checkpoint
from .smoke import grpo_loss


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    provenance(out, vars(args))
    started = time.monotonic()
    if not 35 < args.max_seconds <= 2 * 3600:
        raise ValueError('RL pilots retain a finite time cap of at most two hours')
    deadline = started + args.max_seconds - 35
    requested = []
    previous = signal.signal(signal.SIGTERM, lambda *_: requested.append(True))
    book = json.loads(Path(args.codebook).read_text())
    fit = load_split(args.data, 'fit')
    ids = balanced_order(fit, args.seed, 0)[:24]
    rows = {r['id']: r for r in fit}
    write_json(out / 'panel.json', ids, immutable=True)
    native = NativeModel(args.model, book, args.adapter, args.seed)
    native.add_reference()
    torch, model = native.torch, native.model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    before = native.adapter_hash()
    groups, rewards_all, summaries = [], [], []
    reason = 'diagnostic_panel_complete'
    try:
        for offset in range(0, len(ids), 4):
            if time.monotonic() >= deadline or requested:
                reason = 'time_cap' if not requested else 'external_signal'
                break
            optimizer.zero_grad(set_to_none=True)
            batch = ids[offset:offset + 4]
            for video in batch:
                row = rows[video]
                prompt, _ = native.prompt(row, args.grounded)
                sampled, olds, refs = [], [], []
                for j in range(4):
                    answer = native.generate(prompt, sample=True)
                    with torch.no_grad():
                        lp, _ = native.log_probs(prompt, answer['ids'])
                        ref, _ = native.log_probs(prompt, answer['ids'], reference=True)
                    olds.append(lp.detach())
                    refs.append(ref.detach())
                    answer.update(score(answer['text'], row, book, args.grounded, answer['truncated']))
                    answer.update(video=video, group_id=video, rollout_index=j, reward_version=VERSION)
                    append(out / 'rollouts.jsonl', answer)
                    sampled.append(answer)
                    rewards_all.append(answer)
                reward = torch.tensor([s['reward'] for s in sampled], device='cuda:0')
                std = reward.std(unbiased=False)
                adv = (reward - reward.mean()) / (std + 1e-4)
                model.train()
                parts_list = []
                for j, answer in enumerate(sampled):
                    lp, entropy = native.log_probs(prompt, answer['ids'])
                    loss, parts = grpo_loss(torch, lp, olds[j], refs[j], adv[j])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite GRPO diagnostic loss')
                    (loss / (4 * len(batch))).backward()
                    parts_list.append({**parts, 'entropy': float(entropy.detach()), 'loss': float(loss.detach())})
                group = {'video': video, 'reward_std': float(std), 'zero_variance': bool(std == 0),
                         'reward_mean': float(reward.mean()),
                         **{k: sum(s[k] for s in parts_list) / 4 for k in parts_list[0]}}
                groups.append(group)
                append(out / 'groups.jsonl', group)
            native.assert_trainable()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite diagnostic gradient')
            optimizer.step()
            step = {'update': offset // 4 + 1, 'grad_norm': float(norm),
                    'groups': len(groups), 'elapsed_seconds': time.monotonic() - started, **resource_metrics()}
            summaries.append(step)
            append(out / 'steps.jsonl', step)
        spread_fraction = sum(not g['zero_variance'] for g in groups) / len(groups) if groups else 0.
        invalid = sum(not r['valid'] for r in rewards_all) / len(rewards_all) if rewards_all else 1.
        after = native.adapter_hash()
        reference_ok = native.adapter_hash('reference') == native.reference_hash
        passed = (len(groups) == 24 and spread_fraction >= .2 and invalid <= .05 and
                  after != before and reference_ok and any(s['grad_norm'] > 0 for s in summaries))
        result = {'status': 'passed' if passed else 'failed_gate', 'stop_reason': reason,
                  'groups': len(groups), 'responses': len(rewards_all), 'updates': len(summaries),
                  'nonzero_reward_spread_fraction': spread_fraction, 'invalid_or_truncated_fraction': invalid,
                  'policy_hash_before': before, 'policy_hash_after': after, 'reference_hash': native.reference_hash,
                  'reference_unchanged': reference_ok, 'grounded': args.grounded,
                  'elapsed_seconds': time.monotonic() - started, 'reward_version': VERSION, **resource_metrics()}
        save_checkpoint(native, optimizer, None, out / 'checkpoint', result)
        write_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        signal.signal(signal.SIGTERM, previous)
