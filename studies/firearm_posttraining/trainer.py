"""Bounded single-GPU supervised and warm-start GRPO training with STOP selection."""
import json
import math
import signal
import time
from pathlib import Path
from .data import balanced_order, load_split, digest, write_json
from .evaluation import evaluate
from .model import NativeModel
from .rewards import score, VERSION
from .runtime import append, provenance, save_checkpoint, restore_checkpoint, resource_metrics
from .smoke import grpo_loss
from .stopping import StopState


def run(args):
    out = Path(args.output)
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError('resume needs the original run output')
    started = time.monotonic()
    deadline = started + args.max_seconds - 35
    stopping = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    is_rl = args.method in ('RL', 'RG')
    grounded = args.method in ('SG', 'RG')
    if bool(args.grounded) != grounded:
        raise ValueError('method / grounding flag mismatch')
    if is_rl and not args.adapter:
        raise ValueError('GRPO requires an explicit fixed warm-start adapter')
    if is_rl and not args.frozen_protocol:
        raise ValueError('scientific GRPO requires frozen protocol; use rl-panel for feasibility')
    if args.adapter:
        warm_state = json.loads((Path(args.adapter) / 'state.json').read_text())
        if warm_state.get('selection', {}).get('checks') != 0:
            raise ValueError('paired initialization must be a fixed 0.5-pass warm start without STOP selection')
    if args.frozen_protocol:
        protocol = json.loads(Path(args.frozen_protocol).read_text())
        if protocol.get('status') != 'frozen' or args.max_passes != protocol['shared_max_fit_passes']:
            raise ValueError('frozen protocol/pass budget mismatch')
    if not args.resume:
        provenance(out, vars(args))
    book = json.loads(Path(args.codebook).read_text())
    fit, stop = load_split(args.data, 'fit'), load_split(args.data, 'stop')
    start_cursor = len(fit) // 2 if args.adapter else 0
    if args.adapter and warm_state['cursor'] != start_cursor:
        raise ValueError('warm-start exposure mismatch')
    lookup = {r['id']: r for r in fit}
    native = NativeModel(args.model, book, adapter=args.adapter, seed=args.seed)
    torch, model = native.torch, native.model
    write_json(out / 'trainable.json', native.assert_trainable(), immutable=True)
    if is_rl:
        native.add_reference()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6 if is_rl else 5e-5)
    accumulation = 4 if is_rl else args.accumulation
    total_exposures = math.floor(len(fit) * (.5 if args.warmup_only else args.max_passes))
    total_updates = math.ceil((total_exposures - start_cursor) / accumulation)
    warm_steps = max(1, math.ceil(total_updates * .05))
    def multiplier(step):
        if step < warm_steps:
            return (step + 1) / warm_steps
        return max(0., (total_updates - step) / max(1, total_updates - warm_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    selected = StopState()
    cursor, updates, next_check = start_cursor, 0, start_cursor + len(fit) // 2
    elapsed_before = 0.
    bad_kl = bad_format = bad_stop = 0
    initial_stop = None
    if args.resume:
        state = restore_checkpoint(native, optimizer, scheduler, args.resume)
        cursor, updates, next_check = state['cursor'], state['updates'], state['next_check']
        selected = StopState(**state['selection'])
        elapsed_before = state['elapsed_seconds']
        deadline -= elapsed_before
        bad_kl, bad_format, bad_stop = state.get('bad_kl', 0), state.get('bad_format', 0), state.get('bad_stop', 0)
        initial_stop = state.get('initial_stop')
        if is_rl and state['reference_hash'] != native.reference_hash:
            raise ValueError('resumed reference does not match original fixed warm start')
    def state():
        return {'cursor': cursor, 'updates': updates, 'next_check': next_check,
                'selection': selected.snapshot(), 'elapsed_seconds': elapsed_before + time.monotonic() - started,
                'seed': args.seed, 'fit_manifest_hash': digest(Path(args.data) / 'fit.json'),
                'bad_kl': bad_kl, 'bad_format': bad_format, 'bad_stop': bad_stop, 'initial_stop': initial_stop,
                'reference_hash': native.reference_hash if is_rl else None,
                'initial_exposures': start_cursor,
                'reward_version': VERSION if is_rl else None}
    def snapshot(name):
        path = out / 'checkpoints' / name
        save_checkpoint(native, optimizer, scheduler, path, state())
        return str(path)
    try:
        if args.resume and not args.warmup_only and cursor >= next_check:
            # Boundary checkpoints are saved before their STOP query. A resumed
            # run reuses a completed query, never silently adds a new query.
            eval_path = out / f'stop-{cursor:06d}'
            if (eval_path / 'metrics.json').exists():
                measured = json.loads((eval_path / 'metrics.json').read_text())
            elif eval_path.exists():
                raise RuntimeError('partial STOP query requires explicit recovery; do not silently requery')
            else:
                measured = evaluate(native, stop, eval_path, grounded, deadline=deadline,
                                    stop_requested=lambda: bool(stopping))
            if measured is None:
                selected.reason = 'time_cap_during_validation'
            else:
                selected.observe(measured, args.resume, cursor / len(fit))
                next_check += len(fit) // 2
        if not args.resume:
            baseline = snapshot('initial')
            if not args.warmup_only:
                measured = evaluate(native, stop, out / 'stop-initial', grounded, deadline=deadline, stop_requested=lambda: bool(stopping))
                if measured is None:
                    selected.reason = 'time_cap_during_baseline'
                else:
                    initial_stop = measured['balanced_category_accuracy']
                    selected.observe(measured, baseline, 0)
                    append(out / 'selection.jsonl', {'exposures': cursor, 'checkpoint': baseline, **measured})
        while cursor < total_exposures and not selected.reason:
            if time.monotonic() >= deadline or stopping:
                selected.reason = 'time_cap' if not stopping else 'external_signal'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = []
            # Never cross a scheduled STOP boundary within an optimizer update.
            upper = min(total_exposures, cursor + accumulation,
                        next_check if not args.warmup_only else total_exposures)
            for position in range(cursor, upper):
                pass_index, offset = divmod(position, len(fit))
                video = balanced_order(fit, args.seed, pass_index)[offset]
                row = lookup[video]
                prompt, _ = native.prompt(row, grounded)
                t = time.monotonic()
                if not is_rl:
                    ids, kinds, _ = native.completion(row, grounded)
                    loss, parts = native.sft_loss(prompt, ids, kinds)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    (loss / (upper - cursor)).backward()
                    record = {'loss': float(loss.detach()), **parts}
                else:
                    rollouts, old_probs, refs = [], [], []
                    for j in range(4):
                        result = native.generate(prompt, sample=True)
                        with torch.no_grad():
                            old, _ = native.log_probs(prompt, result['ids'])
                            ref, _ = native.log_probs(prompt, result['ids'], reference=True)
                        old_probs.append(old.detach())
                        refs.append(ref.detach())
                        result.update(score(result['text'], row, book, grounded, result['truncated']))
                        result.update({'video': video, 'group_id': f'{position}:{video}', 'rollout_index': j,
                                       'exposure': position, 'update': updates, 'reward_version': VERSION})
                        append(out / 'rollouts.jsonl', result)
                        rollouts.append(result)
                    rewards = torch.tensor([r['reward'] for r in rollouts], device='cuda:0')
                    std = rewards.std(unbiased=False)
                    advantages = (rewards - rewards.mean()) / (std + 1e-4)
                    model.train()
                    parts_list, losses = [], []
                    for j, result in enumerate(rollouts):
                        lp, entropy = native.log_probs(prompt, result['ids'])
                        loss, parts = grpo_loss(torch, lp, old_probs[j], refs[j], advantages[j])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite GRPO loss')
                        (loss / (4 * (upper - cursor))).backward()
                        losses.append(float(loss.detach()))
                        parts_list.append({**parts, 'entropy': float(entropy.detach())})
                    record = {'loss': sum(losses) / 4, 'reward': float(rewards.mean()), 'reward_std': float(std),
                              'zero_variance': bool(std == 0), 'invalid_fraction': sum(not r['valid'] for r in rollouts) / 4,
                              'truncated_fraction': sum(r['truncated'] for r in rollouts) / 4,
                              **{k: sum(p[k] for p in parts_list) / 4 for k in parts_list[0]}}
                record.update(video=video, exposure=position, update=updates, seconds=time.monotonic() - t)
                append(out / 'exposures.jsonl', record)
                batch.append(record)
            native.assert_trainable()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite gradients')
            optimizer.step()
            scheduler.step()
            cursor, updates = upper, updates + 1
            step = {'update': updates, 'exposures': cursor, 'fit_passes': cursor / len(fit),
                    'grad_norm': float(norm), 'learning_rate': scheduler.get_last_lr()[0],
                    'elapsed_seconds': elapsed_before + time.monotonic() - started, **resource_metrics()}
            if is_rl:
                kl = sum(r['kl_per_token'] for r in batch) / len(batch)
                invalid = sum(r['invalid_fraction'] for r in batch) / len(batch)
                bad_kl = bad_kl + 1 if kl > .2 else 0
                bad_format = bad_format + 1 if invalid > .05 else 0
                if bad_kl >= 3 or bad_format >= 3:
                    selected.reason = 'KL_divergence' if bad_kl >= 3 else 'invalid_output_divergence'
                step.update(kl_per_token=kl, invalid_fraction=invalid,
                            zero_variance_group_fraction=sum(r['zero_variance'] for r in batch) / len(batch))
            append(out / 'steps.jsonl', step)
            if not args.warmup_only and cursor >= next_check and not selected.reason:
                check_path = snapshot(f'exposure-{cursor:06d}')
                measured = evaluate(native, stop, out / f'stop-{cursor:06d}', grounded,
                                    deadline=deadline, stop_requested=lambda: bool(stopping))
                if measured is None:
                    selected.reason = 'time_cap_during_validation'
                    break
                selected.observe(measured, check_path, cursor / len(fit))
                append(out / 'selection.jsonl', {'exposures': cursor, 'checkpoint': check_path, **measured})
                next_check += len(fit) // 2
                if is_rl:
                    bad_stop = bad_stop + 1 if measured['balanced_category_accuracy'] < initial_stop - .10 else 0
                    if bad_stop >= 2:
                        selected.reason = 'STOP_accuracy_divergence'
        if not selected.reason:
            selected.reason = 'fixed_warmup_complete' if args.warmup_only else 'maximum_fit_passes'
        final_checkpoint = snapshot(f'final-{cursor:06d}')
        if args.warmup_only:
            selected.best_checkpoint = final_checkpoint
        result = {**state(), 'status': 'stopped', 'stop_reason': selected.reason,
                  'fit_passes': cursor / len(fit), 'best_checkpoint': selected.best_checkpoint,
                  'last_checkpoint': final_checkpoint, 'censored': 'time_cap' in selected.reason,
                  'scientific_role': 'exploratory_pilot' if not args.frozen_protocol else 'frozen_comparison',
                  **resource_metrics()}
        write_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), **state()})
        # Persist only finite states; corrupted/nonfinite checkpoints are not candidates.
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
