"""Real single-GPU engineering gate. Never treats synthetic RL advantages as learning."""
import json
import math
import time
from pathlib import Path
from .data import load_split, balanced_order, write_json
from .model import NativeModel
from .runtime import append, provenance, save_checkpoint, restore_checkpoint, resource_metrics
from .rewards import score


def grpo_loss(torch, logp, old, ref, advantage, beta=.02):
    ratio = (logp - old).exp()
    delta = ref - logp
    kl = delta.exp() - delta - 1
    unclipped = ratio * advantage
    clipped = ratio.clamp(.8, 1.2) * advantage
    loss = -torch.minimum(unclipped, clipped).mean() + beta * kl.mean()
    return loss, {'kl_per_token': float(kl.detach().mean()),
                  'clip_fraction': float(((ratio.detach() < .8) | (ratio.detach() > 1.2)).float().mean())}


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    spec = provenance(out, vars(args))
    book = json.loads(Path(args.codebook).read_text())
    rows = load_split(args.data, 'fit')
    lookup = {r['id']: r for r in rows}
    panel_ids = balanced_order(rows, args.seed, 0)[:12]
    write_json(out / 'panel.json', panel_ids, immutable=True)
    native = NativeModel(args.model, book, seed=args.seed)
    torch = native.torch
    model = native.model
    write_json(out / 'trainable.json', native.assert_trainable())
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    initial_hash = native.adapter_hash()
    row = lookup[panel_ids[0]]
    prompt, prompt_text = native.prompt(row, args.grounded)
    ids, kinds, target = native.completion(row, args.grounded)
    # Check only structured targets are supervised and original prompt stays separate.
    assert len(ids) == len(kinds) and any(k == 1 for k in kinds)
    write_json(out / 'encoding.json', {'prompt': prompt_text, 'target': target,
                'prompt_tokens': int(prompt['input_ids'].shape[1]), 'completion_tokens': len(ids),
                'image_tokens': int((prompt['input_ids'] == model.config.image_token_id).sum()),
                'kinds': kinds, 'video': row['id']})
    def update():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = native.sft_loss(prompt, ids, kinds)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite supervised loss')
        loss.backward()
        native.assert_trainable()
        if any(p.grad is not None for n, p in model.named_parameters() if 'lora_' not in n):
            raise RuntimeError('base received gradient')
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite gradient')
        optimizer.step()
        scheduler.step()
        return float(loss.detach()), {**parts, 'grad_norm': float(norm)}
    losses = []
    for step in range(3):
        loss, parts = update()
        losses.append(loss)
        append(out / 'steps.jsonl', {'stage': 'sft', 'step': step, 'loss': loss, **parts})
    if native.adapter_hash() == initial_hash:
        raise RuntimeError('adapter did not change')
    if losses[-1] >= losses[0] and losses[0] > .02:
        raise RuntimeError('fixture loss did not decrease')
    if any(dict(model.named_parameters())[n]._version != v for n, v in frozen.items()):
        raise RuntimeError('frozen base mutated')
    prediction = native.generate(prompt)
    save_checkpoint(native, optimizer, scheduler, out / 'checkpoint', {'step': 3, 'sampler_cursor': 3,
                    'sampler_order': panel_ids, 'seed': args.seed})
    expected_loss, _ = update()
    expected_hash = native.adapter_hash()
    from peft import get_peft_model_state_dict
    expected_tensors = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    state = restore_checkpoint(native, optimizer, scheduler, out / 'checkpoint')
    if state['sampler_cursor'] != 3 or native.generate(prompt)['ids'] != prediction['ids']:
        raise RuntimeError('save/reload prediction or cursor mismatch')
    # Generation is deterministic and should not advance training RNG; restore again
    # to test the exact next optimizer/scheduler/RNG transition.
    restore_checkpoint(native, optimizer, scheduler, out / 'checkpoint')
    resumed_loss, _ = update()
    resumed_hash = native.adapter_hash()
    if resumed_hash != expected_hash or resumed_loss != expected_loss:
        differences = {k: {'max_abs': float((v.detach().cpu() - expected_tensors[k]).abs().max()),
                           'different_elements': int((v.detach().cpu() != expected_tensors[k]).sum())}
                       for k, v in get_peft_model_state_dict(model).items() if not torch.equal(v.detach().cpu(), expected_tensors[k])}
        write_json(out / 'resume-differences.json', differences)
        raise RuntimeError(f'exact resume failed: {resumed_loss} vs {expected_loss}; hash equal={resumed_hash == expected_hash}')
    # This nonzero SFT warm start is the actual immutable RL reference.
    native.add_reference()
    policy_start = native.adapter_hash()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    model.eval()
    rollouts, old_lps, ref_lps = [], [], []
    for i in range(4):
        result = native.generate(prompt, sample=True)
        with torch.no_grad():
            old, _ = native.log_probs(prompt, result['ids'])
            ref, _ = native.log_probs(prompt, result['ids'], reference=True)
        if not torch.allclose(old, ref, atol=1e-5, rtol=1e-5):
            raise RuntimeError('reference scores differ at identical initialization')
        old_lps.append(old.detach())
        ref_lps.append(ref.detach())
        result.update(score(result['text'], row, book, args.grounded, result['truncated']))
        result.update({'video': row['id'], 'group_id': row['id'], 'rollout_index': i})
        rollouts.append(result)
        append(out / 'rollouts.jsonl', result)
    rewards = torch.tensor([r['reward'] for r in rollouts], device='cuda:0')
    spread = float(rewards.std(unbiased=False))
    # Engineering-only zero-spread fallback proves mechanics, never counts as RL
    # feasibility or real reward learning. The scientific gate remains false.
    actual_advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    synthetic = spread == 0
    advantages = torch.tensor([-1., -.5, .5, 1.], device='cuda:0') if synthetic else actual_advantages
    optimizer.zero_grad(set_to_none=True)
    summaries = []
    model.train()
    for i, result in enumerate(rollouts):
        logp, entropy = native.log_probs(prompt, result['ids'])
        loss, parts = grpo_loss(torch, logp, old_lps[i], ref_lps[i], advantages[i])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite GRPO loss')
        (loss / 4).backward()
        summaries.append({**parts, 'loss': float(loss.detach()), 'entropy': float(entropy.detach())})
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(norm) or float(norm) == 0:
        raise RuntimeError('GRPO gradient not finite and nonzero')
    optimizer.step()
    if native.adapter_hash() == policy_start or native.adapter_hash('reference') != native.reference_hash:
        raise RuntimeError('policy unchanged or reference mutated')
    result = {'status': 'engineering_passed', 'scientific_rl_gate_passed': False,
              'note': 'RL diagnostic-panel reward variance / validity gate still required after supervised pilot',
              'sft_losses': losses, 'resume_exact': True, 'prediction_reload_exact': True,
              'policy_hash_before': policy_start, 'policy_hash_after': native.adapter_hash(),
              'reference_hash': native.reference_hash, 'reward_std': spread,
              'synthetic_advantages_for_engineering_only': synthetic,
              'grpo': summaries, 'grpo_grad_norm': float(norm),
              'elapsed_seconds': time.monotonic() - started, **resource_metrics()}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)
    return result
