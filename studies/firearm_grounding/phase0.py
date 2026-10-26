"""Phase 0 of the grounding pre-registration: real-model checks before any grounding number is read.

(b1) Scorer.generate gives the same text as the runner's batch_chat call for one video.
(b2) An eager-attention replay of prompt + answer predicts every greedy answer token (argmax).
(b3) Category log-probabilities from the eager scoring pass match a FlashAttention pass within 0.05 nats.
(c)  Token-grid orientation: greying one 28x28-px cell of a frame changes the vision embedding of the token
     at that cell the most (top-1 within one cell in at least 9 of 10 seeded trials).
Only validation videos are used; no lift or reliance is computed here.
"""
import json
import random
from pathlib import Path
import numpy as np
from vera_runner.reliance import CELL, GRID, eager_attention, mask_image
from vera_runner.verbalized import InternVL2Calls, learner_prompt
from .measure import Measurer, boxed_frames
from vera_runner.verbalized import sample_indices


def run_phase0(scorer, backend, cfg, root, out, n_videos):
    torch = backend.torch
    measurer = Measurer(scorer, cfg['data']['dir'], cfg['data']['frames'], cfg['data']['dir'], (root / cfg['learner_instruct_file']).read_text())
    question = learner_prompt(measurer.learner_template, (root / cfg['seed_questions_file']).read_text())
    rows = [r for r in measurer.rows['val'] if boxed_frames(r, sample_indices(r['frames_decoded']), measurer.boxes)][:n_videos]
    calls = InternVL2Calls(backend, cfg['runner']['generation'])
    details, b1, b2_agree, b2_total, b3 = [], [], 0, 0, []
    for r in rows:
        images, _ = measurer.images(r)
        gen = scorer.generate(images, question)
        batch_text = calls.learn([images], question)[0][0]
        b1.append(gen['text'] == batch_text)
        answer = [t for t in gen['ids'] if t != scorer.tok.bos_token_id]
        seq = torch.tensor([gen['query_ids'] + answer], device=backend.device)
        with torch.inference_mode(), eager_attention(scorer.lm):
            hidden = scorer.lm.model(inputs_embeds=scorer._embeds(images, seq), return_dict=True).last_hidden_state[0]
            start = len(gen['query_ids']) - 1
            logits = hidden[start:start + len(answer)].float() @ scorer.lm.output.weight.float().T
            pred = logits.argmax(-1).tolist()
        agree = sum(p == a for p, a in zip(pred, answer))
        b2_agree, b2_total = b2_agree + agree, b2_total + len(answer)
        eager = scorer.score(images, gen['query_ids'], gen['ids'], attention=True)
        flash = scorer.score(images, gen['query_ids'], gen['ids'], attention=False, eager=False)
        diff = max(abs(eager['category_logp'][c] - flash['category_logp'][c]) for c in eager['category_logp']) if eager and flash else None
        b3.append(diff)
        details.append({'video': r['id'], 'same_text': b1[-1], 'answer': gen['text'], 'batch_chat': batch_text,
                        'replay_agreement': agree / len(answer), 'max_logp_diff_eager_flash': diff,
                        'eager_p': eager['category_p'] if eager else None, 'flash_p': flash['category_p'] if flash else None})
    rng = random.Random(20261026)
    trials = []
    for t in range(10):
        r = rows[t % len(rows)]
        images, _ = measurer.images(r)
        k, row, col = rng.randrange(8), rng.randrange(GRID), rng.randrange(GRID)
        masked = mask_image(images[k], [[col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL]])
        with torch.inference_mode():
            a = scorer.model.extract_feature(backend.pixels([images[k]]))[0].float()
            b = scorer.model.extract_feature(backend.pixels([masked]))[0].float()
        change = (a - b).norm(dim=-1).cpu().numpy()
        top = int(np.argmax(change))
        rank = int((change > change[row * GRID + col]).sum())
        trials.append({'video': r['id'], 'slot': k, 'cell': [row, col], 'top_token': [top // GRID, top % GRID], 'rank_of_cell': rank,
                       'within_one_cell': abs(top // GRID - row) <= 1 and abs(top % GRID - col) <= 1})
    checks = {
        'b1_generate_equals_batch_chat': {'passed': all(b1), 'equal': sum(b1), 'videos': len(b1)},
        'b2_eager_replay_predicts_greedy_tokens': {'passed': b2_agree == b2_total, 'agreement': b2_agree / b2_total if b2_total else None,
                                                   'tokens': b2_total},
        'b3_eager_matches_flash': {'passed': all(d is not None and d < 0.05 for d in b3), 'max_abs_logp_diff': max(d for d in b3 if d is not None)},
        'c_token_grid_orientation': {'passed': sum(t['within_one_cell'] for t in trials) >= 9,
                                     'within_one_cell': sum(t['within_one_cell'] for t in trials), 'trials': len(trials)}}
    return {'checks': checks, 'videos': details, 'grid_trials': trials}
