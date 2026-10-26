"""Diagnostic for Phase 0 b2/b3: where eager and flash replays disagree, and by how much."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/local/datasets/sp2117/firearm_visilab/code/VERA-grounding')
import contextlib
import run_firearm_grounding as g
from studies.firearm_grounding.measure import Measurer, boxed_frames
from vera_runner.reliance import eager_attention
from vera_runner.verbalized import learner_prompt, sample_indices
cfg = g.config('/local/datasets/sp2117/firearm_visilab/code/VERA-grounding/studies/firearm_grounding/grounding.json')
root = Path('/local/datasets/sp2117/firearm_visilab/code/VERA-grounding')
out = Path(cfg['root']) / 'phase0'
scorer, backend, info = g.open_scorer(cfg, out)
torch = backend.torch
m = Measurer(scorer, cfg['data']['dir'], cfg['data']['frames'], cfg['data']['dir'], (root / cfg['learner_instruct_file']).read_text())
q = learner_prompt(m.learner_template, (root / cfg['seed_questions_file']).read_text())
rows = [r for r in m.rows['val'] if boxed_frames(r, sample_indices(r['frames_decoded']), m.boxes)][:4]
report = []
for r in rows:
    images, _ = m.images(r)
    gen = scorer.generate(images, q)
    ans = [t for t in gen['ids'] if t != scorer.tok.bos_token_id]
    seq = torch.tensor([gen['query_ids'] + ans], device=backend.device)
    res = {}
    for mode in ('eager', 'flash'):
        ctx = eager_attention(scorer.lm) if mode == 'eager' else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            h = scorer.lm.model(inputs_embeds=scorer._embeds(images, seq), return_dict=True).last_hidden_state[0]
            start = len(gen['query_ids']) - 1
            logits = (h[start:start + len(ans)].float() @ scorer.lm.output.weight.float().T)
            top2 = logits.topk(2, -1)
            res[mode] = {'pred': logits.argmax(-1).tolist(), 'margin_of_greedy': [float(logits[i, a] - logits[i].topk(2).values[1] if logits[i].argmax() == a else logits[i, a] - logits[i].max()) for i, a in enumerate(ans)]}
    mism = {mode: [i for i, (p, a) in enumerate(zip(res[mode]['pred'], ans)) if p != a] for mode in res}
    e = scorer.score(images, gen['query_ids'], gen['ids'], attention=False, eager=True)
    f = scorer.score(images, gen['query_ids'], gen['ids'], attention=False, eager=False)
    report.append({'video': r['id'], 'tokens': len(ans), 'mismatch_eager': [(i, scorer.tok.decode([ans[i]]), round(res['eager']['margin_of_greedy'][i], 3), round(res['flash']['margin_of_greedy'][i], 3)) for i in mism['eager']],
                   'mismatch_flash': [(i, scorer.tok.decode([ans[i]]), round(res['flash']['margin_of_greedy'][i], 3)) for i in mism['flash']],
                   'min_margin_flash': round(min(res['flash']['margin_of_greedy']), 3),
                   'eager_logp': e['category_logp'] if e else None, 'flash_logp': f['category_logp'] if f else None,
                   'eager_p': e['category_p'] if e else None, 'flash_p': f['category_p'] if f else None})
    print(json.dumps(report[-1]), flush=True)
(out / 'phase0_diagnostic.json').write_text(json.dumps(report, indent=1) + '\n')
