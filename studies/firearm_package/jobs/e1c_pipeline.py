"""A16 38B probes, declared prompt decisions, then resumable detector arms."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vera_runner.detector import INTERNVL_PROMPT


def choose_prompt(calls):
    count = sum(bool(c['found']) for c in calls if c['category'] != 'No_Gun' and c['prompt'] == INTERNVL_PROMPT)
    return ('detection' if count else 'referring'), count


def main():
    cfg = json.loads((ROOT / 'studies/firearm_package/package.json').read_text())
    out = Path(cfg['out_root'])
    arms = json.loads((ROOT / 'studies/firearm_package/e1_arms.json').read_text())
    names = ('internvl35_38b_off', 'internvl35_38b_on')
    decisions = []
    for name in names:
        path = out / 'probes' / (name + '.json')
        # The completed SDPA off-probe is preserved across this pipeline restart.
        if not path.exists():
            subprocess.run([sys.executable, '-B', 'run_firearm_package.py', 'probe', '--arm', name], cwd=ROOT, check=True)
        probe = json.loads(path.read_text())
        if probe['model'] != arms[name]['model'] or probe['thinking'] != arms[name]['thinking'] or len(probe['calls']) != 12:
            raise ValueError('Incompatible or incomplete probe: ' + name)
        prompt, count = choose_prompt(probe['calls'])
        decisions.append({'arm': name, 'prompt': prompt, 'parseable_firearm_detection_calls': count})
    (out / 'probe_decisions_38b.json').write_text(json.dumps(decisions, indent=2) + '\n')
    # Record the permitted decisions before collecting either 38B ablation arm.
    with (ROOT / 'studies/firearm_package/DEVIATIONS.md').open('a') as log:
        log.write('\n### Automated 38B probe decisions (before ablation)\n\n')
        for d in decisions:
            log.write(f"- `{d['arm']}`: {d['parseable_firearm_detection_calls']} firearm probe calls with a parseable detection-prompt box; use the {d['prompt']} prompt under the declared rule. Raw probes and `package/probe_decisions_38b.json` preserve the evidence.\n")
    manifest = json.loads((out / 'e1_manifest.json').read_text())
    for d in decisions:
        arm = arms[d['arm']]
        expected = {f['frame_id'] for f in manifest if arm['frames'] == 'all' or f[arm['frames']]}
        for attempt in range(3):
            command = [sys.executable, '-B', 'run_firearm_package.py', 'detect', '--arm', d['arm'], '--prompt', d['prompt']]
            result = subprocess.run(command, cwd=ROOT)
            path = out / 'e1' / d['arm'] / 'records.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            successful = {r['frame_id'] for r in records if r['status'] != 'failed'}
            if result.returncode == 0 and successful == expected:
                break
        else:
            raise RuntimeError('Incomplete 38B arm after three attempts: ' + d['arm'])


if __name__ == '__main__':
    main()
