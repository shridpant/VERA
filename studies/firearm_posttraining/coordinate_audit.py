"""Immutable CPU audit of native coordinate targets, masks and scoring equivalence."""
import argparse
import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
from .data import load_split, write_json, digest, canonical_hash
from .model import target_parts, token_kinds, instruction, NativeModel
from . import rewards


def previous_module(path, name):
    spec = importlib.util.spec_from_file_location('studies.firearm_posttraining._coordinate_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(model, data, output, previous_release, codebook):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    old_root = Path(previous_release) / 'studies/firearm_posttraining'
    old_model = previous_module(old_root / 'model.py', 'model')
    old_reward = previous_module(old_root / 'rewards.py', 'rewards')
    # Only representation-related functions may change. Prove the actual
    # scoring functions and training update machinery retain exact source.
    for name in ('iou', 'matching', 'box_quality', 'score', 'parse_labels'):
        assert inspect.getsource(getattr(rewards, name)) == inspect.getsource(getattr(old_reward, name)), name
    for name in ('trainer.py', 'sampling.py', 'fit_format.py', 'rl_panel.py', 'stopping.py', 'evaluation.py', 'report.py'):
        assert digest(Path(__file__).with_name(name)) == digest(old_root / name), name
    for name in ('sft_loss', 'completion', 'log_probs', 'generate'):
        assert inspect.getsource(getattr(NativeModel, name)) == inspect.getsource(getattr(old_model.NativeModel, name)), name
    book = json.loads(Path(codebook).read_text())
    assert instruction(book) == old_model.instruction(book)
    inventory, rows, boxes, unknown, maximum, error = [], 0, 0, 0, 0, 0.
    for split in ('fit', 'stop'):
        for row in load_split(data, split):
            rows += 1
            for grounded in (False, True):
                text, spans = target_parts(row, grounded)
                old_text, _ = old_model.target_parts(row, grounded)
                if not grounded:
                    assert text == old_text
                new, old = json.loads(text), json.loads(old_text)
                if grounded:
                    for a, b in zip(new['grounding'], old['grounding']):
                        assert a['frame_id'] == b['frame_id']
                        assert (a['boxes'] is None) == (b['boxes'] is None)
                        if a['boxes'] is not None:
                            for native, unit in zip(a['boxes'], b['boxes']):
                                boxes += 1
                                assert 0 <= native[0] < native[2] <= 1000 and 0 <= native[1] < native[3] <= 1000
                                error = max(error, max(abs(x / 1000 - y) for x, y in zip(native, unit)))
                        else:
                            # Unknown teacher payloads stay masked, never become
                            # negative annotation. Empty lists here solely make
                            # paired synthetic responses for scoring equivalence.
                            a['boxes'] = b['boxes'] = []
                actual = rewards.score(json.dumps(new), row, book, grounded)
                prior = old_reward.score(json.dumps(old), row, book, grounded)
                assert actual.keys() == prior.keys()
                for key in actual:
                    if type(actual[key]) is float:
                        assert abs(actual[key] - prior[key]) < 1e-12, (row['id'], key)
                    else:
                        assert actual[key] == prior[key], (row['id'], key)
                encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                kinds = token_kinds(encoded['offset_mapping'], spans)
                maximum = max(maximum, len(kinds) + 1)
                for (a, b), kind in zip(encoded['offset_mapping'], kinds):
                    touches = any(a < y and b > x and z == 0 for x, y, z in spans)
                    assert (kind == 0) == touches
                    unknown += touches
                inventory.append([split, row['id'], grounded, canonical_hash(text)])
    assert maximum <= 512 and error < 1e-15
    result = {'status': 'passed', 'coordinate_version': rewards.COORDINATE_VERSION,
              'target_mask': 'known-schema-unknown-payload-v2', 'target_bytes_unchanged': False,
              'label_targets_and_prompt_unchanged': True, 'scoring_source_unchanged': True,
              'training_update_source_unchanged': True, 'fresh_warmups_required': True,
              'videos': rows, 'known_boxes_checked': boxes, 'maximum_unit_roundtrip_error': error,
              'maximum_completion_tokens': maximum, 'unknown_payload_tokens_masked': unknown,
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'previous_release': str(previous_release), 'target_inventory_sha256': canonical_hash(inventory),
              'code_hashes': {name: digest(Path(__file__).with_name(name)) for name in ('model.py', 'rewards.py', 'coordinate_audit.py')},
              'data_hashes': {name: digest(Path(data) / name) for name in ('fit.json', 'stop.json')},
              'no_reporting_split_access': True}
    write_json(output, result, immutable=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'output', 'previous-release'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    a = p.parse_args()
    print(json.dumps(run(a.model, a.data, a.output, a.previous_release, a.codebook), indent=2))
