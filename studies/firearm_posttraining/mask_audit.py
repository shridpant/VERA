"""CPU audit of target bytes and actual pinned-tokenizer supervision masks."""
import argparse
import importlib.util
import inspect
import subprocess
from pathlib import Path
from .data import load_split, write_json, digest, canonical_hash
from .model import target_parts, token_kinds, instruction, NativeModel


def sft_source_audit(reference_module):
    spec = importlib.util.spec_from_file_location('studies.firearm_posttraining._sft_reference', reference_module)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    functions = {'instruction': (instruction, previous.instruction),
                 'target_parts': (target_parts, previous.target_parts),
                 'token_kinds': (token_kinds, previous.token_kinds)}
    for name in ('__init__', 'assert_trainable', 'adapter_hash', 'prompt', 'encode', 'completion',
                 'joined', 'logits', 'sft_loss', 'save_adapter'):
        functions['NativeModel.' + name] = (getattr(NativeModel, name), getattr(previous.NativeModel, name))
    if any(inspect.getsource(a) != inspect.getsource(b) for a, b in functions.values()):
        raise RuntimeError('SFT source changed; old warm-up cannot be reused')
    current_trainer = Path(__file__).with_name('trainer.py')
    if digest(current_trainer) != digest(Path(reference_module).with_name('trainer.py')):
        raise RuntimeError('trainer source changed; old warm-up cannot be reused')
    return {'sft_training_source_unchanged': True, 'sft_unchanged_functions': sorted(functions),
            'sft_reference_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                 cwd=Path(reference_module).resolve().parents[2], text=True).strip(),
            'sft_reference_model_sha256': digest(reference_module),
            'sft_trainer_sha256': digest(current_trainer)}


def run(model, data, output, previous_module, sft_reference_module=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    spec = importlib.util.spec_from_file_location('studies.firearm_posttraining._previous_mask', previous_module)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    rows, max_tokens, retained, unknown, hashes = 0, 0, 0, 0, []
    for split in ('fit', 'stop'):
        for row in load_split(data, split):
            rows += 1
            for grounded in (False, True):
                text, spans = target_parts(row, grounded)
                old_text, old_spans = previous.target_parts(row, grounded)
                if text != old_text:
                    raise RuntimeError('mask correction changed target bytes')
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                offsets = encoded['offset_mapping']
                kinds, old_kinds = token_kinds(offsets, spans), previous.token_kinds(offsets, old_spans)
                max_tokens = max(max_tokens, len(kinds) + 1)
                for (a, b), kind, old in zip(offsets, kinds, old_kinds):
                    touches_unknown = any(a < y and b > x and z == 0 for x, y, z in spans)
                    if (kind == 0) != touches_unknown:
                        raise RuntimeError('known token masked or unknown payload supervised')
                    unknown += touches_unknown
                    retained += old == 0 and kind != 0
                hashes.append([split, row['id'], grounded, canonical_hash(text)])
    if max_tokens > 512:
        raise RuntimeError('common completion cap exceeded')
    result = {'status': 'passed', 'target_mask': 'known-schema-unknown-payload-v2',
              'videos': rows, 'target_bytes_unchanged': True, 'maximum_completion_tokens': max_tokens,
              'formerly_masked_known_tokens_retained': retained, 'unknown_payload_tokens_masked': unknown,
              'previous_model_sha256': digest(previous_module),
              'current_model_sha256': digest(Path(__file__).with_name('model.py')),
              'target_inventory_sha256': canonical_hash(hashes)}
    if sft_reference_module:
        result.update(sft_source_audit(sft_reference_module))
    write_json(output, result, immutable=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'data', 'output', 'previous-module'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--sft-reference-module')
    a = p.parse_args()
    print(run(a.model, a.data, a.output, a.previous_module, a.sft_reference_module))
