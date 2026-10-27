"""One full-support temperature policy for rollouts, likelihoods and reference KL."""
TEMPERATURE = .8
VERSION = 'full-support-temperature-0.8-v1'


def generation_settings(tokenizer, sample, max_new_tokens):
    settings = {'do_sample': sample, 'num_beams': 1, 'max_new_tokens': max_new_tokens,
                'use_cache': True, 'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'bos_token_id': getattr(tokenizer, 'bos_token_id', None),
                'repetition_penalty': 1.}
    if sample:
        settings.update(temperature=TEMPERATURE, top_p=1., top_k=0)
    return settings


def policy_statistics(logits, tokens):
    logp = (logits / TEMPERATURE).log_softmax(-1)
    selected = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return selected, entropy
