"""Compare cached and full-prefix forward passes of a tiny random Qwen2 model."""
import json
import platform

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0)
    model = Qwen2ForCausalLM(config).eval().cpu()
    ids = torch.tensor([[2, 7, 11, 5, 9, 4, 3]], dtype=torch.long)
    prefill_length = 3
    max_error = 0.0
    cache_lengths = []
    greedy_matches = []
    with torch.inference_mode():
        prefix = model(input_ids=ids[:, :prefill_length], use_cache=True)
        past = prefix.past_key_values
        for end in range(prefill_length, ids.shape[1] + 1):
            full = model(input_ids=ids[:, :end], use_cache=False).logits[:, -1, :]
            if end == prefill_length:
                cached = prefix.logits[:, -1, :]
            else:
                result = model(input_ids=ids[:, end-1:end], past_key_values=past, use_cache=True)
                cached, past = result.logits[:, -1, :], result.past_key_values
            torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
            greedy_matches.append(torch.argmax(cached, dim=-1).item() == torch.argmax(full, dim=-1).item())
            cache_lengths.append(past.get_seq_length())
            assert cache_lengths[-1] == end
            max_error = max(max_error, (cached - full).abs().max().item())
    print(json.dumps({'device':'cpu', 'dtype':str(next(model.parameters()).dtype),
        'python':platform.python_version(), 'torch':torch.__version__,
        'transformers':transformers.__version__, 'layers':config.num_hidden_layers,
        'kv_heads':config.num_key_value_heads, 'input_ids':ids.tolist(),
        'prefill_length':prefill_length, 'compared_positions':len(cache_lengths),
        'cache_lengths':cache_lengths, 'greedy_matches':greedy_matches,
        'max_logit_abs_error':max_error,
        'test':'random-initialized model; teacher-forced input, not quality or speed'}, indent=2))


if __name__ == '__main__':
    main()
