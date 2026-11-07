"""Historical training-scaling arithmetic, not inference or model-quality measurement.

Kaplan et al., Eq. 1.5 and Table 2: https://arxiv.org/abs/2001.08361
Hoffmann et al.: https://arxiv.org/abs/2203.15556
The empirical coefficients are dataset, tokenizer, model and procedure dependent.
"""
import math

# Joint fit in Table 2, not the separate single-variable fits from Eq. 1.1/1.2.
N_C = 6.4e13
D_C = 1.8e13
ALPHA_N = 0.076
ALPHA_D = 0.103


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def kaplan_loss(params_b: float, data_t: float) -> float:
    """Eq. 1.5: WebText2 early-stopped test-loss fit, non-embedding N in B, D in T."""
    _positive(params_b, "params_b")
    _positive(data_t, "data_t")
    n, d = params_b * 1e9, data_t * 1e12
    return ((N_C / n) ** (ALPHA_N / ALPHA_D) + D_C / d) ** ALPHA_D


def chinchilla_optimal_tokens(params_b: float) -> float:
    """Illustrative 20 tokens/parameter assumption; result is billion tokens."""
    _positive(params_b, "params_b")
    return params_b * 20


def chinchilla_optimal_flops(params_b: float) -> float:
    """Training FLOPs under the illustrative D=20N and C≈6ND assumptions."""
    d = chinchilla_optimal_tokens(params_b) * 1e9
    return 6 * params_b * 1e9 * d


def flops_to_params_for_budget(flops: float, tokens_per_param: float = 20):
    """Billion parameters satisfying C≈6ND and D=tokens_per_param*N."""
    _positive(flops, "flops")
    _positive(tokens_per_param, "tokens_per_param")
    return math.sqrt(flops / (6 * tokens_per_param)) / 1e9


def self_test():
    print("Historical training-scaling illustration; estimates are not measurements.")
    print("Kaplan Eq. 1.5/Table 2 estimates early-stopped WebText2 loss,")
    print("using non-embedding parameters; it does not predict deployment quality.")
    for params_b, data_t in [(1, 0.02), (7, 0.14), (70, 1.4)]:
        print(f"N={params_b}B (non-embedding), D={data_t}T: fitted loss={kaplan_loss(params_b, data_t):.4f}")
    print("D=20N and C≈6ND are illustrative training-compute assumptions,")
    print("not a universal optimum and not an inference cost model.")
    print(f"70B and D=20N: {chinchilla_optimal_tokens(70):.0f}B training tokens")


if __name__ == "__main__":
    self_test()
