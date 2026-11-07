"""Conditional-expectation toy model for speculative decoding, not a benchmark.

Assume each successive draft token is accepted with fixed conditional probability
alpha. Each round emits accepted draft tokens and one correction or bonus token;
EOS is ignored. Timings are user assumptions, not predictions or measurements.
"""
import math


def expected_output(alpha: float, k: int) -> float:
    """Expected emitted tokens per round, including the correction/bonus token."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not isinstance(k, int) or k < 0:
        raise ValueError("alpha must be in [0, 1] and k a non-negative integer")
    return sum(alpha ** i for i in range(k + 1))


def speedup(alpha: float, k: int, cost_ratio: float, verification_ratio: float = 1.0) -> float:
    """Toy speedup: target verification and draft costs are relative to one baseline step."""
    if not math.isfinite(cost_ratio) or cost_ratio < 0 or not math.isfinite(verification_ratio) or verification_ratio <= 0:
        raise ValueError("cost_ratio must be non-negative; verification_ratio must be positive")
    return expected_output(alpha, k) / (verification_ratio + k * cost_ratio)


def solve_alpha_for_speedup(target_speedup: float, k: int, cost_ratio: float,
                            verification_ratio: float = 1.0) -> float:
    """Solve under fixed costs; reject unattainable or non-unique targets."""
    low = speedup(0, k, cost_ratio, verification_ratio)
    high = speedup(1, k, cost_ratio, verification_ratio)
    if not math.isfinite(target_speedup) or not low <= target_speedup <= high:
        raise ValueError("target outside the attainable interval")
    if k == 0:
        raise ValueError("alpha is not identifiable when k=0")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if speedup(mid, k, cost_ratio, verification_ratio) < target_speedup:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    print("Toy assumptions only: k=4, draft cost per token=0.05, verification cost=1")
    print("Assuming fixed conditional acceptance; EOS ignored; not a hardware measurement")
    print("alpha    expected tokens/round    modeled speedup")
    for alpha in (0.05, 0.70, 0.90):
        print(f"{alpha:5.2f}    {expected_output(alpha, 4):21.4f}    {speedup(alpha, 4, 0.05):15.4f}x")
