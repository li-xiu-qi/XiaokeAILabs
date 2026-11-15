"""UNVALIDATED historical fit. Results and bottleneck attribution cannot be verified.
No throughput prediction is available until raw data and hold-out tests exist.
"""
import json
import glob
import math
import os

# 实测标定的硬件常数（Spark GB10）
PEAK_FLOPS = 89.3e12  # 89.3 TFLOPS (BF16 实测)

# 从数据拟合的参数
PYTHON_LIMIT = 3000.0  # Python 层开销限制的吞吐（小模型）
POWER_LAW_A = 1912.0   # 幂律系数
POWER_LAW_B = 0.241    # 幂律指数

def predict_throughput(n_params_m, batch_size=128, seq_len=128):
    """
    预测给定参数的模型的批处理吞吐。

    两阶段模型：
    1. 小模型（<100M）：throughput = PYTHON_LIMIT
    2. 大模型（>=100M）：throughput = POWER_LAW_A / n_params_m^POWER_LAW_B

    注意：这个模型假设 batch_size 和 seq_len 固定为 128。
    对于其他 batch_size 和 seq_len，需要额外的修正（见文档）。
    """
    raise NotImplementedError("unverified fit; batch and sequence effects are not modeled")

def fit_power_law(model_data):
    """
    从实测数据拟合幂律参数。

    用大模型（>100M）的数据来拟合 throughput = a / n_params^b
    """
    large_models = [d for d in model_data if d['params_m'] > 100]
    if len(large_models) < 2:
        return POWER_LAW_A, POWER_LAW_B

    # 线性回归：log(throughput) = log(a) - b*log(n_params)
    # 即 y = c + m*x，其中 y = log(throughput), x = log(n_params), c = log(a), m = -b
    import math
    xs = [math.log(d['params_m']) for d in large_models]
    ys = [math.log(d['batch_sweep_seq128']['bs128']['throughput_sps']) for d in large_models]

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x*y for x,y in zip(xs,ys))
    sum_x2 = sum(x*x for x in xs)

    # 最小二乘法
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    c = (sum_y - m * sum_x) / n

    a = math.exp(c)
    b = -m

    return a, b

def fit_python_limit(model_data):
    """
    从实测数据拟合 Python 层限制。

    用小模型（<100M）的平均峰值吞吐。
    """
    small_models = [d for d in model_data if d['params_m'] < 100]
    if not small_models:
        return PYTHON_LIMIT

    avg_throughput = sum(d['batch_sweep_seq128']['bs128']['throughput_sps']
                         for d in small_models) / len(small_models)
    return avg_throughput

def main():
    # 加载实测数据
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    model_data = []
    for f in glob.glob(result_dir + '/emb_bench_*.json'):
        try:
            d = json.load(open(f, encoding='utf-8'))
            if 'model' in d and 'batch_sweep_seq128' in d:
                model_data.append(d)
        except:
            continue

    if not model_data:
        print("No valid data found!")
        return

    # 拟合参数
    global PYTHON_LIMIT, POWER_LAW_A, POWER_LAW_B
    PYTHON_LIMIT = fit_python_limit(model_data)
    POWER_LAW_A, POWER_LAW_B = fit_power_law(model_data)

    print(f"=== Embedding 吞吐理论模型（Spark GB10, BF16）===")
    print(f"硬件常数：peak_FLOPS={PEAK_FLOPS/1e12:.1f} TFLOPS")
    print(f"拟合参数：python_limit={PYTHON_LIMIT:.0f} sent/s, "
          f"power_law: throughput = {POWER_LAW_A:.0f} / n_params^{POWER_LAW_B:.3f}")
    print()

    print(f"{'模型':<25} {'参数量':>8} {'层数':>6} {'实测吞吐':>10} {'理论吞吐':>10} {'误差':>8} {'瓶颈':>12}")
    print("-" * 95)

    for d in model_data:
        name = d['model']
        n_params_m = d['params_m']
        layers = d['layers']

        # 实测峰值吞吐（bs=128）
        actual_throughput = d['batch_sweep_seq128']['bs128']['throughput_sps']

        # 理论吞吐
        predicted_throughput = predict_throughput(n_params_m, batch_size=128, seq_len=128)

        # 误差
        error_pct = abs(predicted_throughput - actual_throughput) / actual_throughput * 100

        # 判断瓶颈
        if n_params_m < 100:
            bottleneck = "Python层"
        else:
            bottleneck = "GPU计算"

        print(f"{name:<25} {n_params_m:>6}M {layers:>6} "
              f"{actual_throughput:>8.0f}/s {predicted_throughput:>8.0f}/s "
              f"{error_pct:>6.1f}% {bottleneck:>12}")

    print()
    print("=== 外推能力测试（参数扫描，bs=128, seq=128）===")
    print(f"{'参数量':>8} {'预测吞吐':>10} {'预测延迟':>12} {'瓶颈':>12}")
    print("-" * 50)

    for n_params_m in [10, 22, 50, 100, 200, 500, 1000, 2000, 5000]:
        throughput = predict_throughput(n_params_m, batch_size=128, seq_len=128)
        latency_ms = 128 / throughput * 1000

        if n_params_m < 100:
            bottleneck = "Python层"
        else:
            bottleneck = "GPU计算"

        print(f"{n_params_m:>6}M {throughput:>8.0f}/s {latency_ms:>10.2f}ms {bottleneck:>12}")

    print()
    print("=== 批处理大小修正（基于实测数据）===")
    print("注：实测显示 bs=1 到 bs=128 吞吐提升约 2-6 倍，取决于模型大小")
    print("小模型（<100M）受 Python 层开销限制，batch 提升有限")
    print("大模型（>100M）受 GPU 计算限制，batch 提升明显")

    # 用实测数据展示 batch sweep
    print()
    print("=== 实测批处理扫描对比 ===")
    for d in model_data:
        name = d['model']
        n_params_m = d['params_m']
        br = d['batch_sweep_seq128']
        print(f"\n{name} ({n_params_m}M):")
        for k in sorted(br.keys()):
            if isinstance(br[k], dict):
                print(f"  {k}: {br[k]['throughput_sps']:.0f} sent/s")

if __name__ == '__main__':
    raise SystemExit("UNVALIDATED: original benchmark data missing; prediction disabled")
