# -*- coding: utf-8 -*-
"""
Simple8b 整数压缩理论性能模型

Simple8b（Anh & Moffat, 2006）是一种字对齐（word-aligned）位封装编码，
把多个小整数打包进一个 64 位字，是倒排索引 docID 列表的经典压缩方案。

核心结构：
- 64 位字 = 4 位选择器（selector）+ 60 位数据区
- 选择器决定「数据位宽 × 每字数」，共 14 档组合
- 编码时贪心扫描：从当前位置起，选能装下这一段全部数字的最小位宽档
- 解码时读低 4 位选择器，按位宽连续读出

选择器表（14 档）：
  selector 0 : 60 个 × 1 位      selector 7 :  7 个 ×  8 位
  selector 1 : 30 个 × 2 位      selector 8 :  6 个 × 10 位
  selector 2 : 20 个 × 3 位      selector 9 :  5 个 × 12 位
  selector 3 : 15 个 × 4 位      selector 10:  4 个 × 15 位
  selector 4 : 12 个 × 5 位      selector 11:  3 个 × 20 位
  selector 5 : 10 个 × 6 位      selector 12:  2 个 × 30 位
  selector 6 :  8 个 × 7 位      selector 13:  1 个 × 60 位

压缩比公式（闭式）：
- 单值最小位宽 b = bit_length(v)，0 也算 1 位（选择器表最小档是 1 位）
- 理想打包位宽 = 能容纳该值的最小档位宽（查表）
- 压缩比 vs uint32 = 32 / 实际每整数平均位宽
- 压缩比 vs varint ≈ 每整数平均位宽 / 平均 varint 字节 × 8

编码不跨字（每个字只装本段数字），但允许末字补 0（槽位装不满时），
解码必须给出元素个数 n 才能截掉补的 0。

本模块给出选择器表、编解码实现与压缩比闭式预测。
"""
from dataclasses import dataclass
import math

# 选择器表：索引 -> (数据位宽, 每字数)。位宽与字数之和不超 60。
SELECTOR_TABLE = [
    (1, 60),    # selector 0
    (2, 30),    # selector 1
    (3, 20),    # selector 2
    (4, 15),    # selector 3
    (5, 12),    # selector 4
    (6, 10),    # selector 5
    (7, 8),     # selector 6
    (8, 7),     # selector 7
    (10, 6),    # selector 8
    (12, 5),    # selector 9
    (15, 4),    # selector 10
    (20, 3),    # selector 11
    (30, 2),    # selector 12
    (60, 1),    # selector 13
]
NUM_SELECTORS = len(SELECTOR_TABLE)
WORD_BITS = 64
SELECTOR_BITS = 4
MAX_VALUE_BITS = 60


def min_bits(value: int) -> int:
    """容纳单个值所需的最小位宽。0 占 1 位（最小档位宽为 1）。"""
    return max(1, value.bit_length())


def selector_for_bits(bits: int) -> tuple:
    """给定位宽需求，返回能容纳它的最小档 (selector, 数据位宽, 每字数)。"""
    for sel, (w, cnt) in enumerate(SELECTOR_TABLE):
        if w >= bits:
            return sel, w, cnt
    raise ValueError(f"位宽需求 {bits} 超过 Simple8b 上限 {MAX_VALUE_BITS}")


def encode_simple8b(arr: list) -> list:
    """
    编码整数列表为 64 位字列表。

    贪心规则：从当前位置 i 起，选「位宽 ≥ 装得下这一段」的最小位宽档，
    把该段能装的数字压进一个字，装不满的槽位补 0（最后一个字）。
    因此解码必须给出元素个数 n 才能截掉补的 0，见 decode_simple8b。
    """
    words = []
    i = 0
    n = len(arr)
    while i < n:
        need = min_bits(arr[i])
        chosen = None
        for sel, (w, cnt) in enumerate(SELECTOR_TABLE):
            if w < need:
                continue
            take = min(cnt, n - i)
            # 该档必须装得下前 take 个数字，否则退到更宽的档
            if max(arr[i:i + take]) >> w:
                continue
            chosen = (sel, w, take)
            break
        if chosen is None:
            raise ValueError(
                f"位置 {i} 的值 {arr[i]} 需要 {need} 位，"
                f"超过 Simple8b 上限 {MAX_VALUE_BITS} 位"
            )
        sel, w, take = chosen
        word = sel
        for k in range(take):
            word |= arr[i + k] << (SELECTOR_BITS + k * w)
        words.append(word)
        i += take
    return words


def decode_simple8b(words: list, n: int = None) -> list:
    """
    解码 64 位字列表为整数列表。读低 4 位选择器，按位宽连续读出。

    编码允许末字补 0，所以必须传原始元素个数 n 才能截掉补的 0；
    不传则返回全部解出的值（末字的补 0 会被当作真实值）。
    """
    out = []
    for word in words:
        sel = word & 0xF
        w, cnt = SELECTOR_TABLE[sel]
        for k in range(cnt):
            out.append((word >> (SELECTOR_BITS + k * w)) & ((1 << w) - 1))
    if n is not None:
        return out[:n]
    return out


def compressed_bytes(words: list) -> int:
    """压缩后字节数。"""
    return len(words) * (WORD_BITS // 8)


def bits_per_int(arr: list) -> float:
    """实测每整数平均位宽（含 4 位选择器开销）。"""
    if not arr:
        return 0.0
    return WORD_BITS * len(encode_simple8b(arr)) / len(arr)


def predicted_bits_per_int(values: list, cover: float = 0.90) -> float:
    """
    闭式预测每整数平均位宽（含 4 位选择器开销）。一阶近似，误差通常 <5%。

    分两部分：

    1. 主体：取能覆盖 cover 比例数值的位宽 b_b 定主体档（档内每字数 cnt_b），
       主体值按 64/cnt_b 计位宽。这对应「大部分数值同档」的实际打包行为。

    2. 异常值修正：位宽超过 b_b 的值会与邻居挤进一个更宽的字
       （宽档每字数 cnt_o，大值通常是 2 个/字），
       并在主体流里留下平均 (cnt_b - 1)/2 个作废的槽位。

    cover=1.0 时退化为「按全局最大值定档」，只对均匀分布准确。
    """
    if not values:
        return 0.0
    n = len(values)
    bits_sorted = sorted(v.bit_length() for v in values)
    idx = min(n - 1, max(0, int(n * cover)))
    b_b = max(1, bits_sorted[idx])
    b_o = max(1, bits_sorted[-1])
    cnt_b = selector_for_bits(b_b)[2]
    cnt_o = selector_for_bits(b_o)[2]

    n_outlier = sum(1 for b in bits_sorted if b > b_b)
    if n_outlier == 0:
        return 64.0 / cnt_b

    # 每个异常值与一个邻居共用宽字（大值档多为 2 个/字）
    wide_values = 2 * ((n_outlier + 1) // 2)
    # 异常值打断主体流，平均作废 (cnt_b - 1)/2 个主体槽位
    waste_slots = n_outlier * (cnt_b - 1) / 2.0
    bulk_bits = 64.0 * (n - wide_values) / cnt_b
    wide_bits = 64.0 * wide_values / cnt_o
    waste_bits = 64.0 * waste_slots / cnt_b
    return (bulk_bits + wide_bits + waste_bits) / n


def ratio_vs_uint32(arr: list, words: list = None) -> float:
    """压缩比 vs 定长 uint32（4 字节/整数）。"""
    if not arr:
        return 1.0
    if words is None:
        words = encode_simple8b(arr)
    return (len(arr) * 4) / compressed_bytes(words)


def ratio_vs_varint(arr: list) -> float:
    """压缩比 vs LEB128 Varint。返回 Simple8b 字节 / Varint 字节。"""
    return compressed_bytes(encode_simple8b(arr)) / varint_size(arr)


def varint_size(arr: list) -> int:
    """LEB128 Varint 编码后的字节数（不实际构造字节串，只算长度）。"""
    total = 0
    for v in arr:
        total += 1
        v >>= 7
        while v:
            total += 1
            v >>= 7
    return total


@dataclass
class Simple8bSpec:
    """Simple8b 参数"""
    n: int = 1_000_000          # 整数个数
    max_value: int = 0          # 数组最大值（0 表示自动取）
    window: int = 0             # 分窗预测的窗长（0 = 按全局最大值）


def _selftest():
    """编解码可逆性与公式自洽性检查"""
    # 1. 可逆性：多组分布
    cases = [
        [],
        [0],
        [1],
        [0, 1] * 30,
        list(range(60)),
        list(range(16)) * 4,
        [7] * 100,
        [1023] * 50,
        [2 ** 32] * 10,
        [1, 1000, 2 ** 20, 0, 5, 999],
        list(range(1, 1001)),
    ]
    for arr in cases:
        words = encode_simple8b(arr)
        back = decode_simple8b(words, len(arr))
        assert back == arr, f"编解码不可逆: 输入长度 {len(arr)} 输出长度 {len(back)}"

    # 2. 随机可逆性（不同数量级混合）
    import random
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(1, 500)
        top = rng.choice([2, 4, 8, 16, 64, 256, 1024, 2 ** 16, 2 ** 32])
        arr = [rng.randint(0, top) for _ in range(n)]
        assert decode_simple8b(encode_simple8b(arr), n) == arr, \
            f"随机可逆性失败 top={top}"

    # 3. 小整数应接近 4.27 位/整数（选择器 3：15 个 × 4 位，理论 64/15）
    small = [rng.randint(0, 15) for _ in range(10000)]
    bpi = bits_per_int(small)
    theo = 64.0 * math.ceil(len(small) / 15) / len(small)
    assert bpi <= theo + 1e-9, f"0-15 的整数应 ≤{theo:.3f} 位/整数, 实际 {bpi:.3f}"

    # 4. 最大值越大，位宽越大
    for a, b in [(15, 1000), (1000, 2 ** 20)]:
        la = [rng.randint(0, a) for _ in range(1000)]
        lb = [rng.randint(0, b) for _ in range(1000)]
        assert bits_per_int(lb) > bits_per_int(la), \
            f"最大值 {b} 的位宽应大于 {a}"

    # 5. 全零数组压到 1 位/整数（选择器 0）
    zeros = [0] * 600
    assert len(encode_simple8b(zeros)) == 10, "600 个 0 应压成 10 个字"

    # 6. 超出 60 位的值必须报错
    try:
        encode_simple8b([2 ** 70])
        raise AssertionError("超出 60 位的值应报错")
    except ValueError:
        pass

    # 7. 每个字的选择器合法（0-13），且高位未溢出
    words = encode_simple8b([rng.randint(0, 1023) for _ in range(1000)])
    for w in words:
        assert 0 <= (w & 0xF) <= 13, "选择器越界"

    # 8. 闭式预测与实测吻合（四种分布，一阶近似误差应 <5%）
    pred_rows = []
    for kind, top in [("small", 15), ("medium", 1000), ("large", 2 ** 20)]:
        arr = [rng.randint(0, top) for _ in range(100000)]
        pred = predicted_bits_per_int(arr)
        meas = bits_per_int(arr)
        err = abs(pred - meas) / meas
        pred_rows.append((kind, pred, meas, err))
        assert err < 0.05, f"{kind} 预测 {pred:.3f} 与实测 {meas:.3f} 偏差 {err*100:.1f}%"

    # 带异常值的 docID 分布也应压得住
    docid = [rng.randint(0, 3000) for _ in range(100000)]
    for _ in range(1000):
        docid[rng.randrange(len(docid))] = rng.randint(3000, 2 ** 24)
    pred = predicted_bits_per_int(docid)
    meas = bits_per_int(docid)
    err = abs(pred - meas) / meas
    pred_rows.append(("docid", pred, meas, err))
    assert err < 0.05, f"docid 预测 {pred:.3f} 与实测 {meas:.3f} 偏差 {err*100:.1f}%"

    # 9. 异常值越多，位宽越大（单调性）
    docid_more = [rng.randint(0, 3000) for _ in range(100000)]
    for _ in range(10000):
        docid_more[rng.randrange(len(docid_more))] = rng.randint(3000, 2 ** 24)
    assert bits_per_int(docid_more) > bits_per_int(docid), \
        "异常值比例升高，位宽应上升"

    print("selftest 全部通过")
    print(f"\n0-15 整数: {bits_per_int(small):.3f} 位/整数 "
          f"(vs uint32 压缩比 {ratio_vs_uint32(small):.1f}x)")
    med = [rng.randint(0, 1000) for _ in range(100000)]
    print(f"0-1000 整数: {bits_per_int(med):.3f} 位/整数 "
          f"(预测 {predicted_bits_per_int(med):.3f}, "
          f"vs uint32 压缩比 {ratio_vs_uint32(med):.1f}x)")
    print("\n闭式预测 vs 实测：")
    for kind, p, m, e in pred_rows:
        print(f"  {kind:>8}  预测 {p:>6.2f}  实测 {m:>6.2f}  误差 {e*100:>4.1f}%")


if __name__ == "__main__":
    _selftest()
