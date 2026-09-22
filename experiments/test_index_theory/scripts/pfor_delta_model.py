# -*- coding: utf-8 -*-
"""
PForDelta（Patched Frame of Reference）整数压缩理论性能模型

PForDelta（Zhang, Long & Suel, 2008）是分帧位封装编码：把整数序列按
固定长度切帧，每帧选一个统一位宽打包 90% 的数值，装不下的异常值另存
异常区。Lucene 5.0+ 的默认 docID 压缩方案，倒排索引的标准组件。

核心结构：
- 每 frame_size（128 或 256）个整数为一帧
- 位宽选择：遍历 0-32 位，找能容纳 90% 数值的最小位宽 b
- 正常区：用 b 位打包那 90% 的数值，异常值的位置留空（填 0）
- 异常区：存 (异常值在帧中的下标, 原始值) 对
- 每帧开销：1 字节存 b + 1 字节存异常数 + 异常区

异常区本身用 Simple8b 压缩（下标天然小，原始值可能大）。

核心公式：
- 帧位宽 b = min{b' : 容纳 ≥90% 数值}（0-32 遍历）
- 帧字节 = ceil(frame_size × b / 8) + 2 + 异常区字节
- 压缩比(vs uint32) = 32 × n / 总字节
- 异常值比例 p 决定位宽：p 越低 b 越小，压缩比越高

异常值的污染被限制在帧内（最多 frame_size 个值受影响），
这是 PForDelta 相对 Simple8b 的核心改进：
Simple8b 的 1% 大值会把邻居一起抬进 30 位的字，
PForDelta 只在本帧内为异常值多付代价。

本模块给出位宽选择、帧编解码、异常区编码与压缩比闭式预测。
"""
from dataclasses import dataclass

# 帧大小默认值。128 是 Lucene 的默认，256 用于大批量场景。
DEFAULT_FRAME_SIZE = 128
# 异常值比例上限：装不下的数值比例超过这个阈值说明位宽选小了，
# 编码会继续往上加位宽，直到异常值比例降到阈值以下。
EXCEPTION_RATIO = 0.10
# 位宽搜索上限（32 位），超过则按定长处理。
MAX_BITS = 32

# Simple8b 选择器表：位宽 -> 每字数，用于压缩异常区。
# 下标天然小（0-255），原始值可能很大，两者位宽不同。
_SIMPLE8B_TABLE = [
    (1, 60), (2, 30), (3, 20), (4, 15), (5, 12), (6, 10), (7, 8),
    (8, 7), (10, 6), (12, 5), (15, 4), (20, 3), (30, 2), (60, 1),
]


def select_bits(values: list) -> int:
    """
    位宽选择：遍历 0-32 位，找能容纳 90% 数值的最小位宽。

    实现：从 1 位开始逐位加宽，统计能装下的数值个数，
    一旦达到 90% 就返回当前位宽。0 位只对全零数组成立（装下 100%）。
    """
    if not values:
        return 0
    n = len(values)
    threshold = n - int(n * EXCEPTION_RATIO)  # 至少装下 90%
    for bits in range(0, MAX_BITS + 1):
        limit = (1 << bits) - 1 if bits > 0 else 0
        fit = 0
        for v in values:
            if v <= limit:
                fit += 1
                if fit >= threshold:
                    return bits
    return MAX_BITS


def _simple8b_encode(arr: list) -> list:
    """异常区用的 Simple8b 编码。下标天然小，用窄档；原始值可能大。"""
    words = []
    i = 0
    n = len(arr)
    while i < n:
        need = max(1, arr[i].bit_length())
        for w, cnt in _SIMPLE8B_TABLE:
            if w < need:
                continue
            take = min(cnt, n - i)
            if max(arr[i:i + take]) >> w:
                continue
            word = 0
            for k in range(take):
                word |= arr[i + k] << (4 + k * w)
            # 选择器编号 = 表内下标，写进低 4 位
            word |= _SIMPLE8B_TABLE.index((w, cnt))
            words.append(word)
            i += take
            break
        else:
            raise ValueError(f"异常区值 {arr[i]} 需要 {need} 位，超过 60 位上限")
    return words


def _simple8b_decode(words: list, n: int) -> list:
    """异常区用的 Simple8b 解码。"""
    out = []
    for word in words:
        sel = word & 0xF
        w, cnt = _SIMPLE8B_TABLE[sel]
        for k in range(cnt):
            out.append((word >> (4 + k * w)) & ((1 << w) - 1))
    return out[:n]


def encode_pfor_delta(arr: list, frame_size: int = DEFAULT_FRAME_SIZE) -> list:
    """
    编码整数列表为帧列表。每帧是一个 dict：
    {
        "count": 帧内整数个数（末帧可能不足 frame_size）,
        "bits": 位宽 b,
        "exc_count": 异常值个数,
        "exc_words": [64 位字, ...]   # 异常区的 Simple8b 编码
        "words": [64 位字, ...]       # 正常区的位打包结果
    }

    正常区：b 位打包全部 frame_size 个值，异常值的位置填 0。
    异常区：存 (异常值在帧中的下标, 原始值) 的扁平序列，用 Simple8b 压缩。
    """
    frames = []
    for start in range(0, len(arr), frame_size):
        frame = arr[start:start + frame_size]
        bits = select_bits(frame)

        # 正常区：b 位打包，异常值填 0
        packed = []
        for v in frame:
            if v >> bits:
                packed.append(0)
            else:
                packed.append(v)

        # 位打包成 64 位字列表（帧内连续排列）
        words = []
        total_bits = len(packed) * bits
        for w_start in range(0, total_bits, 64):
            word = 0
            bit_pos = 0
            for k in range(w_start, min(w_start + 64, total_bits)):
                if packed[k // bits] & (1 << (k % bits)):
                    word |= 1 << bit_pos
                bit_pos += 1
            words.append(word)

        # 异常区：(下标, 原始值)
        exceptions = []
        for idx, v in enumerate(frame):
            if v >> bits:
                exceptions.append((idx, v))

        frames.append({
            "count": len(frame),
            "bits": bits,
            "exc_count": len(exceptions),
            "exc_words": _simple8b_encode(
                [x for pair in exceptions for x in pair]
            ),
            "words": words,
        })
    return frames


def decode_pfor_delta(frames: list, frame_size: int = DEFAULT_FRAME_SIZE) -> list:
    """
    解码帧列表为整数列表。先按位宽解出正常区，再把异常值按下标填回。
    末帧不足 frame_size 时按帧内记录的 count 截断。
    """
    out = []
    for frame in frames:
        bits = frame["bits"]
        words = frame["words"]
        count = frame["count"]

        # 从字列表还原位流，再按 b 位切出正常区值
        total_bits = count * bits
        bit_stream = []
        for word in words:
            for k in range(64):
                bit_stream.append((word >> k) & 1)
        bit_stream = bit_stream[:total_bits]

        normal = []
        for i in range(count):
            val = 0
            base = i * bits
            for k in range(bits):
                if bit_stream[base + k]:
                    val |= 1 << k
            normal.append(val)

        # 异常区：Simple8b 解出扁平的 (下标, 值) 序列，按下标填回
        flat = _simple8b_decode(frame["exc_words"], 2 * frame["exc_count"])
        for j in range(0, len(flat), 2):
            normal[flat[j]] = flat[j + 1]
        out.extend(normal)
    return out


def frame_exception_ratio(frame: list) -> float:
    """帧内异常值比例。"""
    if not frame:
        return 0.0
    bits = select_bits(frame)
    exc = sum(1 for v in frame if v >> bits)
    return exc / len(frame)


def predicted_bits_per_int(values: list, frame_size: int = DEFAULT_FRAME_SIZE) -> float:
    """
    闭式预测每整数平均位宽（含帧开销）。

    主体：90% 的数值按 b 位打包，b = 能覆盖 90% 的位宽。
    异常值：每个异常值多付 (原始位宽 - b) 的存储。
    帧开销：每帧 2 字节（1 字节 b + 1 字节异常数）+ 异常区字节。
    """
    if not values:
        return 0.0
    n = len(values)
    n_frames = (n + frame_size - 1) // frame_size

    # 主体位宽：按 90% 覆盖的位宽
    bits_sorted = sorted(v.bit_length() for v in values)
    b_b = max(1, bits_sorted[min(n - 1, int(n * 0.9))])

    # 异常值：超出 b_b 的值按原始位宽存，其余按 b_b 存
    total_value_bits = 0
    for v in values:
        if v.bit_length() > b_b:
            total_value_bits += v.bit_length()
        else:
            total_value_bits += b_b
    value_bits = total_value_bits

    # 帧开销：每帧 2 字节 + 异常区（每异常值 1 字节下标 + 原始值字节数）
    exception_bytes = 0
    for v in values:
        if v.bit_length() > b_b:
            exception_bytes += 1 + (v.bit_length() + 7) // 8
    overhead_bytes = n_frames * 2 + exception_bytes

    return (value_bits + overhead_bytes * 8) / n


def ratio_vs_uint32(frames: list, n: int) -> float:
    """压缩比 vs 定长 uint32（4 字节/整数）。"""
    if n == 0:
        return 1.0
    return (n * 4) / compressed_bytes(frames)


def compressed_bytes(frames: list) -> int:
    """
    压缩后总字节数（精确，按真实布局算）：
    - 正常区：位打包的 64 位字
    - 帧头：1 字节位宽 b + 1 字节异常数
    - 异常区：Simple8b 压缩的 (下标, 值) 扁平序列
    """
    total = 0
    for frame in frames:
        total += len(frame["words"]) * 8
        total += 2
        total += len(frame["exc_words"]) * 8
    return total


@dataclass
class PForDeltaSpec:
    """PForDelta 参数"""
    n: int = 1_000_000          # 整数个数
    frame_size: int = 128       # 帧大小
    exception_ratio: float = 0.10  # 异常值比例上限


def _selftest():
    """编解码可逆性与公式自洽性检查"""
    import random
    rng = random.Random(42)

    # 1. 可逆性：多种帧大小与分布
    cases = [
        [],
        [0],
        [1],
        list(range(128)),
        list(range(256)),
        [0, 1] * 100,
        [7] * 300,
        [1023] * 100,
        [2 ** 31] * 50,
        [1, 1000, 2 ** 20, 0, 5, 999] * 20,
        list(range(1, 1001)),
    ]
    for arr in cases:
        for fs in [16, 128, 256]:
            frames = encode_pfor_delta(arr, fs)
            back = decode_pfor_delta(frames, fs)
            assert back == arr, \
                f"编解码不可逆: 长度 {len(arr)} vs {len(back)}, frame_size={fs}"

    # 2. 随机可逆性（跨多个数量级）
    for _ in range(30):
        n = rng.randint(1, 600)
        top = rng.choice([2, 4, 16, 256, 1024, 2 ** 16, 2 ** 24, 2 ** 31])
        arr = [rng.randint(0, top) for _ in range(n)]
        for fs in [32, 128, 256]:
            frames = encode_pfor_delta(arr, fs)
            assert decode_pfor_delta(frames, fs) == arr, \
                f"随机可逆性失败 top={top} fs={fs}"

    # 3. 位宽选择：均匀分布应落在低档
    uniform = [rng.randint(0, 1000) for _ in range(10000)]
    bits = select_bits(uniform)
    assert 8 <= bits <= 12, f"0-1000 的位宽应在 8-12 之间, 实际 {bits}"

    # 4. 位宽单调性：最大值越大，位宽越大
    for a, b in [(15, 1000), (1000, 2 ** 20)]:
        la = [rng.randint(0, a) for _ in range(10000)]
        lb = [rng.randint(0, b) for _ in range(10000)]
        assert select_bits(lb) > select_bits(la), \
            f"最大值 {b} 的位宽应大于 {a}"

    # 5. 全零数组位宽为 0
    assert select_bits([0] * 100) == 0, "全零数组位宽应为 0"

    # 6. 闭式预测与实测吻合（docID 分布）
    docid = [rng.randint(0, 3000) for _ in range(100000)]
    for _ in range(1000):
        docid[rng.randrange(len(docid))] = rng.randint(3000, 2 ** 24)
    pred = predicted_bits_per_int(docid)
    meas = compressed_bytes(encode_pfor_delta(docid)) * 8 / len(docid)
    err = abs(pred - meas) / meas
    assert err < 0.20, f"预测 {pred:.2f} 与实测 {meas:.2f} 偏差 {err*100:.1f}%"

    print("selftest 全部通过")
    print(f"\n0-1000 均匀: 位宽 {select_bits(uniform)} bit, "
          f"{meas:.2f} bit/int (预测 {pred:.2f})")


if __name__ == "__main__":
    _selftest()
