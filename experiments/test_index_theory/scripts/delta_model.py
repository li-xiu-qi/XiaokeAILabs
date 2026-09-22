# -*- coding: utf-8 -*-
"""
Delta Encoding（差分编码）理论性能模型（核心计算器）

存储相邻元素的差值而非原始值：d[i] = a[i] - a[i-1]，a[0] 原样存。

核心公式：
- 差分：d[0] = a[0]，d[i] = a[i] - a[i-1]
- 还原：a[0] = d[0]，a[i] = a[i-1] + d[i]
- 升序数组差值小且为正，压缩效果极好（倒排索引 docID 列表的标准做法）
- 无序数组差值大且正负交替，比定长还差（差分会放大数值范围）

压缩比取决于差值分布，不取决于原始值分布。四种模式的闭式期望字节数：
- 等差升序：全部差值 = d，bytes = varint_bytes(d)
- 有序均匀（排序后的随机样本）：差值 ~ 指数分布，均值 m = 值域/n，
  E[bytes] = 1 + sum_{k>=1} exp(-2^(7k) / m)
- 大量重复：差值几乎全 0，varint(0) = 1 字节，bytes ≈ 1 + itemsize/n
- 无序均匀：差值 ~ 均匀(-N, N)，zigzag 后需 varint_bytes(2N) 字节，比定长差

配合 Varint 使用效果更佳（小差值 → 少字节）。负差值必须先走 Zigzag：
zigzag(n) = (n << 1) ^ (n >> 63)（把符号位挪到最低位，小绝对值占小编码）
"""
import math

from varint_model import encode_varint, decode_varint, varint_bytes


def delta_encode(arr) -> list:
    """
    差分编码。返回差值列表，长度与输入相同。
    arr 需支持索引与逐元素减法，不修改原数组。
    """
    if len(arr) == 0:
        return []
    out = [arr[0]]
    for i in range(1, len(arr)):
        out.append(arr[i] - arr[i - 1])
    return out


def delta_decode(deltas) -> list:
    """差分解码（前缀和还原）。返回值列表。"""
    if len(deltas) == 0:
        return []
    out = [deltas[0]]
    for i in range(1, len(deltas)):
        out.append(out[-1] + deltas[i])
    return out


def zigzag_encode(n: int) -> int:
    """
    有符号整数映射为非负整数。(n << 1) ^ (n >> 63) 在 Python 的
    任意精度算术右移下同样成立（负数右移补 1）。
    zigzag(0)=0, zigzag(-1)=1, zigzag(1)=2, zigzag(-2)=3
    """
    return (n << 1) ^ (n >> 63)


def zigzag_decode(z: int) -> int:
    """zigzag 的逆映射。(z >> 1) ^ -(z & 1)。"""
    return (z >> 1) ^ -(z & 1)


def encode_delta_varint(arr) -> bytes:
    """差分 + Varint。差值用 Varint 编码（要求非负，即升序输入）。"""
    return b"".join(encode_varint(d) for d in delta_encode(arr))


def decode_delta_varint(data: bytes, count: int) -> list:
    """差分 + Varint 的解码端。"""
    out = []
    pos = 0
    for _ in range(count):
        value, consumed = decode_varint(data, pos)
        out.append(value)
        pos += consumed
    return delta_decode(out)


def encode_delta_zigzag_varint(arr) -> bytes:
    """差分 + Zigzag + Varint。差值先 zigzag 再 Varint，可处理负差值。"""
    return b"".join(encode_varint(zigzag_encode(d)) for d in delta_encode(arr))


def decode_delta_zigzag_varint(data: bytes, count: int) -> list:
    """差分 + Zigzag + Varint 的解码端。"""
    out = []
    pos = 0
    for _ in range(count):
        value, consumed = decode_varint(data, pos)
        out.append(zigzag_decode(value))
        pos += consumed
    return delta_decode(out)


def ratio_ascending_arith(itemsize: int, step: int) -> float:
    """
    等差升序（全部差值 = step）的压缩比。ratio = itemsize / bytes(step)。
    step < 128 时相对 uint32 有 4 倍压缩。
    """
    return itemsize / varint_bytes(step)


def ratio_sorted_uniform(itemsize: int, n: int, value_range: int) -> float:
    """
    有序均匀（n 个样本均匀撒在 [0, value_range) 后排序）的压缩比。
    差值近似指数分布，均值 m = value_range / n。
    E[bytes] = 1 + sum_{k>=1} exp(-2^(7k) / m)
    """
    m = value_range / n
    if m <= 0:
        return 1.0
    expected = 1.0
    k = 1
    while True:
        term = math.exp(-(2 ** (7 * k)) / m)
        if term < 1e-12:
            break
        expected += term
        k += 1
    return itemsize / expected


def ratio_heavy_duplicates(itemsize: int, n: int) -> float:
    """
    大量重复（差值几乎全为 0）的压缩比。n 个值占 n 字节（varint(0)=1B）
    加首值 itemsize 字节，大 n 下 ratio ≈ itemsize。
    注：纯 Varint 对全 0 序列只能到 4 倍（uint32 基准），
    叠加 RLE 才能突破（连续重复存三元组）。
    """
    return itemsize * n / (itemsize + (n - 1))


def ratio_unsorted_uniform(itemsize: int, value_range: int) -> float:
    """
    无序均匀的压缩比（差分失效的边界情形）。
    差值 ~ 均匀(-N, N)，zigzag 后落在 [0, 2N)，需 bytes(2N) 字节。
    value_range = 2^32 时约 0.8 倍，即比定长 uint32 差 25%。
    """
    return itemsize / varint_bytes(2 * value_range)


def _selftest():
    """公式自洽性检查"""
    # 1. 差分编解码可逆（三种模式各查一个）
    arrays = {
        "等差升序": list(range(0, 1000, 7)),
        "含重复": [5, 5, 5, 5, 6, 6, 100, 100, 100, 101],
        "无序随机": [1000, 3, 5000, 7, 42, 999, 0, 88],
        "负差值": [100, 50, 0, -30, -80, 10, -1000],
    }
    for name, arr in arrays.items():
        assert delta_decode(delta_encode(arr)) == arr, f"{name} 差分往返失败"
        # 差分 + zigzag + varint 全程可逆
        encoded = encode_delta_zigzag_varint(arr)
        assert decode_delta_zigzag_varint(encoded, len(arr)) == arr, \
            f"{name} 差分+zigzag+varint 往返失败"

    # 2. zigzag 可逆且保持小绝对值占小编码
    for n in [-2, -1, 0, 1, 2, 127, -128, 16383, -16384, 2**31 - 1, -(2**31)]:
        assert zigzag_decode(zigzag_encode(n)) == n, f"zigzag 往返失败: {n}"
    assert zigzag_encode(0) == 0 and zigzag_encode(-1) == 1 and zigzag_encode(1) == 2
    # zigzag 后 |n| 越大编码越大，负数不被浪费在符号位上
    assert zigzag_encode(-1) < zigzag_encode(2**31 - 1)

    # 3. 纯非负序列（升序）也能走纯 varint 路径并往返一致
    asc = list(range(0, 5000, 3))
    encoded = encode_delta_varint(asc)
    assert decode_delta_varint(encoded, len(asc)) == asc, "升序差分+varint 往返失败"

    # 4. 闭式公式与实测锚点吻合（itemsize=8，int64 数组的定长基准）
    assert abs(ratio_ascending_arith(8, 127) - 8.0) < 1e-9, "等差 step<128 应 8 倍"
    assert abs(ratio_sorted_uniform(8, 1_000_000, 2**31) - 8 / 1.942) < 0.01, \
        "有序均匀应约 4.12 倍"
    assert ratio_heavy_duplicates(8, 1_000_000) > 7.9, "大量重复应接近 8 倍"
    assert abs(ratio_unsorted_uniform(8, 2**32) - 8 / 5) < 1e-9, \
        "无序均匀应 8/5 = 1.60 倍"

    # 5. 升序等差压缩比远优于无序均匀（差分的价值与边界）
    assert ratio_ascending_arith(8, 100) > ratio_unsorted_uniform(8, 2**32), \
        "升序应远优于无序"

    print("selftest 全部通过")
    print(f"\n差分可逆性: {len(arrays)} 种模式往返一致（含负差值）")
    print(f"  zigzag(-1)={zigzag_encode(-1)}  zigzag(1)={zigzag_encode(1)}  "
          f"zigzag(-2)={zigzag_encode(-2)}  zigzag(2)={zigzag_encode(2)}")
    print("\n各模式相对 uint64 的压缩比（闭式）:")
    print(f"  等差升序 step=1     : {ratio_ascending_arith(8, 1):.2f}x")
    print(f"  等差升序 step=100   : {ratio_ascending_arith(8, 100):.2f}x")
    print(f"  有序均匀 n=100万/2^31: {ratio_sorted_uniform(8, 1_000_000, 2**31):.2f}x")
    print(f"  大量重复 n=100万    : {ratio_heavy_duplicates(8, 1_000_000):.2f}x")
    print(f"  无序均匀 [0,2^32)   : {ratio_unsorted_uniform(8, 2**32):.2f}x")


if __name__ == "__main__":
    _selftest()
