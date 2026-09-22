# -*- coding: utf-8 -*-
"""
Varint（变长整数编码）理论性能模型（核心计算器）

每个字节用低 7 位存数据，最高位是 continuation bit（还有后续字节则为 1）。
1-127 用 1 字节，128-16383 用 2 字节，16384-2097151 用 3 字节，以此类推。

核心公式：
- 编码字节数：bytes(n) = ceil(bits(n) / 7)，bits(n) = max(1, n.bit_length())
- 每字节开销：continuation bit 占 1/8，等效数据 7/8 bit，开销 1/7 ≈ 14.3%（比特口径）
- 压缩比：raw_bytes / varint_bytes，raw_bytes = n × 定长字节数
- 分组压缩比：同组值全落在 [0, U] 时 ratio = itemsize / bytes(U)

小整数（<128）压缩效率最高（相对 uint32 4 倍）。大整数满量程时比定长 uint32 多 25%
（32 位需 5 字节，定长只需 4 字节，字节口径）。两种口径不矛盾：比特口径 14.3% 是
每字节 8 比特里 1 个 continuation bit，字节口径 25% 是 5 字节 vs 4 字节。

参数：
- itemsize: 定长编码字节数（uint32 = 4，uint64 = 8）
- U: 组内数值上界（同组数据的最大值），决定组内压缩比

Google Protocol Buffers、Apache Avro、gRPC、LevelDB、RocksDB 采用。
"""
import math


def encode_varint(n: int) -> bytes:
    """把非负整数编码为 varint 字节串。小端字节序，高位为 continuation bit。"""
    if n < 0:
        raise ValueError("varint 只编码非负整数，负数先走 zigzag")
    if n == 0:
        return b"\x00"
    out = bytearray()
    while n:
        byte = n & 0x7F
        n >>= 7
        if n:
            byte |= 0x80
        out.append(byte)
    return bytes(out)


def decode_varint(data: bytes, offset: int = 0) -> tuple:
    """从 offset 起解一个 varint，返回 (值, 消耗的字节数)。"""
    result = 0
    shift = 0
    consumed = 0
    while True:
        if offset + consumed >= len(data):
            raise ValueError("varint 被截断")
        byte = data[offset + consumed]
        consumed += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, consumed
        shift += 7
        if shift > 63:
            raise ValueError("varint 超过 64 位")


def varint_bytes(n: int) -> int:
    """n 的 varint 字节数（闭式解）。bytes = ceil(bits(n) / 7)。"""
    if n < 0:
        raise ValueError("负数无闭式解，先 zigzag")
    bits = max(1, n.bit_length())
    return math.ceil(bits / 7)


def varint_bytes_measured(n: int) -> int:
    """实测编码字节数，用于校验闭式解。"""
    return len(encode_varint(n))


def compression_ratio(raw_bytes: int, encoded_bytes: int) -> float:
    """压缩比 = 原始字节 / 编码后字节。"""
    return raw_bytes / encoded_bytes if encoded_bytes > 0 else 0.0


def group_ratio(itemsize: int, upper_bound: int) -> float:
    """
    同组压缩比（同组全部值落在 [0, upper_bound]）。
    定长占 itemsize 字节，varint 固定 bytes(U) 字节，ratio = itemsize / bytes(U)。
    """
    return itemsize / varint_bytes(upper_bound)


def per_byte_overhead() -> float:
    """
    每字节开销（相对有效数据比特）。每字节 8 比特含 1 个 continuation bit，
    有效数据 7 比特，开销 = 1/7 ≈ 14.3%。
    """
    return 1.0 / 7.0


def expected_bytes_uniform(lo: int, hi: int) -> float:
    """
    均匀分布 [lo, hi) 的期望 varint 字节数（闭式解）。
    按 7 比特分组的边界切开，E = sum_k k × P(bytes = k)。
    实测锚点：[0,1000) 得 1.872，[0,2^32) 得 4.937。
    """
    total = 0.0
    n = hi - lo
    if n <= 0:
        return 1.0
    # 值 v 的字节数 = ceil(bits(v)/7)，分组边界在 2^(7k)
    pos = lo
    group = 1
    while pos < hi:
        boundary = min(hi, 1 << (7 * group))   # bytes=group 的值域上界
        count = boundary - pos
        if count > 0:
            total += group * count
        pos = boundary
        group += 1
    return total / n


def expected_bytes_zipf(a: float, support_max: int = 2 ** 31) -> float:
    """
    Zipf 分布（P(X=k) ∝ k^-a，k ∈ [1, support_max]）的期望 varint 字节数（闭式解）。
    用尾部概率 P(X >= m) 累加分组，避免逐值求和。
    实测锚点：a=1.3 时约 1.21 字节/值（绝大多数值 < 128）。
    """
    # 归一化常数 sum_{k=1}^{support_max} k^-a
    # 用积分近似尾部 + 精确和算头部，避免大循环
    def tail_from(m: int) -> float:
        """sum_{k=m}^{support_max} k^-a，用欧拉-麦克劳林近似"""
        if m > support_max:
            return 0.0
        if a == 1.0:
            # 调和数尾部（积分近似），m=1 时下限取 0.5 而非 0
            upper = support_max + 0.5
            lower = m - 0.5 if m > 1 else 0.5
            return math.log(upper) - math.log(lower) + 0.5 * (1 / lower - 1 / upper)
        # 积分近似：∫_{m-0.5}^{N+0.5} x^-a dx
        upper = support_max + 0.5
        lower = m - 0.5
        return (upper ** (1 - a) - lower ** (1 - a)) / (1 - a)

    z = tail_from(1)
    total = 0.0
    group = 1
    while True:
        boundary = 1 << (7 * group)
        m_lo = 1 if group == 1 else (1 << (7 * (group - 1)))
        m_hi = min(support_max, boundary - 1)
        if m_lo > support_max:
            break
        # P(m_lo <= X <= m_hi) = tail(m_lo) - tail(m_hi+1)
        p = (tail_from(m_lo) - tail_from(m_hi + 1)) / z
        total += group * p
        group += 1
    return total


def uniform_ratio(itemsize: int, max_value: int) -> float:
    """
    均匀分布 [0, max_value] 的期望压缩比（按组内上界 U = max_value 估算）。
    均匀分布的期望 bytes 略低于固定 U 的情形，这里给上界估算的保守值。
    """
    return group_ratio(itemsize, max_value)


def selftest_roundtrip():
    """编解码可逆性抽查。返回抽查的样本数。"""
    samples = [0, 1, 127, 128, 16383, 16384, 2097151, 2**31 - 1, 2**63 - 1,
               2**64 - 1, 123456789]
    for n in samples:
        data = encode_varint(n)
        value, consumed = decode_varint(data)
        assert value == n, f"往返失败: {n} -> {data} -> {value}"
        assert consumed == len(data), "continuation bit 边界错"
    # 连续编解码（字节串串联）也要可逆
    buf = b"".join(encode_varint(n) for n in samples)
    pos = 0
    for n in samples:
        value, consumed = decode_varint(buf, pos)
        assert value == n, f"串联解码失败: {n}"
        pos += consumed
    assert pos == len(buf), "解码长度与编码长度不一致"
    return len(samples)


def _selftest():
    """公式自洽性检查"""
    # 1. 编解码可逆
    n_samples = selftest_roundtrip()

    # 2. 闭式解与实测字节数一致
    for n in [0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 2**32 - 1]:
        assert varint_bytes(n) == varint_bytes_measured(n), \
            f"字节数公式与实测不符: {n}"

    # 3. 分组边界正确
    assert varint_bytes(127) == 1, "127 应占 1 字节"
    assert varint_bytes(128) == 2, "128 应占 2 字节"
    assert varint_bytes(16383) == 2, "16383 应占 2 字节"
    assert varint_bytes(16384) == 3, "16384 应占 3 字节"

    # 4. 小整数相对 uint32 有 4 倍压缩
    assert group_ratio(4, 127) == 4.0, "小整数相对 uint32 应为 4 倍"

    # 5. 大整数相对 uint32 膨胀（满量程 32 位需 5 字节，定长只需 4 字节）
    r32 = group_ratio(4, 2**32 - 1)
    assert r32 < 1.0, f"uint32 上界下 varint 应膨胀, 实际 {r32:.3f}"
    assert abs(r32 - 4 / 5) < 1e-9, "膨胀倍数应为 4/5 = 0.8（即 1.25 倍膨胀）"

    # 6. 定长开销公式：itemsize 字节等价 itemsize×8 比特，
    #    等效数据比特 = itemsize×8×7/8 = itemsize×7，除以 7 得每字节 1 组
    assert group_ratio(8, 127) == 8.0, "小整数相对 uint64 应为 8 倍"

    # 7. 每字节开销 = 1/7
    assert abs(per_byte_overhead() - 1.0 / 7.0) < 1e-12

    # 8. 闭式期望字节数与实测锚点吻合
    assert abs(expected_bytes_uniform(0, 1000) - 1.872) < 0.01, \
        f"均匀 [0,1000) 期望字节数应约 1.872, 实际 {expected_bytes_uniform(0, 1000)}"
    assert abs(expected_bytes_uniform(0, 2 ** 32) - 4.937) < 0.01, \
        f"均匀 [0,2^32) 期望字节数应约 4.937, 实际 {expected_bytes_uniform(0, 2**32)}"
    z13 = expected_bytes_zipf(1.3)
    assert 1.1 < z13 < 1.4, f"Zipf a=1.3 期望字节数应约 1.2, 实际 {z13:.3f}"

    print("selftest 全部通过")
    print(f"\n编解码可逆性: {n_samples} 个样本往返一致（含串联解码）")
    print(f"  varint(0)      = {encode_varint(0).hex()}")
    print(f"  varint(127)    = {encode_varint(127).hex()}  ({varint_bytes(127)}B)")
    print(f"  varint(128)    = {encode_varint(128).hex()}  ({varint_bytes(128)}B)")
    print(f"  varint(16383)  = {encode_varint(16383).hex()}  ({varint_bytes(16383)}B)")
    print(f"  varint(16384)  = {encode_varint(16384).hex()}  ({varint_bytes(16384)}B)")
    print(f"  varint(2^32-1) = {encode_varint(2**32-1).hex()}  ({varint_bytes(2**32-1)}B)")
    print("\n相对定长编码的压缩比（按组内上界）:")
    for itemsize, name in [(2, "uint16"), (4, "uint32"), (8, "uint64")]:
        row = "  "
        for U, tag in [(127, "<128"), (16383, "<16K"), (2097151, "<2M"),
                       (2**32 - 1, "uint32上界"), (2**64 - 1, "uint64上界")]:
            r = group_ratio(itemsize, U)
            row += f"{tag}:{r:>6.2f}x  "
        print(f"  {name:>7} {row}")
    print("\n大整数膨胀有两种口径，不要混用：")
    print(f"  每字节开销（相对 7 比特有效数据）: {per_byte_overhead()*100:.1f}%")
    print(f"  满量程 uint32 -> varint: 5 字节 vs 4 字节 = "
          f"{1/group_ratio(4, 2**32-1):.3f}x（字节口径 25% 膨胀）")


if __name__ == "__main__":
    _selftest()
