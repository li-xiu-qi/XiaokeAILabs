# -*- coding: utf-8 -*-
"""
算法：Dynamic Token Size Chunking（DFC），复现 2603.06976 Table 2

论文 spec（原文）："Chunk sizes vary dynamically within predefined bounds to
balance granularity and context. Each chunk size Ki is selected such that
Ki ∈ [Kmin, Kmax]"，配置 min=50, max=200 tokens。

**论文欠定义处**：spec 只给了取值区间，没给 Ki 的选择机制。本实现采用
论文自己的 CDAC（Content Density Adaptive Chunking）密度概念作为解释性
选择机制（论文 Table 2 中 CDAC 与 DFC 相邻，均属 Adaptive/Dynamic 类）：

    ρ(x) = |V(x)| / |x|        （局部词汇密度：唯一 token 数 / 总 token 数）
    Ki ∝ ρ^−1                  （越密 → 目标块越小，越稀 → 越大）
    Ki = clamp(K_mid * ρ_base / ρ_local, Kmin, Kmax)

    K_mid = (Kmin+Kmax)/2 = 125 为区间中点；ρ_base 为文档级密度作参考基准。

封块规则：句子贪心累积，达到当前块的 Ki 即封；若封出块 < Kmin 则并入
后续（不产生断点）。口径在报告中标注为「论文欠定义下的解释性实现」。

与 fixed 的区别：块大小随局部内容密度在 [50,200] 内浮动，边界对齐句子。
与 semantic 的区别：不用 embedding，纯词汇统计信号。
"""
from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class DTCSplitter(BaseSplitter):
    name = "dtc"
    kind = "lexical"

    def __init__(self, k_min=50, k_max=200, density_window=3):
        if k_min >= k_max:
            raise ValueError("k_min must be < k_max")
        self.k_min = k_min
        self.k_max = k_max
        self.k_mid = (k_min + k_max) // 2
        self.density_window = density_window  # 局部密度估计的句数窗口

    def _density(self, token_id_lists, idx0, idx1):
        """[idx0, idx1) 句的词汇密度 = 唯一 token 数 / 总 token 数。"""
        all_ids = []
        for ids in token_id_lists[idx0:idx1]:
            all_ids.extend(ids)
        if not all_ids:
            return 0.0
        return len(set(all_ids)) / len(all_ids)

    def split(self, text):
        enc = TextEncoder.get()
        tok = enc.tok
        spans = sentence_token_spans(text, tok)
        if len(spans) < 2:
            return Segmentation(algo=self.name, boundaries=[],
                                n_tokens=spans[-1][2] if spans else 0)

        # 每句的 token 数与 token id（用于密度估计）
        sent_ids = [tok(s, add_special_tokens=False)["input_ids"]
                    for s, _, _ in spans]
        sent_sizes = [len(ids) for ids in sent_ids]
        total = spans[-1][2]

        # 文档级密度作参考基准
        rho_base = self._density(sent_ids, 0, len(sent_ids))
        if rho_base <= 0:
            rho_base = 1.0

        bounds = []
        i = 0
        n_sent = len(spans)
        while i < n_sent:
            # 当前块起点处的局部密度（往后看 density_window 句）
            j_end = min(i + self.density_window, n_sent)
            rho_local = self._density(sent_ids, i, j_end)
            if rho_local <= 0:
                rho_local = rho_base
            # 密度反比定目标块大小，钳制到 [k_min, k_max]
            k_target = max(self.k_min,
                           min(self.k_max,
                               int(round(self.k_mid * rho_base / rho_local))))

            # 贪心累积句子：达到 k_target 封块，或加下句会超 k_max 也封
            cur = 0
            j = i
            while j < n_sent:
                if cur + sent_sizes[j] > self.k_max and cur > 0:
                    break
                cur += sent_sizes[j]
                j += 1
                if cur >= k_target:
                    break

            # 封块检查：不足 k_min 且后面还有内容 → 并入下一轮（不产生断点）
            if cur < self.k_min and j < n_sent:
                i = j
                continue

            # 在最后一句的末尾产生断点（j-1 句的结束 token 位置）
            pos = spans[j - 1][2]
            if 0 < pos < total:
                bounds.append(Boundary(
                    pos=pos,
                    reason=f"DTC 密度目标 {k_target}tok（块 {cur}tok）"))
            i = j

        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
