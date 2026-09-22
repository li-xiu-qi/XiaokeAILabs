# -*- coding: utf-8 -*-
"""生成随机向量用于扩展数据规模测试。"""
import numpy as np

def gen(n, dim, seed=42):
    """生成 n 条 dim 维随机 float32 向量。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype("float32")
