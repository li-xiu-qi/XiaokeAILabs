# BFPRT 选择算法：Python 与 Rust 双实现

同一个 Top-K 选择问题的两种语言实现，对比的是算法落地后的真实性能，不是算法本身的两次抄写。BFPRT（中位数的中位数）解决的是快速选择在最坏情况下退化到 O(n²) 的问题，通过精心挑选主元保证最坏 O(n)。

## 子目录

| 目录 | 实现 | 内容 |
|---|---|---|
| `python/` | Python | `test_bfprt.ipynb`：BFPRT、快速选择、插入排序完整实现，正确性与边界测试，两者实测性能对比和复杂度分析 |
| `rust/` | Rust | `src/` 算法实现，`benches/selection_algorithms.rs` 用 criterion 做基准，plotly 出对比图 |

## 这个实验真正回答的问题

BFPRT 理论上最坏 O(n)，快速选择最坏 O(n²)，但平均都是 O(n)。工程上该选哪个，取决于数据分布和实现常数：BFPRT 的分组、找中位数、递归选主元带来不小的常数开销，随机分布数据上快速选择几乎不会遇到最坏情况，反而更快；只有在数据可能被构造出恶意序列（比如在线服务接收外部输入）时，BFPRT 的最坏情况保证才值得这笔开销。python notebook 的「算法选择建议」一节用实测验证了这个取舍。

## 运行

```bash
# Python：直接跑 notebook
# Rust 基准
cd rust
cargo bench
```
