# 数据量级扩展对比训练实测表

> 职责声明：本报告来自 climbmix / smoltalk 真实语料在单张 GB10 级 GPU 上的历史运行，测了分词器域偏差、词表规模与显存批次基准，没测多机分布式与最终模型能力；后续做词表选型、显存预算与数据量级规划时可引用，引用时注明为单卡历史实测。

生成时间: 2026-11-11T20:24:08+08:00

集群: NVIDIA GB10 (node0) + NVIDIA RTX Spark N1X 64GB (node1)
统一模型: {'d_model': 320, 'n_layers': 6, 'n_heads': 8, 'd_ff': 1280, 'block_size': 256}
统一留出验证集: karpathy/climbmix-400b-shuffle: shard_06542.parquet (2,000,000 chars 上限)

## 各量级预训练指标

| 量级 | max_chars | docs | train_tokens | tokens_seen | world_size | wall_s | tokens/s | final_val_loss | best_val_loss | BPB | PPL | Peak VRAM (MB) | CPU Data (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single/t1_micro_100k | 100,000 | 44 | 26,447 | 2,457,600 | 1 | 35.3 | 69,620.4 | 10.9045 | 9.0413 | 3.697447 | 98309.4259 | 1086.22 | 0.202 |
| single/t2_small_2m | 2,000,000 | 782 | 506,151 | 2,457,600 | 1 | 35.5 | 69,228.2 | 7.2568 | 7.0872 | 2.596178 | 3203.1122 | 1086.22 | 3.862 |
| single/t3_mid_20m | 20,000,000 | 7537 | 5,072,481 | 7,372,800 | 1 | 105.7 | 69,752.1 | 6.1866 | 6.1451 | 2.301836 | 1282.7073 | 1086.22 | 38.7 |
| failover/t2_small_2m | 2,000,000 | 782 | 506,151 | 2,457,600 | 1 | 35.4 | 69,423.7 | 7.2568 | 7.0872 | 2.596178 | 3203.1122 | 1086.22 | 3.862 |

## 接续 SFT 下游效果

| 量级 | SFT wall_s | SFT val_loss | 答案命中率 | SFT 后 BPB | SFT 后 PPL | SFT Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| single/t1_micro_100k | 13.2 | 5.3935 | 0.0 | 2.209108 | 961.4331 | 1017.47 |
| single/t2_small_2m | 13.2 | 4.7142 | 0.0 | 2.05992 | 604.6071 | 1017.47 |
| single/t3_mid_20m | 13.1 | 4.471 | 0.0 | 1.88374 | 349.6066 | 1017.47 |
| failover/t2_small_2m | 13.1 | 4.7142 | 0.0 | 2.05992 | 604.6071 | 1017.47 |

## 故障转移与断点续训记录

| 模式/量级 | 阶段 | 原因 | 故障前 world_size | 恢复位置 | 续训起始 step |
|---|---|---|---|---|---|
| failover/t2_small_2m | preflight | node1_unreachable_preflight | 2 | node0 (world_size=1) | 0 |
| failover/t2_small_2m | pretrain | exit_1 | 1 | node0 (world_size=1) | 200 |
