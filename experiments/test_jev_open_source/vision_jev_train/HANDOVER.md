# 交接：多模态 Jev 训练流水线

状态时间：2026-09-24 23:10（远端时间）

## 当前状态：训练进行中

dgx-spark 已由用户重启恢复，内存 118 GB 可用，无残留进程。

训练命令（正在跑，batch 2，1200 步）：

```bash
cd /home/ke/vision_jev && ~/xinfer-env/bin/python train_vision_v2.py \
  --model /home/ke/models/Qwen3.5-2B \
  --data /home/ke/vision_jev/prepared \
  --out /home/ke/vision_jev/runs/v1 \
  --steps 1200 --bs 2 --mem-cap 45 --mem-min-run 45
```

日志在 `/home/ke/vision_jev/v1.log`，逐步记录在 `/home/ke/vision_jev/runs/v1/events.jsonl`。

## 关键实测数据

Qwen3.5-2B，ScienceQA，平均每问题 3.2 候选：

| batch | 峰值内存 | 可用余量 | 结局 |
|---|---|---|---|
| 2 | 33.89 GB | 78 GB | 跑通 smoke 25 步 |
| 3 | 56.51 GB | 50.6 GB | 第 63 步被闸门拦下 |
| 4 | 55.92 GB | 48.6 GB | 第 3 步被闸门拦下 |

4B 模型单独加载只占 8.46 GB，加载后很空，第一个 backward 就 OOM，说明 4B 的问题在激活不在权重。2B 是 24 层 hidden 2048，激活量约为 4B 的 0.48 倍。

完整的显存纪律和方法论写在 pkm-hub 的 `Projects/001_AI与工具/AI协作学/Qwen3.5多模态微调显存纪律.md`。

## 修掉的三个 bug

1. `truncation=True` 会把图像占位符序列截断，导致 text 与 input_ids 的 image token 数不一致，processor 抛 ValueError。改为拼接前按字符裁 state。serve.py 同因，一并改。
2. 内存闸门写成 `avail < (128 - mem_cap)`，在 121 GB 机器上反推出 78 GB 阈值，与实际可用值相等导致每次误判停训。改为独立的 `--mem-min-run` 绝对阈值。
3. 相对路径：`nohup` 的 cwd 是 home 不是 `~/vision_jev`，`--data prepared` 解析不到文件，报 FileNotFoundError。所有路径用绝对路径。

## 已产出的实测结果（2026-09-25 00:18）

训练：1200 步完成，46.6 分钟，峰值 40.93 GB，可用内存全程 63 GB 以上，闸门未触发。

Adapter：`/home/ke/vision_jev/runs/v1/final/adapter_model.safetensors`，43.7 MB。

校准温度：best_T=4.5（网格 0.5 到 4.5 的 41 点，calib_soft_ce=1.11191）。
train.py 内置网格上限只有 2.0，会触顶给出假的最优值。

| 划分 | 条数 | acc | soft_ce | brier | ECE | acc(校准后) | soft_ce(校准后) | ECE(校准后) |
|---|---|---|---|---|---|---|---|---|
| test | 1000 | 0.3930 | 1.4793 | 0.3434 | 0.2172 | 0.3930 | 1.1889 | 0.0679 |
| dev | 400 | 0.3700 | 1.4619 | 0.3560 | 0.2714 | 0.3700 | 1.1093 | 0.0535 |
| calibration | 400 | 0.4225 | 1.3922 | 0.3253 | 0.2216 | 0.4225 | 1.1109 | 0.0735 |

test acc 0.3930 高于 dev 的 0.3700，没有明显过拟合。test 的 ECE 从 0.2172
降到 0.0679，校准有效且泛化到了未见数据。

acc 比随机基线（ScienceQA 平均 3.2 选项，随机约 0.31）高得有限，原因：
1200 步 LoRA batch 2 只训了约 0.31 个 epoch。教程里可以直接讲，想提准确率
需要加步数或换更大底座。ECE 从 0.27 降到 0.05 说明温度校准确实有效。

评测耗时 21.16 分钟（基于 `evaluate.py` 的图像预 resize 优化；train.py 内置
评测跑 1000 条要十几分钟以上，因为 processor 的 smart_resize 是单线程且
同一张图被每个候选重复处理）。

## 醒来后的验收

```bash
ssh ke@192.168.1.170 "tail -30 /home/ke/vision_jev/v1.log; echo '--- report ---'; cat /home/ke/vision_jev/runs/v1/report.json"
```

report.json 里看 `temperature`（校准温度）、`test_calibrated.acc`（校准后测试准确率）、`peak_gb`（峰值内存 GB）、`minutes`（耗时）。

如果还在训练，看 events.jsonl 的 loss 趋势和 mem_peak_gb。峰值超过 45 GB 或可用低于 45 GB 时闸门会自动停训并保存，不会再把机器打死。

## 下一步

1. 训练完成后记录准确率、峰值内存（GB）、耗时，作为教程素材。
2. 训练期间没留配图，下次跑训练类教程要在开跑前规划截图位：loss 曲线、内存曲线、report 数字、推理示例。
3. 文章目录参照文本版建在 `pkm-hub-write/datawhale-公众号写作项目/02_文章/非商业稿/` 下。
