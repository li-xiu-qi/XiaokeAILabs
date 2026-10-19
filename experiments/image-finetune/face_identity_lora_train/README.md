# 人脸身份 LoRA 微调实验

> 用 20 张照片让 SDXL 底模学会画一个它本来画不出来的人。这里写清测什么、
> 结论是什么、怎么复现。

## 效果

数据：CelebA 单人身份（`celeb_id=14`），20 张训练 / 10 张测试。下面四列共用同一个
提示词、同一个随机种子、同一套生成参数，唯一差别是有没有挂 LoRA。
相似度 = 生成图与本人测试集的 ArcFace embedding 余弦相似度。

![微调前后对比](docs/assets_celeb/face_compare.jpg)

| 生成方式 | vs 本人 | 说明 |
|---|---|---|
| 底模（无 LoRA） | **−0.017** | 画出来的是另一个人 |
| 微调后 1000 步（最佳检查点） | **0.3232** | 同一个人 |
| 微调后 2000 步（过训） | 0.2461 | 训练过量的反例 |
| 参照：这个人自己的照片之间 | 0.7159 | 「同一个人」的水平线 |

## 怎么证明是微调的功劳

相似度从 −0.017 到 0.3232 看起来够大，但「人脸之间本来就相似」可能造成假阳性，
所以评估同时算两个对照（原始数据在 `docs/assets_celeb/baseline.json` 与
`scan_v1.json`）：

- **负对照**：同一批微调后的生成图，与另一个人的相似度只有 0.0546——模型学到的
  是这个特定身份，不是一张通用的人脸。0.3232 与 0.0546 的净差 0.27。
- **正对照**：这个人自己的训练集 vs 测试集相似度 0.7159，是「同一个人」的上界。
  微调后 0.3232 约为上界的 45%——方向对，但没学满，这也是下一步的靶子。

底模对这一身份的输出与对陌生人无法区分（−0.017 vs 0.0249），说明靶子是真的：
提示词没法把一张不存在的脸描述出来，而 20 张照片的 LoRA 可以。

## 数据

20 张训练图全部来自 CelebA（HuggingFace `flwrlabs/celeba`），按 `celeb_id` 切出、
交替切分，测试集不进训练。原图仅 178×218，按原始分辨率训练——放大到 512 训练
只会教插值模糊，这一点后来被证实是瓶颈。

![训练集联系表](docs/assets_celeb/celeb_14_train.jpg)

训练前对数据做过审查：身份一致性、跨身份污染、重复与训练/测试泄漏、质量
四项机器检查全部通过，联系表经人工目检后才开训。

## 训练配置

- 底模：SDXL base
- LoRA：linear rank 32 / alpha 32 / dropout 0.05，lr 1e-4 adamw，bf16，EMA 0.99
- 步数 2000（20 张 × 100 epoch），save_every 500；分辨率 512
- caption：`vkqzm, a photo of a person`
- 评估与基线严格同口径：同提示词（仅多 trigger）、同 seed（1000+i）、同生成参数

## 结论与局限

1. **微调有效**：底模对该身份的输出与陌生人无法区分（−0.017 vs 0.0249），
   微调后 0.3232，与负对照的净差 0.27。
2. **只学到部分身份**：0.3232 对正对照 0.7159 约 45%；检查点扫描每轮生成 10 张，
   人脸检出 6 到 9 张（1000 步 7 张）。
3. **瓶颈是数据分辨率，不是训练参数**：检查点扫描 500 / 1000 / 1500 / 2000 步 →
   0.3110 / 0.3232 / 0.2518 / 0.2461，1000 步见顶后下滑（过拟合）；第二组实验
   降 lr、减步数、加 caption dropout 反而略降（800 步 0.2623）。原图仅 178×218，
   放大训练，LoRA 学到的是插值模糊。

下一步：换 1024² 数据源（如 CelebA-HQ）或同一身份更多张，重跑同口径评估。

## 复现

环境：Python 3.10+、CUDA GPU、[ai-toolkit](https://github.com/ostris/ai-toolkit)。

```bash
# 0. 依赖：onnxruntime + insightface，并检查 ArcFace 权重是否就位
bash scripts/setup_env.sh

# 1. 数据：从 HF parquet 切出一个身份，交替划分训练/测试，写训练 caption
python scripts/prepare_data.py --parquet /path/to/celeba_shard0.parquet --also-negative

# 2. 审查：四项机器检查（不过审不开训），再出联系表人工目检
python scripts/audit_data.py
python scripts/make_contact_sheet.py

# 3. 训练（配置见 config/face_identity_lora.yaml）
AI_TOOLKIT=/path/to/ai-toolkit/run.py bash scripts/train.sh

# 4. 评估：底模基线 → 微调后 → 全检查点扫描
python scripts/evaluate.py baseline
python scripts/evaluate.py lora --lora output/face_identity_lora/face_identity_lora.safetensors
python scripts/evaluate.py scan --run-dir output/face_identity_lora

# 5. 出图：四列对比图（本页首图），数字直接读自上一步的 JSON
python scripts/make_figure.py
```

路径都走环境变量，不写死在脚本里：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FACE_DATA_DIR` | `./data` | 数据根，下面按 `celeb_<id>/{train,test}` 组织 |
| `FACE_MODELS_DIR` | `~/.insightface/models/buffalo_l` | ArcFace 两个 onnx 的位置 |
| `FACE_OUT_DIR` | `./output` | 评估产物与图片 |
| `SDXL_BASE` | `stabilityai/stable-diffusion-xl-base-1.0` | 底模，本地路径或 HF id |
| `AI_TOOLKIT` | `~/ai-toolkit/run.py` | ai-toolkit 入口 |

评估栈：ArcFace `w600k_r50.onnx`（R50，512 维）+ RetinaFace `det_10g.onnx`，
onnxruntime + insightface。两个权重从 InsightFace `buffalo_l` 发布包获取，
本仓不转存。

## 目录结构

```
config/
  face_identity_lora.yaml        v1：2000 步 / lr 1e-4 / 无 caption dropout
  face_identity_lora_v2.yaml     v2：800 步 / lr 5e-5 / dropout 0.15（对过拟合的尝试）
docs/
  assets_celeb/                  对比图、训练/测试集、基线图、原始 metrics
scripts/
  common.py                      共享：路径解析、ArcFace 编码器、身份原型
  prepare_data.py                切出身份、交替划分、写 caption、生成 manifest
  audit_data.py                  四项机器检查，不过审以非零码退出
  make_contact_sheet.py          训练/测试联系表，供人工目检
  train.sh                       ai-toolkit 训练入口（路径全部可配）
  setup_env.sh                   安装评估栈并校验权重
  evaluate.py                    三个子命令：baseline / lora / scan
  make_figure.py                 四列对比图，数字读自 JSON
```

`data/` 与 `output/` 不入库：数据由 `prepare_data.py` 从 CelebA 重新导出，
评估产物按 run 可重新生成。
