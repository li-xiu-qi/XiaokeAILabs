# 人脸身份 LoRA 微调实验

> 本目录是图片微调学习线当前唯一在做的实验：用 20 张照片让 SDXL 底模学会画一个
> 它本来画不出来的人。这里写清测什么、当前结论、怎么复现。前两个风格实验已终止，
> 结论见 `../README.md`。

## 效果

数据：CelebA 单人身份（`celeb_id=14`），20 张训练 / 10 张测试。下面四列共用同一个
提示词、同一个随机种子、同一套生成参数，唯一差别是有没有挂 LoRA。
相似度 = 生成图与本人测试集的 ArcFace embedding 余弦相似度（与本人测试集均值算）。

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

- 底模：SDXL base（本地合并 `vkqzm` 词元的版本，复用时直接当 trigger）
- LoRA：linear rank 32 / alpha 32 / dropout 0.05，lr 1e-4 adamw，bf16，EMA 0.99
- 步数 2000（20 张 × 100 epoch），save_every 500；分辨率 512
- caption：`vkqzm, a photo of a man`
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

按顺序跑（脚本内的路径变量按作者训练机编写，复现时按自己的环境调整）：

```bash
# 1. 数据：从 HF parquet 按 celeb_id 切出，交替划分训练/测试
bash scripts/export_celeb_person.sh 14

# 2. 数据审查：四项机器检查 + 生成联系表
bash scripts/qc_celeb_annotations.sh
bash scripts/make_celeb_sheet.sh

# 3. 训练（ai-toolkit，配置见 config/face_identity_sdxl_lora.yaml）
bash scripts/train_face_proart.sh

# 4. 评估：三对照（底模基线 / 微调后 / 负对照）
bash scripts/face_baseline.sh          # 底模基线，三个对照一起算
bash scripts/face_eval.sh              # 微调后，与基线同提示词同 seed
bash scripts/face_ckpt_scan.sh         # 检查点扫描，零训练成本定最优点

# 5. 出图：四列对比图（本页首图）
bash scripts/make_face_compare.sh
```

评估栈：ArcFace `w600k_r50.onnx`（R50，512 维）+ RetinaFace `det_10g.onnx`，
onnxruntime + insightface；安装见 `scripts/install_face_stack.sh` 与
`scripts/install_face_models.sh`（走阿里云 pypi 镜像）。

## 硬件与网络

| 机器 | GPU | 角色 |
|---|---|---|
| proart-temp | RTX Spark N1X，64 GB，WSL2 arm64 | 训练与评估 |
| dgx-spark | GB10 | 备用，被本地 llama-server 占 72 GiB |

训练机不通 huggingface.co 与 commons.wikimedia.org，只通 metmuseum.org 与
mirrors.aliyun.com；HF 数据走「本地镜像下载 → scp 到训练机」。

## 目录结构

```
config/
  face_identity_sdxl_lora.yaml        v1：2000 步 / lr 1e-4 / 无 caption dropout
  face_identity_sdxl_lora_v2.yaml     v2：800 步 / lr 5e-5 / dropout 0.15
docs/
  assets_celeb/                       数据集联系表、基线图、微调对比图、原始 metrics
scripts/
  # 数据获取与切分
  pick_celeb.sh                        读 parquet，按 celeb_id 找样本量合适的身份
  export_celeb_person.sh               导出指定身份的图，交替切分训练/测试
  # 数据审查
  qc_celeb_annotations.sh              四项机器检查：身份一致性/跨类污染/重复泄漏/质量
  make_celeb_sheet.sh                  训练+测试合并联系表
  make_celeb_split_sheet.sh            训练/测试分开放大，供人工目检
  # 底模能力探针（立项前用）
  probe_candidates.sh                  候选风格各生成 3 seed，判断底模会不会
  make_probe_sheet.sh                 探针结果联系表
  # 训练
  train_face_proart.sh                 v1 训练入口
  train_face_v2_proart.sh              v2 训练入口
  # 评估
  face_baseline.sh                     底模基线（三对照里的 B）
  face_eval.sh                         微调后评估，与基线同提示词同 seed
  face_ckpt_scan.sh  face_ckpt_scan_v2.sh   检查点扫描，零成本定最优点
  make_face_compare.sh                 微调前后对比图（本页首图）
  # 人脸识别栈安装
  install_face_stack.sh                onnxruntime + insightface（走阿里云 pypi 镜像）
  install_face_models.sh               ArcFace 权重装入 insightface 模型目录
```
