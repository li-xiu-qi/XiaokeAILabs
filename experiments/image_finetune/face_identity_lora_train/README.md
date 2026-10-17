# 人脸身份 LoRA 微调实验

图片微调学习线当前唯一在做的实验。前两个实验（通用浮世绘风格、歌川国芳个人风格）
已终止并归档，结论见 `pkm-hub-lab/image-style-lora/`。

## 实验目的

**证明微调有效。**

判据只有一条，写在这里防止范围蔓延：

> 如果底模本来就能生成这类图，这个数据集就没有实验价值。

要找一个底模**怎么调提示词都做不出来**的对象，微调之后做出来了。
选的是 CelebA 里的单个身份：底模的权重里没有这个具体的人，
提示词无法把一张不存在的脸描述出来，而 20 张照片的 LoRA 可以。

配套的客观指标是人脸识别模型的 embedding 余弦相似度（ArcFace 512 维），
不依赖参照集选择，判定门槛可以事先写死。

## 当前结论

数据：CelebA `celeb_id=14`，20 张训练 / 10 张测试（交替切分，测试集不进训练）。

| 量 | 值 |
|---|---|
| 正对照（本人训练集 vs 测试集，同一人的基准线） | 0.7159 |
| 底模基线 vs 本人 | **−0.0172** |
| 底模基线 vs 陌生人（负对照） | 0.0249 |
| v1 最佳（1000 步）vs 本人 | **0.3232** |
| v2 最佳（800 步，lr 5e-5 + caption dropout）vs 本人 | 0.2623 |

1. **微调有效**：−0.017 → 0.3232，同一批生成图对陌生人的相似度只有 0.05，净差 0.27。
2. **只学到部分身份**：0.3232 对正对照 0.7159，约 45%。
3. **瓶颈是数据分辨率不是训练参数**：v1 检查点扫描 500/1000/1500/2000 步 →
   0.3110 / **0.3232** / 0.2518 / 0.2461，1000 步见顶后下滑（过拟合）；
   v2 降 lr、减步数、加 caption dropout 反而略降；10 张生成只有 5 张能被人脸检测器认出。
   原图仅 178×218，放大 2.9 倍到 512 训练，LoRA 学到的是插值模糊。

下一步：换 CelebA-HQ（1024×1024）或同一身份更多张，重跑同口径评估。
数据通路已验证（本地 hf-mirror 下载 → scp 到训练机），换数据一小时內可出结果。

## 目录结构

```
config/
  face_identity_sdxl_lora.yaml        v1：2000 步 / lr 1e-4 / 无 caption dropout
  face_identity_sdxl_lora_v2.yaml     v2：800 步 / lr 5e-5 / dropout 0.15
docs/
  assets_celeb/                       数据集联系表、基线图、微调对比图、原始 metrics
  assets_probe/                       底模能力探针图（候选风格该不该做）
  assets_data/   assets_thangka/      已放弃路线的数据摊牌，留作反例
scripts/
  # 数据获取与切分
  pick_celeb.sh                        读 parquet，按 celeb_id 找样本量合适的身份
  export_celeb_person.sh               导出指定身份的图，交替切分训练/测试
  # 数据审查（对应 pkm-hub-lab 的数据审查规范）
  qc_celeb_annotations.sh              四项机器检查：身份一致性/跨类污染/重复泄漏/质量
  make_celeb_sheet.sh                  训练+测试合并联系表
  make_celeb_split_sheet.sh            训练/测试分开放大，供人工目检
  # 底模能力探针（立项前用）
  probe_candidates.sh                  候选风格各生成 3 seed，判断底模会不会
  probe_commons2.sh  probe_hf.sh       数据可得性摸底
  make_probe_sheet.sh                 探针结果联系表
  # 训练
  train_face_proart.sh                 v1 训练入口
  train_face_v2_proart.sh              v2 训练入口
  # 评估
  face_baseline.sh                     底模基线（三对照里的 B）
  face_eval.sh                         微调后评估，与基线同提示词同 seed
  face_ckpt_scan.sh  face_ckpt_scan_v2.sh   检查点扫描，零成本定最优点
  make_face_compare.sh                 微调前后对比图（交付主图）
  # 人脸识别栈安装
  install_face_stack.sh                onnxruntime + insightface（走阿里云 pypi 镜像）
  install_face_models.sh               ArcFace 权重装入 insightface 模型目录
```

## 硬件与网络

| 机器 | GPU | 角色 |
|---|---|---|
| proart-temp（192.168.1.12） | RTX Spark N1X，64 GB，WSL2 arm64 | 训练与评估 |
| dgx-spark（192.168.1.170） | GB10 | 备用，被本地 llama-server 占 72 GiB |

网络实况：两台训练机都不通 huggingface.co 与 commons.wikimedia.org，
只通 metmuseum.org 与 mirrors.aliyun.com；本地 Windows 通 hf-mirror.com（30 MB/s）。
所以 HF 数据一律「本地下载 → scp 到训练机」，proart 装包走阿里云 pypi 镜像。

## 流程纪律

数据进训练前必须走完：抓取 → 机器四项检查 → 联系表 → **用户目检 → 用户批准**。
完整规范见 `pkm-hub-lab/image-style-lora/notes/数据审查规范.md`。
