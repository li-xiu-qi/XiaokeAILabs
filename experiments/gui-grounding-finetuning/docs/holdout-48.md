# GUI 定位：48 条分层留出样本的配对评估

[English](holdout-48.en.md) | [实验入口](../README.md)

2026-11-07，我们增加了一组未参与训练、也不属于先前 30 条评估记录的 ScreenSpot 图像，检验原有 LoRA 是否只在那 30 条上有效。评估只判断预测坐标是否落入目标框，不执行点击，也不代表任务成功。

| 配对结果 | 数量 |
| --- | ---: |
| 基座命中 | 34/48 |
| LoRA 命中 | 42/48 |
| 从错变对 | 8 |
| 从对变错 | 0 |

差值为 +16.7 个百分点。对这 48 组差值进行 10,000 次配对 bootstrap，固定种子 20261107，百分位区间为 +6.25 至 +27.08 个百分点；它只描述这批样本的重采样波动，不是总体准确率或跨应用泛化的保证。每个来源只有 6 条，也不能据此比较应用之间的表现。

## 数据怎样选出

[ScreenSpot 数据卡](https://huggingface.co/datasets/bevaya/ScreenSpot)标注 Apache-2.0；界面截图内的文字、图像或个人信息可能受其他权利与隐私约束，本仓不发布截图、指令、坐标、原始数据行号和逐例预测。仓内提供[去标识的逐条命中结果](../evidence/holdout-48-paired.json)，每条仅含自编号、来源、元素类型和两次命中与否；它不足以重建截图或预测。

来源为 Android、forum、GitLab、iOS、macOS、shop、tool、Windows，按来源与 icon/text 类型共 16 层各选 3 条。种子 20261107；先按随机顺序检查候选图，与训练图、旧 30 图以及已经选入的新图比较：相同 SHA-256，或 64 位差值哈希汉明距离不超过 5 时排除。本次排除 4 个精确重复和 2 个近重复。感知哈希可能误拒或漏掉相似界面，排重不能排除语义重复或基座预训练污染。该组是去重后的分层小样本，不是随机抽取的总体代表。

本次使用数据集服务当前行，其接口不固定数据 revision，无法仅靠 seed 重建完全相同数据。实际评估 manifest 的 SHA-256 为 `60ed4f4d021dcf95f73ffa190cc0abe5be997d7aecf93693650d1139320e4471`；两次评估在开始时分别记录了该哈希、提示词哈希和模型路径。目录路径一致不证明权重内容一致；本次使用的 adapter 文件 SHA-256 为 `5b6a43819db604ea8b9f41fdefb551d5ab6177814e49ad14629c2e817184e2d4`。旧实验的基座模型、训练超参和软件版本见实验入口。因上游数据行可能改变，下次生成的新清单应视为新的评估，不得称为同一批样本。

## 复算与复现

只复算本文公布的指标，不需要模型和图片：

```bash
python scripts/audit_holdout_results.py
```

要重建评估，请先按实验入口准备训练图、旧评估图、基座模型和 adapter，并在仓库外创建新的输出目录。以下命令在实验目录执行；`TRAIN_DIR`、`PREVIOUS_DIR` 指向各自含 `manifest.json` 和图片的目录，`HOLDOUT_DIR` 必须不存在。请按设备状况安排执行；评估脚本默认在可用系统内存低于 50 GiB 时停止，此值是本次 GB10 的运行保护，不是通用硬件要求。

```bash
python scripts/prepare_stratified_holdout.py --training "$TRAIN_DIR" \
  --previous "$PREVIOUS_DIR" --output "$HOLDOUT_DIR" --per-stratum 3
export GUI_EVAL_DIR="$HOLDOUT_DIR"
export GUI_ROOT="$HOME/gui-holdout-results"
export GUI_MODEL="/path/to/Qwen3-VL-2B-Instruct"
mkdir -p "$GUI_ROOT"
unset GUI_ADAPTER
python scripts/gui_eval_holdout.py
export GUI_ADAPTER="/path/to/adapter"
python scripts/gui_eval_holdout.py
python scripts/gui_summarize_holdout.py
```

两次评估顺序运行、使用同一清单、提示词、图像缩放范围和贪心解码；汇总程序检查数据清单哈希、提示词哈希、基座路径、adapter 角色和逐例标注。脚本不检查目录中模型或 adapter 权重是否被替换，执行期间必须固定权重并自行记录哈希。评估输出包括输入标注与预测，应保存在私有目录。

汇总文件使用独占创建，重复执行前请使用新的 `GUI_ROOT`；避免覆盖已经保存的结果。

本次 GPU 评估使用这些校验增强之前的常规运行版本；新增的防覆盖与显式校验仅做了 CPU 验证，尚未重跑 GPU。
