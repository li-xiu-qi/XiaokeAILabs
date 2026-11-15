# 未参与教师选择的合成页面复测

[English](README.en.md)

本次针对上级实验的三个既有模型，在此前没有用于训练、调参或教师选择的新页面上比较检索结果。数据由 [evaluate.py](evaluate.py) 用种子 20261108 生成，共 64 页、每页 3 个查询，每次从相同 64 页中寻找已知配对页。脚本核验新旧数据的实体、字段值和图片 SHA-256 均不重叠。

| 模型 | Recall@1 | Recall@5 |
|---|---:|---:|
| 原始 ColSmol | 149/192 = 0.7760 | 186/192 = 0.9688 |
| 监督微调 | 154/192 = 0.8021 | 187/192 = 0.9740 |
| 监督微调 + 蒸馏 | 151/192 = 0.7865 | 186/192 = 0.9688 |

相对监督组，蒸馏组有 2 条查询进入首位，5 条退出首位，净少 3 条。这个结果没有显示蒸馏带来提升；单次训练、单一教师和单次数据生成都不足以证明普遍结论。每张页面的三个查询相互关联，不能把 192 条视为完全独立样本。

新页面与原实验共享页面版式、字段、查询模板，但改用此前没有出现的六位数实体与数值。它可以检验原结论在新合成样本上的表现，不能证明对真实文档、不同版式或语义匹配的泛化。教师和模型的选择仍使用过前一组测试集，这一新集合没有参与此前选择。

本次使用 NVIDIA DGX Spark，和主实验相同的 ColSmol 底模、初始 adapter 及两份训练后的 adapter，推理使用 BF16、ColPali Engine 0.3.13。PyTorch 峰值分配 3,021,570,048 字节；独立监控观测到的最低系统可用内存为 90,478,160 KiB。数字为单次运行，不能视为通用设备要求。

## 复算已记录指标

[results.json](results.json) 保存匿名逐条排名；[recompute.py](recompute.py) 只用 Python 标准库，不需要 GPU、图片或权重。以下命令从本实验根目录执行：

```bash
python independent-evaluation/recompute.py
```

脚本检查 192 条排名及汇总指标，输出两组相对变化；它不能重新验证模型打分或图片标签。

## 环境准备与独立复测

原始旧数据、旧图片和训练后 adapter 均未随仓库分发，**仅凭本仓库不能复现本次模型推理**。需要原始 `synthetic/records.json` 与其同级子目录中的旧图片，以及与原实验对应的三份 adapter 和底模：初始 ColSmol adapter、监督训练后的 adapter、监督加蒸馏后的 adapter。初始模型和底模的来源、固定 revision 与历史软件版本见[主实验环境准备](../README.md#环境准备)；另外两份必须来自原运行的保存产物。主实验的蒸馏训练入口目前直接停止，无法用当前代码重新生成缺失的蒸馏 adapter。新生成的训练产物不应冒充原模型。

执行环境也遵循主实验的 Linux、CUDA BF16 和至少 50 GiB 系统可用内存保护条件。`evaluate.py` 固定从以下路径读取两个字体，需事先安装 DejaVu Sans；图片像素依赖字体版本，排名也可能受软件和设备影响：

```bash
test -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
test -f /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
```

确保 `records.json` 中每条记录的 `image` 相对于该文件所在目录均指向实际旧图片，例如 `synthetic/train/page-000.png`。脚本会检查所有引用的图片存在，并计算它们与新图片的 SHA-256；缺少图片或记录为空时会先报错。脚本还根据 `records.json` 检查实体和字段值是否重叠。

激活主实验准备好的环境后，从实验根目录运行；替换全部六个路径，`--out` 必须是尚不存在的新目录：

```bash
python independent-evaluation/evaluate.py \
  --source /path/to/original/synthetic/records.json \
  --model /path/to/original-adapter \
  --base /path/to/base-model \
  --supervised /path/to/supervised-adapter \
  --distilled /path/to/distilled-adapter \
  --out /path/to/new-output
```

完成后检查新目录的 `manifest.json` 和 `results.json`；中途停止可能只留下 `results.partial.json`。该脚本直接进行三组模型推理，不是轻量导入检查。它以 `local_files_only=True` 读取模型，执行前须备齐本地文件。结果需结合相同模型快照、源数据和字体版本解释。

## Independent synthetic holdout

The three existing models were evaluated on 64 newly generated pages with 192 queries, none used for the previous teacher selection. Top-1 counts were 149 for the original model, 154 for supervised fine-tuning and 151 after distillation. Compared with supervised fine-tuning, distillation moved 2 queries into first place and 5 out; this run shows no improvement from distillation.

The new pages share the earlier template and each page contributes three correlated queries. This is a fresh synthetic holdout, not evidence of generalization to real or differently formatted documents. The model weights are not distributed here; the recorded ranks can be checked using recompute.py.

