
## 动手实践：DPO 训练示例

本目录包含一个用于演示 DPO（Direct Preference Optimization）训练流程的简单脚本，包含数据解析、参考模型预计算、训练循环与结果可视化等基础模块，适合快速上手与实验验证。

### 环境准备

推荐使用 Python 3.8+，有 GPU 则可显著加速训练。安装依赖可通过项目内的 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

### 数据准备

脚本默认通过 `datasets` 库从 `trl-lib/ultrafeedback_binarized` 加载训练样本。若需使用自定义数据，请修改脚本中的 `select_and_extract_dataset` 函数或直接替换数据加载逻辑。注意：更换数据集通常需要对数据预处理做对齐（例如 `extract_triple` 函数中期望的字段和对话结构、分词与截断策略等），以保证训练输入格式一致。

### 模型和 Tokenizer 加载

默认模型由 `Config.model_name` 指定，脚本会使用 `transformers` 的 `from_pretrained` 接口加载模型与 tokenizer。如需替换模型或数据集，这里给出一个简单示例：在脚本中将 `Config.model_name` 改为 `gpt2`，并在数据加载处改为 `Helsinki-NLP/opus_books`：

```python
# 在 Config 中或运行前修改
cfg = Config()
cfg.model_name = "gpt2"

# 在数据加载处使用不同的数据集
# raw_dataset = load_dataset(path="Helsinki-NLP/opus_books", split="train")
```

如果替换了数据集，请务必检查并调整数据预处理步骤（例如解析样本的字段名、对话索引、文本截断或拼接策略等），否则可能导致解析错误或不合理的训练输入。

### 配置与训练

配置项集中在 `Config` dataclass 中，包含学习率、批大小、训练轮数、beta 等超参数，直接修改 dataclass或在脚本中覆盖字段即可。训练可在本目录下运行脚本启动：

```bash
python dpo.py
```

训练结束后（若启用保存），最终模型与 tokenizer 会保存到 `Config.output_dir` 指定的目录，脚本也会导出训练曲线图片用于查看训练过程。

### 说明与注意事项

该示例偏向教学和实验用途：会预计算参考模型得分、使用梯度累积，并在支持的设备上尝试启用 bfloat16 加速。运行前请确保能访问 Hugging Face 的模型与数据源，且已安装 `requirements.txt` 中列出的依赖。

### 训练结果可视化演示

![](https://oss-liuchengtu.hudunsoft.com/userimg/64/641653370d117660c8a3274fb26b09b6.jpg)
> 三个 epoch 训练出来的 loss 变化图

![](https://oss-liuchengtu.hudunsoft.com/userimg/f5/f5299289db2dc89e8b6abf78707fa5e2.jpg)
> 200 steps 训练出来的 loss 变化图
![](https://oss-liuchengtu.hudunsoft.com/userimg/66/6693c8d434fa410a3b7f32ca0cc0b891.jpg)
> 200 steps 训练时的 advantage 变化图
![](https://oss-liuchengtu.hudunsoft.com/userimg/d9/d9f688c9303748bde74efc7469e6f9c1.jpg)
> 200 steps 训练时的偏好概率变化图
