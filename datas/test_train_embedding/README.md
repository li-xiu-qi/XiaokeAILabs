# 文本嵌入模型微调项目

> 本项目基于 PyTorch，提供了完整的文本嵌入（Embedding）模型微调代码，核心目标是通过**对比学习**提升模型对语义相似度的捕捉能力，适用于向量检索、文本匹配等下游任务。

---

## 目录

- 项目原理简介
- 环境依赖
- 快速开始
- 自定义数据训练
- 命令行参数说明
- 代码结构概览
- 许可证

---

## 项目原理简介

文本嵌入模型（如 BGE、M3E）可将文本映射到高维向量空间。为适应特定领域或任务，需对其进行微调。

本项目采用主流的 **In-batch Negatives** 对比学习策略：

- 每个 Query 的正样本为其对应的 Document。
- 同批次内其他 Document 均视为负样本。
- 优化目标：正样本距离更近，负样本距离更远。

---

## 环境依赖

请使用 pip 安装项目所需库：

```bash
pip install torch transformers sentence-transformers tqdm
```

---

## 快速开始

1. 下载代码：保存 `train_embedding_model.py` 到本地。
2. 执行训练：终端运行如下命令，脚本会自动创建并使用演示数据集 `demo_data.jsonl`。

   ```bash
   python train_embedding_model.py
   ```

   训练完成后，模型检查点和最终版本将保存在 `./models/bge-finetuned` 目录下。

3. 运行输出示例（本项目示例在 NVIDIA A6000 显卡上运行）：

   > 训练时显卡显存需求约为 6767MiB（约 7GB），请确保显卡资源充足，大概需要1分钟左右就训练完成了。

   ```
   Using device: cuda
   --- Training Started ---
   Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.56s/batch, loss=0.0000, acc=100.00%]
   Epoch 1 finished. Average Loss: 0.0000
   Saving model checkpoint to ./models/bge-finetuned/checkpoint-epoch-1
   Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.55batch/s, loss=0.0000, acc=100.00%]
   Epoch 2 finished. Average Loss: 0.0000
   Saving model checkpoint to ./models/bge-finetuned/checkpoint-epoch-2
   Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.49batch/s, loss=0.0000, acc=100.00%]
   Epoch 3 finished. Average Loss: 0.0000
   Saving model checkpoint to ./models/bge-finetuned/checkpoint-epoch-3
   --- Training Finished ---
   Final model saved to: ./models/bge-finetuned/final_model
   ```

   > 每个 epoch 会显示进度条、当前 batch 的损失和准确率，训练结束后会保存模型和分词器。

---

## 自定义数据训练

1. 数据格式要求：
   - 数据文件为 `.jsonl` 格式，每行一个 JSON 对象，包含 `"query"` 和 `"pos_doc"` 字段。

     示例：

     ```json
     {"query": "什么是大语言模型？", "pos_doc": "大语言模型（LLM）是指在一个极大规模的文本语料库上训练的，参数数量巨大的语言模型。"}
     {"query": "如何预防感冒？", "pos_doc": "预防感冒需要注意保暖、勤洗手、保持室内空气流通并加强体育锻炼。"}
     ```

2. 启动训练：
   - 通过 `--train_dataset` 参数指定数据文件路径，可用其他参数调整训练配置。

     ```bash
     python train_embedding_model.py \
       --model_name_or_path "BAAI/bge-base-zh-v1.5" \
       --train_dataset "path/to/your/my_dataset.jsonl" \
       --output_dir "./models/my-custom-model" \
       --epochs 5 \
       --batch_size 16 \
       --lr 1e-5
     ```

---

## 命令行参数说明

| 参数                  | 默认值                     | 说明                                   |
|-----------------------|----------------------------|----------------------------------------|
| `--model_name_or_path`| `BAAI/bge-base-zh-v1.5`    | Hugging Face上的预训练模型名称或本地路径 |
| `--train_dataset`     | `demo_data.jsonl`          | 训练数据集的文件路径                   |
| `--output_dir`        | `./models/bge-finetuned`   | 模型检查点和输出的保存目录             |
| `--epochs`            | `3`                        | 训练的总轮次                           |
| `--lr`                | `2e-5`                     | 学习率                                 |
| `--batch_size`        | `4`                        | 训练批次大小                           |
| `--query_max_len`     | `64`                       | 查询文本的最大Token长度                |
| `--passage_max_len`   | `256`                      | 文档段落的最大Token长度                |

---

## 代码结构概览

- **`EmbeddingModel` 类**：核心模型，封装 `SentenceTransformer` 并实现对比学习损失计算。
- **`TextDataset` 类**：数据集加载器，负责文件读取。
- **`collate_fn` 函数**：批次文本预处理，转为模型张量格式。
- **`Trainer` 类**：训练流程执行者，包含训练循环、日志记录和模型保存。
- **`main()` 函数**：主入口，参数解析、对象初始化和启动训练。

---

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权。
