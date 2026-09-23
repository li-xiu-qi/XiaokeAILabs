# CLIP 轻量微调：冻结部分层训练

在 CLIP 图文模型上做参数高效微调：冻结视觉或文本塔的大部分层，只训练少量参数，降低显存与训练成本。`finetune_clip_with_freeze.py` 是训练主脚本，`get_dataset.py` 与 `explore_dataset_flickr30k-v2.ipynb` 处理 Flickr30k 数据集（含 train 集划分），`data_common.py` 共享数据逻辑。

依赖见 `requirements.txt`。微调思路同样适用于其他对比学习模型。
