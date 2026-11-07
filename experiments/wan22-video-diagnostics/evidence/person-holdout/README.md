# 虚构人物 LoRA 的留出场景检查

一组 200 步 LoRA 能否让触发词产生训练角色的特征？训练素材是同一虚构二维角色的 16 段合成视频，角色有深色头发、红上衣和黄色围巾；训练描述包含红上衣和黄色围巾。本次两条留出提示只写触发词 `mivora`、人物、场景与动作，不提供这些外观词。每条提示使用预选种子 610、611，以同参数分别生成基模和 LoRA 视频，共 4 对、8 条。610 曾用于此前排障，因此不能视为全新随机种子。

一位 AI 评审在不看模型组别映射的情况下，查看每条视频第 0、12、24、36、48 帧，逐帧标注人物是否可辨及三项外观为 present、absent 或 unknown。遮挡、分辨率不足与无人物均记 unknown，不当作特征缺失。评分封存后才揭盲；原始评分见 [scores.csv](scores.csv)，随机编号与视频 SHA-256 映射见 [blind-mapping.json](blind-mapping.json)，配置见 [eval-plan.json](eval-plan.json)。

| 留出场景 / 种子 | 基模 | LoRA |
| --- | --- | --- |
| 图书馆 / 610 | 五帧均未辨认到人物，特征 unknown | 五帧均未辨认到人物，特征 unknown |
| 图书馆 / 611 | 五帧深发、红衣 present；围巾 unknown | 五帧深发 present；红衣和围巾 absent |
| 湖边长椅 / 610 | 五帧深发及黄围巾 present，红衣 absent | 五帧红衣及黄围巾 absent；头发 unknown |
| 湖边长椅 / 611 | 五帧人物可辨，三项外观均 unknown | 五帧深发 present；红衣和围巾 unknown |

八条分镜分别为 [V01](contact-sheets/V01.jpg)、[V02](contact-sheets/V02.jpg)、[V03](contact-sheets/V03.jpg)、[V04](contact-sheets/V04.jpg)、[V05](contact-sheets/V05.jpg)、[V06](contact-sheets/V06.jpg)、[V07](contact-sheets/V07.jpg)、[V08](contact-sheets/V08.jpg)。原视频未随此页公开；分镜为缩放后的抽帧，仅能粗查评审。随机映射仅用于评分盲化，并不让本次自动化评审变成人类评价。

在这 4 对例子中，LoRA 没有获得红衣或黄围巾的 present 标注；基模分别在两条不同视频中获得这两项标注。画面遮挡和无法判断的条目较多，且同一视频内五帧并非独立样本，不能据此估计总体成功率或比较模型的统计显著性。此实验使用虚构人物及单次 AI 评分，不能证明真实人物身份保持，也不能概括其他训练设置。

## English

We compared a base model and its 200-step LoRA on two held-out scenes and two selected seeds. The prompts name the trigger but omit the reference character's dark hair, red shirt and yellow scarf. One AI reviewer rated five sampled frames per video before receiving the model mapping; obscured features were marked unknown.

Neither model produced a discernible person for the library prompt at seed 610. Across the four LoRA videos, the reviewer marked no red shirt or yellow scarf as present; each of those traits was observed in a different base-model video. This is a four-pair, single-reviewer observation with correlated frames and substantial unknowns, not an estimate of general identity retention or evidence about real people. The score file, mapping and reduced contact sheets above allow a coarse review; the original videos are not included here.

训练角色由本项目用程序绘制，不含真实人物照片。生成所用 [Comfy-Org Wan 2.2](https://huggingface.co/Comfy-Org/Wan-2.2) 与 [Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 模型卡标注 Apache-2.0；此处只公开生成结果的缩放分镜，不包含模型权重或完整视频。
