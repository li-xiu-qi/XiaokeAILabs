# 07 生成结果

> 职责声明：本报告来自 37KB 玩具合成语料上的历史运行（2.07M 参数、128 token 上下文），测的是全链路流程与教学现象，没测真实数据上的语言能力；后续只有讲解链路流程时可引用，任何能力结论都不要引用本文件。


- 主模型：`results\checkpoints\rl.pt`（stage=rl_grpo）
- 采样参数：temperature=0.8, top_k=20, top_p=1.0, max_new=24, seed=1234


## 贪心解码（temperature=0）

| prompt | 生成 | 新 token 数 | tok/s |
|---|---|---|---|
| `User: What is the capital of Japan?\nAssistant:` | Tokyo | 2 | 4.2 |
| `User: Which city is the capital of Egypt?\nAssistant:` | Cairo | 2 | 50.6 |
| `User: What color is a banana?\nAssistant:` | The banana is yellow. | 6 | 40.2 |
| `User: What does Carol do?\nAssistant:` | Grace is a pilot. | 6 | 81.4 |
| `User: What is 7 plus 5?\nAssistant:` | 12 | 2 | 55.1 |
| `User: Hello\nAssistant:` | Hi there! | 4 | 71.5 |
| `User: Who are you?\nAssistant:` | I am XiaoLM, a small language model. | 10 | 85.4 |
| `The capital of Peru is` | a Lima | 3 | 67.1 |
| `The small robot reads the` | is a pilot. | 5 | 74.9 |

## 温度采样

| prompt | 生成 | 新 token 数 | tok/s |
|---|---|---|---|
| `User: What is the capital of Japan?\nAssistant:` | Tokyo | 2 | 11.5 |
| `User: Which city is the capital of Egypt?\nAssistant:` | Cairo | 2 | 53.8 |
| `User: What color is a banana?\nAssistant:` | The banana is yellow. | 6 | 65.4 |
| `User: What does Carol do?\nAssistant:` | Grace is a pilot. | 6 | 75.6 |
| `User: What is 7 plus 5?\nAssistant:` | 12 | 2 | 58.1 |
| `User: Hello\nAssistant:` | Hi there! | 4 | 57.4 |
| `User: Who are you?\nAssistant:` | I am XiaoLM, a small language model. | 10 | 69.9 |
| `The capital of Peru is` | a Lima | 3 | 48.9 |
| `The small robot reads the` | is a pilot. | 5 | 65.5 |

## 对照：midtrain 模型贪心解码

| prompt | 生成 | 新 token 数 | tok/s |
|---|---|---|---|
| `User: What is the capital of Japan?\nAssistant:` |  | 1 | 44.0 |
| `User: Which city is the capital of Egypt?\nAssistant:` | : : color a model : berry blue: berry blue: berry blue | 18 | 84.7 |
| `User: What color is a banana?\nAssistant:` | banana yellow : is.User What the of?Assistant The is.User What the of?Assistant BrasiliaUser How | 24 | 82.0 |
| `User: What does Carol do?\nAssistant:` | is engineer : is capital Norway : color a model : | 15 | 89.8 |
| `User: What is 7 plus 5?\nAssistant:` | orange orange orange orange orange orange : color a color a.User What is lemon : lemon : lemon yellow | 24 | 88.1 |
| `User: Hello\nAssistant:` | there | 2 | 53.3 |
| `User: Who are you?\nAssistant:` | am, you : am tiny trained a model on small.User GoodbyeAssistant GoodbyeAssistant Goodbye see later :, | 24 | 74.7 |
| `The capital of Peru is` | .User What : | 6 | 79.0 |
| `The small robot reads the` | .User What youAssistant?Assistant I fine thank.User What Bob?Assistant The is.User What.User What | 24 | 91.0 |

## 对照：预训练模型贪心解码

| prompt | 生成 | 新 token 数 | tok/s |
|---|---|---|---|
| `User: What is the capital of Japan?\nAssistant:` |  | 1 | 47.3 |
| `User: Which city is the capital of Egypt?\nAssistant:` | : : does keep.The does keep.The does keep.The does keep.The does make.The | 24 | 96.5 |
| `User: What color is a banana?\nAssistant:` | banana yellow.User What the.User What the.User What the.User What the.The does do : | 24 | 59.6 |
| `User: What does Carol do?\nAssistant:` | is engineer : is capital Norway : is capital Norway : | 15 | 92.1 |
| `User: What is 7 plus 5?\nAssistant:` | can do : can do : can do : can do : can simple about.User What the. | 24 | 91.6 |
| `User: Hello\nAssistant:` | there | 2 | 58.2 |
| `User: Who are you?\nAssistant:` | am, youAssistant I XiaoLM a language.User What the.The bird the.The cat the.The farmer the | 24 | 83.8 |
| `The capital of Peru is` | .Lima the.The cat the.The cat the.The cat the.The bird the.The boy the. | 24 | 78.6 |
| `The small robot reads the` | .The farmer the.The farmer the.The farmer the.The student not songs  you water  give?Do | 24 | 72.6 |

## 温度效应对照（续写 prompt，T=0.8 vs T=2.5）

| prompt | T=0.8 | T=2.5 |
|---|---|---|
| `The capital of Peru is` | a Lima | a Lima wel, I am |
| `The small robot reads the` | is a farmer. | n name�Erin 17ghErin? teacherCarolot fine |
| `The quiet student finds the` | is a small language model | apple a Who questionsotThey! read am later city day |

## KV cache 提速对比（生成长度 96 token）

| 场景 | KV cache tokens/s | naive tokens/s | 提速 |
|---|---|---|---|
| batch=1 | 89.9 | 75.3 | 1.19x |
| batch=8（服务场景） | 461.0 | 512.7 | 0.9x |


