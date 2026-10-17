## 忠实度

**忠实度**指标衡量`回答`与`检索上下文`之间的事实一致性。它的范围从0到1，分数越高表示一致性越好。

如果回答中的所有声明都能被检索的上下文支持，则该回答被视为**忠实**。

计算方法如下：

1\. 识别回答中的所有声明。

2\. 检查每个声明是否可以从检索的上下文中推断出来。

3\. 使用以下公式计算忠实度分数：

\[
\text{忠实度分数} = \frac{\text{回答中能被检索上下文支持的声明数量}}{\text{回答中的总声明数量}}
\]

### 示例

```
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
    user_input="第一届超级碗是什么时候举行的？",
    response="第一届超级碗于1967年1月15日举行",
    retrieved_contexts=[
        "第一届AFL-NFL世界冠军赛是一场美式足球比赛，于1967年1月15日在洛杉矶纪念体育场举行。"
    ]
    )
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出

## 使用HHEM-2.1-Open的忠实度

Vectara的HHEM-2.1-Open是一个分类器模型（T5），经过训练可检测LLM生成文本中的幻觉。该模型可用于计算忠实度的第二步，即当声明与给定上下文交叉检查以确定是否可以从上下文中推断出来。该模型是免费、小型且开源的，使其在生产用例中非常高效。要使用该模型计算忠实度，可以使用以下代码片段：

```
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM

sample = SingleTurnSample(
    user_input="第一届超级碗是什么时候举行的？",
    response="第一届超级碗于1967年1月15日举行",
    retrieved_contexts=[
        "第一届AFL-NFL世界冠军赛是一场美式足球比赛，于1967年1月15日在洛杉矶纪念体育场举行。"
    ]
    )
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

您可以通过设置`device`参数将模型加载到指定设备上，并使用`batch_size`参数调整推理的批处理大小。默认情况下，模型在CPU上加载，批处理大小为10

```
my_device = "cuda:0"
my_batch_size = 10

scorer = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
await scorer.single_turn_ascore(sample)
```

### 计算方法

示例

**问题**：爱因斯坦在何时何地出生？

**上下文**：阿尔伯特·爱因斯坦（1879年3月14日出生）是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一

**高忠实度回答**：爱因斯坦于1879年3月14日出生在德国。

**低忠实度回答**：爱因斯坦于1879年3月20日出生在德国。

让我们看看如何计算低忠实度回答的忠实度：

- **步骤1：** 将生成的回答分解为单独的陈述。
  - 陈述：
    - 陈述1："爱因斯坦出生在德国。"
    - 陈述2："爱因斯坦出生于1879年3月20日。"
- **步骤2：** 对于每个生成的陈述，验证它是否可以从给定的上下文中推断出来。
  - 陈述1：是
  - 陈述2：否
- **步骤3：** 使用上述公式计算忠实度。

\[
\text{忠实度} = \frac{\text{1}}{\text{2}} = 0.5
\]
