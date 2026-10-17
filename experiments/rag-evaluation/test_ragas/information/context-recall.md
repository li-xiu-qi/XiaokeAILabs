
**标题：** Context Recall - Ragas（上下文召回率 - Ragas）

**内容：**

上下文召回率（Context Recall）衡量的是成功检索到的相关文档（或信息片段）的数量。它关注的是不遗漏重要的结果。召回率越高，意味着遗漏的相关文档越少。  
简而言之，召回率关注的是不遗漏任何重要的内容。由于它关注的是不遗漏，因此计算上下文召回率总是需要一个参考标准来进行对比。

## 基于LLM的上下文召回率

`LLMContextRecall` 是通过 `用户输入`、`参考答案` 和 `检索到的上下文` 来计算的，其值介于0到1之间，值越高表示性能越好。这种指标使用 `参考答案` 作为 `参考上下文` 的代理，这也使得它更容易使用，因为标注参考上下文可能非常耗时。为了从 `参考答案` 中估算上下文召回率，参考答案会被分解为多个主张（claims），每个主张都会被分析以确定它是否可以归因于检索到的上下文。在理想情况下，参考答案中的所有主张都应该能够归因于检索到的上下文。

上下文召回率的计算公式如下：

\[ \text{上下文召回率} = \frac{\text{参考答案中被检索到的上下文支持的主张数量}}{\text{参考答案中的总主张数量}} \]

### 示例

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall

sample = SingleTurnSample(
    user_input="埃菲尔铁塔位于哪里？",
    response="埃菲尔铁塔位于巴黎。",
    reference="埃菲尔铁塔位于巴黎。",
    retrieved_contexts=["巴黎是法国的首都。"],
)

context_recall = LLMContextRecall(llm=evaluator_llm)
await context_recall.single_turn_ascore(sample)
```

输出

## 非基于LLM的上下文召回率

`NonLLMContextRecall` 指标是通过 `检索到的上下文` 和 `参考上下文` 来计算的，其值介于0到1之间，值越高表示性能越好。这种指标使用非LLM的字符串比较方法来判断检索到的上下文是否相关。你可以使用任何非LLM的指标作为距离度量来判断检索到的上下文是否相关。

上下文召回率的计算公式如下：

\[ \text{上下文召回率} = \frac{\| \text{检索到的相关上下文数量} \|}{\| \text{参考上下文的总数量} \|} \]

### 示例

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["巴黎是法国的首都。"],
    reference_contexts=["巴黎是法国的首都。", "埃菲尔铁塔是巴黎最著名的地标之一。"]
)

context_recall = NonLLMContextRecall()
await context_recall.single_turn_ascore(sample)
```
