**标题：** Context Precision - Ragas

**内容：**

上下文精确度（Context Precision）是一种衡量 `retrieved_contexts` 中相关片段比例的指标。它是通过对上下文中每个片段的 precision@k 进行平均计算得出的。Precision@k 是在排名为 k 的位置上，相关片段的数量与总片段数量的比率。

\[
\text{上下文精确度@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{排名前 } K \text{ 结果中的相关项目总数}}
\]

\[
\text{Precision@k} = \frac{\text{真正例@k}}{\text{真正例@k} + \text{假正例@k}}
\]

其中，\(K\) 是 `retrieved_contexts` 中的片段总数，\(v_k \in \{0, 1\}\) 是排名为 \(k\) 的相关性指示器。

---

### 基于 LLM 的上下文精确度

以下指标使用 LLM 来判断检索到的上下文是否相关。

#### 无参考的上下文精确度

`LLMContextPrecisionWithoutReference` 指标适用于同时具有检索到的上下文和与 `user_input` 相关联的参考上下文的场景。该方法通过将 `retrieved_contexts` 中的每个检索到的上下文或片段与 `response` 进行比较，使用 LLM 来估算检索到的上下文是否相关。

**示例**

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="埃菲尔铁塔在哪里？",
    response="埃菲尔铁塔位于巴黎。",
    retrieved_contexts=["埃菲尔铁塔位于巴黎。"],
)

await context_precision.single_turn_ascore(sample)
```

**输出**

---

#### 带参考的上下文精确度

`LLMContextPrecisionWithReference` 指标适用于同时具有检索到的上下文和与 `user_input` 相关联的参考答案的场景。该方法通过将 `retrieved_contexts` 中的每个检索到的上下文或片段与 `reference` 进行比较，使用 LLM 来估算检索到的上下文是否相关。

**示例**

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="埃菲尔铁塔在哪里？",
    reference="埃菲尔铁塔位于巴黎。",
    retrieved_contexts=["埃菲尔铁塔位于巴黎。"],
)

await context_precision.single_turn_ascore(sample)
```

**输出**

---

### 非 LLM 基础的上下文精确度

该指标使用传统方法来判断检索到的上下文是否相关。它依赖于非 LLM 基础的度量方法来评估检索到的上下文的相关性。

#### 带参考上下文的上下文精确度

`NonLLMContextPrecisionWithReference` 指标适用于同时具有检索到的上下文和参考上下文的场景。该方法通过将 `retrieved_contexts` 中的每个检索到的上下文或片段与 `reference_contexts` 中的每个上下文进行比较，使用非 LLM 基础的相似性度量方法来判断检索到的上下文是否相关。

**示例**

```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["埃菲尔铁塔位于巴黎。"],
    reference_contexts=["巴黎是法国的首都。", "埃菲尔铁塔是巴黎最著名的地标之一。"],
)

await context_precision.single_turn_ascore(sample)
```
