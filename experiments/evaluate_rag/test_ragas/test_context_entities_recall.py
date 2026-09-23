from ragas import SingleTurnSample
from ragas.metrics import BleuScore

import jieba

# 先进行中文分词
test_data = {
    "user_input": "总结给定的文本\n" + " ".join(jieba.cut("该公司报告2024年第三季度增长了8%，主要得益于亚洲市场的强劲表现...")),
    "response": " ".join(jieba.cut("该公司2024年第三季度增长了8%，主要由于有效的营销策略和产品适应性...")),
    "reference": " ".join(jieba.cut("该公司报告2024年第三季度增长了8%，主要由亚洲市场强劲销售推动..."))
}
metric = BleuScore()
test_data = SingleTurnSample(**test_data)
score = metric.single_turn_score(test_data)
print(score)