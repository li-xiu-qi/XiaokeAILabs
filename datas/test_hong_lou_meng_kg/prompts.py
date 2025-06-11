# # -*- coding: utf-8 -*-
# """
# prompts.py
# 存放所有大模型调用的提示词模板
# """

# # 红楼梦实体与关系抽取-YAML格式
# HONGLOUMENG_KG_EXTRACTION_YAML_PROMPT = """
# 请从以下红楼梦文本中自动抽取所有实体（如人物、地点、物品、概念等）和它们之间的关系。

# 文本：
# {text}...

# 请严格用如下格式返回，并用 markdown 代码块包裹（即以```yaml开头，以```结尾）：
# ```yaml
# entities:
#   - name: "实体名称"
#     type: "person"/"place"/"object"/"concept"/"other"
#     aliases: ["别名1", "别名2"]
#     attributes:
#       描述: "实体描述"
#       特征: "主要特征"
#     mentions:
#       - "提及的上下文1"
#       - "提及的上下文2"
# relations:
#   - source: "实体1"
#     target: "实体2"
#     relation_type: "关系类型"
#     confidence: 0.7
#     context: "关系出现的上下文（不超过80字，必须用英文双引号包裹）"
# ```

# 注意事项（必须遵守，否则视为无效）：
# 1. 所有字符串（包括 name、aliases、mentions、context 等）都必须用英文双引号包裹，不能有多余缩进或注释。
# 2. mentions 字段必须为字符串列表（如 - "内容"），每个字符串不超过80字，不能写成 - key: value，也不能嵌套。
# 3. context 字段内容必须用英文双引号包裹，且不超过80字。
# 4. 每个列表项必须独占一行，且前面只允许两个空格缩进。
# 5. 只输出 markdown 代码块内的 YAML 内容，不要输出任何解释、注释、空行或其他内容。
# 6. 请确保输出的 YAML 代码块可以被 Python 的 yaml.safe_load() 正确解析，否则会被判为无效。

# 错误示例（不要这样写）：
# mentions:
#   - key: value
#   - "内容1"
#   - 内容2

# 正确示例：
# mentions:
#   - "内容1"
#   - "内容2"
# """
# -*- coding: utf-8 -*-
"""
prompts.py
存放所有大模型调用的提示词模板
"""

# 红楼梦实体与关系抽取-YAML格式（简化版）
HONGLOUMENG_KG_EXTRACTION_YAML_PROMPT = """
请从以下红楼梦文本中自动抽取所有实体（如人物、地点、物品、概念等）和它们之间的关系。

文本：
{text}...

请严格用如下格式返回，并用 markdown 代码块包裹（即以```yaml开头，以```结尾）：
```yaml
entities:
  - name: "实体名称"
    type: "person"/"place"/"object"/"concept"/"other"
    aliases: ["别名1", "别名2"]
    attributes:
      描述: "实体描述"
      特征: "主要特征"
relations:
  - source: "实体1"
    target: "实体2"
    relation_type: "关系类型"
    confidence: 0.7
```

注意事项（必须遵守，否则视为无效）：
1. 所有字符串（包括 name、aliases、attributes 值等）都必须用英文双引号包裹。
2. aliases 必须为字符串列表格式：["别名1", "别名2"]，如果没有别名则写 []。
3. type 字段只能是：person、place、object、concept、other 中的一个。
4. confidence 为数字（0.1-1.0），不需要引号。
5. 每个列表项必须独占一行，且前面只允许两个空格缩进。
6. 只输出 markdown 代码块内的 YAML 内容，不要输出任何解释、注释、空行或其他内容。
7. 请确保输出的 YAML 代码块可以被 Python 的 yaml.safe_load() 正确解析。

错误示例（不要这样写）：
aliases: [别名1, 别名2]  # 缺少引号
type: 人物  # 应该用英文

正确示例：
aliases: ["别名1", "别名2"]
type: "person"
"""
