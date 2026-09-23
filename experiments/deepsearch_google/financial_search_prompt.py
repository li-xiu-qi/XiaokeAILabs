# ================== 提示词 ==================
INDUSTRY_RESEARCH_PROMPT = """
针对 {industry} 行业研究，分析已收集的信息：{context_yaml_str}

历史搜索记录：
{previous_search_terms_str}

当前已完成 {search_round} 轮信息搜索。
请基于已有信息，判断还需要搜索哪些关键信息来深入了解这个行业。

请以如下结构化文本格式输出，并用```custom_structrue_text包围：
```custom_structrue_text
[continue_search] true/false
[reason] 继续搜索的原因和目标
[search_inputs]
- 搜索输入1
- 搜索输入2
- 搜索输入3
```

注意：
- 每个搜索输入可以包含多个关键词，支持如 site:、inurl: 等搜索引擎语法。
- 不要包含 filetype:pdf、filetype:doc 等文件型检索。
- 只聚焦信息内容检索。
- 搜索输入要尽量多样化，覆盖行业规模、竞争格局、技术动态、政策环境、市场机会等。
- 不要重复历史搜索记录中的内容。
"""

JUDGE_LINK_USEFULNESS_PROMPT = """
你是一名行业研究员，正在收集关于“{industry}”的资料。

请判断以下链接内容是否对行业研究有用：
标题：{title}
摘要：{body}
链接：{href}

请只输出如下结构化文本格式，并用```custom_structrue_text包围：
```custom_structrue_text
[reason] 你的简要理由
[useful] true/false
```
"""
