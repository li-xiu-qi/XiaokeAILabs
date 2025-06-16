# deepsearch-google

![alt text](<assets/image copy 2.png>)
![alt text](assets/image.png)
![alt text](<assets/image copy.png>)

## 工作流程图

```mermaid
flowchart TD
    %% 开始节点
    start([开始]) --> research

    %% 主流程节点
    research[搜索决策]
    search[信息搜索]
    summary[搜索报告汇总]
    early_stop[提前终止]
    
    %% 主流程连接
    research -->|继续搜索| search
    research -->|完成收集| summary
    research -->|连续无效超限| early_stop
    research -->|达到最大搜索次数| early_stop
    search -->|返回决策| research
    summary --> end_success([成功结束])
    early_stop --> end_early([提前结束])

    %% 研究决策详细流程
    subgraph research_detail [搜索决策详细流程]
        direction TB
        ctx_prep[准备上下文信息]
        progress_analysis[分析当前搜索进度]
        continue_decision{是否继续搜索?}
        gen_queries[生成搜索查询列表]
        prepare_summary[准备生成汇总]
        check_invalid[检查连续无效轮数]
        check_max_search[检查是否达到最大搜索次数]
        
        ctx_prep --> progress_analysis
        progress_analysis --> continue_decision
        continue_decision -->|是| gen_queries
        continue_decision -->|否| prepare_summary
        continue_decision -->|检查轮数| check_invalid
        check_invalid -->|超限| prepare_summary
        check_invalid -->|未超限| check_max_search
        check_max_search -->|未超限| gen_queries
        check_max_search -->|已超限| prepare_summary
    end

    %% 信息搜索详细流程
    subgraph search_detail [信息搜索详细流程]
        direction TB
        param_prep[准备搜索参数]
        execute_search[执行搜索查询]
        relevance_judge{内容相关性判断}
        has_body{有摘要?}
        collect_useful[收集有用信息]
        record_useless[记录无用链接]
        update_context[更新上下文]
        count_useful[统计有用信息数量]
        
        param_prep --> execute_search
        execute_search --> relevance_judge
        relevance_judge --> has_body
        has_body -->|有摘要| collect_useful
        has_body -->|无摘要| record_useless
        collect_useful --> update_context
        record_useless --> update_context
        update_context --> count_useful
    end

    %% 搜索报告详细流程
    subgraph summary_detail [搜索报告详细流程]
        direction TB
        analyze_data[分析收集的数据]
        generate_report[生成搜索报告]
        save_output[保存输出文件]
        
        analyze_data --> generate_report
        generate_report --> save_output
    end

    %% 样式定义
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef mainNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef processNode fill:#e8f5e8,stroke:#2e7d32,stroke-width:1px
    classDef decisionNode fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef terminalNode fill:#ffebee,stroke:#c62828,stroke-width:2px

    %% 应用样式
    class start,end_success,end_early startEnd
    class research,search,summary,early_stop mainNode
    class ctx_prep,progress_analysis,gen_queries,prepare_summary,check_invalid,check_max_search,param_prep,execute_search,collect_useful,record_useless,update_context,count_useful,analyze_data,generate_report,save_output processNode
    class continue_decision,relevance_judge,has_body decisionNode
```

## 工作流程说明

### 主流程概述

整个搜索流程包含四个主要节点：

1. **搜索决策** - 分析当前状态，决定下一步行动
2. **信息搜索** - 根据决策执行具体的信息检索
3. **搜索报告汇总** - 整理和生成最终搜索报告（含详细参考文献和搜索时间记录）
4. **提前终止** - 在无效搜索过多、或达到最大搜索次数时提前结束流程

### 详细节点说明

#### 1. 搜索决策节点 (research)

- 分析已收集的信息质量和完整性
- 评估当前搜索进度
- 决定是否需要继续搜索或开始汇总
- 检查无效轮数和总搜索次数，防止无限循环搜索或超出预设搜索量

**输入参数**: industry, context, search_round, invalid_search_rounds, max_invalid_rounds, total_search_count, max_search_count

**输出结果**: continue_search, search_queries

#### 2. 信息搜索节点 (search)

- 执行具体的网络搜索操作（当前使用 Google Search）
- 判断搜索结果的相关性（有摘要才判定，否则直接无用）
- 分类和存储有用信息
- 统计分析本轮有效信息数量
- 累计总搜索次数

**输入参数**: search_queries, industry, useful_links, useless_links, search_pool

**输出结果**: 更新的上下文信息、链接分类、本轮有用信息统计、更新后的 total_search_count

#### 3. 搜索报告汇总节点 (summary)

- 分析所有收集的信息
- 生成结构化的搜索报告
- 报告中包含所有有用搜索结果的详细参考文献（标题、链接、摘要）以及搜索过程中的时间记录
- 保存输出文件

**输入参数**: industry, context, search_round, search_times

**输出结果**: Markdown 格式搜索报告（含参考文献和搜索时间记录）、文件路径

#### 4. 提前终止节点 (early_stop)

- 连续无效搜索轮数 >= 最大允许无效轮数时触发
- 或，总搜索次数 >= 最大允许搜索次数时触发
- 避免无效的重复搜索或超出预算，节省资源
- 可输出部分结果

---

> **参考文献区块说明**：
>
> 生成的搜索报告会自动附带所有有用搜索结果的详细参考文献区块，每条包含标题、原文链接和摘要，便于溯源和查阅。
>
> **搜索时间记录说明**：
>
> 报告中还会包含各主要搜索阶段的时间戳记录。

如需调整提前终止策略，只需修改 `max_invalid_rounds` 或 `max_search_count` 参数即可。
