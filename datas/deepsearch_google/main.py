# ================== 代码主体 ================

import os
import time
import yaml
import openai
from pocketflow import Node, Flow, build_mermaid
from dotenv import load_dotenv
import random

from financial_search_prompt import INDUSTRY_RESEARCH_PROMPT, JUDGE_LINK_USEFULNESS_PROMPT
from google_search import GoogleSearchPool # 新增：导入GoogleSearchPool

# 加载环境变量
load_dotenv()

# 从环境变量中初始化 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def call_llm(prompt: str) -> str:
    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=8196,  # 16k=16384,8k=8196根据需要调整
    )
    return response.choices[0].message.content.strip()


def search_web(term: str, search_pool: GoogleSearchPool): # 修改：添加 search_pool 参数
    """使用 GoogleSearchPool 执行网络搜索"""
    try:
        # Google API 每次最多返回10条，这里与原DDGS的max_results=10保持一致
        results = search_pool.search(term, num_results=10)
        # search_pool.display_results(results) # 可选：显示原始结果
        # search_pool.display_usage_stats() # 可选：显示使用统计
        return results
    except Exception as e:
        print(f"Google 搜索 '{term}' 时出错: {e}")
        return []


def parse_custom_structured_text(text, fields):
    """
    解析自定义结构化文本，fields为需要提取的字段列表。
    返回dict。
    """
    # 提取```custom_structrue_text包围内容
    if '```custom_structrue_text' in text:
        text = text.split('```custom_structrue_text',1)[1].split('```',1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {k: None for k in fields}
    for idx, line in enumerate(lines):
        for field in fields:
            tag = f'[{field}]'
            if line.startswith(tag):
                val = line.split(']',1)[1].strip()
                if field == 'search_inputs':
                    # 后续的-行都是搜索输入
                    search_inputs = []
                    for l in lines[idx+1:]:
                        if l.startswith('- '):
                            search_inputs.append(l[2:].strip())
                        else:
                            break
                    result['search_inputs'] = search_inputs
                else:
                    result[field] = val
    return result


# ================== 工作流核心部分 ==================

class SearchDecisionFlow(Node):  # 搜索决策节点
    """
    搜索决策节点：
    - 分析当前已收集的信息和历史搜索记录
    - 判断是否需要继续搜索，还是可以进入搜索报告汇总
    - 生成下一轮搜索的关键词列表
    - 控制最大轮数和无效轮数提前终止
    """
    def prep(self, shared):
        # 提取上下文、搜索主题、当前轮次
        search_context = shared.get("search_context", [])
        context_str = yaml.dump(search_context, allow_unicode=True)
        search_topic = shared["search_topic"]
        search_round = shared.get("search_round", 0)
        return search_topic, context_str, search_round

    def exec(self, inputs):
        search_topic, context_yaml_str, search_round = inputs
        print(f"\n正在分析 {search_topic} 搜索进度... (第 {search_round + 1} 轮)")
        # 解析历史搜索关键词，避免重复
        previous_search_terms = []
        if search_round > 0:
            try:
                context_data = yaml.safe_load(context_yaml_str)
                if isinstance(context_data, list):
                    for search_batch in context_data:
                        if isinstance(search_batch, dict) and 'search_query' in search_batch:
                            previous_search_terms.append(search_batch['search_query'])
                        elif isinstance(search_batch, list):
                            for item in search_batch:
                                if isinstance(item, dict) and 'search_query' in item:
                                    previous_search_terms.append(item['search_query'])
            except Exception:
                # It's generally better to log the exception here or handle it more specifically
                pass # Silently ignore parsing errors for now
        previous_search_terms_str = '\n'.join(f'- {term}' for term in previous_search_terms)
        if not previous_search_terms_str:
            previous_search_terms_str = '（无）'
        # 构造 LLM 提示词，自动生成下一轮搜索建议
        prompt = INDUSTRY_RESEARCH_PROMPT.format(
            industry=search_topic,
            context_yaml_str=context_yaml_str,
            previous_search_terms_str=previous_search_terms_str,
            search_round=search_round
        )
        llm_response = call_llm(prompt)
        result = parse_custom_structured_text(llm_response, ['continue_search', 'reason', 'search_inputs'])
        # 类型转换
        result['continue_search'] = str(result['continue_search']).strip().lower() == 'true'
        if result['search_inputs'] is None:
            result['search_inputs'] = []
        print(f"继续搜索: {result['continue_search']}")
        print(f"搜索原因: {result['reason']}")
        print("本轮搜索输入:", result["search_inputs"])
        return result

    def post(self, shared, prep_res, exec_res):
        # 控制最大轮数和无效轮数，决定是否提前终止
        max_rounds = shared.get("max_rounds")
        max_invalid_rounds = shared.get("max_invalid_rounds", 3)
        current_round = shared.get("search_round", 0)
        invalid_search_rounds = shared.get("invalid_search_rounds", 0)
        # 新增：最大搜索次数判断
        max_search_count = shared.get("max_search_count")
        total_search_count = shared.get("total_search_count", 0)
        if max_rounds is not None and current_round >= max_rounds:
            print(f"\n=== 已达到最大轮数({max_rounds})，搜索流程终止 ===")
            return "complete"
        if max_invalid_rounds is not None and invalid_search_rounds >= max_invalid_rounds:
            print(f"\n=== 连续{invalid_search_rounds}轮无有效信息，提前终止搜索流程 ===")
            return "complete"
        if max_search_count is not None and total_search_count >= max_search_count:
            print(f"\n=== 已达到最大搜索次数({max_search_count})，搜索流程终止 ===")
            return "complete"
        if exec_res.get("continue_search", True):
            shared["search_inputs"] = exec_res.get("search_inputs", [])
            shared["search_round"] = current_round + 1
            print("\n=== 开始新一轮搜索 ===")
            return "search"
        else:
            print("\n=== 搜索流程完成 ===")
            return "complete"


class SearchInfo(Node):  # 信息搜索节点
    """
    信息搜索节点：
    - 根据决策节点生成的关键词，实际执行网络检索
    - 对每条搜索结果调用 LLM 判断其有用性
    - 自动去重，累计有用/无用链接
    - 支持无摘要时直接判为无用链接
    """
    def prep(self, shared):
        # 获取本轮搜索关键词、搜索主题、历史有用/无用链接、搜索池
        return (
            shared.get("search_inputs", []),
            shared.get("search_topic"),
            shared.get("useful_links", []),
            shared.get("useless_links", []),
            shared.get("search_pool") # 新增：获取 search_pool
        )

    def exec(self, inputs):
        search_queries, search_topic, useful_links, useless_links, search_pool = inputs # 新增：接收 search_pool
        all_search_results = []
        total_queries = len(search_queries)
        print(f"\n本轮搜索输入: {', '.join(search_queries)}")
        seen_urls = set(link['href'] for link in useful_links + useless_links if link.get('href'))
        # 新增：累计搜索次数
        shared = self.shared if hasattr(self, 'shared') else None
        search_count_this_round = 0
        for idx, search_query in enumerate(search_queries, 1):
            print(f"\n搜索输入 ({idx}/{total_queries}): {search_query}")
            if search_pool is None:
                print("错误：search_pool 未初始化，无法执行搜索。")
                all_search_results.append({"search_query": search_query, "results": [], "error": "Search pool not available"})
                continue
            try:
                search_results_list = search_web(search_query, search_pool) # 修改：传递 search_pool
                search_count_this_round += 1  # 每执行一次search_web，+1
                print(f"找到 {len(search_results_list)} 条相关信息")
                round_useful_links = []
                for search_result in search_results_list:
                    url = search_result.get("link", "") # 修改：Google API 使用 "link"
                    title = search_result.get("title", "")
                    snippet = search_result.get("snippet", "") # 修改：Google API 使用 "snippet"

                    if not url or url in seen_urls:
                        continue
                    
                    # 构造 link_info 给 judge_link_usefulness，它期望 'body' 和 'href'
                    temp_link_info_for_judge = {
                        "title": title,
                        "body": snippet, # 将 snippet 映射到 body
                        "href": url     # 将 link 映射到 href
                    }
                    is_useful, reason = judge_link_usefulness(temp_link_info_for_judge, search_topic)
                    
                    # 存储时，我们保留原始字段名或统一的字段名，这里用 'body' 和 'href' 以保持一致性
                    link_info = {
                        "search_query": search_query,
                        "title": title,
                        "body": snippet, # 存储为 body
                        "href": url,     # 存储为 href
                        "reason": reason
                    }
                    seen_urls.add(url)
                    if is_useful:
                        useful_links.append(link_info)
                        round_useful_links.append(link_info)
                    else:
                        useless_links.append(link_info)
                all_search_results.append({"search_query": search_query, "results": round_useful_links})
                if round_useful_links:
                    print("有用搜索结果预览:")
                    for j, useful_result in enumerate(round_useful_links[:3]):
                        print(f"  {j+1}. {useful_result.get('title', '无标题')}")
                        print(f"     {useful_result.get('body', '无摘要')[:100]}...") # 使用 body
            except Exception as e:
                print(f"搜索 '{search_query}' 时出错: {e}")
                all_search_results.append({"search_query": search_query, "results": [], "error": str(e)})
            if idx < total_queries:
                sleep_time = random.randint(16, 30)
                print(f"等待秒{sleep_time}后继续搜索...")
                time.sleep(sleep_time)
        # 更新累计搜索次数
        if shared is not None:
            shared["total_search_count"] = shared.get("total_search_count", 0) + search_count_this_round
        else:
            # 兼容原有逻辑
            pass
        return all_search_results, useful_links, useless_links

    def post(self, shared, prep_res, exec_res):
        # 更新上下文和有用/无用链接统计，累计无效轮数
        all_results, useful_links, useless_links = exec_res
        search_context = shared.get("search_context", [])
        search_context.extend(all_results)
        shared["search_context"] = search_context
        shared["useful_links"] = useful_links
        shared["useless_links"] = useless_links
        total_results = sum(len(item.get("results", [])) for item in all_results)
        print("\n本轮搜索完成！") # Changed to normal string
        print(f"共收集到 {total_results} 条有用信息")
        print(f"累计有用链接: {len(useful_links)}，无用链接: {len(useless_links)}")
        print("\n返回决策节点，准备下一轮搜索...") # Changed to normal string
        if total_results == 0:
            shared["invalid_search_rounds"] = shared.get("invalid_search_rounds", 0) + 1
        else:
            shared["invalid_search_rounds"] = 0
        # 记录本轮搜索时间
        record_search_time(shared, event="search_round_end")
        # The redundant update to total_search_count has been removed.
        return "search"


class SearchSummary(Node):  # 搜索报告汇总节点
    """
    搜索报告汇总节点：
    - 汇总所有已收集的信息，统计轮次和条数
    - 生成结构化的搜索报告
    - 保存为 Markdown 文件，便于后续分析
    """
    def prep(self, shared):
        # 获取搜索主题、上下文、轮次
        return (
            shared.get("search_topic"),
            shared.get("search_context", []),
            shared.get("search_round", 0)
        )

    def exec(self, inputs):
        search_topic, search_context, search_round = inputs
        print(f"\n=== {search_topic} 搜索报告汇总 ===")
        print(f"完成搜索轮次: {search_round}")
        # 统计收集的信息条数
        total_results = 0
        for search_batch in search_context:
            if isinstance(search_batch, dict):
                total_results += len(search_batch.get("results", []))
        # 生成详细的参考文献内容
        reference_md = ""
        ref_idx = 1
        for search_batch in search_context:
            if isinstance(search_batch, dict):
                results = search_batch.get("results", [])
                for search_result in results:
                    title = search_result.get("title", "无标题")
                    url = search_result.get("href", "") # 保持 href
                    body = search_result.get("body", "") # 保持 body
                    reference_md += (
                        f"\n【第{ref_idx}篇参考文章开始】\n"
                        f"[{ref_idx}] 标题：{title}\n"
                        f"[{ref_idx}] 原文链接: {url}\n"
                        f"[{ref_idx}] 摘要：{body}\n"
                        f"【第{ref_idx}篇参考文章结束】\n"
                    )
                    ref_idx += 1
        # 获取所有搜索时间
        search_times = inputs[3] if len(inputs) > 3 else []
        search_time_md = "\n".join([
            f"- {item['event']}: {item['timestamp']}" for item in search_times
        ]) if search_times else "无"
        # 生成搜索报告
        summary = f"""
# {search_topic} 搜索报告

## 收集统计
- 搜索轮次: {search_round}
- 收集信息条数: {total_results}

## 搜索时间记录
{search_time_md}

## 搜索完成时间
{time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 参考文献与搜索结果
{reference_md}

---
注: 所有详细信息已保存在系统内存中，可用于后续分析处理。
"""
        print("搜索报告汇总:") # Changed to normal string
        print(f"- 搜索轮次: {search_round}")
        print(f"- 收集信息条数: {total_results}")
        return summary

    def post(self, shared, prep_res, exec_res):
        # 保存搜索报告到本地文件
        print("\n=== 搜索流程完成！===") # Changed to normal string
        shared["summary"] = exec_res
        filename = f"{shared['search_topic']}_搜索报告.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(exec_res)
        print(f"搜索报告已保存到 '{filename}' 文件中")
        return None




def judge_link_usefulness(link_info, industry):
    """
    调用 LLM 判断链接是否有用。
    link_info: dict, 包含 title, body, href (由 SearchInfo.exec 构造)
    industry: 行业名
    返回: (is_useful: bool, reason: str)
    """
    summary = link_info.get('body', '').strip() # 使用 body
    if not summary:
        return False, '无摘要，无法判断有用信息，视为无用链接'
    prompt = JUDGE_LINK_USEFULNESS_PROMPT.format(
        industry=industry,
        title=link_info.get('title', ''),
        body=summary, # 使用 body
        href=link_info.get('href', '') # 使用 href
    )
    resp = call_llm(prompt)
    result = parse_custom_structured_text(resp, ['reason', 'useful'])
    is_useful = str(result.get('useful','')).strip().lower() == 'true'
    return is_useful, result.get('reason','')


"""
示例用法
"""

# ================== 支持多主题批量搜索 ==================
SEARCH_TOPICS = [
    # {
    #     "search_topic": "智能风控&大数据征信服务",
    #     "max_rounds": 100,
    #     "desc": "聚合行业发展相关数据，行业生命周期与结构解读，政策影响、技术演进，行业进入与退出策略建议，上游原材料价格，行业规模变动、竞争格局"
    # },
    {
        "search_topic": "生成式AI基建与算力投资趋势（2023-2026）",
        "max_rounds": 6,
        "max_search_count": 15,  # 新增：最大搜索次数限制
        "desc": "政策信息，政策联动与区域对比信息，美联储利率变动对全球资本流动的影响，灰犀牛事件"
    },
    # {
    #     "search_topic": "商汤科技",
    #     "max_rounds": 100,
    #     "desc": "主营业务、核心竞争力与行业地位，行业对比分析，竞争分析，公开数据与管理层信息，治理结构与发展战略"
    # }
]

# ================== 搜索时间记录工具 ==================

def record_search_time(shared, event="search"):
    if "search_times" not in shared:
        shared["search_times"] = []
    shared["search_times"].append({
        "event": event,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })


# ================== 主流程支持多主题批量搜索 ==================
if __name__ == "__main__":
    decision = SearchDecisionFlow()
    search = SearchInfo()
    summary = SearchSummary()
    decision - "search" >> search
    decision - "complete" >> summary
    search - "search" >> decision
    flow = Flow(start=decision)

    # 初始化 Google 搜索池
    # 请替换为您的真实 API 密钥和搜索引擎 ID
    # 您可以添加多个客户端以实现配额轮换
    google_search_pool = GoogleSearchPool()
    try:
        google_search_pool.add_client(
            api_key=os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_1"), # 修改：使用 GOOGLE_API_KEY
            search_engine_id=os.getenv("GOOGLE_ENGINE_ID", "YOUR_GOOGLE_SEARCH_ENGINE_ID_1"), # 修改：使用 GOOGLE_ENGINE_ID
            daily_limit=90 # 示例每日限制
        )
        # 如果有更多API密钥，可以继续添加，确保环境变量名称也相应更新
        # 例如，如果您有 GOOGLE_API_KEY_2, GOOGLE_ENGINE_ID_2
        # google_search_pool.add_client(
        #     api_key=os.getenv("GOOGLE_API_KEY_2", "YOUR_GOOGLE_API_KEY_2"),
        #     search_engine_id=os.getenv("GOOGLE_ENGINE_ID_2", "YOUR_GOOGLE_SEARCH_ENGINE_ID_2"),
        #     daily_limit=90
        # )
        if not google_search_pool.clients:
            print("警告：未配置任何 Google 搜索客户端。搜索功能将不可用。")
            print("请在代码中或通过环境变量 GOOGLE_API_KEY, GOOGLE_ENGINE_ID 等配置客户端。") # 修改：更新提示信息
            # exit(1) # 如果没有客户端，可以选择退出
    except Exception as e:
        print(f"初始化 Google 搜索池时出错: {e}")
        # exit(1) # 初始化失败，可以选择退出


    for topic_cfg in SEARCH_TOPICS:
        shared_state = {
            "search_topic": topic_cfg["search_topic"],
            "max_rounds": topic_cfg["max_rounds"],
            "max_invalid_rounds": topic_cfg.get("max_invalid_rounds", 5),
            "max_search_count": topic_cfg.get("max_search_count"),
            "total_search_count": 0,
            "search_pool": google_search_pool # 新增：将搜索池实例传递给共享状态
        }
        print("\n=== 开始搜索流程 ===")
        print(f"目标主题: {shared_state['search_topic']}")
        print(f"说明: {topic_cfg['desc']}")
        print("注意: 此程序将持续收集信息，不会生成分析内容")
        print("程序将通过多轮搜索深入收集相关信息")
        try:
            result = flow.run(shared_state)
            if result:
                print(f"\n=== 搜索流程已完成 ===")
            else:
                print(f"\n=== 搜索流程正在进行中 ===")
        except Exception as e:
            print(f"\n!!! 搜索流程发生异常: {e}")
            print("正在保存当前已收集的搜索状态和报告...")
            # 尝试提前输出搜索报告
            try:
                summary_report = summary.exec((shared_state.get("search_topic"), shared_state.get("search_context", []), shared_state.get("search_round", 0)))
                filename = f"{shared_state['search_topic']}_搜索报告_异常中断.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(summary_report)
                print(f"异常中断时的搜索报告已保存到 '{filename}' 文件中")
            except Exception as e2:
                print(f"保存异常报告时再次出错: {e2}")
