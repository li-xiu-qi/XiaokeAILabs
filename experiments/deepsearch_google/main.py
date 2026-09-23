import os
import time
import yaml
import openai
from pocketflow import Node, Flow, build_mermaid
from dotenv import load_dotenv
import random
from financial_search_prompt import INDUSTRY_RESEARCH_PROMPT, JUDGE_LINK_USEFULNESS_PROMPT
from google_search import GoogleSearchPool

# 加载环境变量
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def call_llm(prompt: str) -> str:
    """调用 LLM（如 OpenAI GPT）进行对话生成"""
    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=8196,
    )
    return response.choices[0].message.content.strip()

def search_web(term: str, search_pool: GoogleSearchPool):
    """使用 GoogleSearchPool 进行网页搜索"""
    try:
        results = search_pool.search(term, num_results=10)
        return results
    except Exception as e:
        print(f"Google 搜索 '{term}' 时出错: {e}")
        return []

def parse_custom_structured_text(text, fields):
    """解析 LLM 返回的自定义结构化文本"""
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

class SearchDecisionFlow(Node):
    """搜索决策节点，判断是否继续搜索及生成新搜索输入"""
    def prep(self, shared):
        search_context = shared.get("search_context", [])
        context_str = yaml.dump(search_context, allow_unicode=True)
        search_topic = shared["search_topic"]
        search_round = shared.get("search_round", 0)
        return search_topic, context_str, search_round
    def exec(self, inputs):
        search_topic, context_yaml_str, search_round = inputs
        print(f"\n正在分析 {search_topic} 搜索进度... (第 {search_round + 1} 轮)")
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
                pass
        previous_search_terms_str = '\n'.join(f'- {term}' for term in previous_search_terms)
        if not previous_search_terms_str:
            previous_search_terms_str = '（无）'
        prompt = INDUSTRY_RESEARCH_PROMPT.format(
            industry=search_topic,
            context_yaml_str=context_yaml_str,
            previous_search_terms_str=previous_search_terms_str,
            search_round=search_round
        )
        llm_response = call_llm(prompt)
        result = parse_custom_structured_text(llm_response, ['continue_search', 'reason', 'search_inputs'])
        result['continue_search'] = str(result['continue_search']).strip().lower() == 'true'
        if result['search_inputs'] is None:
            result['search_inputs'] = []
        print(f"继续搜索: {result['continue_search']}")
        print(f"搜索原因: {result['reason']}")
        print("本轮搜索输入:", result["search_inputs"])
        return result
    def post(self, shared, prep_res, exec_res):
        # 判断是否达到终止条件，否则继续下一轮搜索
        max_rounds = shared.get("max_rounds")
        max_invalid_rounds = shared.get("max_invalid_rounds", 3)
        current_round = shared.get("search_round", 0)
        invalid_search_rounds = shared.get("invalid_search_rounds", 0)
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

class SearchInfo(Node):
    """执行实际搜索并筛选有用链接"""
    def prep(self, shared):
        return (
            shared.get("search_inputs", []),
            shared.get("search_topic"),
            shared.get("useful_links", []),
            shared.get("useless_links", []),
            shared.get("search_pool")
        )
    def exec(self, inputs):
        search_queries, search_topic, useful_links, useless_links, search_pool = inputs
        all_search_results = []
        total_queries = len(search_queries)
        print(f"\n本轮搜索输入: {', '.join(search_queries)}")
        seen_urls = set(link['href'] for link in useful_links + useless_links if link.get('href'))
        shared = self.shared if hasattr(self, 'shared') else None
        search_count_this_round = 0
        for idx, search_query in enumerate(search_queries, 1):
            print(f"\n搜索输入 ({idx}/{total_queries}): {search_query}")
            if search_pool is None:
                print("错误：search_pool 未初始化，无法执行搜索。")
                all_search_results.append({"search_query": search_query, "results": [], "error": "Search pool not available"})
                continue
            try:
                search_results_list = search_web(search_query, search_pool)
                search_count_this_round += 1
                print(f"找到 {len(search_results_list)} 条相关信息")
                round_useful_links = []
                for search_result in search_results_list:
                    url = search_result.get("link", "")
                    title = search_result.get("title", "")
                    snippet = search_result.get("snippet", "")
                    if not url or url in seen_urls:
                        continue
                    temp_link_info_for_judge = {
                        "title": title,
                        "body": snippet,
                        "href": url
                    }
                    is_useful, reason = judge_link_usefulness(temp_link_info_for_judge, search_topic)
                    link_info = {
                        "search_query": search_query,
                        "title": title,
                        "body": snippet,
                        "href": url,
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
                        print(f"     {useful_result.get('body', '无摘要')[:100]}...")
            except Exception as e:
                print(f"搜索 '{search_query}' 时出错: {e}")
                all_search_results.append({"search_query": search_query, "results": [], "error": str(e)})
            if idx < total_queries:
                sleep_time = random.randint(16, 30)
                print(f"等待秒{sleep_time}后继续搜索...")
                time.sleep(sleep_time)
        if shared is not None:
            shared["total_search_count"] = shared.get("total_search_count", 0) + search_count_this_round
        else:
            pass
        return all_search_results, useful_links, useless_links
    def post(self, shared, prep_res, exec_res):
        # 更新搜索上下文和有用/无用链接
        all_results, useful_links, useless_links = exec_res
        search_context = shared.get("search_context", [])
        search_context.extend(all_results)
        shared["search_context"] = search_context
        shared["useful_links"] = useful_links
        shared["useless_links"] = useless_links
        total_results = sum(len(item.get("results", [])) for item in all_results)
        print("\n本轮搜索完成！")
        print(f"共收集到 {total_results} 条有用信息")
        print(f"累计有用链接: {len(useful_links)}，无用链接: {len(useless_links)}")
        print("\n返回决策节点，准备下一轮搜索...")
        if total_results == 0:
            shared["invalid_search_rounds"] = shared.get("invalid_search_rounds", 0) + 1
        else:
            shared["invalid_search_rounds"] = 0
        record_search_time(shared, event="search_round_end")
        return "search"

class SearchSummary(Node):
    """搜索流程结束后，生成搜索报告"""
    def prep(self, shared):
        return (
            shared.get("search_topic"),
            shared.get("search_context", []),
            shared.get("search_round", 0)
        )
    def exec(self, inputs):
        search_topic, search_context, search_round = inputs
        print(f"\n=== {search_topic} 搜索报告汇总 ===")
        print(f"完成搜索轮次: {search_round}")
        total_results = 0
        for search_batch in search_context:
            if isinstance(search_batch, dict):
                total_results += len(search_batch.get("results", []))
        reference_md = ""
        ref_idx = 1
        for search_batch in search_context:
            if isinstance(search_batch, dict):
                results = search_batch.get("results", [])
                for search_result in results:
                    title = search_result.get("title", "无标题")
                    url = search_result.get("href", "")
                    body = search_result.get("body", "")
                    reference_md += (
                        f"\n【第{ref_idx}篇参考文章开始】\n"
                        f"[{ref_idx}] 标题：{title}\n"
                        f"[{ref_idx}] 原文链接: {url}\n"
                        f"[{ref_idx}] 摘要：{body}\n"
                        f"【第{ref_idx}篇参考文章结束】\n"
                    )
                    ref_idx += 1
        search_times = inputs[3] if len(inputs) > 3 else []
        search_time_md = "\n".join([
            f"- {item['event']}: {item['timestamp']}" for item in search_times
        ]) if search_times else "无"
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
        print("搜索报告汇总:")
        print(f"- 搜索轮次: {search_round}")
        print(f"- 收集信息条数: {total_results}")
        return summary
    def post(self, shared, prep_res, exec_res):
        print("\n=== 搜索流程完成！===")
        shared["summary"] = exec_res
        filename = f"{shared['search_topic']}_搜索报告.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(exec_res)
        print(f"搜索报告已保存到 '{filename}' 文件中")
        return None

def judge_link_usefulness(link_info, industry):
    """判断链接是否有用，调用 LLM 进行判别"""
    summary = link_info.get('body', '').strip()
    if not summary:
        return False, '无摘要，无法判断有用信息，视为无用链接'
    prompt = JUDGE_LINK_USEFULNESS_PROMPT.format(
        industry=industry,
        title=link_info.get('title', ''),
        body=summary,
        href=link_info.get('href', '')
    )
    resp = call_llm(prompt)
    result = parse_custom_structured_text(resp, ['reason', 'useful'])
    is_useful = str(result.get('useful','')).strip().lower() == 'true'
    return is_useful, result.get('reason','')

# 搜索主题配置
SEARCH_TOPICS = [
    {
        "search_topic": "生成式AI基建与算力投资趋势（2023-2026）",
        "max_rounds": 6,
        "max_search_count": 15,
        "desc": "政策信息，政策联动与区域对比信息，美联储利率变动对全球资本流动的影响，灰犀牛事件"
    },
]

def record_search_time(shared, event="search"):
    """记录每轮搜索的时间点"""
    if "search_times" not in shared:
        shared["search_times"] = []
    shared["search_times"].append({
        "event": event,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == "__main__":
    # 构建流程节点
    decision = SearchDecisionFlow()
    search = SearchInfo()
    summary = SearchSummary()
    decision - "search" >> search
    decision - "complete" >> summary
    search - "search" >> decision
    flow = Flow(start=decision)
    # 初始化 Google 搜索池
    google_search_pool = GoogleSearchPool()
    try:
        google_search_pool.add_client(
            api_key=os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_1"),
            search_engine_id=os.getenv("GOOGLE_ENGINE_ID", "YOUR_GOOGLE_SEARCH_ENGINE_ID_1"),
            daily_limit=90
        )
        if not google_search_pool.clients:
            print("警告：未配置任何 Google 搜索客户端。搜索功能将不可用。")
            print("请在代码中或通过环境变量 GOOGLE_API_KEY, GOOGLE_ENGINE_ID 等配置客户端。")
    except Exception as e:
        print(f"初始化 Google 搜索池时出错: {e}")
    # 遍历每个搜索主题，执行流程
    for topic_cfg in SEARCH_TOPICS:
        shared_state = {
            "search_topic": topic_cfg["search_topic"],
            "max_rounds": topic_cfg["max_rounds"],
            "max_invalid_rounds": topic_cfg.get("max_invalid_rounds", 5),
            "max_search_count": topic_cfg.get("max_search_count"),
            "total_search_count": 0,
            "search_pool": google_search_pool
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
            try:
                summary_report = summary.exec((shared_state.get("search_topic"), shared_state.get("search_context", []), shared_state.get("search_round", 0)))
                filename = f"{shared_state['search_topic']}_搜索报告_异常中断.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(summary_report)
                print(f"异常中断时的搜索报告已保存到 '{filename}' 文件中")
            except Exception as e2:
                print(f"保存异常报告时再次出错: {e2}")
