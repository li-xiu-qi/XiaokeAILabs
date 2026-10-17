# ================== 代码主体 ================

import os
import time
import yaml
import openai
import streamlit as st
from pocketflow import Node, Flow, build_mermaid # 移除 DDGS
from dotenv import load_dotenv
import random

from financial_search_prompt import INDUSTRY_RESEARCH_PROMPT, JUDGE_LINK_USEFULNESS_PROMPT
from google_search import GoogleSearchPool # 新增：导入GoogleSearchPool

# 设置页面配置
st.set_page_config(page_title="深度搜索助手", layout="wide")

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
        # Google API 每次最多返回10条
        results = search_pool.search(term, num_results=10)
        return results
    except Exception as e:
        st.error(f"Google 搜索 '{term}' 时出错: {e}") # 使用 Streamlit 的错误提示
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
        st.info(f"正在分析 {search_topic} 搜索进度... (第 {search_round + 1} 轮)")
        # 解析历史搜索关键词，避免重复
        previous_search_terms = []
        if search_round > 0:
            try:
                context_data = yaml.safe_load(context_yaml_str)
                if isinstance(context_data, list):
                    for search_batch in context_data:
                        if isinstance(search_batch, dict) and 'search_query' in search_batch: # 修改 'term' 为 'search_query'
                            previous_search_terms.append(search_batch['search_query'])
                        elif isinstance(search_batch, list):
                            for item in search_batch:
                                if isinstance(item, dict) and 'search_query' in item: # 修改 'term' 为 'search_query'
                                    previous_search_terms.append(item['search_query'])
            except Exception:
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
        with st.spinner("正在分析已收集信息并生成新的搜索关键词..."):
            llm_response = call_llm(prompt)
            result = parse_custom_structured_text(llm_response, ['continue_search', 'reason', 'search_inputs'])
        
        # 类型转换
        result['continue_search'] = str(result['continue_search']).strip().lower() == 'true'
        if result['search_inputs'] is None:
            result['search_inputs'] = []
        
        st.write(f"继续搜索: {result['continue_search']}")
        st.write(f"搜索原因: {result['reason']}")
        st.write("本轮搜索输入:", result["search_inputs"])
        return result

    def post(self, shared, prep_res, exec_res):
        # 控制最大轮数和无效轮数，决定是否提前终止
        max_rounds = shared.get("max_rounds")
        max_invalid_rounds = shared.get("max_invalid_rounds", 3)
        current_round = shared.get("search_round", 0)
        invalid_search_rounds = shared.get("invalid_search_rounds", 0)
        # 最大搜索次数判断
        max_search_count = shared.get("max_search_count")
        total_search_count = shared.get("total_search_count", 0)
        if max_rounds is not None and current_round >= max_rounds:
            st.warning(f"已达到最大轮数({max_rounds})，搜索流程终止")
            return "complete"
        if max_invalid_rounds is not None and invalid_search_rounds >= max_invalid_rounds:
            st.warning(f"连续{invalid_search_rounds}轮无有效信息，提前终止搜索流程")
            return "complete"
        if max_search_count is not None and total_search_count >= max_search_count:
            st.warning(f"已达到最大搜索次数({max_search_count})，搜索流程终止")
            return "complete"
        if exec_res.get("continue_search", True):
            shared["search_inputs"] = exec_res.get("search_inputs", [])
            shared["search_round"] = current_round + 1
            st.success("开始新一轮搜索")
            return "search"
        else:
            st.success("搜索流程完成")
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
        
        st.subheader(f"本轮搜索输入: {', '.join(search_queries)}")
        seen_urls = set(link['href'] for link in useful_links + useless_links if link.get('href'))
        
        progress_bar = st.progress(0)
        shared = self.shared if hasattr(self, 'shared') else None # 保留 shared 引用
        search_count_this_round = 0

        for idx, search_query in enumerate(search_queries, 1):
            st.write(f"搜索输入 ({idx}/{total_queries}): {search_query}")
            if search_pool is None:
                st.error("错误：search_pool 未初始化，无法执行搜索。")
                all_search_results.append({"search_query": search_query, "results": [], "error": "Search pool not available"})
                continue
            try:
                with st.spinner(f"正在搜索 '{search_query}'..."):
                    search_results_list = search_web(search_query, search_pool) # 修改：传递 search_pool
                search_count_this_round += 1
                
                st.write(f"找到 {len(search_results_list)} 条相关信息")
                round_useful_links = []
                
                with st.expander(f"查询: {search_query} 的搜索结果"):
                    for i, search_result_item in enumerate(search_results_list): # 重命名变量避免与外层冲突
                        url = search_result_item.get("link", "") # 修改：Google API 使用 "link"
                        title = search_result_item.get("title", "")
                        snippet = search_result_item.get("snippet", "") # 修改：Google API 使用 "snippet"

                        if not url or url in seen_urls:
                            continue
                        
                        st.text(f"正在分析结果 {i+1}/{len(search_results_list)}...")
                        
                        temp_link_info_for_judge = {
                            "title": title,
                            "body": snippet, # 将 snippet 映射到 body
                            "href": url     # 将 link 映射到 href
                        }
                        is_useful, reason = judge_link_usefulness(temp_link_info_for_judge, search_topic)  
                        
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
                            st.success(f"**有用**: {link_info['title']}")
                            st.write(f"原因: {reason}")
                        else:
                            useless_links.append(link_info)
                            st.error(f"**无用**: {link_info['title']}")
                            st.write(f"原因: {reason}")
                
                all_search_results.append({"search_query": search_query, "results": round_useful_links})
                
                if round_useful_links:
                    st.success("有用搜索结果预览:")
                    for j, useful_result in enumerate(round_useful_links[:3]):
                        st.write(f"{j+1}. {useful_result.get('title', '无标题')}")
                        st.write(f"   {useful_result.get('body', '无摘要')[:100]}...")
            except Exception as e:
                st.error(f"搜索 '{search_query}' 时出错: {str(e)}")
                all_search_results.append({"search_query": search_query, "results": [], "error": str(e)})
            
            # 更新进度条
            progress_bar.progress((idx) / total_queries)
            
            if idx < total_queries:
                sleep_time = random.randint(16, 30)
                with st.spinner(f"等待{sleep_time}秒后继续搜索..."):
                    time.sleep(sleep_time)
        
        # 更新累计搜索次数
        if shared is not None:
            shared["total_search_count"] = shared.get("total_search_count", 0) + search_count_this_round
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
        
        st.success("本轮搜索完成！")
        st.write(f"共收集到 {total_results} 条有用信息")
        st.write(f"累计有用链接: {len(useful_links)}，无用链接: {len(useless_links)}")
        st.info("返回决策节点，准备下一轮搜索...")
        
        if total_results == 0:
            shared["invalid_search_rounds"] = shared.get("invalid_search_rounds", 0) + 1
        else:
            shared["invalid_search_rounds"] = 0
        # 记录本轮搜索时间
        record_search_time(shared, event="search_round_end")
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
            shared.get("search_round", 0),
            shared.get("search_times", [])
        )

    def exec(self, inputs):
        search_topic, search_context, search_round, search_times = inputs
        st.header(f"{search_topic} 搜索报告汇总")
        st.write(f"完成搜索轮次: {search_round}")
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
                for search_result_item in results: # 变量名修改避免冲突
                    title = search_result_item.get("title", "无标题")
                    url = search_result_item.get("href", "") # 保持 href
                    body = search_result_item.get("body", "") # 保持 body
                    reference_md += (
                        f"\n【第{ref_idx}篇参考文章开始】\n"
                        f"[{ref_idx}] 标题：{title}\n"
                        f"[{ref_idx}] 原文链接: {url}\n"
                        f"[{ref_idx}] 摘要：{body}\n"
                        f"【第{ref_idx}篇参考文章结束】\n"
                    )
                    ref_idx += 1
        # 获取所有搜索时间
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
        st.write(f"搜索报告汇总:")
        st.write(f"- 搜索轮次: {search_round}")
        st.write(f"- 收集信息条数: {total_results}")
        return summary

    def post(self, shared, prep_res, exec_res):
        # 保存搜索报告到本地文件
        st.success("搜索流程完成！")
        shared["summary"] = exec_res
        filename = f"{shared['search_topic']}_搜索报告.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(exec_res)
        st.success(f"搜索报告已保存到 '{filename}' 文件中")
        
        # 提供下载链接
        st.download_button(
            label="下载搜索报告",
            data=exec_res,
            file_name=filename,
            mime="text/markdown",
        )
        return None


def judge_link_usefulness(link_info, industry):
    """
    调用 LLM 判断链接是否有用。
    link_info: dict, 包含 title, body, href (由 SearchInfo.exec 构造)
    industry: 行业名
    focus_areas: 关注领域
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


# ================== Streamlit页面主体 ==================
def main():
    st.title("深度搜索助手")
    st.markdown("""
    这是一个使用人工智能增强的深度搜索工具，可以通过多轮迭代搜索深入收集行业信息。
    """)

    # 初始化 Google 搜索池 (在会话状态中缓存)
    if "google_search_pool" not in st.session_state:
        pool = GoogleSearchPool()
        try:
            # 从环境变量加载第一个客户端
            api_key_1 = os.getenv("GOOGLE_API_KEY")
            engine_id_1 = os.getenv("GOOGLE_ENGINE_ID")
            if api_key_1 and engine_id_1:
                pool.add_client(api_key_1, engine_id_1, daily_limit=90) # 示例限制
            
            # 可以添加更多客户端的逻辑，例如 GOOGLE_API_KEY_2, GOOGLE_ENGINE_ID_2
            # api_key_2 = os.getenv("GOOGLE_API_KEY_2")
            # engine_id_2 = os.getenv("GOOGLE_ENGINE_ID_2")
            # if api_key_2 and engine_id_2:
            #     pool.add_client(api_key_2, engine_id_2, daily_limit=90)

        except Exception as e:
            st.sidebar.error(f"初始化Google搜索池失败: {e}")
        st.session_state.google_search_pool = pool
        
    google_search_pool = st.session_state.google_search_pool

    # 侧边栏设置
    with st.sidebar:
        st.header("搜索配置")
        search_topic = st.text_input("搜索主题", value="生成式AI基建与算力投资趋势（2023-2026）")
        max_rounds = st.slider("最大搜索轮数", min_value=1, max_value=10, value=6)
        max_invalid_rounds = st.slider("最大无效轮数", min_value=1, max_value=5, value=3)
        # 新增：最大搜索次数
        max_search_count = st.number_input("最大搜索次数", min_value=1, max_value=100, value=15)
        topic_desc = st.text_area("搜索说明", value="政策信息，政策联动与区域对比信息，美联储利率变动对全球资本流动的影响，灰犀牛事件")
        start_button = st.button("开始搜索")
        
        # 显示API密钥状态
        api_key_status = "已配置 ✓" if openai.api_key else "未配置 ✗"
        st.info(f"OpenAI API密钥: {api_key_status}")

        google_api_configured = bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_ENGINE_ID"))
        google_api_status = "已配置 ✓" if google_api_configured else "未配置 ✗"
        st.info(f"Google API凭据: {google_api_status}")
        if not google_api_configured:
            st.warning("Google API密钥或搜索引擎ID未在.env文件中配置。搜索功能将不可用。")
        
        if google_search_pool and google_search_pool.clients:
            st.success(f"Google搜索池已加载 {len(google_search_pool.clients)} 个客户端。")
            # google_search_pool.display_usage_stats() # 不适合在sidebar里直接打印，信息太多
        else:
            st.error("Google搜索池未成功加载客户端。")

    # 初始化会话状态
    if "search_running" not in st.session_state:
        st.session_state.search_running = False
    if "search_complete" not in st.session_state:
        st.session_state.search_complete = False
    if "shared_state" not in st.session_state:
        st.session_state.shared_state = {}
    
    # 处理搜索按钮点击事件
    if start_button:
        if not openai.api_key:
            st.error("请先配置OpenAI API密钥")
        elif not google_api_configured or not google_search_pool.clients:
            st.error("Google API未配置或搜索池无客户端，无法开始搜索。请检查.env文件和配置。")
        else:
            st.session_state.search_running = True
            st.session_state.search_complete = False
            st.session_state.shared_state = {
                "search_topic": search_topic,
                "max_rounds": max_rounds,
                "max_invalid_rounds": max_invalid_rounds,
                "max_search_count": max_search_count,  # 新增
                "total_search_count": 0,               # 新增
                "search_context": [],
                "search_round": 0,
                "useful_links": [],
                "useless_links": [],
                "search_times": [],
                "search_pool": google_search_pool # 新增：将搜索池实例传递给共享状态
            }
            record_search_time(st.session_state.shared_state, event="search_start")
            st.rerun()  # 更新: 从 experimental_rerun 到 rerun
    
    # 显示搜索状态
    if st.session_state.search_running:
        if not st.session_state.search_complete:
            run_search_workflow()
        else:
            display_search_results()
    
    # 显示About信息
    with st.expander("关于深度搜索助手"):
        st.markdown("""
        ### 深度搜索助手
        
        这是一个使用人工智能增强的搜索工具，通过多轮迭代搜索深入收集相关信息。
        
        **特点:**
        - 智能分析已收集信息并生成新搜索关键词
        - 自动判断搜索结果的有用性
        - 去重并累计有用/无用链接
        - 生成结构化搜索报告
        
        **使用方法:**
        1. 在侧边栏输入搜索主题
        2. 设置最大搜索轮数和无效轮数
        3. 点击"开始搜索"按钮
        4. 等待搜索完成并查看结果
        """)


def run_search_workflow():
    """执行搜索工作流"""
    # 创建工作流节点
    decision = SearchDecisionFlow()
    search = SearchInfo()
    summary = SearchSummary()
    
    # 设置工作流
    shared_state = st.session_state.shared_state
    current_round = shared_state.get("search_round", 0)
    
    # 创建进度指示器
    st.subheader("搜索进度")
    st.write(f"当前轮次: {current_round + 1}/{shared_state['max_rounds']}")
    
    # 根据当前状态执行相应的节点
    if "current_node" not in shared_state:
        shared_state["current_node"] = "decision"
    
    if shared_state["current_node"] == "decision":
        with st.spinner("正在分析并决策下一步..."):
            prep_res = decision.prep(shared_state)
            exec_res = decision.exec(prep_res)
            next_node = decision.post(shared_state, prep_res, exec_res)
            shared_state["current_node"] = next_node
            st.rerun()  # 更新: 从 experimental_rerun 到 rerun
    
    elif shared_state["current_node"] == "search":
        with st.spinner("正在执行搜索..."):
            prep_res = search.prep(shared_state)
            exec_res = search.exec(prep_res)
            next_node = search.post(shared_state, prep_res, exec_res)
            shared_state["current_node"] = next_node
            st.rerun()  # 更新: 从 experimental_rerun 到 rerun
    
    elif shared_state["current_node"] == "complete":
        with st.spinner("正在生成搜索报告..."):
            prep_res = summary.prep(shared_state)
            exec_res = summary.exec(prep_res)
            summary.post(shared_state, prep_res, exec_res)
            st.session_state.search_complete = True
            st.rerun()  # 更新: 从 experimental_rerun 到 rerun


def display_search_results():
    """显示搜索结果"""
    shared_state = st.session_state.shared_state
    
    st.header("🎉 搜索完成!")
    st.subheader(f"主题: {shared_state['search_topic']}")
    
    total_results = 0
    if "useful_links" in shared_state:
        total_results = len(shared_state["useful_links"])
    
    st.write(f"总共搜索轮次: {shared_state.get('search_round', 0)}")
    st.write(f"收集到有用信息: {total_results} 条")
    
    # 显示搜索报告
    if "summary" in shared_state and shared_state["summary"]:
        with st.expander("查看搜索报告", expanded=True):
            st.markdown(shared_state["summary"])
        
        # 提供下载按钮
        st.download_button(
            label="下载搜索报告",
            data=shared_state["summary"],
            file_name=f"{shared_state['search_topic']}_搜索报告.md",
            mime="text/markdown",
        )
    
    # 添加重新开始按钮
    if st.button("开始新的搜索"):
        st.session_state.search_running = False
        st.session_state.search_complete = False
        st.session_state.shared_state = {}
        st.rerun()  # 更新: 从 experimental_rerun 到 rerun


if __name__ == "__main__":
    main()
