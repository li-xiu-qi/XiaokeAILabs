import os
import time
import json
import openai
from duckduckgo_search import DDGS
from pocketflow import Node, Flow
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

class SearchNode(Node):
    """基础搜索节点：执行关键词搜索"""
    def prep(self, shared):
        query = shared.get('current_query')
        depth = shared.get('current_depth', 0)
        return query, depth

    def exec(self, inputs):
        query, depth = inputs
        print(f"\n[Level {depth}] 搜索关键词: {query}")
        
        # 执行搜索
        results = list(search_web(query))
        print(f"找到 {len(results)} 条结果")
        
        return {
            'query': query,
            'depth': depth,
            'results': results
        }

    def post(self, shared, prep_res, exec_res):
        # 保存搜索结果
        all_results = shared.get('all_results', [])
        all_results.append(exec_res)
        shared['all_results'] = all_results
        return 'analyze'

class AnalyzeNode(Node):
    """分析节点：评估搜索质量并决定下一步"""
    def prep(self, shared):
        all_results = shared.get('all_results', [])
        original_query = shared.get('original_query')
        max_depth = shared.get('max_depth', 3)
        current_depth = shared.get('current_depth', 0)
        return all_results, original_query, max_depth, current_depth

    def exec(self, inputs):
        all_results, original_query, max_depth, current_depth = inputs
        
        if not all_results:
            return {'action': 'stop', 'reason': '无搜索结果'}
        
        latest_results = all_results[-1]['results']
        
        # 使用LLM分析搜索质量和相关性
        prompt = f"""
原始查询: {original_query}
当前搜索深度: {current_depth}
最大深度限制: {max_depth}

最新搜索结果摘要:
{json.dumps([r.get('title', '') + ' | ' + r.get('body', '')[:100] for r in latest_results[:5]], ensure_ascii=False, indent=2)}

请分析:
1. 当前结果的质量和相关性 (1-10分)
2. 是否需要继续深入搜索
3. 如果需要，建议新的搜索策略

请以JSON格式回复:
{{
    "quality_score": 数字1-10,
    "relevance_score": 数字1-10,
    "action": "continue" 或 "refine" 或 "stop",
    "reason": "详细原因",
    "new_query": "如果action是continue或refine，提供新的搜索词",
    "search_strategy": "搜索策略说明"
}}
"""

        response = call_llm(prompt)
        try:
            # 提取JSON部分
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            else:
                json_str = response
            
            analysis = json.loads(json_str)
            print(f"质量评分: {analysis['quality_score']}/10")
            print(f"相关性评分: {analysis['relevance_score']}/10") 
            print(f"决策: {analysis['action']} - {analysis['reason']}")
            
            return analysis
            
        except Exception as e:
            print(f"分析解析错误: {e}")
            return {'action': 'stop', 'reason': 'AI分析失败'}

    def post(self, shared, prep_res, exec_res):
        action = exec_res.get('action', 'stop')
        current_depth = shared.get('current_depth', 0)
        max_depth = shared.get('max_depth', 3)
        
        if action == 'stop' or current_depth >= max_depth:
            return 'complete'
        elif action in ['continue', 'refine']:
            # 更新查询和深度
            shared['current_query'] = exec_res.get('new_query', shared['current_query'])
            shared['current_depth'] = current_depth + 1
            return 'search'
        else:
            return 'complete'

class SummarizeNode(Node):
    """总结节点：整理最终结果"""
    def prep(self, shared):
        return shared.get('all_results', []), shared.get('original_query')

    def exec(self, inputs):
        all_results, original_query = inputs
        print(f"\n=== 深度搜索完成 ===")
        print(f"原始查询: {original_query}")
        print(f"总共执行了 {len(all_results)} 轮搜索")
        
        # 统计总结果数
        total_results = sum(len(r['results']) for r in all_results)
        print(f"总共获得 {total_results} 条信息")
        
        return {
            'original_query': original_query,
            'search_rounds': len(all_results),
            'total_results': total_results,
            'detailed_results': all_results
        }

    def post(self, shared, prep_res, exec_res):
        shared['final_summary'] = exec_res
        return None

def call_llm(prompt: str) -> str:
    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def search_web(term: str):
    with DDGS() as ddgs:
        results = ddgs.text(keywords=term, region="cn-zh", max_results=15)
    return results

if __name__ == "__main__":
    # 构建节点
    search = SearchNode()
    analyze = AnalyzeNode()  
    summarize = SummarizeNode()
    
    # 连接流程
    search - 'analyze' >> analyze
    analyze - 'search' >> search
    analyze - 'complete' >> summarize
    
    # 初始化
    flow = Flow(start=search)
    shared_state = {
        'original_query': 'AIGC在金融行业的应用',
        'current_query': 'AIGC在金融行业的应用',
        'current_depth': 0,
        'max_depth': 3,
        'all_results': []
    }
    
    # 运行深度搜索
    result = flow.run(shared_state)
    
    # 保存结果
    if shared_state.get('final_summary'):
        with open("深度搜索结果.json", "w", encoding="utf-8") as f:
            json.dump(shared_state['final_summary'], f, ensure_ascii=False, indent=2)
        print(f"\n深度搜索结果已保存到 '深度搜索结果.json'")
