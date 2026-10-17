# -*- coding: utf-8 -*-
"""
红楼梦知识图谱构建器
基于NLP技术提取红楼梦中的人物、地点、关系等信息，构建知识图谱
"""

import os
import re
import json
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
import asyncio
from dotenv import load_dotenv
import yaml  # 新增：用于解析YAML


from semantic_splitter import SemanticSplitter
from fallback_openai_client import AsyncFallbackOpenAIClient
from prompts import HONGLOUMENG_KG_EXTRACTION_YAML_PROMPT  # 导入提示词


# 加载环境变量
load_dotenv()

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

@dataclass
class Entity:
    """实体类"""
    name: str
    type: str  # person, place, object, concept
    aliases: Set[str]
    attributes: Dict[str, Any]
    mentions: List[str]  # 出现的上下文

@dataclass
class Relation:
    """关系类"""
    source: str
    target: str
    relation_type: str
    confidence: float
    context: str

class HongLouMengKGBuilder:
    """红楼梦知识图谱构建器（全流程大模型自动抽取，无预设，无分词）"""
    
    def __init__(self, text_file: str = "红楼梦.txt"):
        """
        初始化知识图谱构建器
        
        Args:
            text_file: 红楼梦文本文件路径
        """
        self.text_file = text_file
        self.text = ""
        self.entities = {}  # name -> Entity
        self.relations = []  # List[Relation]
        self.graph = nx.DiGraph()
        self.llm_client = None
        self._init_llm_client()
        self._load_text()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            primary_api_key = os.getenv("ZHIPU_API_KEY")
            primary_base_url = os.getenv("ZHIPU_BASE_URL")
            primary_model = os.getenv("ZHIPU_MODEL", "glm-4-flash")
            
            fallback_api_key = os.getenv("GUIJI_API_KEY")
            fallback_base_url = os.getenv("GUIJI_BASE_URL")
            fallback_model = os.getenv("GUIJI_MODEL", "THUDM/GLM-4-9B-0414")
            
            if primary_api_key and primary_base_url:
                self.llm_client = AsyncFallbackOpenAIClient(
                    primary_api_key=primary_api_key,
                    primary_base_url=primary_base_url,
                    primary_model_name=primary_model,
                    fallback_api_key=fallback_api_key,
                    fallback_base_url=fallback_base_url,
                    fallback_model_name=fallback_model
                )
                print("✅ LLM客户端初始化成功")
            else:
                print("⚠️ 未找到LLM配置，将使用基础NLP方法")
        except Exception as e:
            print(f"⚠️ LLM客户端初始化失败: {e}")
            
    def _load_text(self):
        """加载红楼梦文本"""
        if os.path.exists(self.text_file):
            with open(self.text_file, 'r', encoding='utf-8') as f:
                self.text = f.read()
            print(f"✅ 成功加载文本，总长度: {len(self.text)} 字符")
        else:
            raise FileNotFoundError(f"找不到文件: {self.text_file}")
    
    async def extract_entities_and_relations_with_llm(self, chunk_size: int = 1000, max_concurrency: int = 200) -> Tuple[Dict[str, Entity], List[Relation]]:
        """
        使用LLM自动抽取实体和关系（全流程自动，无任何预设/分词/规则/词性标注），支持并发处理。
        """
        if not self.llm_client:
            print("⚠️ LLM客户端不可用，跳过LLM实体与关系抽取")
            return self.entities, self.relations
        print("\n🤖 开始用LLM自动抽取实体和关系...")
        splitter = SemanticSplitter(target_chunk_size=chunk_size)
        chunks = splitter.split_text(self.text)
        print(f"文本分为 {len(chunks)} 个块，开始LLM并发处理...")

        import asyncio
        semaphore = asyncio.Semaphore(max_concurrency)
        results = [None] * len(chunks)

        async def process_chunk(i, chunk):
            async with semaphore:
                try:
                    print(f"处理第 {i+1}/{len(chunks)} 个文本块...")
                    entities, relations = await self._extract_entities_and_relations_from_chunk(chunk)
                    print(f"块 {i+1} 抽取到实体: {len(entities)}，关系: {len(relations)}")
                    print(f"示例实体: {entities[:1] if entities else '无'}")
                    print(f"示例关系: {relations[:1] if relations else '无'}")
                    results[i] = (entities, relations)
                except Exception as e:
                    print(f"处理块 {i+1} 时出错: {e}")
                    results[i] = ([], [])

        await asyncio.gather(*(process_chunk(i, chunk) for i, chunk in enumerate(chunks)))        # 合并所有结果
        for entities, relations in results:
            # 合并实体
            if entities is None:
                entities = []
            if relations is None:
                relations = []
                
            for entity_data in entities:
                if not isinstance(entity_data, dict):
                    continue
                name = entity_data.get("name", "").strip()
                if name and len(name) >= 2:
                    # 确保 attributes 是字典类型
                    entity_attributes = entity_data.get("attributes", {})
                    if not isinstance(entity_attributes, dict):
                        entity_attributes = {}
                    
                    if name in self.entities:
                        self.entities[name].attributes.update(entity_attributes)
                    else:
                        # 确保 aliases 和 mentions 也是正确类型
                        aliases = entity_data.get("aliases", [])
                        if not isinstance(aliases, list):
                            aliases = []
                        
                        mentions = entity_data.get("mentions", [])
                        if not isinstance(mentions, list):
                            mentions = []
                            
                        self.entities[name] = Entity(
                            name=name,
                            type=entity_data.get("type", "unknown"),
                            aliases=set(aliases),
                            attributes=entity_attributes,
                            mentions=mentions
                        )
            # 合并关系
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                if rel.get("source") and rel.get("target"):
                    self.relations.append(
                        Relation(
                            source=rel["source"],
                            target=rel["target"],
                            relation_type=rel.get("relation_type", "未知"),
                            confidence=rel.get("confidence", 0.7),
                            context=rel.get("context", "")
                        )
                    )
        print(f"✅ LLM实体与关系抽取完成，当前实体数: {len(self.entities)}，关系数: {len(self.relations)}")
        return self.entities, self.relations
    
    async def _extract_entities_and_relations_from_chunk(self, chunk: str) -> Tuple[list, list]:
        """
        用LLM从文本块中抽取实体和关系，要求输出YAML格式
        """
        prompt = HONGLOUMENG_KG_EXTRACTION_YAML_PROMPT.format(text=chunk[:800])
        try:
            response = await self.llm_client.chat_completions_create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096
            )
            content = response.choices[0].message.content.strip()

            # 优先用split提取markdown代码块中的yaml内容
            if '```yaml' in content:
                parts = content.split('```yaml')
                if len(parts) > 1:
                    yaml_part = parts[1].split('```')[0]
                    cleaned_content = yaml_part.strip()
                else:
                    cleaned_content = content
            else:
                cleaned_content = self._clean_yaml_content(content)

            # 尝试解析YAML
            yaml_match = re.search(r'(entities:|relations:)[\s\S]*', cleaned_content, re.IGNORECASE)
            if yaml_match:
                yaml_text = yaml_match.group()
                try:
                    data = yaml.safe_load(yaml_text)
                    if isinstance(data, dict):
                        return data.get("entities", []), data.get("relations", [])
                except yaml.YAMLError as yaml_error:
                    print(f"YAML解析错误: {yaml_error}")
                    # 直接fallback到正则解析，并传递报错信息
                    return await self._fallback_parse(cleaned_content, str(yaml_error))

            # 如果没有找到YAML格式，尝试备用解析
            return await self._fallback_parse(cleaned_content)

        except Exception as e:
            print(f"LLM抽取实体和关系时出错: {e}")
            return [], []
    
    async def _fallback_parse(self, content: str, error_msg: str = None) -> Tuple[list, list]:
        """
        使用LLM修正格式错误的内容为标准YAML并解析。
        可选地将报错信息一并输入，便于大模型修复。
        """        # 构造修正提示，包含报错信息
        fix_prompt = (
            "你刚才输出的内容格式有误，请将下面内容修正为严格的 YAML 格式，并用 markdown 代码块包裹，只保留 entities 和 relations 字段，"
            "所有字符串必须用英文双引号包裹。不要输出任何解释。\n"
        )
        if error_msg:
            fix_prompt += f"\nYAML解析报错信息如下：\n{error_msg}\n"
        fix_prompt += f"\n原始内容：\n{content[:2000]}"
        try:
            response = await self.llm_client.chat_completions_create(
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
                max_tokens=4096
            )
            fixed_content = response.choices[0].message.content.strip()
            # 提取yaml代码块
            if '```yaml' in fixed_content:
                parts = fixed_content.split('```yaml')
                if len(parts) > 1:
                    yaml_part = parts[1].split('```')[0]
                    cleaned_content = yaml_part.strip()
                else:
                    cleaned_content = fixed_content
            else:
                cleaned_content = fixed_content
            # 解析
            yaml_match = re.search(r'(entities:|relations:)[\s\S]*', cleaned_content, re.IGNORECASE)
            if yaml_match:
                yaml_text = yaml_match.group()
                try:
                    data = yaml.safe_load(yaml_text)
                    if isinstance(data, dict):
                        return data.get("entities", []), data.get("relations", [])
                except Exception as e:
                    print(f"二次LLM修正后YAML解析仍失败: {e}")
            print("二次LLM修正后仍无法解析，返回空")
            return [], []
        except Exception as e:
            print(f"LLM二次修正内容时出错: {e}")
            return [], []
    def build_graph(self):
        """构建知识图谱"""
        print("\n📊 开始构建知识图谱...")
        
        # 添加节点
        for name, entity in self.entities.items():
            # 过滤掉None值的属性
            clean_attributes = {k: v for k, v in entity.attributes.items() if v is not None}
            
            node_attrs = {
                "type": entity.type or "unknown",
                "frequency": clean_attributes.get("frequency", 0),
                **clean_attributes
            }
            
            self.graph.add_node(name, **node_attrs)
        
        # 添加边
        for relation in self.relations:
            if relation.source in self.graph and relation.target in self.graph:
                # 确保所有边属性都不是None
                edge_attrs = {
                    "relation_type": relation.relation_type or "unknown",
                    "confidence": relation.confidence if relation.confidence is not None else 0.5,
                    "context": (relation.context[:100] + "..." if len(relation.context) > 100 else relation.context) if relation.context else ""
                }
                
                self.graph.add_edge(
                    relation.source,
                    relation.target,
                    **edge_attrs
                )
        
        print(f"✅ 知识图谱构建完成")
        print(f"   节点数: {self.graph.number_of_nodes()}")
        print(f"   边数: {self.graph.number_of_edges()}")
    
    def analyze_graph(self) -> Dict[str, Any]:
        """分析知识图谱"""
        print("\n📈 开始图谱分析...")
        
        analysis = {}
        
        # 基本统计
        analysis["basic_stats"] = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }
        
        # 中心性分析
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            analysis["top_degree_centrality"] = sorted(
                degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]
            
            analysis["top_betweenness_centrality"] = sorted(
                betweenness_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]
        
        # 实体类型分布
        entity_types = defaultdict(int)
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "unknown")
            entity_types[node_type] += 1
        
        analysis["entity_type_distribution"] = dict(entity_types)
        
        # 关系类型分布
        relation_types = defaultdict(int)
        for edge in self.graph.edges():
            relation_type = self.graph.edges[edge].get("relation_type", "unknown")
            relation_types[relation_type] += 1
        
        analysis["relation_type_distribution"] = dict(relation_types)
        
        print("✅ 图谱分析完成")
        return analysis
    
    def visualize_graph(self, output_file: str = "hongloumeng_kg.png", 
                       max_nodes: int = 50, min_frequency: int = 3):
        """可视化知识图谱"""
        print(f"\n🎨 开始可视化知识图谱，保存到 {output_file}")
        
        # 创建子图（只显示高频实体）
        important_nodes = [
            node for node in self.graph.nodes()
            if self.graph.nodes[node].get("frequency", 0) >= min_frequency
        ][:max_nodes]
        
        subgraph = self.graph.subgraph(important_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print("⚠️ 没有符合条件的节点用于可视化")
            return
        
        plt.figure(figsize=(20, 16))
        
        # 计算布局
        try:
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        except:
            pos = nx.random_layout(subgraph)
        
        # 根据实体类型设置颜色
        color_map = {
            "person": "#FF6B6B",
            "place": "#4ECDC4", 
            "object": "#45B7D1",
            "concept": "#96CEB4",
            "unknown": "#FECA57"
        }
        
        node_colors = [
            color_map.get(subgraph.nodes[node].get("type", "unknown"), "#FECA57")
            for node in subgraph.nodes()
        ]
        
        # 根据频率设置节点大小
        node_sizes = [
            max(100, min(2000, subgraph.nodes[node].get("frequency", 1) * 50))
            for node in subgraph.nodes()
        ]
        
        # 绘制图
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7,
            arrowsize=20
        )
        
        plt.title("红楼梦知识图谱", fontsize=16, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            plt.scatter([], [], c=color, s=100, label=type_name)
            for type_name, color in color_map.items()
            if any(subgraph.nodes[node].get("type") == type_name for node in subgraph.nodes())
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 知识图谱可视化完成，保存到 {output_file}")
    
    def save_results(self, output_dir: str = "kg_output"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 开始保存结果到 {output_dir}")
        
        # 保存实体
        entities_data = {
            name: {
                "type": entity.type,
                "aliases": list(entity.aliases),
                "attributes": entity.attributes,
                "mentions": entity.mentions[:5]  # 限制数量
            }
            for name, entity in self.entities.items()
        }
        
        with open(os.path.join(output_dir, "entities.json"), 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        
        # 保存关系
        relations_data = [
            {
                "source": relation.source,
                "target": relation.target,
                "relation_type": relation.relation_type,
                "confidence": relation.confidence,
                "context": relation.context[:200] + "..." if len(relation.context) > 200 else relation.context
            }
            for relation in self.relations
        ]
        
        with open(os.path.join(output_dir, "relations.json"), 'w', encoding='utf-8') as f:
            json.dump(relations_data, f, ensure_ascii=False, indent=2)
        
        # 保存图谱
        nx.write_gexf(self.graph, os.path.join(output_dir, "knowledge-graph.gexf"))
        
        # 保存分析结果
        analysis = self.analyze_graph()
        with open(os.path.join(output_dir, "analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果保存完成")
        print(f"   实体文件: {os.path.join(output_dir, 'entities.json')}")
        print(f"   关系文件: {os.path.join(output_dir, 'relations.json')}")
        print(f"   图谱文件: {os.path.join(output_dir, 'knowledge-graph.gexf')}")
        print(f"   分析文件: {os.path.join(output_dir, 'analysis.json')}")
    
    async def build_complete_kg(self, output_dir: str = "kg_output"):
        """完整的知识图谱构建流程（全自动，无预设，实体和关系均用LLM）"""
        print("🚀 开始构建红楼梦知识图谱（全流程大模型自动抽取）")
        print("=" * 50)
        
        try:
            await self.extract_entities_and_relations_with_llm()
            self.build_graph()
            analysis = self.analyze_graph()
            self.visualize_graph()
            self.save_results(output_dir)
            self._print_summary(analysis)
            
            print("\n🎉 红楼梦知识图谱构建完成！")
            
        except Exception as e:
            print(f"❌ 构建过程中出现错误: {e}")
            raise
        finally:
            if self.llm_client:
                await self.llm_client.close()
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """打印总结信息"""
        print("\n📊 知识图谱构建总结")
        print("=" * 30)
        
        print(f"📈 基本统计:")
        basic_stats = analysis.get("basic_stats", {})
        print(f"   实体数量: {basic_stats.get('nodes', 0)}")
        print(f"   关系数量: {basic_stats.get('edges', 0)}")
        print(f"   图密度: {basic_stats.get('density', 0):.4f}")
        
        print(f"\n🏆 核心人物（按度中心性）:")
        top_degree = analysis.get("top_degree_centrality", [])
        for i, (name, centrality) in enumerate(top_degree[:5], 1):
            print(f"   {i}. {name}: {centrality:.4f}")
        
        print(f"\n🌉 关键中介（按介数中心性）:")
        top_betweenness = analysis.get("top_betweenness_centrality", [])
        for i, (name, centrality) in enumerate(top_betweenness[:5], 1):
            print(f"   {i}. {name}: {centrality:.4f}")
        
        print(f"\n📊 实体类型分布:")
        entity_dist = analysis.get("entity_type_distribution", {})
        for entity_type, count in entity_dist.items():
            print(f"   {entity_type}: {count}")
        
        print(f"\n🔗 关系类型分布:")
        relation_dist = analysis.get("relation_type_distribution", {})
        for relation_type, count in sorted(relation_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {relation_type}: {count}")


async def main():
    """主函数"""
    # 检查文件是否存在
    text_file = "红楼梦.txt"
    if not os.path.exists(text_file):
        print(f"❌ 找不到文件: {text_file}")
        return
    
    # 创建知识图谱构建器
    kg_builder = HongLouMengKGBuilder(text_file)
    
    # 构建知识图谱
    await kg_builder.build_complete_kg(
        output_dir="hongloumeng_kg_output"
    )


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
