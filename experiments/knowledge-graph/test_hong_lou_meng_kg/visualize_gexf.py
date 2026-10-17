# -*- coding: utf-8 -*-
"""
GEXF知识图谱可视化工具
支持多种可视化方式：NetworkX、Pyvis交互式图、Gephi风格布局等
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from pyvis.network import Network
import pandas as pd
from collections import defaultdict
import webbrowser
import json

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class GEXFVisualizer:
    """GEXF图谱可视化器"""
    
    def __init__(self, gexf_file: str):
        """
        初始化可视化器
        
        Args:
            gexf_file: GEXF文件路径
        """
        self.gexf_file = gexf_file
        self.graph = None
        self._load_graph()
    
    def _load_graph(self):
        """加载GEXF图谱"""
        try:
            self.graph = nx.read_gexf(self.gexf_file)
            print(f"✅ 成功加载图谱：{self.gexf_file}")
            print(f"   节点数: {self.graph.number_of_nodes()}")
            print(f"   边数: {self.graph.number_of_edges()}")
        except Exception as e:
            print(f"❌ 加载图谱失败: {e}")
            raise
    
    def analyze_graph(self):
        """分析图谱基本信息"""
        print("\n📊 图谱分析结果：")
        print(f"节点数量: {self.graph.number_of_nodes()}")
        print(f"边数量: {self.graph.number_of_edges()}")
        print(f"图密度: {nx.density(self.graph):.4f}")
        
        # 节点类型分布
        node_types = defaultdict(int)
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] += 1
        
        print(f"\n实体类型分布:")
        for type_name, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_name}: {count}")
        
        # 关系类型分布
        edge_types = defaultdict(int)
        for edge in self.graph.edges():
            edge_type = self.graph.edges[edge].get('relation_type', 'unknown')
            edge_types[edge_type] += 1
        
        print(f"\n关系类型分布:")
        for type_name, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {type_name}: {count}")
        
        # 度中心性分析
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            print(f"\n🏆 度中心性最高的节点:")
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (node, centrality) in enumerate(top_nodes, 1):
                print(f"  {i}. {node}: {centrality:.4f}")
    
    def visualize_static(self, output_file: str = "knowledge-graph_static.png", 
                        max_nodes: int = 100, min_degree: int = 1,
                        figsize: tuple = (20, 16)):
        """
        静态图可视化（使用matplotlib）
        
        Args:
            output_file: 输出文件名
            max_nodes: 最大显示节点数
            min_degree: 最小度数筛选
            figsize: 图片尺寸
        """
        print(f"\n🎨 开始静态图可视化...")
        
        # 筛选重要节点
        degrees = dict(self.graph.degree())
        important_nodes = [
            node for node, degree in degrees.items() 
            if degree >= min_degree
        ]
        
        # 按度数排序并限制数量
        important_nodes = sorted(important_nodes, 
                               key=lambda x: degrees[x], 
                               reverse=True)[:max_nodes]
        
        if not important_nodes:
            print("⚠️ 没有符合条件的节点")
            return
        
        # 创建子图
        subgraph = self.graph.subgraph(important_nodes)
        
        plt.figure(figsize=figsize)
        
        # 计算布局
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(subgraph, seed=42)
        
        # 设置颜色映射
        color_map = {
            'person': '#FF6B6B',      # 红色 - 人物
            'place': '#4ECDC4',       # 青色 - 地点
            'object': '#45B7D1',      # 蓝色 - 物品
            'concept': '#96CEB4',     # 绿色 - 概念
            'other': '#FECA57',       # 黄色 - 其他
            'unknown': '#95A5A6'      # 灰色 - 未知
        }
        
        # 获取节点颜色
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            node_colors.append(color_map.get(node_type, '#95A5A6'))
        
        # 根据度数设置节点大小
        node_sizes = [max(100, min(3000, degrees[node] * 100)) for node in subgraph.nodes()]
        
        # 绘制图
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color='gray',
                              alpha=0.5,
                              width=0.5,
                              arrows=True,
                              arrowsize=10)
        
        # 添加标签（只为重要节点）
        top_nodes = sorted(subgraph.nodes(), 
                          key=lambda x: degrees[x], 
                          reverse=True)[:20]
        
        labels = {node: node for node in top_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels,
                               font_size=8,
                               font_weight='bold',
                               font_family='sans-serif')
        
        plt.title("红楼梦知识图谱", fontsize=20, fontweight='bold', pad=20)
        
        # 添加图例
        legend_elements = []
        used_types = set(subgraph.nodes[node].get('type', 'unknown') for node in subgraph.nodes())
        
        for type_name, color in color_map.items():
            if type_name in used_types:
                legend_elements.append(
                    plt.scatter([], [], c=color, s=200, label=type_name, alpha=0.8)
                )
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 静态图保存到: {output_file}")
    
    def visualize_interactive(self, output_file: str = "knowledge-graph_interactive.html",
                            max_nodes: int = 200, min_degree: int = 1):
        """
        交互式图可视化（使用Pyvis）
        
        Args:
            output_file: 输出HTML文件名
            max_nodes: 最大显示节点数
            min_degree: 最小度数筛选
        """
        print(f"\n🌐 开始交互式图可视化...")
        
        # 筛选重要节点
        degrees = dict(self.graph.degree())
        important_nodes = [
            node for node, degree in degrees.items() 
            if degree >= min_degree
        ]
        
        # 按度数排序并限制数量
        important_nodes = sorted(important_nodes, 
                               key=lambda x: degrees[x], 
                               reverse=True)[:max_nodes]
        
        if not important_nodes:
            print("⚠️ 没有符合条件的节点")
            return
        
        # 创建子图
        subgraph = self.graph.subgraph(important_nodes)
        
        # 创建Pyvis网络
        net = Network(height="800px", width="100%", 
                     bgcolor="#ffffff", font_color="black")
        
        # 设置物理引擎
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        # 颜色映射
        color_map = {
            'person': '#FF6B6B',      # 红色 - 人物
            'place': '#4ECDC4',       # 青色 - 地点  
            'object': '#45B7D1',      # 蓝色 - 物品
            'concept': '#96CEB4',     # 绿色 - 概念
            'other': '#FECA57',       # 黄色 - 其他
            'unknown': '#95A5A6'      # 灰色 - 未知
        }
        
        # 添加节点
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            color = color_map.get(node_type, '#95A5A6')
            
            # 计算节点大小
            degree = degrees[node]
            size = max(10, min(50, degree * 3))
            
            # 创建hover信息
            title = f"节点: {node}\n类型: {node_type}\n度数: {degree}"
            if node_data.get('frequency'):
                title += f"\n频率: {node_data['frequency']}"
            
            net.add_node(node, 
                        label=node,
                        color=color,
                        size=size,
                        title=title)
        
        # 添加边
        for edge in subgraph.edges():
            source, target = edge
            edge_data = subgraph.edges[edge]
            
            # 创建边的hover信息
            relation_type = edge_data.get('relation_type', '未知关系')
            confidence = edge_data.get('confidence', 0.5)
            context = edge_data.get('context', '')[:100]
            
            title = f"关系: {relation_type}\n置信度: {confidence:.2f}"
            if context:
                title += f"\n上下文: {context}..."
            
            net.add_edge(source, target, 
                        title=title,
                        width=max(1, confidence * 3))
        
        # 保存HTML文件
        net.save_graph(output_file)
        
        print(f"✅ 交互式图保存到: {output_file}")
        print(f"💡 在浏览器中打开 {output_file} 查看交互式图谱")
        
        # 尝试在浏览器中打开
        try:
            abs_path = os.path.abspath(output_file)
            webbrowser.open(f'file://{abs_path}')
        except:
            pass
    
    def export_to_formats(self, output_dir: str = "export"):
        """
        导出为多种格式
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📁 导出图谱为多种格式到: {output_dir}")
        
        # 导出为GraphML (可用于Cytoscape)
        graphml_file = os.path.join(output_dir, "knowledge-graph.graphml")
        nx.write_graphml(self.graph, graphml_file)
        print(f"✅ GraphML: {graphml_file}")
        
        # 导出为边列表
        edgelist_file = os.path.join(output_dir, "knowledge-graph_edges.csv")
        edges_data = []
        for edge in self.graph.edges():
            source, target = edge
            edge_data = self.graph.edges[edge]
            edges_data.append({
                'source': source,
                'target': target,
                'relation_type': edge_data.get('relation_type', ''),
                'confidence': edge_data.get('confidence', 0.5),
                'context': edge_data.get('context', '')
            })
        
        pd.DataFrame(edges_data).to_csv(edgelist_file, index=False, encoding='utf-8-sig')
        print(f"✅ 边列表CSV: {edgelist_file}")
        
        # 导出节点信息
        nodes_file = os.path.join(output_dir, "knowledge-graph_nodes.csv")
        nodes_data = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            nodes_data.append({
                'name': node,
                'type': node_data.get('type', ''),
                'frequency': node_data.get('frequency', 0),
                'degree': self.graph.degree(node)
            })
        
        pd.DataFrame(nodes_data).to_csv(nodes_file, index=False, encoding='utf-8-sig')
        print(f"✅ 节点信息CSV: {nodes_file}")
        
        # 导出统计信息
        stats_file = os.path.join(output_dir, "graph_statistics.json")
        stats = {
            'basic_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(), 
                'density': nx.density(self.graph)
            },
            'node_types': {},
            'edge_types': {}
        }
        
        # 节点类型统计
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # 边类型统计
        for edge in self.graph.edges():
            edge_type = self.graph.edges[edge].get('relation_type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 统计信息JSON: {stats_file}")


def main():
    """主函数"""
    # GEXF文件路径
    gexf_file = "hongloumeng_kg_output/knowledge-graph.gexf"
    
    if not os.path.exists(gexf_file):
        print(f"❌ 找不到GEXF文件: {gexf_file}")
        print("请确保已经运行了知识图谱构建程序")
        return
    
    # 创建可视化器
    visualizer = GEXFVisualizer(gexf_file)
    
    # 分析图谱
    visualizer.analyze_graph()
    
    # 静态图可视化
    visualizer.visualize_static(
        output_file="knowledge-graph_static.png",
        max_nodes=100,
        min_degree=2
    )
    
    # 交互式图可视化
    try:
        visualizer.visualize_interactive(
            output_file="knowledge-graph_interactive.html",
            max_nodes=150,
            min_degree=1
        )
    except ImportError:
        print("⚠️ 未安装pyvis，跳过交互式可视化")
        print("安装命令: pip install pyvis")
    
    # 导出多种格式
    visualizer.export_to_formats("graph_exports")
    
    print("\n🎉 图谱可视化完成！")
    print("📋 输出文件:")
    print("  - knowledge-graph_static.png (静态图)")
    print("  - knowledge-graph_interactive.html (交互式图)")
    print("  - graph_exports/ (多种导出格式)")


if __name__ == "__main__":
    main()
