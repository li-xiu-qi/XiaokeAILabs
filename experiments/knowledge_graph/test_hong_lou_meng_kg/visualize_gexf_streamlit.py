# -*- coding: utf-8 -*-
"""
基于 Streamlit 的红楼梦知识图谱交互式可视化
"""
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import tempfile
import os

# GEXF 文件路径
gexf_file = "hongloumeng_kg_output/knowledge_graph.gexf"

st.set_page_config(page_title="红楼梦知识图谱可视化", layout="wide")
st.title("红楼梦知识图谱可视化 (Streamlit 版)")

# 加载图谱
def load_graph(gexf_path):
    G = nx.read_gexf(gexf_path)
    return G

graph = load_graph(gexf_file)

# 侧边栏参数
st.sidebar.header("可视化参数")
max_nodes = st.sidebar.slider("最大节点数", 10, 500, 150, 10)
min_degree = st.sidebar.slider("最小度数", 1, 10, 1, 1)
show_static = st.sidebar.checkbox("显示静态图 (matplotlib)", value=True)
show_interactive = st.sidebar.checkbox("显示交互图 (pyvis)", value=True)

# 节点筛选
degrees = dict(graph.degree())
important_nodes = [n for n, d in degrees.items() if d >= min_degree]
important_nodes = sorted(important_nodes, key=lambda x: degrees[x], reverse=True)[:max_nodes]
subG = graph.subgraph(important_nodes)

# 统计信息
st.subheader("图谱统计信息")
st.write(f"节点数: {subG.number_of_nodes()}  |  边数: {subG.number_of_edges()}  |  密度: {nx.density(subG):.4f}")

node_types = pd.Series([subG.nodes[n].get('type', 'unknown') for n in subG.nodes()]).value_counts()
st.write("**实体类型分布：**")
st.dataframe(node_types)

edge_types = pd.Series([subG.edges[e].get('relation_type', 'unknown') for e in subG.edges()]).value_counts()
st.write("**关系类型分布：**")
st.dataframe(edge_types)

# 静态图
if show_static:
    st.subheader("静态知识图谱 (matplotlib)")
    plt.figure(figsize=(16, 12))
    try:
        pos = nx.spring_layout(subG, k=3, iterations=50, seed=42)
    except:
        pos = nx.random_layout(subG, seed=42)
    color_map = {
        'person': '#FF6B6B',
        'place': '#4ECDC4',
        'object': '#45B7D1',
        'concept': '#96CEB4',
        'other': '#FECA57',
        'unknown': '#95A5A6'
    }
    node_colors = [color_map.get(subG.nodes[n].get('type', 'unknown'), '#95A5A6') for n in subG.nodes()]
    node_sizes = [max(100, min(3000, degrees[n] * 100)) for n in subG.nodes()]
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(subG, pos, edge_color='gray', alpha=0.5, width=0.5, arrows=True, arrowsize=10)
    top_nodes = sorted(subG.nodes(), key=lambda x: degrees[x], reverse=True)[:20]
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(subG, pos, labels, font_size=8, font_weight='bold', font_family='sans-serif')
    plt.axis('off')
    st.pyplot(plt)

# 交互式图
if show_interactive:
    st.subheader("交互式知识图谱 (pyvis)")
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    color_map = {
        'person': '#FF6B6B',
        'place': '#4ECDC4',
        'object': '#45B7D1',
        'concept': '#96CEB4',
        'other': '#FECA57',
        'unknown': '#95A5A6'
    }
    for n in subG.nodes():
        node_type = subG.nodes[n].get('type', 'unknown')
        color = color_map.get(node_type, '#95A5A6')
        degree = degrees[n]
        size = max(10, min(50, degree * 3))
        title = f"节点: {n}<br>类型: {node_type}<br>度数: {degree}"
        if subG.nodes[n].get('frequency'):
            title += f"<br>频率: {subG.nodes[n]['frequency']}"
        net.add_node(n, label=n, color=color, size=size, title=title)
    for u, v, d in subG.edges(data=True):
        relation_type = d.get('relation_type', '未知关系')
        confidence = d.get('confidence', 0.5)
        context = d.get('context', '')
        title = f"关系: {relation_type}<br>置信度: {confidence}" + (f"<br>上下文: {context[:100]}..." if context else "")
        net.add_edge(u, v, title=title, width=max(1, float(confidence) * 3 if confidence else 1))
    # 临时文件保存 html
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_file.flush()
        st.components.v1.html(open(tmp_file.name, encoding="utf-8").read(), height=700, scrolling=True)
    os.unlink(tmp_file.name)

st.info("可通过侧边栏调整参数，体验不同的知识图谱视图。点击交互图节点可查看详细信息。")
