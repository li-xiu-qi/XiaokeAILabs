# -*- coding: utf-8 -*-
"""
基于 Bokeh 的红楼梦知识图谱交互式可视化
"""
import networkx as nx
from bokeh.io import show
from bokeh.models import (Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxZoomTool, ResetTool, PanTool, WheelZoomTool, ColumnDataSource, LabelSet)
from bokeh.plotting import from_networkx, output_file, save
from bokeh.layouts import column
from bokeh.models.widgets import Div
import pandas as pd
import os

# GEXF 文件路径
gexf_file = "hongloumeng_kg_output/knowledge_graph.gexf"

# 加载图谱
def load_graph(gexf_path):
    G = nx.read_gexf(gexf_path)
    return G

graph = load_graph(gexf_file)

# 筛选参数
def get_subgraph(G, max_nodes=150, min_degree=1):
    degrees = dict(G.degree())
    important_nodes = [n for n, d in degrees.items() if d >= min_degree]
    important_nodes = sorted(important_nodes, key=lambda x: degrees[x], reverse=True)[:max_nodes]
    return G.subgraph(important_nodes)

max_nodes = 150
min_degree = 1
subG = get_subgraph(graph, max_nodes=max_nodes, min_degree=min_degree)

# 颜色映射
def get_node_color(node_type):
    color_map = {
        'person': '#FF6B6B',
        'place': '#4ECDC4',
        'object': '#45B7D1',
        'concept': '#96CEB4',
        'other': '#FECA57',
        'unknown': '#95A5A6'
    }
    return color_map.get(node_type, '#95A5A6')

node_types = [subG.nodes[n].get('type', 'unknown') for n in subG.nodes()]
node_colors = [get_node_color(t) for t in node_types]
node_labels = list(subG.nodes())
# 计算节点半径（与度数相关，范围0.05~0.15）
node_radii = [0.05 + 0.01 * min(10, max(1, graph.degree(n))) for n in subG.nodes()]

# Bokeh plot
plot = Plot(width=1200, height=900, x_range=Range1d(-2, 2), y_range=Range1d(-2, 2),
            background_fill_color="#f8f8f8", toolbar_location="above")

plot.title.text = "红楼梦知识图谱 (Bokeh 交互式)"

# 布局
pos = nx.spring_layout(subG, k=3, iterations=50, seed=42)

# 转换为Bokeh图
graph_renderer = from_networkx(subG, pos, scale=1, center=(0, 0))
graph_renderer.node_renderer.data_source.data['node_color'] = node_colors
graph_renderer.node_renderer.data_source.data['node_label'] = node_labels
graph_renderer.node_renderer.data_source.data['node_type'] = node_types
graph_renderer.node_renderer.data_source.data['node_radius'] = node_radii
graph_renderer.node_renderer.glyph = Circle(radius='node_radius', fill_color='node_color', line_color='white', line_width=2)
graph_renderer.edge_renderer.glyph = MultiLine(line_color="#bbb", line_alpha=0.5, line_width=2)

# 悬浮提示
hover = HoverTool(tooltips=[
    ("节点", "@node_label"),
    ("类型", "@node_type"),
])
plot.add_tools(hover, TapTool(), BoxZoomTool(), ResetTool(), PanTool(), WheelZoomTool())

# 标签
x, y = zip(*[pos[n] for n in subG.nodes()])
labels = LabelSet(x='x', y='y', text='node_label',
                  source=ColumnDataSource({'x': x, 'y': y, 'node_label': node_labels}),
                  text_font_size="10pt", text_color="#222", text_align="center", text_baseline="middle")
plot.renderers.append(graph_renderer)
plot.add_layout(labels)

# 统计信息
node_types_count = pd.Series(node_types).value_counts()
edge_types = pd.Series([subG.edges[e].get('relation_type', 'unknown') for e in subG.edges()]).value_counts()

stats_html = f"""
<div style='font-size:16px;'>
<b>节点数：</b>{subG.number_of_nodes()} &nbsp; <b>边数：</b>{subG.number_of_edges()} &nbsp; <b>密度：</b>{nx.density(subG):.4f}<br>
<b>实体类型分布：</b><br>{node_types_count.to_frame().to_html(header=False)}<br>
<b>关系类型分布：</b><br>{edge_types.to_frame().to_html(header=False)}
</div>
"""

# 输出到HTML
output_file("knowledge_graph_bokeh.html", title="红楼梦知识图谱 (Bokeh)")
layout = column(Div(text=stats_html), plot)
save(layout)

print("✅ Bokeh 交互式知识图谱已导出为 knowledge_graph_bokeh.html，可用浏览器打开查看！")
