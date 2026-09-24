# -*- coding: utf-8 -*-
"""
基于 Dash + dash-cytoscape 的红楼梦知识图谱交互式可视化
"""
import os
import networkx as nx
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import pandas as pd
import json

# GEXF 文件路径
gexf_file = "hongloumeng_kg_output/knowledge_graph.gexf"

# 加载图谱
def load_graph(gexf_path):
    G = nx.read_gexf(gexf_path)
    return G

def nx_to_cytoscape(G, max_nodes=200, min_degree=1):
    degrees = dict(G.degree())
    important_nodes = [n for n, d in degrees.items() if d >= min_degree]
    important_nodes = sorted(important_nodes, key=lambda x: degrees[x], reverse=True)[:max_nodes]
    subG = G.subgraph(important_nodes)
    
    # 节点类型颜色映射
    color_map = {
        'person': '#FF6B6B',
        'place': '#4ECDC4',
        'object': '#45B7D1',
        'concept': '#96CEB4',
        'other': '#FECA57',
        'unknown': '#95A5A6'
    }
    
    nodes = []
    for n in subG.nodes():
        data = subG.nodes[n]
        node_type = data.get('type', 'unknown')
        color = color_map.get(node_type, '#95A5A6')
        nodes.append({
            'data': {
                'id': n,
                'label': n,
                'type': node_type,
                'frequency': data.get('frequency', ''),
                'desc': data.get('描述', ''),
                'feature': data.get('特征', '')
            },
            'classes': node_type,
            'style': {'background-color': color, 'width': 30, 'height': 30}
        })
    
    edges = []
    for u, v, d in subG.edges(data=True):
        edges.append({
            'data': {
                'source': u,
                'target': v,
                'relation_type': d.get('relation_type', ''),
                'confidence': d.get('confidence', ''),
                'context': d.get('context', '')
            },
            'classes': 'edge'
        })
    return nodes + edges

# Dash App 初始化
app = dash.Dash(__name__)
app.title = "红楼梦知识图谱交互式可视化"

# 加载图谱数据
graph = load_graph(gexf_file)
cyto_data = nx_to_cytoscape(graph, max_nodes=200, min_degree=1)

# 统计信息
node_types = pd.Series([graph.nodes[n].get('type', 'unknown') for n in graph.nodes()]).value_counts().to_dict()
edge_types = pd.Series([graph.edges[e].get('relation_type', 'unknown') for e in graph.edges()]).value_counts().to_dict()

# 布局选项
layouts = [
    {'label': '力导向(spring)', 'value': 'cose'},
    {'label': '圆形(circle)', 'value': 'circle'},
    {'label': '网格(grid)', 'value': 'grid'},
    {'label': '同心(concentric)', 'value': 'concentric'},
    {'label': '随机(random)', 'value': 'random'}
]

app.layout = html.Div([
    html.H1("红楼梦知识图谱交互式可视化", style={'textAlign': 'center'}),
    html.Div([
        html.Label("布局方式："),
        dcc.Dropdown(
            id='layout-dropdown',
            options=layouts,
            value='cose',
            clearable=False,
            style={'width': '200px', 'display': 'inline-block'}
        ),
        html.Label("最小度数筛选：", style={'marginLeft': '30px'}),
        html.Div([
            dcc.Slider(
                id='min-degree-slider',
                min=1, max=10, step=1, value=1,
                marks={i: str(i) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True},
            )
        ], style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Label("最大节点数：", style={'marginLeft': '30px'}),
        dcc.Input(
            id='max-nodes-input',
            type='number',
            min=10, max=500, step=10, value=200,
            style={'width': '100px', 'display': 'inline-block'}
        ),
    ], style={'margin': '20px'}),
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=cyto_data,
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '800px', 'background': '#f8f8f8'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'font-size': '12px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': '#222',
                    'background-opacity': 0.9,
                    'border-width': 2,
                    'border-color': '#fff',
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'line-color': '#bbb',
                    'width': 2,
                    'target-arrow-color': '#bbb',
                    'arrow-scale': 1.2,
                }
            },
            {
                'selector': '.person',
                'style': {'background-color': '#FF6B6B'}
            },
            {
                'selector': '.place',
                'style': {'background-color': '#4ECDC4'}
            },
            {
                'selector': '.object',
                'style': {'background-color': '#45B7D1'}
            },
            {
                'selector': '.concept',
                'style': {'background-color': '#96CEB4'}
            },
            {
                'selector': '.other',
                'style': {'background-color': '#FECA57'}
            },
            {
                'selector': '.unknown',
                'style': {'background-color': '#95A5A6'}
            },
        ]
    ),
    html.Div(id='node-info', style={'margin': '20px', 'fontSize': '16px', 'color': '#333'}),
    html.Hr(),
    html.Div([
        html.H4("实体类型分布"),
        html.Pre(json.dumps(node_types, ensure_ascii=False, indent=2)),
        html.H4("关系类型分布"),
        html.Pre(json.dumps(edge_types, ensure_ascii=False, indent=2)),
    ], style={'margin': '20px', 'background': '#f0f0f0', 'padding': '10px', 'borderRadius': '8px'})
])

@app.callback(
    Output('cytoscape-graph', 'elements'),
    [Input('min-degree-slider', 'value'),
     Input('max-nodes-input', 'value')]
)
def update_graph(min_degree, max_nodes):
    return nx_to_cytoscape(graph, max_nodes=max_nodes, min_degree=min_degree)

@app.callback(
    Output('cytoscape-graph', 'layout'),
    [Input('layout-dropdown', 'value')]
)
def update_layout(layout_name):
    return {'name': layout_name}

@app.callback(
    Output('node-info', 'children'),
    [Input('cytoscape-graph', 'tapNodeData')]
)
def display_node_info(data):
    if not data:
        return "点击节点查看详细信息"
    info = [f"节点: {data.get('label', '')}", f"类型: {data.get('type', '')}"]
    if data.get('frequency'):
        info.append(f"频率: {data['frequency']}")
    if data.get('desc'):
        info.append(f"描述: {data['desc']}")
    if data.get('feature'):
        info.append(f"特征: {data['feature']}")
    return html.Div([html.P(line) for line in info])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
