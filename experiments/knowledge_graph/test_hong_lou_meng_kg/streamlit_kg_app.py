import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import defaultdict
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="红楼梦知识图谱探索器",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def load_graph(gexf_file):
    """加载图谱数据"""
    return nx.read_gexf(gexf_file)

def main():
    st.title("📚 红楼梦知识图谱交互式探索器")
    
    # 侧边栏控制
    st.sidebar.title("📊 控制面板")
    
    # 加载图谱
    gexf_file = "hongloumeng_kg_output/knowledge_graph.gexf"
    if not os.path.exists(gexf_file):
        st.error(f"找不到图谱文件: {gexf_file}")
        return
    
    graph = load_graph(gexf_file)
    
    # 基本信息
    st.sidebar.write(f"**节点数**: {graph.number_of_nodes()}")
    st.sidebar.write(f"**边数**: {graph.number_of_edges()}")
    st.sidebar.write(f"**密度**: {nx.density(graph):.4f}")
    
    # 筛选控制
    max_nodes = st.sidebar.slider("最大节点数", 50, 500, 200)
    min_degree = st.sidebar.slider("最小度数", 1, 20, 2)
    
    # 节点类型筛选
    node_types = set()
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        node_types.add(node_type)
    
    selected_types = st.sidebar.multiselect(
        "选择节点类型",
        options=list(node_types),
        default=list(node_types)
    )
    
    # 搜索功能
    search_term = st.sidebar.text_input("搜索节点")
    
    # 主要内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 网络图", "📊 统计分析", "🔍 节点详情", "💾 数据导出"])
    
    with tab1:
        st.subheader("交互式网络图")
        
        # 筛选图谱
        filtered_graph = filter_graph(graph, max_nodes, min_degree, selected_types, search_term)
        
        if filtered_graph.number_of_nodes() > 0:
            # 创建可视化
            fig = create_network_plot(filtered_graph)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("没有符合条件的节点")
    
    with tab2:
        st.subheader("网络统计分析")
        show_statistics(graph)
    
    with tab3:
        st.subheader("节点详细信息")
        show_node_details(graph, search_term)
    
    with tab4:
        st.subheader("数据导出")
        show_export_options(graph)

def filter_graph(graph, max_nodes, min_degree, selected_types, search_term):
    """筛选图谱"""
    # 度数筛选
    degrees = dict(graph.degree())
    filtered_nodes = [node for node, degree in degrees.items() if degree >= min_degree]
    
    # 类型筛选
    if selected_types:
        filtered_nodes = [
            node for node in filtered_nodes
            if graph.nodes[node].get('type', 'unknown') in selected_types
        ]
    
    # 搜索筛选
    if search_term:
        filtered_nodes = [
            node for node in filtered_nodes
            if search_term.lower() in node.lower()
        ]
    
    # 限制数量
    filtered_nodes = sorted(filtered_nodes, key=lambda x: degrees[x], reverse=True)[:max_nodes]
    
    return graph.subgraph(filtered_nodes)

def create_network_plot(graph):
    """创建网络图"""
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # 边数据
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # 节点数据
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []
    node_sizes = []
    
    color_map = {
        'person': '#FF6B6B',
        'place': '#4ECDC4',
        'object': '#45B7D1',
        'concept': '#96CEB4',
        'other': '#FECA57',
        'unknown': '#95A5A6'
    }
    
    degrees = dict(graph.degree())
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        node_data = graph.nodes[node]
        node_type = node_data.get('type', 'unknown')
        degree = degrees[node]
        
        info = f"节点: {node}<br>类型: {node_type}<br>度数: {degree}"
        if node_data.get('frequency'):
            info += f"<br>频率: {node_data['frequency']}"
        node_info.append(info)
        
        node_colors.append(color_map.get(node_type, '#95A5A6'))
        node_sizes.append(max(10, min(50, degree * 2)))
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           text=node_text,
                           textposition="middle center",
                           hoverinfo='text',
                           hovertext=node_info,
                           marker=dict(size=node_sizes,
                                     color=node_colors,
                                     line=dict(width=2, color="white")))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor="white"))
    
    return fig

def show_statistics(graph):
    """显示统计信息"""
    col1, col2 = st.columns(2)
    
    with col1:
        # 度分布
        degrees = [d for n, d in graph.degree()]
        fig_deg = px.histogram(x=degrees, nbins=30, title="度分布")
        st.plotly_chart(fig_deg, use_container_width=True)
        
        # 节点类型分布
        node_types = defaultdict(int)
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            node_types[node_type] += 1
        
        fig_type = px.bar(x=list(node_types.keys()), y=list(node_types.values()),
                         title="节点类型分布")
        st.plotly_chart(fig_type, use_container_width=True)
    
    with col2:
        # 中心性分析
        try:
            centrality = nx.betweenness_centrality(graph)
            degree_centrality = nx.degree_centrality(graph)
            
            df_cent = pd.DataFrame({
                'node': list(centrality.keys()),
                'betweenness': list(centrality.values()),
                'degree_centrality': list(degree_centrality.values())
            })
            
            fig_cent = px.scatter(df_cent, x='degree_centrality', y='betweenness',
                                 hover_name='node', title="中心性分析")
            st.plotly_chart(fig_cent, use_container_width=True)
        except Exception as e:
            st.error(f"中心性计算失败: {e}")

def show_node_details(graph, search_term):
    """显示节点详情"""
    if search_term:
        matching_nodes = [node for node in graph.nodes() 
                         if search_term.lower() in node.lower()]
        
        if matching_nodes:
            for node in matching_nodes[:10]:  # 限制显示数量
                with st.expander(f"📍 {node}"):
                    node_data = graph.nodes[node]
                    st.write(f"**类型**: {node_data.get('type', '未知')}")
                    st.write(f"**度数**: {graph.degree(node)}")
                    if node_data.get('frequency'):
                        st.write(f"**频率**: {node_data['frequency']}")
                    
                    # 相邻节点
                    neighbors = list(graph.neighbors(node))
                    if neighbors:
                        st.write(f"**相邻节点** ({len(neighbors)}个):")
                        st.write(", ".join(neighbors[:20]))  # 限制显示数量
        else:
            st.write("未找到匹配的节点")

def show_export_options(graph):
    """显示导出选项"""
    st.write("### 🔽 数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("导出节点信息为CSV"):
            nodes_data = []
            for node in graph.nodes():
                node_data = graph.nodes[node]
                nodes_data.append({
                    'name': node,
                    'type': node_data.get('type', ''),
                    'frequency': node_data.get('frequency', 0),
                    'degree': graph.degree(node)
                })
            
            df_nodes = pd.DataFrame(nodes_data)
            csv = df_nodes.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载节点CSV文件",
                data=csv,
                file_name="knowledge_graph_nodes.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("导出边信息为CSV"):
            edges_data = []
            for edge in graph.edges():
                source, target = edge
                edge_data = graph.edges[edge]
                edges_data.append({
                    'source': source,
                    'target': target,
                    'relation_type': edge_data.get('relation_type', ''),
                    'confidence': edge_data.get('confidence', 0.5)
                })
            
            df_edges = pd.DataFrame(edges_data)
            csv = df_edges.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载边CSV文件",
                data=csv,
                file_name="knowledge_graph_edges.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
