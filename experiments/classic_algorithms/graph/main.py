import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 设置中文字体，使用SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    print("图算法演示程序")
    print("-" * 30)
    
    # 创建一个简单的无向图
    G = nx.Graph()
    
    # 添加节点
    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    
    # 添加边
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)])
    
    # 绘制图
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 定义节点位置布局
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=800, edge_color='gray', font_size=15, font_weight='bold')
    plt.title("简单无向图示例", fontsize=16)
    plt.savefig("simple_graph.png")
    plt.show()
    
    # 演示图的遍历
    print("\n深度优先搜索遍历:")
    dfs_path = list(nx.dfs_preorder_nodes(G, source=1))
    print(f"DFS路径: {dfs_path}")
    
    print("\n广度优先搜索遍历:")
    bfs_path = list(nx.bfs_tree(G, source=1).nodes())
    print(f"BFS路径: {bfs_path}")
    
    # 创建一个加权图
    print("\n创建加权图...")
    G_weighted = nx.Graph()
    weighted_edges = [(1, 2, 0.6), (1, 3, 0.9), (2, 3, 0.4), 
                    (3, 4, 0.8), (4, 5, 0.1), (5, 6, 0.7), (6, 1, 0.5)]
    G_weighted.add_weighted_edges_from(weighted_edges)
    
    # 计算最短路径
    print("\n计算最短路径:")
    shortest_path = nx.shortest_path(G_weighted, source=1, target=5, weight='weight')
    shortest_path_length = nx.shortest_path_length(G_weighted, source=1, target=5, weight='weight')
    print(f"从节点1到节点5的最短路径: {shortest_path}")
    print(f"路径长度: {shortest_path_length}")
    
    # 计算最小生成树
    print("\n计算最小生成树:")
    mst = nx.minimum_spanning_tree(G_weighted, weight='weight')
    print(f"最小生成树的边: {list(mst.edges(data=True))}")
    
    print("\n程序运行完毕，请查看图像结果。")

if __name__ == "__main__":
    main()
