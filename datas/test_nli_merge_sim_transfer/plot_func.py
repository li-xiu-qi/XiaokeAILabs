# 可视化函数
from typing import Dict, List
import matplotlib
from matplotlib import pyplot as plt


def plot_results(results: List[Dict[str, any]], title: str): # 输入类型改为 List[Dict]
    """
    可视化实验结果（余弦相似度），并用颜色标示与第一个结果的差异。
    Args:
        results: 一个字典列表，每个字典包含 'text_a', 'text_b', 'similarity'。
        title: 图表的标题（字符串）。
    """
    # 设置 Matplotlib 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    
    # 从字典列表中提取用于绘图的数据
    keys = [f"{item['text_a'][:20]}... vs {item['text_b'][:20]}..." for item in results]
    vals = [item['similarity'] for item in results]

    if not vals: # 处理空结果的情况
        print(f"没有结果可供绘制: {title}")
        return

    # 创建图表，设置更合适的画布大小
    fig, ax = plt.subplots(figsize=(max(12, len(keys) * 1.5), 6)) # 根据标签数量调整宽度

    # 定义颜色
    base_color = 'lightblue'
    higher_color = 'lightcoral' # 比基准高的颜色
    lower_color = 'lightgreen' # 比基准低的颜色

    # 获取基准值（第一个结果的相似度）
    base_value = vals[0]

    # 绘制条形图，并根据与基准值的比较设置颜色
    colors = [base_color] # 第一个柱子用基准色
    for val in vals[1:]:
        if val > base_value:
            colors.append(higher_color)
        elif val < base_value:
            colors.append(lower_color)
        else:
            colors.append(base_color) # 与基准值相等也用基准色

    bars = ax.bar(keys, vals, color=colors, width=0.6)

    # 设置图表标题，并调整字体大小
    ax.set_title(title, fontsize=16, pad=20)

    # 设置Y轴范围，留出顶部空间显示数值
    min_val = min(vals) if vals else 0
    max_val = max(vals) if vals else 1
    ax.set_ylim(min(0, min_val - 0.1), max(1, max_val + 0.15)) # 动态调整Y轴范围
    # 添加Y轴标签
    ax.set_ylabel("余弦相似度", fontsize=12)

    # 设置X轴刻度标签，旋转角度以便阅读，并调整字体大小
    ax.set_xticks(np.arange(len(keys))) # 确保刻度位置正确
    ax.set_xticklabels(keys, rotation=30, ha='right', fontsize=10) # 调整旋转角度

    # 添加水平网格线，设置样式和透明度，提高可读性
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7) # 使用虚线网格

    # 在每个条形上方精确显示数值和内部显示与基准的差异
    for i, bar in enumerate(bars):
        y = bar.get_height()
        # 显示数值 (在柱子上方)
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01, f"{y:.3f}",
                 ha='center', va='bottom', fontsize=9)
        # 显示与基准的差异（第一个柱子除外，显示在柱子内部靠近顶部）
        if i > 0:
            diff = y - base_value
            diff_text = f"{diff:+.3f}" # 带符号显示差异
            diff_color = 'red' if diff < 0 else 'green' if diff > 0 else 'gray'
            # 将差异文本放在柱子内部，稍微低于顶部
            ax.text(bar.get_x() + bar.get_width() / 2, y - 0.01 if y > 0 else y + 0.02, diff_text, # 调整负值文本位置
                     ha='center', va='top' if y > 0 else 'bottom', fontsize=8, color=diff_color, fontweight='bold')

    # 添加图例说明颜色含义，并将其放置在图表外部右侧
    # 更新图例中的基准值显示
    legend_elements = [Patch(facecolor=base_color, edgecolor='gray', label=f'基准值 ({base_value:.3f})'),
                       Patch(facecolor=higher_color, edgecolor='gray', label='高于基准'),
                       Patch(facecolor=lower_color, edgecolor='gray', label='低于基准')]
    # 将图例放置在轴域 (Axes) 的右上方外部
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0.)

    # 自动调整子图参数，使之填充整个图像区域，防止标签重叠
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # 调整布局以容纳图例

    # 显示图表
    plt.show()