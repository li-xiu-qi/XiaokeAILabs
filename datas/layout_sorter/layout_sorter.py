#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image


class LayoutSorter:
    """版面布局元素自适应排序处理器"""

    def __init__(
        self,
        threshold_column_assign: float = 0.9,
        threshold_cross: float = 0.3,
        max_columns: int = 4,
    ):
        """
        初始化排序器

        Args:
            threshold_column_assign: 判定元素属于某栏的阈值(0-1)，默认0.9
            threshold_cross: 判定元素跨栏的阈值(0-1)，默认0.3
            max_columns: 考虑的最大栏数，默认4
        """
        self.threshold_column_assign = threshold_column_assign
        self.threshold_cross = threshold_cross
        self.max_columns = max_columns

    def get_image_width(self, image_path: str) -> int:
        """获取图片宽度"""
        try:
            with Image.open(image_path) as img:
                return img.width
        except Exception as e:
            raise ValueError(f"无法读取图片宽度，错误：{e}")

    def sort_layout(
        self,
        layout_result: Union[str, dict],
        image_path: str = None,
        page_width: float = None,
    ) -> Dict:
        """对版面检测结果进行排序，自动检测栏数"""
        # 加载检测结果
        if isinstance(layout_result, str):
            with open(layout_result, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = layout_result

        # 确定页面宽度
        if image_path is not None:
            actual_page_width = self.get_image_width(image_path)
        elif page_width is not None:
            actual_page_width = page_width
        else:
            raise ValueError("必须提供image_path或page_width参数")

        # 获取元素列表
        elements = result.get("boxes", [])

        # 排序元素（采用新的混合布局处理方法）
        sorted_elements = self._sort_elements_adaptive(elements, actual_page_width)

        # 将排序后的元素更新回原始结果
        result["boxes"] = sorted_elements
        return result

    def _detect_column_count(self, elements: List[Dict], page_width: float) -> int:
        """
        基于内容特征自动检测版面栏数

        Args:
            elements: 元素列表
            page_width: 页面宽度

        Returns:
            检测到的栏数(1-4)
        """
        valid_elements = [
            elem
            for elem in elements
            if "coordinate" in elem and len(elem["coordinate"]) == 4
        ]

        if not valid_elements:
            return 1  # 默认单栏

        # 提取元素特征
        element_features = []
        title_widths = []
        text_widths = []
        center_xs = []

        for elem in valid_elements:
            x1, y1, x2, y2 = elem["coordinate"]
            width = x2 - x1
            center_x = (x1 + x2) / 2
            width_ratio = width / page_width

            center_xs.append(center_x)

            # 区分标题和正文
            elem_type = elem.get("type", "").lower()
            is_title = any(t in elem_type for t in ["title", "heading", "标题"])
            is_figure = any(
                t in elem_type for t in ["figure", "image", "picture", "图"]
            )

            # 收集宽度信息
            if is_title:
                title_widths.append(width_ratio)
            elif not is_figure:  # 排除图片
                text_widths.append(width_ratio)
                element_features.append(
                    {
                        "width_ratio": width_ratio,
                        "center_x_ratio": center_x / page_width,
                    }
                )

        # 方法1: 基于文本宽度分布
        column_count_by_width = self._estimate_columns_by_width(text_widths, page_width)

        # 方法2: 基于水平位置聚类
        column_count_by_cluster = self._estimate_columns_by_clustering(
            center_xs, page_width
        )

        # 方法3: 分析标题长度
        column_count_by_titles = self._estimate_columns_by_titles(title_widths)

        # 综合判断最可能的栏数
        column_counts = [
            column_count_by_width,
            column_count_by_cluster,
            column_count_by_titles,
        ]
        # 使用众数作为最终结果
        counts = {}
        for count in column_counts:
            counts[count] = counts.get(count, 0) + 1

        # 找出出现次数最多的栏数
        most_common_count = max(counts.items(), key=lambda x: x[1])[0]

        return most_common_count

    def _estimate_columns_by_width(
        self, text_widths: List[float], page_width: float
    ) -> int:
        """基于文本宽度分布估计栏数"""
        if not text_widths:
            return 1

        # 计算文本宽度的分布
        width_ratios = [w for w in text_widths]

        if not width_ratios:
            return 1

        # 计算平均宽度比例
        avg_width_ratio = sum(width_ratios) / len(width_ratios)

        # 根据平均宽度比例估计栏数
        if avg_width_ratio > 0.85:  # 大部分元素接近页宽
            return 1
        elif avg_width_ratio > 0.55:  # 元素宽度约为页宽一半多一点
            return 2
        elif avg_width_ratio > 0.35:  # 元素宽度约为页宽三分之一多一点
            return 3
        else:  # 元素较窄
            return 4

    def _estimate_columns_by_clustering(
        self, center_xs: List[float], page_width: float
    ) -> int:
        """使用K-means聚类分析水平位置分布"""
        if len(center_xs) < 4:
            return 1

        # 归一化中心点位置
        normalized_xs = np.array([[x / page_width] for x in center_xs])

        best_k = 1
        best_score = float("inf")

        # 尝试不同的k值(1-4栏)
        for k in range(1, min(5, len(center_xs))):
            if len(normalized_xs) < k:
                continue

            kmeans = KMeans(n_clusters=k, random_state=0).fit(normalized_xs)
            score = kmeans.inertia_  # 聚类内平方和

            # 使用肘部法则评估最佳k值
            if k > 1 and best_score / score < 1.5:  # 如果改进不明显
                break

            best_score = score
            best_k = k

        return best_k

    def _estimate_columns_by_titles(self, title_widths: List[float]) -> int:
        """基于标题宽度估计栏数"""
        if not title_widths:
            return 2  # 默认值

        # 计算标题宽度平均值
        avg_title_width = sum(title_widths) / len(title_widths)

        # 较宽的标题暗示栏数较少
        if avg_title_width > 0.85:
            return 1
        elif avg_title_width > 0.65:
            return 2
        elif avg_title_width > 0.4:
            return 3
        else:
            return 4

    def _identify_cross_column_elements(self, elements: List[Dict], page_width: float) -> List[Dict]:
        """
        识别跨栏元素
        
        Args:
            elements: 元素列表
            page_width: 页面宽度
            
        Returns:
            处理后的元素列表，添加了is_separator标记
        """
        for elem in elements:
            x1, y1, x2, y2 = elem["coordinate"]
            width = x2 - x1
            width_ratio = width / page_width
            
            # 判断元素类型
            elem_type = elem.get("type", "").lower()
            is_figure = any(t in elem_type for t in ["figure", "image", "picture", "图"])
            is_title = any(t in elem_type for t in ["title", "heading", "标题"])
            
            # 识别宽元素和图片作为可能的分隔符
            is_wide = width_ratio > 0.6  # 宽度超过页面60%
            
            # 图片或宽元素可能是跨栏的分隔符
            elem["is_separator"] = (is_figure and width_ratio > 0.4) or (is_wide and (is_figure or is_title))
            
        return elements

    def _divide_into_vertical_regions(self, elements: List[Dict]) -> List[List[Dict]]:
        """
        将页面元素划分为垂直区域
        
        Args:
            elements: 按y坐标排序的元素列表
            
        Returns:
            区域列表，每个区域是一个元素列表
        """
        regions = []
        current_region = []
        
        for elem in elements:
            if elem.get("is_separator", False):
                # 如果当前区域有内容，保存它
                if current_region:
                    regions.append(current_region)
                    current_region = []
                
                # 分隔符单独作为一个区域
                regions.append([elem])
            else:
                current_region.append(elem)
        
        # 添加最后一个区域
        if current_region:
            regions.append(current_region)
            
        return regions

    def _detect_region_column_count(self, region: List[Dict], page_width: float) -> int:
        """
        检测区域的栏数
        
        Args:
            region: 区域内的元素列表
            page_width: 页面宽度
            
        Returns:
            区域栏数
        """
        # 排除分隔符区域
        if len(region) == 1 and region[0].get("is_separator", False):
            return 1
            
        return self._detect_column_count(region, page_width)

    def _sort_region_elements(self, region: List[Dict], page_width: float, region_columns: int) -> List[Dict]:
        """
        对区域内元素进行排序
        
        Args:
            region: 区域内的元素列表
            page_width: 页面宽度
            region_columns: 区域栏数
            
        Returns:
            排序后的元素列表
        """
        # 如果是单个分隔符区域，直接返回
        if len(region) == 1 and region[0].get("is_separator", False):
            return region
            
        # 计算列宽
        column_width = page_width / region_columns
        
        # 分配元素到相应的栏
        columns = [[] for _ in range(region_columns)]
        
        for elem in region:
            x1, _, x2, _ = elem["coordinate"]
            elem_width = x2 - x1
            center_x = (x1 + x2) / 2
            
            # 计算元素与各栏的重叠情况
            overlaps = []
            for col in range(region_columns):
                col_start = col * column_width
                col_end = (col + 1) * column_width
                
                overlap = max(0, min(x2, col_end) - max(x1, col_start))
                overlaps.append(overlap)
                
            # 找出重叠最多的栏
            best_col = overlaps.index(max(overlaps))
            columns[best_col].append(elem)
            
        # 按垂直位置排序每一栏内的元素
        for col in columns:
            col.sort(key=lambda e: e["coordinate"][1])
            
        # 按栏合并元素
        sorted_elements = []
        for col in columns:
            sorted_elements.extend(col)
            
        return sorted_elements

    def _sort_elements_adaptive(self, elements: List[Dict], page_width: float) -> List[Dict]:
        """
        自适应混合布局排序算法

        Args:
            elements: 元素列表
            page_width: 页面宽度

        Returns:
            排序后的元素列表
        """
        # 筛选有效元素
        valid_elements = [
            elem
            for elem in elements
            if "coordinate" in elem and len(elem["coordinate"]) == 4
        ]
        
        if not valid_elements:
            return []
        
        # 1. 按Y坐标排序所有元素
        valid_elements.sort(key=lambda e: e["coordinate"][1])
        
        # 2. 识别跨栏元素
        valid_elements = self._identify_cross_column_elements(valid_elements, page_width)
        
        # 3. 划分垂直区域
        regions = self._divide_into_vertical_regions(valid_elements)
        
        # 4. 对每个区域单独处理
        result = []
        prev_region_columns = None  # 记录前一个区域的栏数
        
        for i, region in enumerate(regions):
            # 单个分隔符区域直接添加
            if len(region) == 1 and region[0].get("is_separator", False):
                result.append(region[0])
                continue
                
            # 检测区域栏数
            region_columns = self._detect_region_column_count(region, page_width)
            
            # 对小区域使用连续性原则：如果区域内元素少于5个，且不是第一个区域，
            # 可能是因为跨栏元素导致的错误分割，尝试使用前一个区域的栏数
            if len(region) < 5 and prev_region_columns is not None:
                # 如果是文章末尾小段落，保持前一区域的栏数
                region_columns = prev_region_columns
                
            # 排序当前区域的元素
            sorted_region = self._sort_region_elements(region, page_width, region_columns)
            result.extend(sorted_region)
            
            # 更新前一个区域栏数（仅当区域内元素足够多时）
            if len(region) >= 5:
                prev_region_columns = region_columns
                
            print(f"区域 {i+1}: 检测到栏数 {region_columns}")
        
        # 清除临时标记
        for elem in result:
            if "is_separator" in elem:
                del elem["is_separator"]
                
        return result

    def _sort_elements(
        self, elements: List[Dict], page_width: float, num_columns: int = None
    ) -> List[Dict]:
        """
        传统的固定栏数排序方法 (向后兼容)
        
        建议使用新的_sort_elements_adaptive方法来处理混合布局
        """
        # 使用自适应混合布局排序替代
        return self._sort_elements_adaptive(elements, page_width)


if __name__ == "__main__":
    # 使用示例
    sorter = LayoutSorter()
    # sorted_result = sorter.sort_layout(layout_result, image_path=image_path)