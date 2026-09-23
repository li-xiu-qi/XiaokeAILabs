#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Dict, Optional, Union
from PIL import Image

class LayoutSorter:
    """版面布局元素排序处理器"""
    
    def __init__(self, 
                 threshold_left_right: float = 0.9,
                 threshold_cross: float = 0.3):
        """
        初始化排序器
        
        Args:
            threshold_left_right: 判定元素属于左/右栏的阈值(0-1)，默认0.9
            threshold_cross: 判定元素跨栏的阈值(0-1)，默认0.3
        """
        self.threshold_left_right = threshold_left_right
        self.threshold_cross = threshold_cross
    
    def get_image_width(self, image_path: str) -> int:
        """
        获取图片宽度
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片宽度
        """
        try:
            with Image.open(image_path) as img:
                return img.width
        except Exception as e:
            raise ValueError(f"无法读取图片宽度，错误：{e}")
    
    def sort_layout(self, layout_result: Union[str, dict], image_path: str = None, page_width: float = None) -> Dict:
        """
        对版面检测结果进行排序
        
        Args:
            layout_result: JSON文件路径或者包含检测结果的字典
            image_path: 图片路径，如果提供则自动获取页面宽度
            page_width: 页面宽度，如果不提供图片路径则必须提供此参数
            
        Returns:
            包含排序后元素的完整layout_result对象
        """
        # 加载检测结果
        if isinstance(layout_result, str):
            with open(layout_result, 'r', encoding='utf-8') as f:
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
        elements = result.get('boxes', [])
        sorted_elements = self._sort_elements(elements, actual_page_width)
        
        # 将排序后的元素更新回原始结果
        result['boxes'] = sorted_elements
        return result
    
    
    def _sort_elements(self, elements: List[Dict], page_width: float) -> List[Dict]:
        """
        对元素按照左右栏进行排序
        
        Args:
            elements: 元素列表
            page_width: 页面宽度,必须提供
            
        Returns:
            排序后的元素列表
        """
        # 筛选有效元素
        valid_elements = [
            elem for elem in elements 
            if "coordinate" in elem and len(elem["coordinate"]) == 4
        ]
        
        if not valid_elements:
            return []
            
        page_center_x = page_width / 2
        left_column = []
        right_column = []
        
        # 分配元素到左右栏
        for elem in valid_elements:
            x1, _, x2, _ = elem["coordinate"]
            elem_width = x2 - x1
            
            # 计算左右覆盖比例
            left_part = max(0, min(x2, page_center_x) - x1)
            right_part = max(0, x2 - max(x1, page_center_x))
            
            left_ratio = left_part / elem_width if elem_width > 0 else 0
            right_ratio = right_part / elem_width if elem_width > 0 else 0
            
            # 根据覆盖比例分配
            if left_ratio >= self.threshold_left_right:
                left_column.append(elem)
            elif right_ratio >= self.threshold_left_right:
                right_column.append(elem)
            elif left_ratio > self.threshold_cross and right_ratio > self.threshold_cross:
                left_column.append(elem)
            else:
                elem_center_x = (x1 + x2) / 2
                if elem_center_x <= page_center_x:
                    left_column.append(elem)
                else:
                    right_column.append(elem)
                    
        # 按垂直位置排序
        left_column.sort(key=lambda e: e["coordinate"][1])
        right_column.sort(key=lambda e: e["coordinate"][1])
        
        return left_column + right_column
    

if __name__ == "__main__":
    # 使用示例
    sorter = LayoutSorter()
    # sorted_elements = sorter.sort_layout(layout_result, page_width)
