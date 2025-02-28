#!/usr/bin/env python  
# -*- coding: utf-8 -*-
# author：筱可
# 2023-07-01
"""
#### 使用说明：
将PDF文档转换为Markdown格式，支持OCR和文本模式。

#### 主要功能：
1. 将PDF文件转换为Markdown格式
2. 提取PDF中的图片并保存
3. 根据PDF类型自动选择处理方法（OCR或文本模式）

#### 参数说明：
mineru_pdf2md函数：
    - pdf_file_path: PDF文件的绝对路径
    - md_output_path: 输出Markdown文件的目录绝对路径
    - return_path: 是否返回生成的Markdown文件路径，默认为False
    - 返回值: 如果return_path=True，返回生成的Markdown文件路径；否则返回Markdown内容

#### 注意事项：
- 依赖magic_pdf库进行PDF解析和转换
- 输入的PDF文件路径和输出路径必须为绝对路径
- 自动创建输出目录和图片目录
"""

import os
import uuid

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def mineru_pdf2md(pdf_file_path, md_output_path, return_path=False):
    """
    将PDF文件转换为Markdown格式
    
    参数:
        pdf_file_path: PDF文件的绝对路径
        md_output_path: 输出Markdown文件的目录绝对路径
        return_path: 是否返回生成的Markdown文件路径，默认为False
        
    返回:
        如果return_path=True，返回生成的Markdown文件路径；否则返回Markdown内容
        
    异常:
        ValueError: 当路径不是绝对路径时
        FileNotFoundError: 当PDF文件不存在时
    """
    # 检查是否为绝对路径
    if not os.path.isabs(pdf_file_path):
        raise ValueError(f"PDF文件路径必须是绝对路径: {pdf_file_path}")
    if not os.path.isabs(md_output_path):
        raise ValueError(f"Markdown输出路径必须是绝对路径: {md_output_path}")

    # 检查PDF文件是否存在
    if not os.path.isfile(pdf_file_path):
        raise FileNotFoundError(f"PDF文件未找到: {pdf_file_path}")

    # 获取不带后缀的文件名
    name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]

    # 使用传入的绝对路径
    md_dir = md_output_path
    image_dir = os.path.join(md_output_path, "images")
    md_suff = f"{name_without_suff}.md"
    md_path = os.path.join(md_dir, md_suff.replace("-", ""))
    print(md_path)

    # 创建目录
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)

    # 初始化写入器
    image_writer = FileBasedDataWriter(image_dir)
    md_writer = FileBasedDataWriter(md_dir)

    # 读取PDF文件
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_path)

    # 创建数据集
    ds = PymuDocDataset(pdf_bytes)

    # 根据PDF类型进行处理
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # OCR模式处理
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        # 文本模式处理
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    
    # 获取图片目录的相对路径
    show_image_dir = os.path.basename(image_dir)
    
    # 获取Markdown内容
    md_content = pipe_result.get_markdown(show_image_dir)
    
    # 如果不需要返回路径，直接返回Markdown内容
    if not return_path:
        return md_content
    
    # 保存Markdown内容到文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return md_path
