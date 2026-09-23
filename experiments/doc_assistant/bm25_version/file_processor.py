#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：筱可
# 2024-05-16 10:22:35
"""
#### 使用说明：
该模块提供了一系列函数，用于处理上传的文件，包括创建临时文件、提取PDF文件内容（支持多种方法），以及提取文本文件内容。

#### 主要功能：
1.  创建临时文件，用于存储上传的文件流。
2.  使用PyMuPDF提取PDF文件内容。
3.  使用pymupdf4llm提取PDF文件内容。
4.  使用mineru提取PDF文件内容。
5.  提取上传文件的内容，并返回内容和元数据。

#### 参数说明：
**create_temp_file函数参数说明：**
*   `file_stream` (文件流对象):  上传的文件流对象。
*   `original_filename` (str):  原始文件名。
*   **返回值**:  临时文件路径 (str) | None: 临时文件路径，如果创建失败则返回 None。

**process_pdf_with_pymupdf函数参数说明：**
*   `file_path` (str):  文件路径。
*   **返回值**:  提取的文本内容 (str) | None: 提取的文本内容, 如果处理失败则返回 None。

**process_pdf_with_pymupdf4llm函数参数说明：**
*   `file_path` (str):  文件路径。
*   **返回值**:  提取的文本内容 (str) | None: 提取的文本内容, 如果处理失败则返回 None。

**process_pdf_with_mineru函数参数说明：**
*   `file_path` (str):  文件路径。
*   **返回值**:  提取的文本内容 (str) | None: 提取的文本内容, 如果处理失败则返回 None。

**extract_uploaded_file_content函数参数说明：**
*   `uploaded_file` (文件对象):  上传的文件对象，支持 PDF 和 TXT 格式。
*   `pdf_process_method` (str):  PDF处理方法，可选 'pymupdf', 'pymupdf4llm', 'mineru'，默认为 'pymupdf'。
*   **返回值**:  文件内容的字符串 (str) | None, 文件元数据 (dict) | None: 文件内容的字符串和文件元数据，如果处理失败则返回 (None, None)。

#### 注意事项：
*   需要安装 fitz (PyMuPDF), streamlit, pymupdf4llm, 和 mineru_convert 依赖。
*   确保有创建临时文件的权限。
"""
import fitz  # PyMuPDF
import streamlit as st
import os
import pymupdf4llm
import uuid
from mineru_convert import mineru_pdf2md


def create_temp_file(file_stream, original_filename: str) -> str | None:
    """
    创建临时文件并返回文件路径

    参数:
        file_stream: 文件流对象
        original_filename: 原始文件名

    返回值:
        str | None: 临时文件路径，如果创建失败则返回 None
    """
    try:
        current_dir: str = os.getcwd()
        upload_dir: str = os.path.join(current_dir, "upload_files")
        # 如果目录不存在，则先创建
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        filename, ext = os.path.splitext(original_filename)
        tmp_file_path: str = os.path.join(upload_dir, f"{filename}_{uuid.uuid4().hex}{ext}")

        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(file_stream)

        return tmp_file_path
    except PermissionError as pe:
        st.error(f"权限不足，无法创建临时文件: {str(pe)}")
        return None
    except OSError as oe:
        st.error(f"操作系统错误，无法创建临时文件: {str(oe)}")
        return None
    except Exception as e:
        st.error(f"创建临时文件失败: {str(e)}")
        return None


def process_pdf_with_pymupdf(file_path: str) -> str | None:
    """
    使用 PyMuPDF 处理 PDF 文件并提取内容

    参数:
        file_path: 文件路径

    返回值:
        str | None: 提取的文本内容
    """

    try:
        content: str = ""
        doc = fitz.open(file_path)
        for page in doc:
            content += page.get_text()
        return content
    except Exception as e:
        st.error(f"PyMuPDF 处理 PDF 失败: {str(e)}")
        return None


def process_pdf_with_pymupdf4llm(file_path: str) -> str | None:
    """
    使用 pymupdf4llm 处理 PDF 文件并提取内容

    参数:
        file_path: 文件路径

    返回值:
        str | None: 提取的文本内容
    """
    try:
        # 使用 pymupdf4llm 处理临时文件
        content: str = pymupdf4llm.to_markdown(file_path)
        return content
    except Exception as e:
        st.error(f"pymupdf4llm 处理 PDF 失败: {str(e)}")
        return None


def process_pdf_with_mineru(file_path: str) -> str | None:
    """
    使用 mineru 处理 PDF 文件并提取内容

    参数:
        file_path: 文件路径

    返回值:
        str | None: 提取的文本内容
    """
    # 获取文件名和文件名无后缀（使用os.path模块，跨平台兼容）
    pdf_relative: str = file_path
    pdf_name: str = os.path.basename(pdf_relative)
    # 从路径中提取文件名（正确处理完整路径）
    pdf_name: str = os.path.basename(pdf_relative)
    # 移除文件扩展名以仅获取文件名
    name_without_suff: str = os.path.splitext(pdf_name)[0]

    # 设置输出目录
    output_dir: str = "test_outputs"

    # 生成唯一ID
    unique_id: uuid.UUID = uuid.uuid4()
    output_subdir: str = f"{name_without_suff}_{unique_id}".replace("-", "")

    # 创建输出目录和图片目录
    md_relative: str = os.path.join(output_dir, output_subdir)
    os.makedirs(md_relative, exist_ok=True)

    image_relative: str = os.path.join(md_relative, "images")
    os.makedirs(image_relative, exist_ok=True)

    # 转换为绝对路径
    pdf_path: str = os.path.abspath(pdf_relative)
    md_output_path: str = os.path.abspath(md_relative)

    # 使用绝对路径调用函数
    md: str = mineru_pdf2md(
        pdf_file_path=pdf_path,
        md_output_path=md_output_path,
    )
    return md


def extract_uploaded_file_content(uploaded_file, pdf_process_method: str = "pymupdf") -> tuple[str | None, dict | None]:
    """
    功能描述: 处理上传的文件并返回完整内容和文件元数据。

    参数:
        uploaded_file: 上传的文件对象，支持 PDF 和 TXT 格式
        file_content_length (int): 文件内容长度
        pdf_method (str): PDF处理方法，可选 'pymupdf', 'pymupdf4llm', 'mineru'，默认为 'pymupdf'

    返回值:
        tuple[str | None, dict | None]: 文件内容的字符串和文件元数据，如果处理失败则返回 (None, None)
    """
    try:
        original_filename: str = uploaded_file.name
        file_stream = uploaded_file.read()
        tmp_file_path: str | None = create_temp_file(file_stream, original_filename)
        if tmp_file_path is None:
            return None, None

        if uploaded_file.type == "application/pdf":
            # 根据指定的 pdf_method 处理 PDF
            if pdf_process_method == "pymupdf":
                content: str | None = process_pdf_with_pymupdf(tmp_file_path)
            elif pdf_process_method == "pymupdf4llm":
                content: str | None = process_pdf_with_pymupdf4llm(tmp_file_path)
            elif pdf_process_method == "mineru":
                content: str | None = process_pdf_with_mineru(tmp_file_path)
            else:
                st.error(f"不支持的 PDF 处理方法: {pdf_process_method}")
                return None, None
        else:
            # 处理文本文件
            content: str = uploaded_file.getvalue().decode("utf-8")

        file_metadata: dict = {
            "name": original_filename,
            "type": uploaded_file.type.split('/')[-1].upper(),
            "size": len(file_stream)
        }

        return content, file_metadata
    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")
        return None, None
