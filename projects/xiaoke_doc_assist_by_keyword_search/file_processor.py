import fitz  # PyMuPDF
import streamlit as st
import os
import pymupdf4llm
import uuid

def create_temp_file(file_stream, original_filename) -> str | None:
    """
    创建临时文件并返回文件路径

    参数:
        file_stream: 文件流对象
        original_filename: 原始文件名

    返回值:
        str | None: 临时文件路径，如果创建失败则返回 None
    """
    try:
        current_dir = os.getcwd()
        upload_dir = os.path.join(current_dir, "upload_files")
        # 如果目录不存在，则先创建
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        filename, ext = os.path.splitext(original_filename)
        tmp_file_path = os.path.join(upload_dir, f"{filename}_{uuid.uuid4().hex}{ext}")

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

def process_pdf_with_pymupdf(file_path) -> str | None:
    """
    使用 PyMuPDF 处理 PDF 文件并提取内容

    参数:
        file_path: 文件路径

    返回值:
        str | None: 提取的文本内容
    """
    
    try:
        content = ""
        doc = fitz.open(file_path)
        for page in doc:
            content += page.get_text()
        return content
    except Exception as e:
        st.error(f"PyMuPDF 处理 PDF 失败: {str(e)}")
        return None

def process_pdf_with_pymupdf4llm(file_path) -> str | None:
    """
    使用 pymupdf4llm 处理 PDF 文件并提取内容

    参数:
        file_path: 文件路径

    返回值:
        str | None: 提取的文本内容
    """
    try:
        # 使用 pymupdf4llm 处理临时文件
        content = pymupdf4llm.to_markdown(file_path)
        return content
    except Exception as e:
        st.error(f"pymupdf4llm 处理 PDF 失败: {str(e)}")
        return None

def process_pdf_with_mineru(file_path) -> str | None:
    """
    使用 mineru 处理 PDF 文件并提取内容（待实现）

    参数:
        file_stream: 文件流对象

    返回值:
        str | None: 提取的文本内容
    """
    # TODO: 等待 mineru 实现
    st.warning("mineru 处理方法尚未实现")
    return None

def extract_uploaded_file_content(uploaded_file, pdf_method: str = "pymupdf") -> tuple[str | None, dict | None]:
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
        original_filename = uploaded_file.name
        file_stream = uploaded_file.read()
        tmp_file_path = create_temp_file(file_stream, original_filename)
        if tmp_file_path is None:
            return None, None

        if uploaded_file.type == "application/pdf":
            # 根据指定的 pdf_method 处理 PDF
            if pdf_method == "pymupdf":
                content = process_pdf_with_pymupdf(tmp_file_path)
            elif pdf_method == "pymupdf4llm":
                content = process_pdf_with_pymupdf4llm(tmp_file_path)
            elif pdf_method == "mineru":
                content = process_pdf_with_mineru(file_stream)
            else:
                st.error(f"不支持的 PDF 处理方法: {pdf_method}")
                return None, None
        else:
            # 处理文本文件
            content = uploaded_file.getvalue().decode("utf-8")

        file_metadata = {
            "name": original_filename,
            "type": uploaded_file.type.split('/')[-1].upper(),
            "size": len(file_stream)
        }

        return content, file_metadata
    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")
        return None, None
