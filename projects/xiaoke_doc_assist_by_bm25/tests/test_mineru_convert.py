"""
用来测试MinerU的PDF转换功能
"""

import os
import uuid

from projects.xiaoke_doc_assist_by_bm25.mineru_convert import mineru_pdf2md

if __name__ == '__main__':

    # 使用示例：将相对路径转换为绝对路径后传入
    try:
        # 原相对路径
        pdf_relative = "test_datas/test_paper.pdf"
        pdf_name = pdf_relative.split("/")[-1]
        name_without_suff = pdf_name.split(".")[-2]

        output_dir = "../test_outputs"
        # 组合输出目录和图片目录

        # # 生成唯一ID
        unique_id = uuid.uuid4()
        output_subdir = f"{name_without_suff}".replace("-", "")

        md_relative = os.path.join(output_dir,output_subdir)

        os.makedirs(md_relative, exist_ok=True)

        image_relative = os.path.join(md_relative, "images")
        os.makedirs(image_relative, exist_ok=True)
        # 转换为绝对路径
        pdf_path = os.path.abspath(pdf_relative)
        md_output_path = os.path.abspath(md_relative)

        # 使用绝对路径调用函数
        md  =  mineru_pdf2md(
            pdf_file_path=pdf_path,
                     md_output_path=md_output_path,
                     return_path=True
                     )
        print(md)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")