#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块名称: main
功能描述: 主程序，整合所有模块并运行 Streamlit 应用。
作者: 筱可
创建日期: 2025-02-20
版本: 1.0
"""

import streamlit as st  # 导入 Streamlit 库用于创建 Web 应用界面
from ui_components import setup_sidebar, setup_main_ui, display_chat_history  # 导入 UI 组件函数
from chat_handler import handle_uploaded_files, init_session, handle_chat  # 导入聊天处理相关函数


def main() -> None:
    """
    功能描述: 主函数，初始化应用并运行聊天助手。
    参数: 无
    返回值: 无
    """
    # 初始化会话状态，确保应用启动时必要的 session_state 变量被创建
    init_session()

    # 设置主界面布局，包含标题、说明等基础 UI 元素
    setup_main_ui()

    # 设置侧边栏并获取用户选择的参数
    setup_sidebar()


    # 创建文件上传控件，支持 PDF 和 TXT 格式的文件
    uploaded_files = st.file_uploader("上传文档（支持PDF/TXT）", type=["pdf", "txt"], accept_multiple_files=True)
    handle_uploaded_files(uploaded_files)

    # 在应用首次加载时显示气球动画效果
    if "first_load" not in st.session_state:
        st.balloons()
        st.session_state.first_load = True

    # 显示聊天历史记录，展示用户与助手的对话内容
    display_chat_history()

    # 创建聊天输入框，获取用户输入的问题
    if query := st.chat_input("请输入问题..."):
        # 处理用户输入的聊天内容，使用所选模型和参数进行回复
        handle_chat(query)


if __name__ == "__main__":
    # 当脚本作为主程序运行时，执行 main 函数
    main()