#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# author：筱可  
# 2025-02-22  
"""
#### 使用说明：
该模块用于设置和展示 Streamlit 用户界面，包括侧边栏配置、主界面标题和作者信息展示，以及聊天记录展示。

#### 主要功能：
1. 设置侧边栏的配置，包括选择模型、调整参数。
2. 设置主界面的标题和作者信息展示。
3. 展示聊天记录，排除系统消息。

#### 参数说明：
1. setup_sidebar 函数：
    - 返回值：
        - selected_model (str): 用户选择的模型。
        - temperature (float): 温度参数，控制生成文本的随机性。
        - max_tokens (int): 最大生成文本长度。
        - context_length (int): 上下文长度。
        - file_content_length (int): 文件内容读取长度。

2. setup_main_ui 函数：
    - 返回值：无

3. display_chat_history 函数：
    - 返回值：无

#### 注意事项：
1. 需要安装 streamlit 库以及相关配置文件。
2. 确保 session_state 已正确初始化，用于存储聊天记录和相关状态。
"""

import streamlit as st
from config import AUTHOR, WECHAT_PLATFORM, MODEL_LIST


def setup_sidebar() -> None:
    """
    功能描述: 设置侧边栏配置，包括模型选择和参数调整。
    """
    with st.sidebar:
        if st.button("新建对话"):
            st.session_state.messages = []
            st.session_state.uploaded_content = None
            st.session_state.current_file = None
            st.success("新对话已创建！")
            
        st.header("配置参数")
        st.session_state.selected_model = st.selectbox("选择模型", options=list(MODEL_LIST.keys()), index=2)
        st.session_state.temperature = st.slider("温度参数", 0.0, 1.0, 0.3, 0.1)
        st.session_state.max_token = st.slider("最大输出长度", 100, 4096, 2048, 100)
        st.session_state.context_length = st.slider("上下文长度", 1000, 100000, 32000, 500)
        st.session_state.language = st.selectbox("选择语言", options=["zh", "en"], index=0)
        st.session_state.retrieval_context_length = st.slider("检索上下文长度", 1000, 50000, 15000, 1000)
     


def setup_main_ui() -> None:
    """
    功能描述: 设置主界面，包括标题和作者信息展示。
    """
    st.title("📑 DeepSeek 智能文档助手 ✨")
    st.markdown("""<hr style="border:2px solid #FFA07A; border-radius: 5px;">""", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='text-align: center; padding: 15px; background: linear-gradient(45deg, #FFD700, #FFA07A); 
        border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;'>
            <h4 style='color: #2F4F4F; margin: 0;'>🐰 作者：{AUTHOR}</h4>
            <p style='color: #800080; margin: 10px 0 0;'>
                🌸 公众号：「<strong style='color: #FF4500;'>{WECHAT_PLATFORM}</strong>」
                <br><span style='font-size:14px; color: #4682B4;'>✨ 探索AI的无限可能 ✨</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_chat_history() -> None:
    """
    功能描述: 显示聊天记录，排除系统消息。
    """
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
