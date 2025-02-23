#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块名称: chat_handler
功能描述: 处理聊天逻辑，包括会话初始化、系统提示词生成和 API 调用。
作者: 筱可
创建日期: 2025-02-20
版本: 1.0
"""

import streamlit as st
from openai import OpenAI
from config  import API_KEY, BASE_URL, CURRENT_DATE, MODEL_LIST,  Document
from file_processor import extract_uploaded_file_content
from bm25 import ChineseBM25, EnglishBM25
from split_by_markdown import merge_markdown_chunks, split_markdown_by_headers
# 初始化 OpenAI 客户端
client: OpenAI = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def init_session() -> None:
    """
    功能描述: 初始化会话状态，确保必要的 session_state 变量存在。
    """
    if "messages" not in st.session_state:
        # 初始化消息列表
        st.session_state.messages = []
    if "uploaded_content" not in st.session_state:
        # 上传的文件内容,主要使用metadata和content组成
        st.session_state.uploaded_content = [] 
    if "context_length" not in st.session_state:
        # 上下文长度
        st.session_state.context_length = 32000
    if "file_content_length" not in st.session_state:
        # 文本块长度限制
        st.session_state.file_content_length = 15000
    if "selected_model" not in st.session_state:
        # 模型选择
        st.session_state.selected_model = list(MODEL_LIST.keys())[2]
    if "temperature" not in st.session_state:
        # 温度控制
        st.session_state.temperature = 0.5
    if "max_tokens" not in st.session_state:
        # 最大生成令牌数
        st.session_state.max_tokens = 4096
    if "language" not in st.session_state:
        # 语言选择
        st.session_state.language = "zh"
    if "file_change" not in st.session_state:
        st.session_state.file_change = 0

def retrieved_information(query="") -> str:
    """
    功能描述: 根据已处理的文件内容生成参考信息文档。

    参数:
        file_metadata: 文件元数据
        processed_content: 已处理的文件内容

    返回值:
        str: 生成的参考信息文档字符串
    """
    uploaded_content = st.session_state.uploaded_content
    documents = []
    for u in uploaded_content:
        content = u.get("content")
        # metadata = u.get("metadata")
        initial_chunks = split_markdown_by_headers(content)
        if not initial_chunks:
            print("有未提取文件出现")
            continue
        # 测试自动检测语言
        merged_chunks_auto = merge_markdown_chunks(initial_chunks, 
                                                   chunk_size=1000, 
                                                   chunk_overlap=100)
        # {header:"",content:"",level:""}
        # todo 给块上面都添加序号，方便后续进行同文件内容重排序
        for chunk_dict in merged_chunks_auto:
            # d = Document(content=chunk_dict.get("content"),metadata=metadata)
            # documents.append(d)
            # todo 将document换成Document类
            documents.append(chunk_dict.get("content"))
    language = st.session_state.language
    if language == "zh":
        # 中文模式
        chinese_bm25 = ChineseBM25(documents)
        retrieved_text_result = chinese_bm25.search(query)

    elif language == "en":
        # 英文模式
        english_bm25 = EnglishBM25(documents)
        retrieved_text_result= english_bm25.search(query)
    
    relate_texts = [documents[doc_id] for doc_id, score in retrieved_text_result]
    retrieval_context_length = st.session_state.retrieval_context_length 
    # 根据检索内容长度限制获取合适数量的文本
    selected_texts = []
    current_length = 0
    for text in relate_texts:
        content_length = len(text)
        if current_length + content_length <= retrieval_context_length:
            selected_texts.append(text)
            current_length += content_length
        else:
            break
    
    # 将选中的文本内容拼接成字符串返回
    if selected_texts:
        return "\n\n".join(text for text in selected_texts)

def get_system_prompt() -> str:
    """
    功能描述: 根据是否有上传文件生成对应的系统提示词。

    参数:
        file_metadata: 文件元数据，默认为 None
        processed_content (str | None): 已处理的文件内容，默认为 None

    返回值:
        str: 生成的系统提示词字符串
    """
    file_metadatas = [file['metadata'] for file in st.session_state.uploaded_content]
    file_info = "\n".join(
        f"【- 用户上传了文档：{file_metadata['name']}\n  - 文档类型：{file_metadata['type']}】"
        for file_metadata in file_metadatas
    )
    
    retrieved_info = retrieved_information()
    if file_metadatas:
        return f"""
<system>
    [当前日期] {CURRENT_DATE}
    [角色] 您是一名专业的文档分析助理，擅长从技术文档中提取关键信息
    [背景] 
    {file_info}
    用户将会向你提问关于这些文档的问题，系统将会根据问题检索对应的文本块内容，并拼接成参考信息文档。
    [参考信息文档]
    【{retrieved_info}】
    [核心任务]
    1. 当用户提问时，优先从参考信息文档中寻找答案
    2. 对复杂问题进行分步骤思考
    3. 不要生成参考信息文档没有提供的信息
    4. 保持专业且易懂的语气
    5. 不确定时，可以请求用户提供更多信息
    [交互要求]
    - 保持专业且易懂的语气
    - 关键数据用**加粗**显示
    - 代码块使用```包裹
</system>
        """
    return f"""
<system>
    [当前日期] {CURRENT_DATE}
    [角色] 您是一名专业的文档分析助理。
    [核心任务]
    1. 当用户提问时，可以询问用户是否需要提供资料增强回答
    2. 对复杂问题进行分步骤思考
    3. 不要生成文档没有提供的信息
    4. 保持专业且易懂的语气
    5. 不确定时，可以请求用户提供更多信息
    [交互要求]
    - 保持专业且易懂的语气
    - 关键数据用**加粗**显示
    - 代码块使用```包裹
</system>
"""

def handle_uploaded_files(uploaded_files) -> None:
    # 处理多个上传文件
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 检查文件是否已经在 uploaded_content 中
            if any(file['metadata']['name'] == uploaded_file.name for file in st.session_state.uploaded_content):
                continue
            with st.spinner("文档处理中..."):
                # 处理上传的文件，返回处理后的内容和文件元数据
                processed_content, file_metadata = extract_uploaded_file_content(uploaded_file, pdf_method="pymupdf4llm")
            # 显示成功消息，提示用户文件已解析完成
            st.success(f"文档 {uploaded_file.name} 解析完成！")
            if processed_content:
                # 将处理后的内容保存到 session_state
                st.session_state.uploaded_content.append({"metadata": file_metadata, "content": processed_content}) 
                # 初始化消息列表，包含系统提示（基于上传文件生成）

def messages_context_manager():
    
    system_message: dict = {"role": "system", "content": get_system_prompt()}
    messages_for_api: list = [{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages if m["role"] != "system"]

    total_length: int = sum(len(m["content"]) for m in messages_for_api)
    while total_length > st.session_state.context_length:
        messages_for_api.pop(0)
        total_length = sum(len(m["content"]) for m in messages_for_api)

    messages_for_api.insert(0, system_message)
    return messages_for_api

def generate_response(selected_model: str, messages_for_api: list, temperature: float, max_tokens: int):
    """
    功能描述: 调用 API 生成回复。

    参数:
        selected_model (str): 选择的模型名称
        messages_for_api (list): 发送给 API 的消息列表
        temperature (float): 温度参数，控制生成多样性
        max_tokens (int): 最大生成令牌数

    返回值:
        generator: 生成回复的生成器
    """
    response = client.chat.completions.create(
        model=MODEL_LIST[selected_model],
        messages=messages_for_api,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def handle_chat(query: str) -> None:
    """
    功能描述: 处理用户输入，调用 API 生成回复并更新聊天记录。

    参数:
        prompt (str): 用户输入的问题
        selected_model (str): 选择的模型名称
        temperature (float): 温度参数，控制生成多样性
        max_tokens (int): 最大生成令牌数
    """

    selected_model: str = st.session_state.selected_model
    temperature: float = st.session_state.temperature
    max_tokens: int = st.session_state.max_tokens
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # 管理上下文信息
    messages_for_api = messages_context_manager()

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            collected_response: list = []
            for token in generate_response(selected_model, messages_for_api, temperature, max_tokens):
                collected_response.append(token)
                response_placeholder.markdown("".join(collected_response) + "▌")
            final_response: str = "".join(collected_response)
            response_placeholder.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
    except Exception as e:
        st.error(f"API请求失败: {str(e)}")