# 作者: 筱可
# 日期: 2025 年 2 月 11 日
# 版权所有 (c) 2025 筱可 & 筱可AI研习社. 保留所有权利.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from datetime import datetime

import os
import openai
from openai import OpenAI
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from dotenv import load_dotenv


import os


def find_dotenv_path(dir_name: str = "XiaokeAILabs"):
    # 首先在当前工作目录下查找 .env 文件
    current_working_dir = os.getcwd()
    env_path = os.path.join(current_working_dir, ".env")
    if os.path.exists(env_path):
        return env_path

    # 如果没有指定目录名，且当前工作目录下没有 .env 文件，则返回 None
    if not dir_name:
        return None

    # 从当前脚本所在目录开始向上查找指定目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        # 检查当前目录是否为指定目录
        if os.path.basename(current_dir) == dir_name:
            # 在指定目录下查找 .env 文件
            env_path_in_specified_dir = os.path.join(current_dir, ".env")
            if os.path.exists(env_path_in_specified_dir):
                return env_path_in_specified_dir
            else:
                return None
        # 到达根目录时停止查找
        if current_dir == os.path.dirname(current_dir):
            break
        # 向上移动一层目录
        current_dir = os.path.dirname(current_dir)

    # 如果没有找到指定目录，返回 None
    return None


DOT_ENV_PATH = find_dotenv_path()
load_dotenv(dotenv_path=DOT_ENV_PATH)
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

# 模型列表
model_list = {
    "DeepSeek-R1-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
}


# 初始化 session_state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_content" not in st.session_state:
        st.session_state.uploaded_content = None
    if "context_length" not in st.session_state:
        st.session_state.context_length = 12000
    if "file_content_length" not in st.session_state:
        st.session_state.file_content_length = 15000


init_session()
AUTHOR = "筱可"
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
WECHAT_PLATFORM = "筱可AI研习社"


# 文件处理函数
def process_uploaded_file(uploaded_file):
    try:
        content = ""
        if uploaded_file.type == "application/pdf":
            # 处理PDF文件
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                content += page.get_text()
        else:
            # 处理文本文件
            content = uploaded_file.getvalue().decode("utf-8")

        # 截取前 file_content_length 个字符
        max_length = st.session_state.file_content_length
        return content[:max_length] + "..." if len(content) > max_length else content

    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")
        return None


# 侧边栏设置
with st.sidebar:
    st.header("配置参数")
    selected_model = st.selectbox("选择模型", options=list(model_list.keys()), index=2)
    temperature = st.slider("温度参数", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("最大长度", 100, 4096, 2048, 100)
    st.session_state.context_length = st.slider("上下文长度", 1000, 100000, 32000, 500)
    st.session_state.file_content_length = st.slider(
        "文件内容读取长度", 1000, 100000, 15000, 500
    )

    # 新建对话按钮
    if st.button("新建对话"):
        st.session_state.messages = []
        st.session_state.uploaded_content = None
        st.session_state.current_file = None
        st.success("新对话已创建！")

st.title("📑 DeepSeek 智能文档助手 ✨")

st.markdown(
    """<hr style="border:2px solid #FFA07A; border-radius: 5px;">""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style='
        text-align: center;
        padding: 15px;
        background: linear-gradient(45deg, #FFD700, #FFA07A);
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    '>
        <h4 style='color: #2F4F4F; margin: 0;'>🐰 作者：{AUTHOR}</h4>
        <p style='color: #800080; margin: 10px 0 0;'>
            🌸 公众号：「<strong style='color: #FF4500;'>{WECHAT_PLATFORM}</strong>」
            <br>
            <span style='font-size:14px; color: #4682B4;'>✨ 探索AI的无限可能 ✨</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "first_load" not in st.session_state:
    st.balloons()
    st.session_state.first_load = True


# 文件上传部件
uploaded_file = st.file_uploader("上传文档（支持PDF/TXT）", type=["pdf", "txt"])
if uploaded_file and uploaded_file != st.session_state.get("current_file"):
    processed_content = process_uploaded_file(uploaded_file)
    if processed_content:
        st.session_state.uploaded_content = processed_content
        st.session_state.current_file = uploaded_file

        # 构建CRISPE框架系统提示
        system_prompt = f"""
<system>
    [当前日期] {CURRENT_DATE}
    [角色] 您是一名专业的文档分析助理，擅长从技术文档中提取关键信息
    
    [背景] 
    - 用户上传了文档：{uploaded_file.name}
    - 文档类型：{uploaded_file.type.split('/')[-1].upper()}
    - 文档内容：{processed_content[:st.session_state.file_content_length]}...
    
    [核心任务]
    1. 当用户提问时，优先从文档中寻找答案
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
        # 清空历史消息
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.success(f"文档 {uploaded_file.name} 解析完成！")
else:
    # 默认系统提示词模板
    default_system_prompt = f"""
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
    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "system", "content": default_system_prompt}
        ]

# 聊天记录显示
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 用户输入处理
if prompt := st.chat_input("请输入问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构建API请求
    keep_messages = 10
    system_message = st.session_state.messages[0]  # 保留第一个system消息
    messages_for_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] != "system"
    ]

    # 截断上下文以满足上下文长度要求
    total_length = sum(len(m["content"]) for m in messages_for_api)
    while total_length > st.session_state.context_length:
        messages_for_api.pop(0)
        total_length = sum(len(m["content"]) for m in messages_for_api)

    messages_for_api.insert(0, system_message)  # 将system消息重新插入到第一条

    try:
        # 生成流式回复
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            collected_response = []

            response = client.chat.completions.create(
                model=model_list[selected_model],
                messages=messages_for_api,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    collected_response.append(token)
                    response_placeholder.markdown("".join(collected_response) + "▌")

            final_response = "".join(collected_response)
            response_placeholder.markdown(final_response)

            # 更新消息记录
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

    except openai.APIError as e:
        error_msg = f"""
        <error>
            [错误分析]
            API请求失败，可能原因：
            1. 上下文过长（当前：{len(str(messages_for_api))}字符）
            
            [修正建议]
            请尝试以下操作：
            - 调整上下文长度至16000字符内
            - 重新组织问题表述
            - 新建对话以重试
        </error>
        """
        st.error(error_msg, icon="🚨")
