import re
import streamlit as st
import os
import openai

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="联网搜索对话系统", layout="wide")
st.title("🔎 联网搜索对话系统 ")

st.sidebar.title("配置选项")
analyze_images_enabled = st.sidebar.checkbox("开启图片分析", value=False)
if st.sidebar.button("清空对话记录", use_container_width=True):
    st.session_state["history"] = []
    st.session_state["search_results"] = []

def get_reference_content():
    # 直接使用一个 markdown 文件作为输入内容
    file_name = "./web_datas/全球公认的三大帅气、高颜值狗，有你家的爱犬吗？_images.md"
    with open(file_name, encoding="utf-8") as f:
        reference_content = f.read()
    return reference_content

def call_guiji_rag_model_stream(query, chat_history=None):  # 删除了 answer_blocks 参数
    client = openai.OpenAI(
        api_key=os.getenv("GUIJI_API_KEY"), base_url=os.getenv("GUIJI_BASE_URL")
    )
    guiji_model = os.getenv("GUIJI_TEXT_MODEL")
    
    # 直接获取参考内容
    reference_content = get_reference_content()
    prompt = f"请根据以下参考内容回答用户的问题：{query}\n\n下面是参考内容：\n{reference_content}"
    
    sys_prompt = """
    你是一个人工智能助手，能够回答用户的问题，并且可以参考提供的内容。请根据用户的问题和参考内容生成准确的回答。
    参考内容可能包含多个段落，请确保回答时充分利用这些信息。
    如果参考内容中有图片，你可以适当选择合适的图片进行引用并使用 markdown 的语法输出到回答中。
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    if chat_history:
        # 将历史记录拼接到消息中
        for chat in chat_history:
            messages.insert(-1, chat)  # 在倒数第二个位置插入，保持用户问题在最后

    response = client.chat.completions.create(
        model=guiji_model, messages=messages, stream=True, max_tokens=4096
    )
    return response

# 搜索输入框
query = st.text_input("请输入你的问题或关键词：", "全球公认的三大帅气、高颜值狗是哪些？")

if "history" not in st.session_state:
    st.session_state["history"] = []
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []

# 显示历史对话
for msg in st.session_state["history"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input and user_input.strip():
    st.session_state["history"].append({"role": "user", "content": user_input.strip()})
    # 示例 User-Agent（目前未使用）
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko Chrome/91.0.4472.124 Safari/537.36"
    with st.status("正在处理内容，请稍候...", expanded=True) as status:  # 更新状态提示消息
        try:
            status.write("📄 正在加载参考内容...")
            answer_blocks = get_reference_content()
            status.write("✅ 内容加载完成。")

            if answer_blocks:
                status.write("🤖 正在调用大模型流式生成回答...")
                chat_history = st.session_state["history"][-5:]
                response = call_guiji_rag_model_stream(
                    user_input, chat_history  # 传入已移除的参数位置
                )
                full_answer = ""
                with st.chat_message("assistant"):
                    stream_placeholder = st.empty()
                    # 流式接收并展示回答
                    for chunk in response:
                        delta = (
                            chunk.choices[0].delta.content
                            if chunk.choices[0].delta
                            else ""
                        )
                        if delta:
                            full_answer += delta
                            stream_placeholder.markdown(full_answer)
                # 保存 AI 回答到历史
                st.session_state["history"].append(
                    {"role": "assistant", "content": full_answer}
                )
                status.write("✅ 回答生成完毕！")
        except Exception as e:
            st.error(f"处理过程中发生错误: {e}")
            status.write("❌ 处理过程中发生错误")
