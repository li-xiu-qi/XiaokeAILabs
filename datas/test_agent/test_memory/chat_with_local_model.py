def get_local_api_config():

    """从环境变量或.env文件获取本地API配置"""
    import os
    api_key = os.getenv("LOCAL_API_KEY", "anything")
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:10002/v1")
    model = os.getenv("LOCAL_TEXT_MODEL", "qwen3")
    return api_key, base_url, model




# 用新版 openai 库的方式调用本地 openai 兼容模型
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def chat_with_local_model(messages, model=None, temperature=0.7, **kwargs):
    """
    用新版 openai 库的方式调用本地 openai 兼容模型
    messages: openai 格式的 messages
    model: 模型名称，默认读取 LOCAL_TEXT_MODEL
    其他参数可透传
    """
    import os
    api_key = os.getenv("LOCAL_API_KEY", "anything")
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:10002/v1")
    model = model or os.getenv("LOCAL_TEXT_MODEL", "qwen3")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs
    )


if __name__ == "__main__":
    # 示例对话
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请用一句话介绍一下你自己。"},
    ]
    result = chat_with_local_model(messages)
    print("Assistant:", result.choices[0].message.content)
