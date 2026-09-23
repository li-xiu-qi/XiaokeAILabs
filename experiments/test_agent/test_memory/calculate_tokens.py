
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"


from modelscope import AutoTokenizer

def count_tokens(text, model_name="Qwen/Qwen3-32B"):
    """
    计算输入文本的 token 数，支持指定模型名。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors=None)["input_ids"]
    return len(tokens)


if __name__ == "__main__":
    # 读取Markdown全文
    with open("2110.02861.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    token_count = count_tokens(md_content)
    print(f"Markdown内容的token数: {token_count}")

    # 判断是否超过7/16上下文（40960 * 7 // 16 ≈ 17920）
    max_tokens = 40960
    threshold = max_tokens * 7 // 16
    if token_count > threshold:
        print("内容超过7/16上下文窗口，建议分块处理。")
    else:
        print("内容未超过7/16上下文窗口，可以整体修复。")