from transformers import AutoTokenizer

tokenizer_directory = r"./tokenize"

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    print("分词器加载成功！")
    print(tokenizer.tokenize("你好，世界！"))
except Exception as e:
    print(f"加载分词器失败：{e}")