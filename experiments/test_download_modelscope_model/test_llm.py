# 导入必要的库
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型路径
model_name = "./Qwen2.5-0.5B-Instruct"

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择数据类型
    device_map="auto",   # 自动分配设备
    trust_remote_code=True,  # 信任远程代码
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 设置用户提问内容
prompt = "请简单介绍一下大型语言模型。"

# 构造对话消息格式
messages = [
    {"role": "system", "content": "你是通义千问，由阿里云开发的AI助手。你是一个有用的助手。"},
    {"role": "user", "content": prompt}
]
# 应用聊天模板，生成输入文本
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 不进行分词，返回字符串
    add_generation_prompt=True  # 添加生成提示符
)

# 对输入文本进行编码，转换为模型输入格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 使用模型生成回答
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512  # 最大生成512个新token
)

# 提取生成的新token（去除输入部分）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 将生成的token解码为文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出模型回答
print(response)