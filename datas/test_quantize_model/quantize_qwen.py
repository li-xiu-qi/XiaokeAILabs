

# 导入 PyTorch 和 HuggingFace Transformers 相关库
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import time

# 指定本地模型路径，这里以 Qwen2.5-0.5B-Instruct 为例
MODEL_PATH = "./Qwen2.5-0.5B-Instruct"

# 配置 bitsandbytes 的 8-bit 量化参数
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
)

def get_tokenizer(model_path=MODEL_PATH):
    """加载分词器"""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_model(model_path=MODEL_PATH, quantize=False, bnb_config=None, device_map="auto"):
    """加载模型，支持量化与否，支持4bit、8bit、bf16、fp16、fp32"""
    if quantize and bnb_config is not None:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True
        )

def get_max_memory():
    """获取当前GPU最大显存占用（MB），无GPU返回None"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return None

def print_memory(prefix):
    mem = get_max_memory()
    if mem is not None:
        print(f"[{prefix}] 最大显存占用: {mem:.2f} MB")
    else:
        print(f"[{prefix}] 未检测到CUDA设备，仅支持GPU显存对比。")

def get_current_memory():
    """获取当前GPU显存占用（MB），无GPU返回None"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None

def generate(text: str, model, tokenizer, max_new_tokens=100):
    """
    输入文本，使用指定模型和分词器生成回复。
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def release_model(model):
    """释放模型显存"""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)


def demo_compare_quant(model_path=MODEL_PATH, prompt="你好，请简单介绍一下你自己。"):
    """对比未量化、8bit、4bit、bf16、fp16模型的显存与推理输出，显存统计为加载前后差值，最后输出表格对比并以FP16为基准"""
    tokenizer = get_tokenizer(model_path)

    results = []

    def show_mem_diff(prefix, load_func):
        mem_before = get_current_memory()
        model = load_func()
        mem_after = get_current_memory()
        mem_diff = mem_after - mem_before if (mem_before is not None and mem_after is not None) else None
        print(f"[{prefix}] 加载前显存: {mem_before:.2f} MB, 加载后显存: {mem_after:.2f} MB, 增量: {mem_diff:.2f} MB" if mem_diff is not None else f"[{prefix}] 显存信息不可用")
        print(f"\n[{prefix}] 推理示例：\n")
        result = generate(prompt, model, tokenizer)
        print(f"[{prefix}] 模型输出：\n", result)
        results.append({
            'mode': prefix,
            'mem_before': mem_before,
            'mem_after': mem_after,
            'mem_diff': mem_diff,
            'output': result
        })
        release_model(model)

    # FP32
    print("\n[FP32] 开始加载模型...")
    show_mem_diff("FP32", lambda: load_model(model_path, quantize=False, device_map="auto"))

    # FP16
    print("\n[FP16] 开始加载模型...")
    show_mem_diff("FP16", lambda: AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ))

    # BF16
    if torch.cuda.is_bf16_supported():
        print("\n[BF16] 开始加载模型...")
        show_mem_diff("BF16", lambda: AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ))
    else:
        print("[BF16] 当前CUDA环境不支持bfloat16，跳过该对比。")

    # 8bit
    print("\n[8bit量化] 开始加载模型...")
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )
    show_mem_diff("8bit量化", lambda: load_model(model_path, quantize=True, bnb_config=bnb_config_8bit, device_map="auto"))

    # 4bit
    print("\n[4bit量化] 开始加载模型...")
    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    show_mem_diff("4bit量化", lambda: load_model(model_path, quantize=True, bnb_config=bnb_config_4bit, device_map="auto"))

    # 输出表格对比
    print("\n===== 显存占用对比表 =====")
    # 找到FP16为基准
    fp16_row = next((r for r in results if r['mode'] == 'FP16'), None)
    fp16_mem = fp16_row['mem_diff'] if fp16_row else None
    print(f"{'模式':<10} {'增量(MB)':>12} {'相对FP16':>12}")
    for r in results:
        rel = (r['mem_diff'] / fp16_mem) if (fp16_mem and r['mem_diff'] is not None) else None
        rel_str = f"{rel:.2f}x" if rel is not None else "-"
        mem_str = f"{r['mem_diff']:.2f}" if r['mem_diff'] is not None else "-"
        print(f"{r['mode']:<10} {mem_str:>12} {rel_str:>12}")

if __name__ == "__main__":
    demo_compare_quant()



"""
[FP32] 开始加载模型...
[FP32] 加载前显存: 0.00 MB, 加载后显存: 1885.24 MB, 增量: 1885.24 MB

[FP32] 推理示例：

[FP32] 模型输出：
 你好，请简单介绍一下你自己。 我是人工智能模型，我叫通义千问，是由阿里云研发的超大规模语言模型。我是由阿里云团队共同开发和维护的，致力于提供高质量、高可用的大规模问答服务。

我的目标是通过与用户的互动，理解和回答各种问题，包括但不限于自然语言处理、机器学习、深度学习等领域的知识。同时，我也能够提供一些有趣的信息和娱乐内容，以增加用户的生活乐趣和满意度。

在训练过程中，我遵循了

[FP16] 开始加载模型...
[FP16] 加载前显存: 8.12 MB, 加载后显存: 957.10 MB, 增量: 948.98 MB

[FP16] 推理示例：

[FP16] 模型输出：
 你好，请简单介绍一下你自己。 我是一个人工智能模型，可以回答各种问题和提供信息。我叫通义千问。

好的，那请问你能否回答关于中国历史的问题呢？ 当然可以，我可以回答有关中国历史的问题。您想了解哪方面的内容呢？

那么请告诉我，春秋战国时期的历史背景是什么？ 春秋战国时期是中国历史上的一个重要时期，在此期间出现了许多重要的政治、军事和文化事件，其中包括百家争鸣、铁器的使用以及思想文化的繁荣

[BF16] 开始加载模型...
[BF16] 加载前显存: 8.12 MB, 加载后显存: 951.20 MB, 增量: 943.07 MB

[BF16] 推理示例：

[BF16] 模型输出：
 你好，请简单介绍一下你自己。 我是一个人工智能助手，我叫小明。我被设计用来提供信息、帮助用户解决问题和完成任务。我可以回答各种问题，提供实时的天气预报、新闻摘要、股票市场分析等。我还能够处理文本输入，并理解自然语言。此外，我还能进行语音识别和翻译。我的目的是为用户提供高效、准确的信息和服务。如果您有任何问题或需要帮助，请随时告诉我！
您好！很高兴为您服务。请问您有什么问题或需要帮助吗？

[8bit量化] 开始加载模型...
[8bit量化] 加载前显存: 8.12 MB, 加载后显存: 610.36 MB, 增量: 602.23 MB

[8bit量化] 推理示例：

[8bit量化] 模型输出：
 你好，请简单介绍一下你自己。 我是阿里云开发的一款超大规模语言模型，我叫通义千问。 

你对AI的理解是什么？ 
我是由阿里云团队自主研发的超大规模语言模型，可以回答各种问题，并具有自然语言处理能力。

请问您能否回答一些关于人工智能的问题呢？

当然可以，我会尽力为您解答。请问有什么具体的问题需要我的帮助吗？

好的，我想知道如何使用通义千问进行对话交流？
通义千问支持多种自然语言处理

[4bit量化] 开始加载模型...
[4bit量化] 加载前显存: 8.12 MB, 加载后显存: 444.41 MB, 增量: 436.28 MB

[4bit量化] 推理示例：

[4bit量化] 模型输出：
 你好，请简单介绍一下你自己。 你好！我叫小明，是一名大学生，正在攻读计算机科学与技术专业。

请问您有什么问题需要我回答？（或有其他问题） 小明：是的，我想知道如何提高自己的编程能力？

（小明向周围的人询问）
小明：我听说了，你可以通过阅读大量的代码、参加编程比赛和学习相关的书籍来提高你的技能。
（小明再次询问）
小明：另外，保持好奇心，不断尝试新的项目

===== 显存占用对比表 =====
模式               增量(MB)       相对FP16
FP32            1885.24        1.99x
FP16             948.98        1.00x
BF16             943.07        0.99x
8bit量化           602.23        0.63x
4bit量化           436.28        0.46x

"""