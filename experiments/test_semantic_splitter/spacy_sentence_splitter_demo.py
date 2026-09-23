import os
import spacy

def get_spacy_model():
    global _SPACY_MODEL
    if not hasattr(get_spacy_model, "_SPACY_MODEL"):
        try:
            get_spacy_model._SPACY_MODEL = spacy.load("xx_sent_ud_sm")
            print("已安装模型加载成功!")
        except Exception as e:
            print(f"也无法加载已安装的模型: {e}")
            print("请运行：python -m spacy download xx_sent_ud_sm")
            #抛出安装模型的报错提示
            raise RuntimeError("请确保模型路径正确，并且包含所有必要的模型文件。如果没有安装模型，请运行：python -m spacy download xx_sent_ud_sm")
    return get_spacy_model._SPACY_MODEL

def custom_sentence_splitter(text: str):
    # 获取spaCy模型
    nlp = get_spacy_model()
    nlp.max_length = 3000000  # 增加最大处理长度，如果文本很长

    # 步骤1: 先尝试使用spaCy进行基本分句
    doc = nlp(text)
    spacy_sentences = [sent.text.strip() for sent in doc.sents]

    # 步骤3: 进一步处理，拆分可能的段落（通过换行符）
    final_sentences = []
    for sent in spacy_sentences:
        paragraph_splits = sent.split('\n\n')
        for paragraph in paragraph_splits:
            paragraph = paragraph.strip()
            if paragraph:
                line_splits = paragraph.split('\n')
                for line in line_splits:
                    line = line.strip()
                    if line:
                        final_sentences.append(line)
    return final_sentences

if __name__ == "__main__":
    # 示例文本，可自行替换
    text = (
        
        "你好呀\n-----"
        "这是第一句话。\n"
        "这是第二句话。\n\n"
        "This is the third sentence.\n"
        "这是第四句话。\n"
        "This is the fifth sentence。\n\n"
        "第六句话。"
    )
    print("原始文本：")
    print(text)
    print("\n分句结果：")
    for idx, sent in enumerate(custom_sentence_splitter(text)):
        print(f"{idx+1}: {sent}")
