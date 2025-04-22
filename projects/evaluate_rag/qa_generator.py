import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import openai
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# 加载环境变量
load_dotenv()

class QAGenerator:
    """问答对生成器类"""
    
    def __init__(self, 
                 model_name: str = "THUDM/GLM-4-32B-0414", 
                 embedding_model: str = "BAAI/bge-m3",
                 api_key: str = None,
                 base_url: str = None,
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200):
        """
        初始化问答对生成器
        
        Args:
            model_name: OpenAI模型名称
            embedding_model: 用于文本嵌入的模型
            api_key: OpenAI API密钥
            base_url: OpenAI API基础URL
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        # 设置API参数
        self.model_name = model_name
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        
        # 文本分块参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 加载Sentence Transformer模型
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"成功加载嵌入模型: {embedding_model}")
        except Exception as e:
            print(f"加载嵌入模型时出错: {e}")
            print("继续执行，但嵌入功能将不可用")
            self.embedding_model = None
    
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        调用大语言模型生成文本
        
        Args:
            prompt: 输入提示
            temperature: 生成温度
            max_tokens: 最大生成标记数
            
        Returns:
            生成的文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                return response.choices[0].message.content.strip()
            else:
                print("警告: API返回了空响应或无效结构")
                return ""
        except Exception as e:
            print(f"调用API时发生错误: {e}")
            return ""
    
    def load_dataset(self, dataset_path: str = None, dataset_name: str = None) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Args:
            dataset_path: 本地数据集路径
            dataset_name: Hugging Face数据集名称
            
        Returns:
            文档列表
        """
        documents = []
        
        if dataset_path and os.path.exists(dataset_path):
            # 加载本地数据集
            try:
                if dataset_path.endswith('.json'):
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            if isinstance(item, dict) and 'text' in item:
                                documents.append(item)
                elif dataset_path.endswith('.txt'):
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents.append({"text": text, "source": dataset_path})
                elif dataset_path.endswith('.csv'):
                    df = pd.read_csv(dataset_path)
                    if 'text' in df.columns:
                        for _, row in df.iterrows():
                            documents.append({
                                "text": row['text'],
                                "source": dataset_path
                            })
            except Exception as e:
                print(f"加载本地数据集时出错: {e}")
        elif dataset_name:
            # 从Hugging Face加载数据集
            try:
                ds = load_dataset(dataset_name, split="train")
                for item in ds:
                    if 'text' in item:
                        documents.append({
                            "text": item['text'],
                            "source": item.get('source', dataset_name)
                        })
            except Exception as e:
                print(f"加载Hugging Face数据集时出错: {e}")
        
        print(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        # 简单的分块逻辑，优先按段落分割
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            # 如果当前块加上这个段落仍然小于chunk_size，就继续添加
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # 如果当前块已经达到chunk_size，保存该块并开始一个新的块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果一个段落超过了chunk_size，我们需要进一步分割它
                if len(paragraph) > self.chunk_size:
                    sentences = paragraph.split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            # 如果单个句子超过chunk_size，我们按单词分割
                            if len(sentence) > self.chunk_size:
                                words = sentence.split(' ')
                                current_chunk = ""
                                for word in words:
                                    if len(current_chunk) + len(word) <= self.chunk_size:
                                        current_chunk += word + " "
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        current_chunk = word + " "
                            else:
                                current_chunk = sentence + ". "
                else:
                    current_chunk = paragraph + "\n\n"
        
        # 不要忘记最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理文档集合，进行分块
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的文档块列表
        """
        processed_docs = []
        for doc in tqdm(documents, desc="处理文档"):
            text = doc["text"]
            source = doc.get("source", "未知来源")
            
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    "page_content": chunk,
                    "metadata": {
                        "source": source,
                        "chunk_id": i,
                        "start_index": text.find(chunk) if chunk in text else -1
                    }
                })
        
        # 去重
        unique_texts = {}
        unique_docs = []
        for doc in processed_docs:
            content = doc["page_content"]
            if content not in unique_texts:
                unique_texts[content] = True
                unique_docs.append(doc)
        
        print(f"处理后得到 {len(unique_docs)} 个唯一文档块")
        return unique_docs

    def generate_qa_prompt(self, context: str) -> str:
        """
        生成用于创建问答对的提示
        
        Args:
            context: 上下文文本
            
        Returns:
            格式化的提示字符串
        """
        prompt = """
以下是中文版指示：
你的任务是根据给定的上下文编写一个事实性问题及其答案。
你的事实性问题应当能够通过上下文中的具体、简洁的事实信息来回答。
你的事实性问题应当以用户在搜索引擎中提问的风格来表述。
这意味着你的事实性问题不能包含"根据文章"或"上下文"等表述。

请按照以下格式提供你的回答：

Output:::
Factoid question: (你的事实性问题)
Answer: (你对该事实性问题的回答)

以下是上下文：

Context: {context}
Output:::"""
        
        return prompt.format(context=context)
    
    def generate_qa_pairs(self, processed_docs: List[Dict[str, Any]], 
                         n_generations: int = 10) -> List[Dict[str, Any]]:
        """
        从处理后的文档中生成问答对
        
        Args:
            processed_docs: 处理后的文档列表
            n_generations: 要生成的问答对数量
            
        Returns:
            生成的问答对列表
        """
        outputs = []
        
        # 随机选择n_generations个文档
        selected_docs = random.sample(processed_docs, min(n_generations, len(processed_docs)))
        
        print(f"正在生成 {len(selected_docs)} 个问答对...")
        for doc in tqdm(selected_docs, desc="生成问答对"):
            # 生成QA对
            prompt = self.generate_qa_prompt(doc["page_content"])
            output_qa_couple = self.call_llm(prompt)
            
            try:
                question = output_qa_couple.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
                answer = output_qa_couple.split("Answer: ")[-1].strip()
                
                # 验证答案不要太长
                if len(answer) < 300:
                    outputs.append({
                        "context": doc["page_content"],
                        "question": question,
                        "answer": answer,
                        "source_doc": doc["metadata"]["source"],
                    })
                else:
                    print(f"跳过过长的答案: {answer[:50]}...")
            except Exception as e:
                print(f"解析QA对失败: {e}")
                continue
                
        return outputs
    
    def critique_prompt(self, question: str, context: str = None, criterion: str = "standalone") -> str:
        """
        生成用于评价问题质量的提示
        
        Args:
            question: 要评价的问题
            context: 问题的上下文（仅在groundedness评价中需要）
            criterion: 评价标准（groundedness、relevance或standalone）
            
        Returns:
            格式化的提示字符串
        """
        if criterion == "groundedness" and context:
            prompt = """
你将得到一个上下文和一个问题。
你的任务是提供一个"总评分"，评估在给定上下文的情况下，问题能在多大程度上得到明确的回答。
请按1到5的等级评分，其中1表示根据上下文完全无法回答该问题，5表示根据上下文可以清晰明确地回答该问题。

请按以下格式提供你的回答：

回答:::
评价理由: (你给出评分的理由，文本形式)
总评分: (你的评分，1到5之间的数字)

你必须在回答中提供"评价理由:"和"总评分:"的值。

以下是问题和上下文。

问题: {question}
上下文: {context}
回答::: """
            return prompt.format(question=question, context=context)
        
        elif criterion == "relevance":
            prompt = """
你将得到一个问题。
你的任务是提供一个"总评分"，代表这个问题对于使用Hugging Face生态系统构建NLP应用的机器学习开发者有多大用处。
请按1到5的等级评分，其中1表示该问题完全没有用处，5表示该问题非常有用。

请按以下格式提供你的回答：

回答:::
评价理由: (你给出评分的理由，文本形式)
总评分: (你的评分，1到5之间的数字)

你必须在回答中提供"评价理由:"和"总评分:"的值。

以下是问题。

问题: {question}
回答::: """
            return prompt.format(question=question)
        
        else:  # standalone
            prompt = """
你将得到一个问题。
你的任务是提供一个"总评分"，代表这个问题在多大程度上是独立于上下文的。
请按1到5的等级评分，其中1表示该问题需要依赖额外信息才能被理解，5表示该问题本身就有意义。
例如，如果问题提到了特定的环境，如"在上下文中"或"在文档中"，评分必须为1。
问题可以包含像Gradio、Hub、Hugging Face或Space这样晦涩的技术名词或缩写，并且仍然可以评为5分：只要对于一个能够查阅文档的操作员来说，问题的含义是清晰的即可。

例如，"ViT模型是从哪个检查点导入的？"应评为1分，因为它隐含地提到了一个上下文，因此问题并非独立于上下文。

请按以下格式提供你的回答：

回答:::
评价理由: (你给出评分的理由，文本形式)
总评分: (你的评分，1到5之间的数字)

你必须在回答中提供"评价理由:"和"总评分:"的值。

以下是问题。

问题: {question}
回答::: """
            return prompt.format(question=question)
    
    def evaluate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估问答对的质量
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            带有评估结果的问答对列表
        """
        print("正在评估问答对质量...")
        for qa_pair in tqdm(qa_pairs, desc="评估问答对"):
            # 对每个问答对进行三个维度的评估
            try:
                # 1. Groundedness - 问题是否可以从上下文中回答
                groundedness_prompt = self.critique_prompt(
                    qa_pair["question"], qa_pair["context"], "groundedness"
                )
                groundedness_eval = self.call_llm(groundedness_prompt)
                
                # 2. Relevance - 问题对用户是否有用
                relevance_prompt = self.critique_prompt(
                    qa_pair["question"], criterion="relevance"
                )
                relevance_eval = self.call_llm(relevance_prompt)
                
                # 3. Standalone - 问题是否可以独立理解
                standalone_prompt = self.critique_prompt(
                    qa_pair["question"], criterion="standalone"
                )
                standalone_eval = self.call_llm(standalone_prompt)
                
                # 解析评估结果
                for eval_name, eval_text in [
                    ("groundedness", groundedness_eval),
                    ("relevance", relevance_eval),
                    ("standalone", standalone_eval)
                ]:
                    try:
                        if "总评分:" in eval_text:
                            score = int(eval_text.split("总评分:")[-1].strip())
                            eval_content = eval_text.split("总评分:")[0].split("评价理由:")[-1].strip()
                        elif "Total rating:" in eval_text:
                            score = int(eval_text.split("Total rating:")[-1].strip())
                            eval_content = eval_text.split("Total rating:")[-2].split("Evaluation:")[-1].strip()
                        else:
                            # 尝试从最后几个字符中提取分数
                            for i in range(1, 6):
                                if str(i) in eval_text[-5:]:
                                    score = i
                                    eval_content = eval_text.replace(str(i), "").strip()
                                    break
                            else:
                                score = 3  # 默认中等评分
                                eval_content = eval_text
                        
                        qa_pair[f"{eval_name}_score"] = score
                        qa_pair[f"{eval_name}_eval"] = eval_content
                    except Exception as e:
                        print(f"解析{eval_name}评估时出错: {e}")
                        qa_pair[f"{eval_name}_score"] = 3  # 默认中等评分
                        qa_pair[f"{eval_name}_eval"] = "评估解析失败"
                        
            except Exception as e:
                print(f"评估问答对时出错: {e}")
                continue
                
        return qa_pairs
    
    def save_results(self, qa_pairs: List[Dict[str, Any]], output_file: str = "qa_pairs.json") -> None:
        """
        保存生成的问答对到文件
        
        Args:
            qa_pairs: 问答对列表
            output_file: 输出文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
            print(f"成功保存 {len(qa_pairs)} 个问答对到 {output_file}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
            # 尝试保存到备用位置
            try:
                backup_file = "qa_pairs_backup.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
                print(f"已保存备份到 {backup_file}")
            except:
                pass

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成问答对')
    parser.add_argument('--model', type=str, default="THUDM/GLM-4-32B-0414", help='OpenAI模型名称')
    parser.add_argument('--embedding_model', type=str, default="BAAI/bge-m3", help='嵌入模型名称')
    parser.add_argument('--api_key', type=str, help='OpenAI API密钥')
    parser.add_argument('--base_url', type=str, help='OpenAI API基础URL')
    parser.add_argument('--dataset_path', type=str, help='本地数据集路径')
    parser.add_argument('--dataset_name', type=str, default="m-ric/huggingface_doc", help='Hugging Face数据集名称')
    parser.add_argument('--n_generations', type=int, default=10, help='要生成的问答对数量')
    parser.add_argument('--output_file', type=str, default="qa_pairs.json", help='输出文件路径')
    parser.add_argument('--chunk_size', type=int, default=2000, help='文本块大小')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='文本块重叠大小')
    parser.add_argument('--skip_evaluation', action='store_true', help='跳过评估步骤')
    
    args = parser.parse_args()
    
    # 初始化问答对生成器
    qa_generator = QAGenerator(
        model_name=args.model,
        embedding_model=args.embedding_model,
        api_key=args.api_key,
        base_url=args.base_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # 加载数据集
    documents = qa_generator.load_dataset(args.dataset_path, args.dataset_name)
    
    # 处理文档
    processed_docs = qa_generator.process_documents(documents)
    
    # 生成问答对
    qa_pairs = qa_generator.generate_qa_pairs(
        processed_docs, 
        n_generations=args.n_generations
    )
    
    # 评估问答对质量
    if not args.skip_evaluation:
        qa_pairs = qa_generator.evaluate_qa_pairs(qa_pairs)
    
    # 保存结果
    qa_generator.save_results(qa_pairs, args.output_file)
    
    # 打印简短摘要
    print("\n生成的问答对摘要:")
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n--- 问答对 {i} ---")
        print(f"问题: {qa['question']}")
        print(f"答案: {qa['answer']}")
        if not args.skip_evaluation:
            print(f"独立性评分: {qa.get('standalone_score', 'N/A')}/5")
            print(f"相关性评分: {qa.get('relevance_score', 'N/A')}/5")
            print(f"扎根性评分: {qa.get('groundedness_score', 'N/A')}/5")
    
    if len(qa_pairs) > 3:
        print(f"\n... 还有 {len(qa_pairs) - 3} 个问答对 ...")

if __name__ == "__main__":
    main()
