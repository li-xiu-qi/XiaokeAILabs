# 作者: 筱可
# 日期: 2025年5月24日
# 版权所有 (c) 2025 筱可 & 筱可AI研习社. 保留所有权利.
# 使用 transformers 库复现 BGE-M3 稀疏向量生成

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from typing import Dict, List, Union, Any
import torch.nn.functional as F

class BGEM3SparseEmbedder:
    """
    使用 transformers 库复现 BGE-M3 的稀疏向量生成
    """
    
    def __init__(self, model_name_or_path: str, use_fp16: bool = True):
        """
        初始化稀疏嵌入器
        
        Args:
            model_name_or_path: 模型路径或名称
            use_fp16: 是否使用半精度浮点数
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 获取模型配置
        self.config = self.model.config
        self.vocab_size = self.config.vocab_size
        
        # 移动到设备并设置精度
        self.model.to(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.model.half()
        
        # 设置为评估模式
        self.model.eval()
        
        # 初始化稀疏嵌入层（模拟BGE-M3的实现）
        self._init_sparse_embedding_layer()
        
        # 获取特殊token ID用于过滤
        self._get_special_tokens()
    
    def _init_sparse_embedding_layer(self):
        """
        初始化稀疏嵌入层
        BGE-M3使用一个线性层来生成稀疏权重
        """
        hidden_size = self.config.hidden_size
        
        # 创建稀疏权重生成层
        self.sparse_linear = nn.Linear(hidden_size, 1)
        
        # 移动到设备并设置精度
        self.sparse_linear.to(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.sparse_linear.half()
        
        # 初始化权重（简单的随机初始化，实际BGE-M3是训练好的）
        with torch.no_grad():
            nn.init.xavier_uniform_(self.sparse_linear.weight)
            nn.init.zeros_(self.sparse_linear.bias)
    
    def _get_special_tokens(self):
        """
        获取需要过滤的特殊token ID
        """
        self.unused_tokens = set()
        special_token_types = ['cls_token', 'eos_token', 'pad_token', 'unk_token', 'sep_token']
        
        for token_type in special_token_types:
            if hasattr(self.tokenizer, token_type):
                token = getattr(self.tokenizer, token_type)
                if token is not None:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(token_id, int):
                        self.unused_tokens.add(token_id)
        
        # 添加常见的特殊token ID
        if hasattr(self.tokenizer, 'special_tokens_map'):
            for token in self.tokenizer.special_tokens_map.values():
                if isinstance(token, str):
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(token_id, int):
                        self.unused_tokens.add(token_id)
        
        print(f"过滤的特殊token ID: {self.unused_tokens}")
    
    def _sparse_embedding(self, hidden_state: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        生成稀疏嵌入权重
        
        Args:
            hidden_state: 模型的隐藏状态 [batch_size, seq_len, hidden_size]
            input_ids: 输入的token ID [batch_size, seq_len]
        
        Returns:
            稀疏权重 [batch_size, seq_len]
        """
        # 通过线性层生成权重
        sparse_weights = self.sparse_linear(hidden_state).squeeze(-1)  # [batch_size, seq_len]
        
        # 应用ReLU激活函数（确保权重为正）
        sparse_weights = F.relu(sparse_weights)
        
        # 应用log(1 + x)激活（BGE-M3的实现方式）
        sparse_weights = torch.log(1 + sparse_weights)
        
        return sparse_weights
    
    def _process_token_weights(self, token_weights: np.ndarray, input_ids: List[int]) -> Dict[str, float]:
        """
        处理token权重，转换为字典格式并过滤特殊token
        
        Args:
            token_weights: token权重数组
            input_ids: 输入的token ID列表
        
        Returns:
            处理后的权重字典 {token_id: weight}
        """
        result = defaultdict(float)
        
        for weight, token_id in zip(token_weights, input_ids):
            # 过滤特殊token和零权重
            if token_id not in self.unused_tokens and weight > 0:
                token_id_str = str(token_id)
                # 保留最大权重
                if weight > result[token_id_str]:
                    result[token_id_str] = float(weight)
        
        return dict(result)
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 8,
        max_length: int = 512,
        return_token_text_mapping: bool = True
    ) -> Dict[str, Any]:
        """
        对句子进行编码，生成稀疏向量
        
        Args:
            sentences: 输入句子
            batch_size: 批处理大小
            max_length: 最大长度
            return_token_text_mapping: 是否返回token文本映射
        
        Returns:
            包含稀疏权重的字典
        """
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        else:
            input_was_string = False
        
        all_lexical_weights = []
        all_token_text_weights = []
        
        # 分批处理
        for start_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start_idx:start_idx + batch_size]
            
            # tokenize
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # 前向传播获取隐藏状态
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # 使用最后一层隐藏状态
            hidden_states = outputs.last_hidden_state
            
            # 生成稀疏权重
            sparse_weights = self._sparse_embedding(hidden_states, inputs['input_ids'])
            
            # 处理每个样本
            for i in range(len(batch_sentences)):
                # 获取有效长度（排除padding）
                attention_mask = inputs['attention_mask'][i].cpu().numpy()
                valid_length = int(attention_mask.sum())
                
                # 提取该样本的权重和token ID
                sample_weights = sparse_weights[i][:valid_length].cpu().numpy()
                sample_input_ids = inputs['input_ids'][i][:valid_length].cpu().numpy().tolist()
                
                # 处理权重
                lexical_weights = self._process_token_weights(sample_weights, sample_input_ids)
                all_lexical_weights.append(lexical_weights)
                
                # 如果需要，生成token文本映射
                if return_token_text_mapping:
                    token_text_weights = {}
                    for token_id_str, weight in lexical_weights.items():
                        token_text = self.tokenizer.decode([int(token_id_str)])
                        token_text_weights[token_text] = weight
                    all_token_text_weights.append(token_text_weights)
        
        # 如果输入是单个字符串，返回单个结果
        if input_was_string:
            result = {
                'lexical_weights': all_lexical_weights[0],
            }
            if return_token_text_mapping:
                result['token_text_weights'] = all_token_text_weights[0]
        else:
            result = {
                'lexical_weights': all_lexical_weights,
            }
            if return_token_text_mapping:
                result['token_text_weights'] = all_token_text_weights
        
        return result
    
    def convert_id_to_token(self, lexical_weights: Union[Dict[str, float], List[Dict[str, float]]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        将token ID转换为token文本
        
        Args:
            lexical_weights: 词汇权重字典或列表
        
        Returns:
            转换后的权重字典
        """
        def _convert_single(weights_dict):
            result = {}
            for token_id_str, weight in weights_dict.items():
                token_text = self.tokenizer.decode([int(token_id_str)])
                result[token_text] = weight
            return result
        
        if isinstance(lexical_weights, dict):
            return _convert_single(lexical_weights)
        elif isinstance(lexical_weights, list):
            return [_convert_single(weights) for weights in lexical_weights]
        else:
            raise ValueError("输入必须是字典或字典列表")


def demo_sparse_embedding():
    """
    演示稀疏嵌入的用法
    """
    print("=" * 60)
    print("BGE-M3 稀疏向量生成演示 (使用 transformers 库)")
    print("=" * 60)
    
    # 测试句子
    sentences = [
        "这是一个测试句子。",
        "BGE-M3 模型用于处理自然语言。",
        "我们正在演示如何使用 BGE-M3 的词汇权重。",
        "机器学习和人工智能技术正在快速发展。"
    ]
    
    # 模型路径（请根据实际情况修改）
    model_path = r'C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3'
    
    try:
        # 初始化嵌入器
        print("正在加载模型...")
        embedder = BGEM3SparseEmbedder(model_path, use_fp16=True)
        print("模型加载完成！")
        print()
        
        # 生成稀疏向量
        print("正在生成稀疏向量...")
        output = embedder.encode(sentences, return_token_text_mapping=True)
        print("稀疏向量生成完成！")
        print()
        
        # 显示结果
        print("稀疏向量结果:")
        print("-" * 50)
        
        lexical_weights = output['lexical_weights']
        token_text_weights = output['token_text_weights']
        
        for i, sentence in enumerate(sentences):
            print(f"句子 {i+1}: \"{sentence}\"")
            print("Token ID权重:")
            for token_id, weight in lexical_weights[i].items():
                print(f"  ID {token_id} -> 权重: {weight:.6f}")
            
            print("Token文本权重:")
            for token_text, weight in token_text_weights[i].items():
                print(f"  '{token_text}' -> 权重: {weight:.6f}")
            print()
        
        # 单句测试
        print("-" * 50)
        print("单句测试:")
        single_sentence = "人工智能改变世界"
        single_output = embedder.encode(single_sentence, return_token_text_mapping=True)
        
        print(f"句子: \"{single_sentence}\"")
        print("Token ID权重:", single_output['lexical_weights'])
        print("Token文本权重:", single_output['token_text_weights'])
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查模型路径是否正确，或者尝试使用在线模型:")
        print("model_path = 'BAAI/bge-m3'")


def compare_with_flagembedding():
    """
    与FlagEmbedding库的结果进行比较
    """
    print("\n" + "=" * 60)
    print("与 FlagEmbedding 库结果比较")
    print("=" * 60)
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        
        # 测试句子
        test_sentence = "这是一个测试句子。"
        model_path = r'C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3'
        
        # 使用我们的实现
        print("使用 transformers 实现:")
        embedder = BGEM3SparseEmbedder(model_path, use_fp16=True)
        our_output = embedder.encode(test_sentence, return_token_text_mapping=True)
        print("Token文本权重:", our_output['token_text_weights'])
        print()
        
        # 使用FlagEmbedding库
        print("使用 FlagEmbedding 库:")
        flag_model = BGEM3FlagModel(model_path, use_fp16=True)
        flag_output = flag_model.encode([test_sentence], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        
        # 转换FlagEmbedding的结果
        flag_weights = flag_output["lexical_weights"][0]
        flag_text_weights = {}
        for token_id, weight in flag_weights.items():
            token_text = flag_model.tokenizer.decode([int(token_id)])
            flag_text_weights[token_text] = weight
        
        print("Token文本权重:", flag_text_weights)
        print()
        
        # 比较结果
        print("结果比较:")
        print("注意: 由于稀疏层参数不同，权重值会有差异，但token分布应该相似")
        print(f"我们的实现包含 {len(our_output['token_text_weights'])} 个token")
        print(f"FlagEmbedding 包含 {len(flag_text_weights)} 个token")
        
    except ImportError:
        print("未安装 FlagEmbedding 库，跳过比较")
    except Exception as e:
        print(f"比较时出错: {e}")


if __name__ == "__main__":
    # 运行演示
    demo_sparse_embedding()
    
    # 比较结果
    compare_with_flagembedding()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("注意: 这是一个简化的实现，用于理解BGE-M3稀疏向量的生成原理")
    print("实际的BGE-M3模型使用训练好的稀疏层参数，会产生不同的权重值")
    print("=" * 60)