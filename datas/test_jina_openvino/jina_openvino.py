import openvino as ov
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from typing import Tuple, Union, Optional, List
import logging

class JinaClipOpenVINO:
    """Jina CLIP OpenVINO推理类"""
    
    def __init__(self, model_path: Union[str, Path] = "./jina_model_path", device: str = "CPU"):
        """
        初始化JinaClipOpenVINO实例
        
        Args:
            model_path: 模型文件夹路径
            device: 推理设备 (CPU/GPU)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.core = ov.Core()
        
        # 延迟加载模型和预处理器
        self._compiled_text_model = None
        self._compiled_vision_model = None
        self._tokenizer = None
        self._image_processor = None
        
        # SentenceTransformers兼容属性
        self.max_seq_length = 512
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """懒加载模型和预处理器"""
        if self._compiled_text_model is None or self._compiled_vision_model is None:
            try:
                # 模型路径
                text_model_path = self.model_path / "jina-clip-text_v2_fp16.xml"
                vision_model_path = self.model_path / "jina-clip-vision_v2_fp16.xml"
                
                if not text_model_path.exists() or not vision_model_path.exists():
                    raise FileNotFoundError(f"模型文件不存在: {text_model_path} 或 {vision_model_path}")
                
                # 编译模型
                self._compiled_text_model = self.core.compile_model(text_model_path, device_name=self.device)
                self._compiled_vision_model = self.core.compile_model(vision_model_path, device_name=self.device)
                
                self.logger.info(f"模型已成功加载到 {self.device} 设备")
                
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise
        
        if self._tokenizer is None or self._image_processor is None:
            try:
                # 加载预处理器
                preprocess_path = self.model_path 
                if not preprocess_path.exists():
                    raise FileNotFoundError(f"预处理器路径不存在: {preprocess_path}")
                
                self._tokenizer = AutoTokenizer.from_pretrained(preprocess_path, trust_remote_code=True)
                self._image_processor = AutoImageProcessor.from_pretrained(preprocess_path, trust_remote_code=True)
                
                self.logger.info("预处理器已成功加载")
                
            except Exception as e:
                self.logger.error(f"预处理器加载失败: {e}")
                raise
    
    def _preprocess_text(self, text: str) -> dict:
        """预处理文本"""
        texts = [text] if isinstance(text, str) else text
        tokenizer_kwargs = {
            "padding": "max_length",
            "max_length": 512,
            "truncation": True
        }
        
        text_inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            **tokenizer_kwargs,
        ).to("cpu")
        
        return text_inputs
    
    def _preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> dict:
        """预处理图像"""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("图像输入必须是文件路径或PIL Image对象")
        
        images = [image]
        vision_inputs = self._image_processor(images=images, return_tensors="pt")
        return vision_inputs
    
    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """L2归一化特征向量"""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norm + 1e-8)  # 添加小常数防止除零
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为特征向量"""
        self._load_models()
        
        try:
            text_inputs = self._preprocess_text(text)
            text_ov_res = self._compiled_text_model(text_inputs["input_ids"])
            text_features = list(text_ov_res.values())[0]
            return self._normalize_features(text_features)
        
        except Exception as e:
            self.logger.error(f"文本编码失败: {e}")
            raise
    
    def encode_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """编码图像为特征向量"""
        self._load_models()
        
        try:
            vision_inputs = self._preprocess_image(image_input)
            vis_ov_res = self._compiled_vision_model(vision_inputs["pixel_values"])
            vision_features = list(vis_ov_res.values())[0]
            return self._normalize_features(vision_features)
        
        except Exception as e:
            self.logger.error(f"图像编码失败: {e}")
            raise
    
    def encode(self, sentences: Union[str, List[str], List[Union[str, Path, Image.Image]]], 
               batch_size: int = 32, show_progress_bar: bool = True, 
               convert_to_numpy: bool = True, convert_to_tensor: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        SentenceTransformers兼容的encode方法
        
        Args:
            sentences: 输入文本、图像路径或PIL图像列表
            其他参数为兼容性参数
            
        Returns:
            特征向量数组
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        for sentence in sentences:
            if isinstance(sentence, str):
                # 判断是否为图像路径
                if isinstance(sentence, str) and sentence.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    embedding = self.encode_image(sentence)
                else:
                    embedding = self.encode_text(sentence)
            elif isinstance(sentence, (Path, Image.Image)):
                embedding = self.encode_image(sentence)
            else:
                embedding = self.encode_text(str(sentence))
            
            embeddings.append(embedding)
        
        embeddings = np.vstack(embeddings)
        
        if convert_to_tensor:
            import torch
            return torch.from_numpy(embeddings)
        
        return embeddings
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """SentenceTransformers兼容的similarity方法"""
        # 确保输入是归一化的
        embeddings1 = self._normalize_features(embeddings1)
        embeddings2 = self._normalize_features(embeddings2)
        return np.dot(embeddings1, embeddings2.T)
    
    def calculate_similarity(self, text: str, image_input: Union[str, Path, Image.Image]) -> float:
        """计算文本和图像的相似度"""
        try:
            text_features = self.encode_text(text)
            vision_features = self.encode_image(image_input)
            
            # 计算余弦相似度
            similarity = np.dot(text_features, vision_features.T)[0, 0]
            return float(similarity)
        
        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            raise
    
    def batch_encode_text(self, texts: list) -> np.ndarray:
        """批量编码文本"""
        features_list = []
        for text in texts:
            features = self.encode_text(text)
            features_list.append(features)
        return np.vstack(features_list)
    
    def batch_encode_image(self, image_inputs: list) -> np.ndarray:
        """批量编码图像"""
        features_list = []
        for image_input in image_inputs:
            features = self.encode_image(image_input)
            features_list.append(features)
        return np.vstack(features_list)

# 兼容性函数，保持向后兼容
def calculate_similarity(text: str, image_path: Union[str, Path]) -> Tuple[float, np.ndarray, np.ndarray]:
    """向后兼容的相似度计算函数"""
    clip = JinaClipOpenVINO()
    similarity = clip.calculate_similarity(text, image_path)
    text_features = clip.encode_text(text)
    vision_features = clip.encode_image(image_path)
    return similarity, text_features, vision_features

def main():
    """示例用法"""
    # 创建JinaClipOpenVINO实例
    clip = JinaClipOpenVINO()
    
    text = "一只小海豹"
    image_path = "./image.png"
    
    try:
        # 原有用法
        similarity = clip.calculate_similarity(text, image_path)
        print(f"余弦相似度: {similarity:.4f}")
        
        # SentenceTransformers兼容用法
        texts = ["一只小海豹", "一只小狗"]
        text_embeddings = clip.encode(texts)
        print(f"文本嵌入形状: {text_embeddings.shape}")
        
        # 混合编码
        mixed_inputs = ["一只小海豹", "./image.png"]
        mixed_embeddings = clip.encode(mixed_inputs)
        print(f"混合嵌入形状: {mixed_embeddings.shape}")
        
        # 相似度矩阵
        similarities = clip.similarity(text_embeddings, mixed_embeddings)
        print(f"相似度矩阵:\n{similarities}")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()
