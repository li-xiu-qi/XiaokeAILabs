"""
多模态模型对图像进行分析，生成标题和描述的异步工具类
"""
import os
import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Union, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import base64 
import re 
import aiofiles # 导入 aiofiles
import tempfile
import requests
from .prompts import MULTIMODAL_PROMPT # 导入提示词模板

load_dotenv()

# 辅助函数：将图片转换为 base64 (使用 aiofiles 实现真正的异步)
async def image_to_base64_async(file_path: str) -> str:
    """将图像文件异步转换为 base64 编码的字符串."""
    try:
        async with aiofiles.open(file_path, "rb") as image_file:
            image_bytes = await image_file.read()
            return base64.b64encode(image_bytes).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"图片文件未找到: {file_path}")
        raise
    except Exception as e:
        logging.error(f"将图片 {file_path} 转换为 base64 时出错: {e}")
        raise

# 辅助函数：从文本中提取标题和描述（自然语言格式）
def extract_title_and_description(text: str) -> Dict[str, Any]:
    """从字符串中提取标题和描述，先用分割符提取包裹内容，再用简单分割和startswith判断。"""
    if not text:
        logging.warning("收到用于标题/描述提取的空文本。")
        return {"title": "错误", "description": "模型返回空响应。", "error": "空响应"}

    # 先用分割符提取包裹内容
    start_flag = "【图片分析开始】"
    end_flag = "【图片分析结束】"
    if start_flag in text and end_flag in text:
        content = text.split(start_flag, 1)[1].split(end_flag, 1)[0].strip()
    else:
        logging.warning("未找到图片分析包裹标记，直接处理原始文本。")
        content = text.strip()

    title = None
    description = None
    for line in content.splitlines():
        line = line.strip()
        if not title and (line.startswith("标题：") or line.lower().startswith("title:")):
            title = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            title = title.strip(' "')
        elif not description and (line.startswith("描述：") or line.lower().startswith("description:")):
            description = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            description = description.strip(' "')

    # 兜底：如果只有一句话，且很短，直接当标题
    if not title and content and len(content) <= 15:
        title = content
    if not description and content and len(content) <= 60:
        description = content

    if not title:
        title = "未提取到标题"
    if not description:
        description = "未提取到描述"

    return {"title": title, "description": description}

class AsyncImageAnalysis:
    """
    异步图像文本提取器类，用于将图像内容转换为文本描述和标题。

    该类使用OpenAI的多模态模型异步分析图像内容，生成描述性文本和标题。
    支持多种API提供商：GUIJI、ZHIPU、VOLCES等
    """

    # 预定义的配置
    PROVIDER_CONFIGS = {
        "guiji": {
            "api_key_env": "GUIJI_API_KEY", # API密钥的环境变量名
            "base_url_env": "GUIJI_BASE_URL", # 基础URL的环境变量名
            "model_env": "GUIJI_VISION_MODEL", # 视觉模型的环境变量名
            "default_models": [ "Pro/Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct",] # 默认模型列表
        },
        "zhipu": {
            "api_key_env": "ZHIPU_API_KEY",
            "base_url_env": "ZHIPU_BASE_URL",
            "model_env": "ZHIPU_VISION_MODEL", 
            "default_models": ["glm-4v-flash", "glm-4v"]
        },
        "volces": {
            "api_key_env": "VOLCES_API_KEY",
            "base_url_env": "VOLCES_BASE_URL",
            "model_env": "VOLCES_VISION_MODEL",
            "default_models": ["doubao-1.5-vision-lite-250315", "doubao-1.5-vision-pro-250328"]
        },
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_API_BASE",
            "model_env": "OPENAI_VISION_MODEL",
            "default_models": ["gpt-4-vision-preview", "gpt-4o"]
        }
    }

    def __init__(
        self,
        provider: str = "zhipu",  # 默认使用智谱
        api_key: str = None,
        base_url: str = None,
        vision_model: str = None,
        prompt: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """
        初始化图像分析器
        
        参数:
            provider (str): API提供商，支持 'guiji', 'zhipu', 'volces', 'openai'
            api_key (str, optional): API密钥，如果不提供则从环境变量读取
            base_url (str, optional): API基础URL，如果不提供则从环境变量读取
            vision_model (str, optional): 视觉模型名称，如果不提供则从环境变量或默认值读取
            prompt (Optional[str], optional): 自定义提示词
            max_concurrent (int): 最大并发数
        """
        self.provider = provider.lower()
        
        if self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"不支持的提供商: {provider}. 支持的提供商: {list(self.PROVIDER_CONFIGS.keys())}")
        
        config = self.PROVIDER_CONFIGS[self.provider]
        
        # 获取API密钥
        self.api_key = api_key or os.getenv(config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"API密钥未提供，请设置 {config['api_key_env']} 环境变量，或传入api_key参数。")

        # 获取基础URL
        self.base_url = base_url or os.getenv(config["base_url_env"])
        if not self.base_url:
            raise ValueError(f"基础URL未提供，请设置 {config['base_url_env']} 环境变量，或传入base_url参数。")
        
        # 获取视觉模型
        self.vision_model = (vision_model or 
                           os.getenv(config["model_env"]) or 
                           config["default_models"][0])
        
        print(f"使用提供商: {self.provider}")
        print(f"API基础URL: {self.base_url}")
        print(f"视觉模型: {self.vision_model}")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 设置提示词
        self._prompt = prompt or MULTIMODAL_PROMPT
        
        # 设置并发限制
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        """进入异步上下文."""
        # 客户端已在 __init__ 中初始化。
        # 如果客户端本身是异步上下文管理器，可以在此处调用其 __aenter__，
        # 但对于 AsyncOpenAI，通常通过在退出时调用 close() 来管理。
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文并关闭客户端."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()

    async def analyze_image(
        self,
        image_url: str = None,
        local_image_path: str = None,
        model: str = None,
        detail: str = "low", # 图像细节级别: 'low' 或 'high'
        prompt: str = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        异步分析图像并返回描述信息。

        参数:
            image_url (str, optional): 在线图片URL
            local_image_path (str, optional): 本地图片路径
            model (str, optional): 使用的视觉模型，默认使用实例的默认模型
            detail (str): 图像细节级别，'low'或'high'
            prompt (str, optional): 自定义提示词
            temperature (float): 模型温度参数

        返回:
            Dict[str, Any]: 包含title和description的字典
        """
        async with self.semaphore:  # 限制并发
            # 基本参数检查
            if not image_url and not local_image_path:
                raise ValueError("必须提供一个图像来源：image_url或local_image_path")
            if image_url and local_image_path:
                raise ValueError("只能提供一个图像来源：image_url或local_image_path")

            # 处理图像来源
            final_image_url = image_url
            image_format = "jpeg"  # 默认格式
            if local_image_path:
                try:
                    from PIL import Image # 条件导入
                    loop = asyncio.get_event_loop()
                    def get_image_format():
                        with Image.open(local_image_path) as img:
                            return img.format.lower() if img.format else "jpeg"
                    image_format = await loop.run_in_executor(None, get_image_format)
                except ImportError:
                    logging.warning(f"Pillow (PIL) 库未找到。无法确定图片 {local_image_path} 的格式，将使用默认jpeg。请安装Pillow以获得更准确的格式检测。")
                except Exception as e:
                    logging.warning(f"无法打开或识别图片格式 {local_image_path}: {e}, 使用默认jpeg")
                base64_image = await image_to_base64_async(local_image_path)
                final_image_url = f"data:image/{image_format};base64,{base64_image}"

            model_to_use = model or self.vision_model
            prompt_text = prompt or self._prompt
            try:
                response = await self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": final_image_url, "detail": detail},
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=300,
                )
                result_content = response.choices[0].message.content
                analysis_result = extract_title_and_description(result_content)
                return analysis_result
            except Exception as e:
                logging.error(f"API调用失败: {e}")
                print(f"❌ 图像分析失败，URL: {final_image_url}")
                print(f"   错误详情: {str(e)}")
                print(f"   使用模型: {model_to_use}")
                return {"error": f"API调用失败: {str(e)}", "title": "", "description": ""}
            # 已无临时文件，无需finally清理

    async def analyze_multiple_images(
        self,
        image_sources: List[Dict[str, str]], 
        model: str = None,
        detail: str = "low",
        prompt: str = None,
        temperature: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        异步分析多个图像并返回其描述。

        参数:
            image_sources (List[Dict[str, str]]): 字典列表，
                每个字典指定一个 image_url 或 local_image_path。
                示例: [{\"image_url\": \"http://...\"}, {\"local_image_path\": \"/path/...\"}]
            model (str, optional): 用于这些分析的特定模型。
            detail (str, optional): 图像分析的详细程度。
            prompt (str, optional): 用于这些分析的特定提示。
            temperature (float, optional): 模型的温度参数。

        返回:
            List[Dict[str, Any]]: 分析结果或错误字典的列表。
        """
        tasks = []
        for image_source in image_sources:
            task = self.analyze_image(
                image_url=image_source.get("image_url"),
                local_image_path=image_source.get("local_image_path"),
                model=model,
                detail=detail,
                prompt=prompt,
                temperature=temperature,
            )
            tasks.append(task)
        
        # Gather results, allowing individual tasks to fail without stopping others
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, res_or_exc in enumerate(results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                source_info = image_sources[i].get("image_url") or image_sources[i].get("local_image_path", "Unknown source")
                logging.error(f"Error analyzing image {source_info}: {res_or_exc}")
                processed_results.append({
                    "error": str(res_or_exc), 
                    "title": "Error", 
                    "description": f"Failed to analyze image: {source_info}"
                })
            elif isinstance(res_or_exc, dict):
                processed_results.append(res_or_exc)
            else: # Should not happen if analyze_image returns dict or raises Exception
                source_info = image_sources[i].get("image_url") or image_sources[i].get("local_image_path", "Unknown source")
                logging.warning(f"Unexpected result type for image {source_info}: {type(res_or_exc)}")
                processed_results.append({
                    "error": "Unexpected result type",
                    "title": "Error",
                    "description": f"Unexpected result from analyzing image: {source_info}"
                })
                
        return processed_results

# 主程序执行区域
