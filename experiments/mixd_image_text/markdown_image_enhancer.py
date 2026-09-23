"""
Markdown图片分析增强器
输入：普通的Markdown文本
输出：带有AI图片分析描述的Markdown文本

独立模块，可用于其他项目
"""

import re
import asyncio
from typing import List, Dict, Any, Optional
from image_utils.async_image_analysis import AsyncImageAnalysis


class MarkdownImageEnhancer:
    """
    Markdown图片增强器
    可以分析Markdown中的图片并添加AI生成的描述
    """
    
    # 匹配Markdown图片语法的正则表达式
    IMG_TAG_RE = re.compile(r'!\[([^\]]*)\]\((https?://[^\)]+)\)', re.IGNORECASE)
    
    def __init__(self, 
                 provider: str = "zhipu",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 max_concurrent: int = 10):
        """
        初始化图片增强器
        
        :param provider: 图片分析API提供商
        :param api_key: API密钥
        :param base_url: API基础URL
        :param vision_model: 视觉模型名称
        :param max_concurrent: 最大并发数
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.vision_model = vision_model
        self.max_concurrent = max_concurrent
    
    def extract_img_urls_with_alt(self, markdown: str) -> List[Dict[str, str]]:
        """
        提取Markdown中所有远程图片URL和对应的alt文本
        
        :param markdown: Markdown文本
        :return: 包含url和alt的字典列表
        """
        matches = self.IMG_TAG_RE.findall(markdown)
        img_info = []
        seen_urls = set()
        
        for alt_text, url in matches:
            if url not in seen_urls:
                img_info.append({
                    'url': url,
                    'alt': alt_text
                })
                seen_urls.add(url)
        
        return img_info
    
    async def analyze_images_batch(self, img_info_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量分析图片
        
        :param img_info_list: 图片信息列表
        :return: 分析结果列表
        """
        if not img_info_list:
            return []
        
        async with AsyncImageAnalysis(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            vision_model=self.vision_model,
            max_concurrent=self.max_concurrent,
        ) as analyzer:
            image_sources = [{"image_url": info['url']} for info in img_info_list]
            results = await analyzer.analyze_multiple_images(image_sources)
            return results
    
    def replace_img_with_analysis(self, markdown: str, img_info_list: List[Dict[str, str]], 
                                 analysis_results: List[Dict[str, Any]]) -> str:
        """
        替换Markdown中的图片为带AI描述的版本
        
        :param markdown: 原始Markdown文本
        :param img_info_list: 图片信息列表
        :param analysis_results: AI分析结果列表
        :return: 增强后的Markdown文本
        """
        def replacement(match):
            original_alt, url = match.groups()
            
            # 查找对应的分析结果
            for i, img_info in enumerate(img_info_list):
                if img_info['url'] == url and i < len(analysis_results):
                    result = analysis_results[i]
                    if result and isinstance(result, dict) and not result.get("error"):
                        # 使用AI分析的标题，如果没有则使用原始alt，最后使用默认值
                        ai_title = result.get("title", "").strip()
                        title = ai_title or original_alt or "图片"
                        
                        # 获取AI描述
                        ai_desc = result.get("description", "").strip()
                        
                        # 构建新的图片Markdown
                        new_img_md = f"![{title}]({url})"
                        if ai_desc:
                            # 将描述作为引用块添加到图片下方
                            desc_lines = ai_desc.splitlines()
                            formatted_desc = "\n".join(f"> {line}" for line in desc_lines)
                            new_img_md += f"\n{formatted_desc}"
                        
                        return new_img_md
            
            # 如果没有找到分析结果，返回原始内容
            return match.group(0)
        
        return self.IMG_TAG_RE.sub(replacement, markdown)
    
    async def enhance_markdown_async(self, markdown: str) -> str:
        """
        异步增强Markdown中的图片（推荐使用）
        
        :param markdown: 原始Markdown文本
        :return: 增强后的Markdown文本
        """
        # 提取图片信息
        img_info_list = self.extract_img_urls_with_alt(markdown)
        
        if not img_info_list:
            print("📷 未发现需要分析的图片")
            return markdown
        
        print(f"🔮 开始分析 {len(img_info_list)} 个图片...")
        
        # 批量分析图片
        analysis_results = await self.analyze_images_batch(img_info_list)
        
        print(f"🎯 分析完成，共获得 {len(analysis_results)} 个结果")
        
        # 替换图片
        enhanced_markdown = self.replace_img_with_analysis(markdown, img_info_list, analysis_results)
        
        print("✅ 图片增强完成")
        return enhanced_markdown
    
    def enhance_markdown(self, markdown: str) -> str:
        """
        同步增强Markdown中的图片（内部使用asyncio）
        
        :param markdown: 原始Markdown文本
        :return: 增强后的Markdown文本
        """
        # 检查是否已经在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，需要创建新的循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_in_new_loop, markdown)
                return future.result()
        except RuntimeError:
            # 没有运行的事件循环，直接创建新的
            return asyncio.run(self.enhance_markdown_async(markdown))
    
    def _run_in_new_loop(self, markdown: str) -> str:
        """在新的事件循环中运行异步方法"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.enhance_markdown_async(markdown))
        finally:
            loop.close()


# 便利函数
def enhance_markdown_images(markdown: str,
                          provider: str = "zhipu",
                          api_key: Optional[str] = None,
                          base_url: Optional[str] = None,
                          vision_model: Optional[str] = None,
                          max_concurrent: int = 10) -> str:
    """
    便利函数：增强Markdown中的图片
    
    :param markdown: 原始Markdown文本
    :param provider: 图片分析API提供商
    :param api_key: API密钥
    :param base_url: API基础URL
    :param vision_model: 视觉模型名称
    :param max_concurrent: 最大并发数
    :return: 增强后的Markdown文本
    """
    enhancer = MarkdownImageEnhancer(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        vision_model=vision_model,
        max_concurrent=max_concurrent
    )
    return enhancer.enhance_markdown(markdown)


async def enhance_markdown_images_async(markdown: str,
                                       provider: str = "zhipu",
                                       api_key: Optional[str] = None,
                                       base_url: Optional[str] = None,
                                       vision_model: Optional[str] = None,
                                       max_concurrent: int = 10) -> str:
    """
    异步便利函数：增强Markdown中的图片
    
    :param markdown: 原始Markdown文本
    :param provider: 图片分析API提供商
    :param api_key: API密钥
    :param base_url: API基础URL
    :param vision_model: 视觉模型名称
    :param max_concurrent: 最大并发数
    :return: 增强后的Markdown文本
    """
    enhancer = MarkdownImageEnhancer(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        vision_model=vision_model,
        max_concurrent=max_concurrent
    )
    return await enhancer.enhance_markdown_async(markdown)


