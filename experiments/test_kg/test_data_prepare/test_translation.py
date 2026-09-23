# -*- coding: utf-8 -*-
"""
红楼梦繁体文本小规模测试脚本
只翻译第一章进行测试
"""

import os
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openai import OpenAI

@dataclass
class TextChunk:
    """用于存储文本片段及相关元数据的类。"""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    translated_content: str = ""
    
    def __str__(self) -> str:
        return f"章节: {self.metadata.get('chapter', '未知')}\n内容长度: {len(self.content)}"

def test_translation():
    """测试翻译功能"""
    
    # 智谱API配置
    client = OpenAI(
        api_key="fbe98b321a134d0781e830589066e292.cRduntVv1MTm3w6d",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    # 从红楼梦文本中提取第一章的一小段
    test_text = """第一回　甄士隱夢幻識通靈　賈雨村風塵怀閨秀
-----------------------------------------------------------------
此開卷第一回也．作者自云：因曾歷過一番夢幻之后，故將真事隱去，
而借"通靈"之說，撰此《石頭記》一書也．故曰"甄士隱"云云．但書中所記
何事何人？自又云："今風塵碌碌，一事無成，忽念及當日所有之女子，一
一細考較去，覺其行止見識，皆出于我之上．何我堂堂須眉，誠不若彼裙釵
哉？實愧則有余，悔又無益之大無可如何之日也！"""
    
    print("🧪 开始测试翻译功能...")
    print(f"📝 原文预览:\n{test_text[:100]}...")
    
    try:
        prompt = f"""请将以下繁体中文文本转换为简体中文，保持原文的文学风格和韵味：

{test_text}

要求：
1. 只进行繁体到简体的转换
2. 保持原有的标点符号和格式
3. 保持古典文学的语言风格
4. 不要添加任何解释或注释
5. 直接输出转换后的简体中文文本"""

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一个专业的中文繁简转换助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        print("✅ 翻译测试成功！")
        print(f"📄 翻译结果:\n{translated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ 翻译测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🏮 红楼梦翻译功能测试")
    print("="*40)
    
    if test_translation():
        print("\n🎉 测试通过！可以开始完整翻译。")
    else:
        print("\n❌ 测试失败，请检查API配置。")
