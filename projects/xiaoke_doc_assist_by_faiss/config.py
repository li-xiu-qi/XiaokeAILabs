#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块名称: config
功能描述: 负责加载环境变量、定义模型列表和全局常量，提供程序运行的基础配置。
作者: 筱可
创建日期: 2025-02-20
版本: 1.0
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# 全局变量声明
DOT_ENV_PATH: str  # .env 文件路径
API_KEY: str  # API 密钥
BASE_URL: str  # API 基础 URL
AUTHOR: str = "筱可"  # 作者名称
WECHAT_PLATFORM: str = "筱可AI研习社"  # 微信公众号名称
CURRENT_DATE: str = datetime.now().strftime("%Y-%m-%d")  # 当前日期，格式为 YYYY-MM-DD
MODEL_LIST: dict  # 模型列表，键为显示名称，值为实际模型标识


class Document:
    def __init__(self,content:str,metadata:dict):
        self.content = content
        self.metadata = metadata




def find_dotenv_path(dir_name: str = "XiaokeAILabs") -> str | None:
    """
    功能描述: 在指定目录中查找 .env 文件的路径。

    参数:
        dir_name (str): 要查找的目标目录名称，默认为 "XiaokeAILabs"

    返回值:
        str | None: .env 文件的完整路径，如果未找到则返回 None
    """
    # 当前工作目录路径
    current_working_dir: str = os.getcwd()
    env_path: str = os.path.join(current_working_dir, ".env")
    if os.path.exists(env_path):
        return env_path

    # 如果未指定目录名且当前目录无 .env 文件，返回 None
    if not dir_name:
        return None

    # 从当前脚本所在目录向上查找
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(current_dir) == dir_name:
            env_path_in_specified_dir: str = os.path.join(current_dir, ".env")
            if os.path.exists(env_path_in_specified_dir):
                return env_path_in_specified_dir
            return None
        if current_dir == os.path.dirname(current_dir):
            break
        current_dir = os.path.dirname(current_dir)
    return None


# 加载环境变量并初始化全局变量
DOT_ENV_PATH = find_dotenv_path()
load_dotenv(dotenv_path=DOT_ENV_PATH)
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 定义模型列表
MODEL_LIST = {
    "DeepSeek-R1-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "Qwen-72B-128k": "Qwen/Qwen2.5-72B-Instruct-128K",
    "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen-7B":"Qwen/Qwen2.5-7B-Instruct",
}
