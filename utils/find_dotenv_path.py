import os


def find_dotenv_path():
    current_dir = os.path.dirname(__file__)
    while current_dir != os.path.dirname(current_dir):  # 循环直到找到顶层目录
        if os.path.basename(current_dir) == "XiaokeAILabs":
            break
        current_dir = os.path.dirname(current_dir)
    return os.path.join(current_dir, ".env")
