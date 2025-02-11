import os

def find_dotenv_path():
    current_working_dir = os.getcwd()
    env_path = os.path.join(current_working_dir, ".env")
    if os.path.exists(env_path):
        return env_path

    current_dir = os.path.dirname(__file__)
    while current_dir != os.path.dirname(current_dir):  # 循环直到找到顶层目录
        if os.path.basename(current_dir) == "XiaokeAILabs":
            break
        current_dir = os.path.dirname(current_dir)
    return os.path.join(current_dir, ".env")
