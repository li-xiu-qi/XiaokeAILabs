"""Jev-Omni 运行时公共配置：缓存目录、离线模式、ffmpeg 补丁。

quickstart.py 与 bench_latency.py 共用，不直接执行。
"""
import glob
import os
import subprocess
import sys

JEV_OMNI_REPO_DIR = "models--akhilaaa3--Jev-Omni"


def setup_environment(models_dir=None):
    """配置权重缓存与离线模式。

    models_dir 为 None 时使用 HuggingFace 默认缓存；否则按 download_models.py
    --target-dir 使用的同一目录读取权重。
    """
    if models_dir:
        os.environ["HF_HUB_CACHE"] = os.path.abspath(models_dir)
    # load_jev_omni 内部仍会调用 snapshot_download，权重已在缓存中，
    # 离线模式下直接命中，不发网络请求。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def patch_ffmpeg():
    """给官方代码中的 ffmpeg 调用补上 -y。

    jev_omni.py 转码音频到临时文件时没有传 -y，同一进程第二次运行音频推理时，
    /tmp 下残留的临时文件会触发 ffmpeg 的覆盖确认，非交互模式下直接失败。
    在运行时补丁比改官方文件可靠，升级权重也不丢失。
    """
    original_run = subprocess.run

    def patched_run(args, *pargs, **kwargs):
        if isinstance(args, (list, tuple)) and args:
            exe = os.path.basename(str(args[0])).lower()
            if exe in ("ffmpeg", "ffmpeg.exe") and "-y" not in args:
                args = [args[0], "-y", *args[1:]]
        return original_run(args, *pargs, **kwargs)

    subprocess.run = patched_run


def ensure_jev_omni_importable(models_dir=None):
    """找到缓存中的 Jev-Omni 快照目录并加入 sys.path。

    jev_omni.py 与 load_model.py 随权重下载，不在脚本目录。默认缓存或
    download_models.py --target-dir 指定的目录都按 hub 缓存结构查找。
    """
    candidates = []
    if models_dir:
        candidates.append(os.path.abspath(models_dir))
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        candidates.append(os.path.abspath(env_cache))
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))

    for cache_root in candidates:
        pattern = os.path.join(cache_root, JEV_OMNI_REPO_DIR, "snapshots", "*")
        snapshots = [p for p in glob.glob(pattern) if os.path.isfile(os.path.join(p, "jev_omni.py"))]
        if snapshots:
            snapshot_path = sorted(snapshots)[-1]
            if snapshot_path not in sys.path:
                sys.path.insert(0, snapshot_path)
            return snapshot_path
    return None


def experiment_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_paths():
    root = experiment_root()
    return {
        "video": os.path.join(root, "assets", "the-wall-23s.mp4"),
        "image": os.path.join(root, "media", "frame.jpg"),
        "audio": os.path.join(root, "media", "audio.wav"),
        "results": os.path.join(root, "results"),
    }
