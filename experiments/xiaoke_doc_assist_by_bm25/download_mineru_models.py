import json
import os

import requests
from modelscope import snapshot_download


def download_json(url):
    """下载JSON配置文件"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_and_setup_models():
    """下载并设置必要的模型文件"""
    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    model_dir = snapshot_download("opendatalab/PDF-Extract-Kit-1.0", allow_patterns=mineru_patterns)
    layoutreader_model_dir = snapshot_download("ppaanngggg/layoutreader")
    model_dir = model_dir + "/models"
    print(f"model_dir is: {model_dir}")
    print(f"layoutreader_model_dir is: {layoutreader_model_dir}")

    json_url = "https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json"
    config_file_name = "magic-pdf.json"
    home_dir = os.path.expanduser("~")
    config_file = os.path.join(home_dir, config_file_name)

    # 下载并修改配置文件
    json_data = download_json(json_url)
    json_data["models-dir"] = model_dir
    json_data["layoutreader-model-dir"] = layoutreader_model_dir

    with open(config_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"The configuration file has been configured successfully, the path is: {config_file}")



if __name__ == "__main__":
    download_and_setup_models()