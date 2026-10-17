import openvino as ov
from pathlib import Path
from transformers import AutoModel
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\jina-clip-v2"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
core = ov.Core()

from PIL import Image
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

def calc_simularity_softmax(embeddings1, embeddings2, apply_softmax=True):
  """
  计算两个嵌入向量列表之间的相似度并可选地应用 softmax。
  参数:
    embeddings1: Iterable，第一个嵌入向量列表。
    embeddings2: Iterable，第二个嵌入向量列表。
    apply_softmax: bool，是否对相似度分数应用 softmax，默认 True。
  返回:
    List[List[float]]，每个 embeddings1 中向量与 embeddings2 中所有向量的相似度分数列表。
  """
  simularity = []
  for emb1 in embeddings1:
    scores = [emb1 @ emb2 for emb2 in embeddings2]
    if apply_softmax:
      scores = softmax(scores)
    simularity.append(scores)
  return simularity

def visionize_result(image: Image.Image, labels: list[str], probs: np.ndarray, top: int = 5):
  """
  可视化零样本分类结果。
  参数:
    image: PIL.Image.Image，输入图像。
    labels: list[str]，分类标签列表。
    probs: np.ndarray，模型输出的 softmax 概率。
    top: int，要展示的最高概率标签数量，默认 5。
  """
  plt.figure(figsize=(8, 8))
  # 取 top 个概率最大的标签
  top_idxs = np.argsort(-probs)[:min(top, probs.shape[0])]
  top_probs = probs[top_idxs]
  # 显示原图
  plt.subplot(2, 1, 1)
  plt.imshow(image)
  plt.axis("off")
  # 显示水平柱状图
  plt.subplot(2, 1, 2)
  y = np.arange(len(top_probs))
  plt.barh(y, top_probs)
  plt.gca().invert_yaxis()
  plt.yticks(y, [labels[i] for i in top_idxs])
  plt.xlabel("相似度")
  plt.grid(True)
  plt.tight_layout()
  
# Ensure notebook_utils is available
utils_path = Path("notebook_utils.py")
if not utils_path.exists():
    resp = requests.get(
        "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
    )
    utils_path.write_text(resp.text)

# Prepare data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download furseal image
furseal_path = data_dir / "furseal.png"
if not furseal_path.exists():
    resp = requests.get(
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/3f779fc1-c1b2-4dec-915a-64dae510a2bb",
        stream=True,
    )
    furseal_path.write_bytes(resp.content)

# Download coco image
coco_path = data_dir / "coco.jpg"
if not coco_path.exists():
    resp = requests.get(
        "https://github.com/user-attachments/assets/1c66a05d-7442-45c2-a34c-bb08b95af7a6",
        stream=True,
    )
    coco_path.write_bytes(resp.content)

# Load images
img_furseal = Image.open(furseal_path)
img_coco   = Image.open(coco_path)

IMAGE_INPUTS = [img_furseal, img_coco]
TEXT_INPUTS  = ["Seal", "Cobra", "Rat", "Penguin", "Dog"]
tokenizer = model.get_tokenizer()

tokenizer_kwargs = dict()
tokenizer_kwargs["padding"] = "max_length"
tokenizer_kwargs["max_length"] = 512
tokenizer_kwargs["truncation"] = True

text_inputs = tokenizer(
    TEXT_INPUTS,
    return_tensors="pt",
    **tokenizer_kwargs,
).to("cpu")


processor = model.get_preprocess()
vision_inputs = processor(images=IMAGE_INPUTS, return_tensors="pt")
# Telemetry
from notebook_utils import collect_telemetry
collect_telemetry("jina-clip.ipynb")

fp16_text_model_path = Path("jina-clip-text_v1_fp16.xml")

if not fp16_text_model_path.exists():
    ov_text_model = ov.convert_model(model.text_model, example_input=text_inputs["input_ids"])
    ov.save_model(ov_text_model, fp16_text_model_path)