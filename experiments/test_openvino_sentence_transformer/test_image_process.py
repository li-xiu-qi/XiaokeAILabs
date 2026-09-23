from transformers import AutoImageProcessor
from PIL import Image
import requests # 用于从 URL 加载图片示例

# 假设您的配置文件和模型文件在同一个目录下
processor_directory = r"./preprocessor"

try:
    # 1. 加载图像预处理器
    image_processor = AutoImageProcessor.from_pretrained(processor_directory, trust_remote_code=True)
    print("图像预处理器加载成功！")

    # 2. 加载一张图片 (示例：从 URL 加载)
    # 您也可以使用 Image.open("path/to/your/local/image.jpg")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" # 示例图片 URL
    image = Image.open(requests.get(url, stream=True).raw)
    # 确保图像是 RGB 格式，如果不是，需要转换
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    print(f"原始图片尺寸: {image.size}, 模式: {image.mode}")

    # 3. 对图片进行预处理
    # "return_tensors='pt'" 表示返回 PyTorch 张量，您也可以用 'np' 返回 NumPy 数组，或 'tf' 返回 TensorFlow 张量
    inputs = image_processor(images=image, return_tensors="np")

    # inputs 会是一个字典，其中 'pixel_values' 键对应的值就是预处理后的图像张量
    pixel_values = inputs.pixel_values
    print(f"预处理后的图像张量形状: {pixel_values.shape}") # 形状通常是 (batch_size, num_channels, height, width)

except Exception as e:
    print(f"处理过程中发生错误：{e}")