import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

def load_image_from_url(url):
    """从URL加载图片"""
    return Image.open(requests.get(url, stream=True).raw)

def main():
    # 加载CLIP模型和处理器
    print("正在加载CLIP模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 准备示例图片
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 一只猫
        "http://images.cocodataset.org/val2017/000000024241.jpg"   # 一只狗
    ]

    print("正在下载测试图片...")
    images = [load_image_from_url(url) for url in image_urls]

    # 准备文本描述
    texts = [
        "一只猫",
        "一只狗",
        "一个人"
    ]

    print("处理图像和文本...")
    # 处理图像和文本
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)

    # 获取特征
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # 打印结果
    print("\n图像-文本匹配概率：")
    for i, image_probs in enumerate(probs):
        print(f"\n图像 {i+1}:")
        for text, prob in zip(texts, image_probs):
            print(f"{text}: {prob:.3f}")

if __name__ == "__main__":
    main() 

"""
1. 使用Hugging Face的预训练CLIP模型（openai/clip-vit-base-patch32）
2. 下载并处理两张测试图片（一只猫和一只狗）
3. 使用三个中文文本描述进行匹配测试
4. 计算并显示图像-文本的匹配概率
"""