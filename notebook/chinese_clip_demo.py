import torch
from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import matplotlib.pyplot as plt

def load_image_from_url(url):
    """从URL加载图片"""
    return Image.open(requests.get(url, stream=True).raw)

def display_results(images, texts, probs):
    """可视化展示结果"""
    n_images = len(images)
    n_texts = len(texts)
    
    plt.figure(figsize=(15, 5))
    for i in range(n_images):
        plt.subplot(1, n_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f'图像 {i+1}\n' + '\n'.join([f'{texts[j]}: {probs[i][j]:.3f}' for j in range(n_texts)]))
    plt.tight_layout()
    plt.show()

def main():
    # 加载Chinese-CLIP模型和处理器
    print("正在加载Chinese-CLIP模型...")
    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)

    # 准备示例图片
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 一只猫
        "http://images.cocodataset.org/val2017/000000024241.jpg",  # 一只狗
        "https://raw.githubusercontent.com/OFA-Sys/Chinese-CLIP/master/examples/pokemon.png"  # 皮卡丘
    ]

    print("正在下载测试图片...")
    images = [load_image_from_url(url) for url in image_urls]

    # 准备更丰富的中文文本描述
    texts = [
        "一只可爱的橘猫",
        "一只黑白相间的狗",
        "一只黄色的皮卡丘",
        "一个正在打篮球的人",
        "一朵红色的玫瑰花"
    ]

    print("处理图像和文本...")
    # 处理图像和文本
    inputs = processor(
        images=images,  # 可以是PIL图像列表或单个图像
        text=texts,     # 可以是文本列表或单个文本
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,   # 对文本进行填充
        truncation=True # 过长的文本会被截断
    )

    # 获取特征
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # 图像-文本的匹配分数
        probs = logits_per_image.softmax(dim=1) # 归一化后的匹配概率

        # 打印结果
        print("\n图像-文本匹配概率：")
        for i, image_probs in enumerate(probs):
            print(f"\n图像 {i+1}:")
            for text, prob in zip(texts, image_probs):
                print(f"{text}: {prob:.3f}")

        # 可视化展示结果
        display_results(images, texts, probs.numpy())

        # 额外展示：提取图像和文本特征，计算相似度矩阵
        print("\n提取特征并计算相似度矩阵...")
        image_features = model.get_image_features(**inputs) # 提取图像特征
        text_features = model.get_text_features(**inputs) # 提取文本特征
    
        # 计算余弦相似度
        similarity = torch.nn.functional.normalize(image_features, dim=1) @ \
                    torch.nn.functional.normalize(text_features, dim=1).T
    
        print("\n图像-文本相似度矩阵：")
        for i, row in enumerate(similarity):
            print(f"\n图像 {i+1}:")
            for j, sim in enumerate(row):
                print(f"{texts[j]}: {sim:.3f}")

if __name__ == "__main__":
    main() 

"""
1. 使用了专门的Chinese-CLIP模型（OFA-Sys/chinese-clip-vit-base-patch16）
2. 添加了更多的测试图片（包括猫、狗和皮卡丘）
3. 使用了更丰富的中文描述
4. 添加了可视化功能，使用matplotlib展示结果
5. 增加了特征提取和相似度矩阵计算的演示

CLIPProcessor（预处理器）的作用
CLIPProcessor 负责将原始的图像和文本数据处理成模型可以接受的格式。它主要完成以下工作：
图像处理：
调整图像大小
将图像转换为张量
标准化像素值
处理批量图像
文本处理：
分词（Tokenization）
添加特殊标记（如[CLS], [SEP]等）
填充（Padding）
截断（Truncation）

CLIPModel（模型）的作用
CLIPModel 是核心模型部分，它包含：
图像编码器：
将图像转换为特征向量
通常基于Vision Transformer (ViT) 或 ResNet 架构
文本编码器：
将文本转换为特征向量
通常基于Transformer架构
"""