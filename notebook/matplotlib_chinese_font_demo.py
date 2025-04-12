import requests
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 方法1：下载字体文件，手动cp到matplotlib的fonts目录下，清除matplotlib的缓存
# 参考：https://blog.csdn.net/takedachia/article/details/131017286
# print(matplotlib.matplotlib_fname())

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(10, 6))
# plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], 'ro-')
# plt.xlabel('X轴')
# plt.ylabel('Y轴')
# plt.title('中文标题')
# plt.show()

# 方法2：在代码中下载字体文件，用matplotlib显示中文时指定字体
# 由于要到github下载字体文件，所以需要联网+翻墙，字体文件10MB左右，不翻墙下载很可能失败
chinese_font = None

def download_font_if_needed(font_url, save_path):
    """下载字体文件如果不存在"""
    font_path = Path(save_path)
    if not font_path.exists():
        print(f"正在下载字体文件到 {save_path}...")
        try:
            response = requests.get(font_url)
            response.raise_for_status()
            
            # 确保目录存在
            font_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存字体文件
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"字体文件已下载到 {save_path}")
            return True
        except Exception as e:
            print(f"下载字体文件失败: {e}")
            return False
    return True

def set_matplotlib_chinese_font():
    """设置matplotlib支持中文显示"""
    # 直接下载并指定中文字体
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    # 保存在当前目录
    font_path = "./SimHei.ttf"
    
    # 下载字体文件如果不存在
    if download_font_if_needed(font_url, font_path):
        # 使用自定义字体文件
        global chinese_font
        chinese_font = FontProperties(fname=font_path)
        print("已成功设置中文字体")
    else:
        print("无法设置中文字体，图表中的中文可能显示不正常")

# 设置matplotlib支持中文
set_matplotlib_chinese_font()

plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], 'ro-')
plt.xlabel('X轴', fontproperties=chinese_font)
plt.ylabel('Y轴', fontproperties=chinese_font)
plt.title('中文标题', fontproperties=chinese_font)
plt.show()