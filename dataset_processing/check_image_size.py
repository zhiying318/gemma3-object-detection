from datasets import load_dataset
import numpy as np
from PIL import Image

import sys

# dataset = load_dataset("ariG23498/license-detection-paligemma", split="train")
dataset = load_dataset("zhiyingzou0202/object_detection_bbox_paligemma", split="train")

# 查看第一条记录的图片
img = dataset[10]['image']  # 这是 PIL.Image.Image 类型

# 获取分辨率 (width, height)
width, height = img.size
print(f"分辨率：{width}x{height}")

# 获取图片在内存中占用的大小（bytes）
img_np = np.array(img)  # 转换为 NumPy 数组
num_bytes = img_np.nbytes  # 内存中占用的字节数
print(f"内存大小：{num_bytes / 1024:.2f} KB ({num_bytes / (1024*1024):.2f} MB)")


def get_sample_memory_size(sample):
    total_bytes = 0

    for key, value in sample.items():
        if isinstance(value, Image.Image):
            # 图像：转为 NumPy 数组获取内存大小
            total_bytes += np.array(value).nbytes
        elif isinstance(value, (str, int, float, list, dict)):
            # 其他字段：用 sys.getsizeof 获取粗略估算
            total_bytes += sys.getsizeof(value)
        else:
            print(f"未知字段类型 {key}: {type(value)}")
    
    return total_bytes

size_bytes = get_sample_memory_size(dataset[10])
print(f"第10条记录内存开销约为：{size_bytes / 1024:.2f} KB")