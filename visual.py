import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam import generate_mask
import os

# 选择一张训练图像
train_dir = "/root/autodl-tmp/Patch-based-dataset/train_data_patch"
classes = os.listdir(train_dir)
class_dir = os.path.join(train_dir, classes[3])  # 第一个类别
images = os.listdir(class_dir)
img_path = os.path.join(class_dir, images[0])  # 第一张图像

# 加载图像
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB

# 生成掩码
mask = generate_mask(image)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(mask, cmap='gray')
ax[1].set_title("Generated Mask")
ax[1].axis('off')

plt.tight_layout()
plt.savefig("mask_visualization.png")
print("Visualization saved to mask_visualization.png")