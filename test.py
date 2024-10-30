from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, AutoModel
import torch
from sklearn.cluster import KMeans
import numpy as np


# 获取所有图片文件名
filenames = glob('images-tmp2/*.png')
# 加载所有图片并转换为RGB格式
imgs = [Image.open(fn).convert('RGB') for fn in filenames]

# 设置子图的行列数（这里设置为一行多列）
fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))  # 调整figsize以适应图片数量

# 循环显示图片
#for img, ax in zip(imgs, axs):
#    ax.imshow(img)
#    ax.axis('off')  # 关闭坐标轴

#plt.show()  # 显示所有图片

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

class KeysExtractor:

    def __init__(self, model, n_layer=-1):
        self.keys = None
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys


ke = KeysExtractor(model)

inputs = processor(images=imgs, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

n_clusters = 3

K = ke.keys[:, 1:, :].reshape(len(imgs) * 256, -1)
# ^ shape: (n_images * 256, n_hidden)
labels = KMeans(n_clusters=n_clusters).fit_predict(K).reshape(len(imgs), 16, 16)
# ^ contains cluster indices; shape: (n_images, 16, 16)

fig, axs = plt.subplots(2, len(imgs), figsize=(10, 3))
for i, (img, img_labels) in enumerate(zip(imgs, labels)):
    axs[0][i].imshow(img)
    axs[1][i].imshow(img_labels)
for row in axs:
    for ax in row:
        ax.axis('off')
plt.show()  # 显示所有图片
