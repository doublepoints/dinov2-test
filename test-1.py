from glob import glob
from PIL import Image
import os
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, AutoModel
import torch
from sklearn.cluster import KMeans
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap


class KeysExtractor:
    def __init__(self, model, n_layer=-1):
        self.keys = None
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys


def process_images(image_paths):
    """处理任意大小的图片"""
    imgs = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        imgs.append(img)
    return imgs


def create_colormap(n_clusters):
    """创建自定义颜色映射"""
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))
    colors = [(r, g, b, 0.3) for r, g, b, _ in colors]  # 设置透明度为0.3
    return LinearSegmentedColormap.from_list('custom_cmap', colors)


def create_overlay_image(original_img, labels, n_clusters):
    """创建带有透明mask的叠加图像"""
    # 转换原始图片为numpy数组
    img_array = np.array(original_img)

    # 创建自定义颜色映射
    cmap = create_colormap(n_clusters)

    # 将labels调整为与原始图片相同的大小
    h, w = original_img.size[1], original_img.size[0]
    mask = Image.fromarray(labels.astype(np.uint8))
    mask = mask.resize((w, h), Image.Resampling.NEAREST)
    mask_array = np.array(mask)

    # 创建彩色mask
    mask_colored = cmap(mask_array / (n_clusters - 1))
    mask_colored = (mask_colored * 255).astype(np.uint8)

    # 创建叠加图像
    overlay = Image.alpha_composite(
        Image.fromarray(np.uint8(img_array)).convert('RGBA'),
        Image.fromarray(mask_colored)
    )

    return overlay


def save_results(original_img, overlay_img, original_path, output_folder):
    """保存原始图片和叠加结果"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取原始文件名
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]

    # 保存叠加图像
    overlay_path = os.path.join(output_folder, f"{name_without_ext}_overlay.png")
    overlay_img.save(overlay_path)

    return overlay_path


def visualize_images(imgs, labels=None, overlays=None, figsize=(15, 5)):
    """可视化原始图片、聚类结果和叠加效果"""
    n_imgs = len(imgs)
    n_rows = 3 if overlays is not None else (2 if labels is not None else 1)

    fig, axs = plt.subplots(n_rows, n_imgs, figsize=(figsize[0], figsize[1] * n_rows))
    if n_imgs == 1:
        axs = axs.reshape(n_rows, 1)

    # 显示原始图片
    for i, img in enumerate(imgs):
        axs[0][i].imshow(img)
        axs[0][i].axis('off')
        axs[0][i].set_title('Original')

    if labels is not None:
        # 显示聚类结果
        for i, img_labels in enumerate(labels):
            axs[1][i].imshow(img_labels)
            axs[1][i].axis('off')
            axs[1][i].set_title('Clusters')

    if overlays is not None:
        # 显示叠加结果
        for i, overlay in enumerate(overlays):
            axs[2][i].imshow(overlay)
            axs[2][i].axis('off')
            axs[2][i].set_title('Overlay')

    plt.tight_layout()
    return fig


def perform_clustering(processor, model, imgs, n_clusters=3):
    """执行图像聚类"""
    ke = KeysExtractor(model)

    # 使用processor处理图片
    inputs = processor(images=imgs, return_tensors="pt")

    # 获取attention keys
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 获取每个图片的patch数量
    n_patches = ke.keys.shape[1] - 1  # 减去CLS token
    patch_size = int(math.sqrt(n_patches))

    # 重塑keys并执行聚类
    K = ke.keys[:, 1:, :].reshape(len(imgs) * n_patches, -1)
    labels = KMeans(n_clusters=n_clusters).fit_predict(K)

    # 重塑标签以匹配图片patch网格
    labels = labels.reshape(len(imgs), patch_size, patch_size)

    return labels


def main(image_folder, output_folder, n_clusters=3):
    """主函数"""
    # 加载模型和processor
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    # 获取图片路径并处理图片
    image_paths = glob(f'{image_folder}/*.png')
    imgs = process_images(image_paths)

    # 执行聚类
    labels = perform_clustering(processor, model, imgs, n_clusters)

    # 创建和保存叠加图像
    overlays = []
    overlay_paths = []
    for img, label, path in zip(imgs, labels, image_paths):
        overlay = create_overlay_image(img, label, n_clusters)
        overlays.append(overlay)
        overlay_path = save_results(img, overlay, path, output_folder)
        overlay_paths.append(overlay_path)

    # 可视化结果
    fig = visualize_images(imgs, labels, overlays, figsize=(15, 5))
    ##plt.show()

    return imgs, labels, overlays, overlay_paths


if __name__ == "__main__":
    # 使用示例
    imgs, labels, overlays, overlay_paths = main(
        image_folder='images-tmp2',
        output_folder='images-output',
        n_clusters=2
    )