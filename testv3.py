##显示剔除前景后的特征点，并与未剔除之前进行比较。

import torch
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torchvision

# 禁用beta警告
torchvision.disable_beta_transforms_warning()


class KeysExtractor:
    def __init__(self, model, n_layer=-1):
        self.keys = None
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys


def get_background_mask(image_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    ke = KeysExtractor(model)

    imgs = [Image.open(fn).convert('RGB') for fn in image_paths]

    inputs = processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    n_clusters = 2
    K = ke.keys[:, 1:, :].cpu().reshape(len(imgs) * 256, -1)
    labels = KMeans(n_clusters=n_clusters).fit_predict(K).reshape(len(imgs), 16, 16)

    masks = []
    for i, img in enumerate(imgs):
        mask = labels[i]
        #background_mask = (mask == 0) | (mask == 1)
        background_mask = (mask == 0)
        # 上采样到原图大小
        background_mask = torch.nn.functional.interpolate(
            torch.from_numpy(background_mask).float().unsqueeze(0).unsqueeze(0),
            size=img.size[::-1],
            mode='nearest'
        ).squeeze().bool()
        background_mask = background_mask.to(device)
        masks.append(background_mask)

    return imgs, masks


def match_background_features(image_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs, background_masks = get_background_mask(image_paths)

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    images = [load_image(path).to(device) for path in image_paths]

    all_feats = []
    all_original_keypoints = []  # 存储原始特征点
    for img, mask in zip(images, background_masks):
        print(f"Mask shape: {mask.shape}")

        feats = extractor.extract(img)
        # 去掉batch维度
        keypoints = feats['keypoints'].squeeze(0).to(device)  # 从[1, 2048, 2]变成[2048, 2]

        # 保存原始特征点
        all_original_keypoints.append(keypoints.cpu())

        print(f"Keypoints shape after squeeze: {keypoints.shape}")

        # 确保keypoints的坐标在图像范围内
        kpts_y = keypoints[:, 1].long().clamp(0, mask.shape[0] - 1)
        kpts_x = keypoints[:, 0].long().clamp(0, mask.shape[1] - 1)

        # 获取这些点是否在背景中
        valid_points = mask[kpts_y, kpts_x]

        print(f"Valid points shape: {valid_points.shape}")
        print(f"Number of valid points: {valid_points.sum().item()}")

        # 更新特征字典，只保留背景点
        feats['keypoints'] = keypoints[valid_points].unsqueeze(0)  # 添加回batch维度
        feats['keypoint_scores'] = feats['keypoint_scores'].squeeze(0)[valid_points].unsqueeze(0)
        feats['descriptors'] = feats['descriptors'][:, valid_points]

        all_feats.append(feats)

        # 打印更新后的特征形状
        print(f"Updated keypoints shape: {feats['keypoints'].shape}")
        print(f"Updated scores shape: {feats['keypoint_scores'].shape}")
        print(f"Updated descriptors shape: {feats['descriptors'].shape}")

    # 检查是否有足够的特征点进行匹配
    if len(all_feats[0]['keypoints'].squeeze(0)) == 0 or len(all_feats[1]['keypoints'].squeeze(0)) == 0:
        raise ValueError("No background features found in one or both images")

    matches01 = matcher({'image0': all_feats[0], 'image1': all_feats[1]})
    feats0, feats1, matches01 = [rbd(x) for x in [all_feats[0], all_feats[1], matches01]]

    matches = matches01['matches']
    points0 = feats0['keypoints'].squeeze(0)[matches[..., 0]].cpu()
    points1 = feats1['keypoints'].squeeze(0)[matches[..., 1]].cpu()

    background_masks = [mask.cpu() for mask in background_masks]

    return points0, points1, imgs, background_masks, all_original_keypoints


def visualize_results(image_paths):
    try:
        points0, points1, imgs, masks, original_keypoints = match_background_features(image_paths)

        # 创建3x2的子图布局
        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        # 显示原始图片
        axs[0][0].imshow(imgs[0])
        axs[0][0].set_title('Image 1')
        axs[0][1].imshow(imgs[1])
        axs[0][1].set_title('Image 2')

        # 显示原始特征点
        axs[1][0].imshow(imgs[0])
        axs[1][0].scatter(original_keypoints[0][:, 0], original_keypoints[0][:, 1], c='r', s=1)
        axs[1][0].set_title('Original Keypoints - Image 1')
        axs[1][1].imshow(imgs[1])
        axs[1][1].scatter(original_keypoints[1][:, 0], original_keypoints[1][:, 1], c='r', s=1)
        axs[1][1].set_title('Original Keypoints - Image 2')

        # 显示背景mask和筛选后的特征点
        axs[2][0].imshow(imgs[0])
        if len(points0) > 0:
            axs[2][0].scatter(points0[:, 0], points0[:, 1], c='r', s=1)
        axs[2][0].set_title('Background Keypoints - Image 1')

        axs[2][1].imshow(imgs[1])
        if len(points1) > 0:
            axs[2][1].scatter(points1[:, 0], points1[:, 1], c='r', s=1)
        axs[2][1].set_title('Background Keypoints - Image 2')

        # 添加mask作为半透明覆盖层，确保使用正确的数据类型
        for i in range(2):
            # 将布尔mask转换为浮点数
            mask_float = masks[i].float().numpy()
            # 创建白色背景的RGB mask（前景为白色，背景为黑色）
            mask_overlay = np.stack([1 - mask_float] * 3, axis=-1)
            # 使用黄色作为前景色
            yellow_overlay = np.zeros_like(mask_overlay)
            yellow_overlay[..., 0] = 1  # Red channel
            yellow_overlay[..., 1] = 1  # Green channel
            # 只在前景区域（mask为False的地方）显示黄色
            mask_overlay = np.where(mask_overlay > 0.5, yellow_overlay, 0)
            axs[2][i].imshow(mask_overlay, alpha=0.3)

        for ax in axs.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        print(f"Original keypoints - Image 1: {len(original_keypoints[0])}")
        print(f"Original keypoints - Image 2: {len(original_keypoints[1])}")
        print(f"Background matching points: {len(points0)}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please try with different images or adjust the background detection parameters")
if __name__ == "__main__":
    image_paths = ['/home/tridot/Projects/dinov2-test/images-tmp2/Screenshot from 2024-10-30 10-47-16.png', '/home/tridot/Projects/dinov2-test/images-tmp2/Screenshot from 2024-10-30 10-47-18.png']
    visualize_results(image_paths)