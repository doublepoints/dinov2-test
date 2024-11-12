from omegaconf import OmegaConf
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
import torch.nn.functional as F
from . import get_model
from .base_model import BaseModel
import gc
import numpy as np

to_ctr = OmegaConf.to_container


class KeysExtractor:
    def __init__(self, model, n_layer=-1):
        self.keys = None
        # 直接获取key模块
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys

    def remove(self):
        """移除hook以防止内存泄漏"""
        if hasattr(self, 'hook') and self.hook is not None:
            self.hook.remove()
        self.keys = None


class TwoViewPipelineMask(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }

    required_data_keys = ["view0", "view1"]
    strict_conf = False
    components = ["extractor", "matcher", "filter", "solver", "ground_truth"]

    def _init(self, conf):
        """初始化模型组件"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化其他组件
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))
            self.extractor = self.extractor.to(self.device)  # 确保 SuperPoint 在 GPU 上
            self.extractor.eval()  # 设置为评估模式

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))
            self.matcher = self.matcher.to(self.device)
            self.matcher.eval()

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))
            self.filter = self.filter.to(self.device)
            self.filter.eval()

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))
            self.solver = self.solver.to(self.device)
            self.solver.eval()

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(to_ctr(conf.ground_truth))
            self.ground_truth = self.ground_truth.to(self.device)
            self.ground_truth.eval()

        # DINOv2相关组件延迟初始化
        self.processor = None
        self.dinov2_model = None
        self.keys_extractor = None

    def get_background_mask(self, image):
        """使用DINOv2生成背景mask"""
        try:
            # 延迟初始化DINOv2
            if not hasattr(self, 'processor') or self.processor is None:
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            if not hasattr(self, 'dinov2_model') or self.dinov2_model is None:
                self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base')
                self.dinov2_model = self.dinov2_model.to(self.device)  # 确保在GPU上
                self.dinov2_model.eval()
            if not hasattr(self, 'keys_extractor') or self.keys_extractor is None:
                self.keys_extractor = KeysExtractor(self.dinov2_model)

            # 确保图像在GPU上且格式正确
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)  # 移到GPU
                # 添加batch维度如果需要
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                # 将图像转换为灰度
                if image.shape[1] == 3:  # RGB
                    scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(self.device)
                    image = (image * scale).sum(1, keepdim=True)

                # 记录原始图像尺寸
                original_height, original_width = image.shape[2:]

                # 保持数据在GPU上直到必须转到CPU
                image_np = image.squeeze(0).squeeze(0).cpu().numpy()
                # 扩展为3通道(DINOv2需要3通道输入)
                image_np = np.stack([image_np] * 3, axis=-1)
            else:
                image_np = image
                original_height, original_width = image_np.shape[:2]

            # 处理图像并确保在GPU上
            inputs = self.processor(images=[image_np], return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 移到GPU

            # 在GPU上运行模型
            with torch.no_grad():
                outputs = self.dinov2_model(**inputs, output_attentions=True)

                if self.keys_extractor.keys is None:
                    raise RuntimeError("Failed to capture keys")

                # 在GPU上处理keys，只在必要时移到CPU进行聚类
                keys = self.keys_extractor.keys[:, 1:, :]  # 保持在GPU
                K = keys.cpu().reshape(256, -1)  # 移到CPU进行聚类

                # 执行KMeans聚类
                labels = KMeans(n_clusters=2).fit_predict(K).reshape(16, 16)

                # 生成背景mask并立即移回GPU
                background_mask = torch.from_numpy(labels == 0).float().unsqueeze(0).unsqueeze(0)
                background_mask = background_mask.to(self.device)

                # 在GPU上进行上采样
                background_mask = F.interpolate(
                    background_mask,
                    size=(original_height, original_width),
                    mode='nearest'
                ).squeeze().bool()

                return background_mask  # 已经在GPU上

        except Exception as e:
            print(f"Error in get_background_mask: {e}")
            print(f"Image shape: {image.shape if isinstance(image, torch.Tensor) else 'unknown'}")
            print(f"Device: {image.device if isinstance(image, torch.Tensor) else 'unknown'}")
            raise
        finally:
            # 清理中间变量
            if 'outputs' in locals():
                del outputs
            if 'inputs' in locals():
                del inputs

    def extract_view(self, data, i):
        """处理单个视图的特征提取和背景过滤"""
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract

        try:
            if self.conf.extractor.name and not skip_extract:
                # 确保图像在GPU上
                if isinstance(data_i.get("image"), torch.Tensor):
                    data_i["image"] = data_i["image"].to(self.device)

                # 获取特征点和描述符（SuperPoint处理）
                feats = self.extractor(data_i)  # SuperPoint已经在GPU上

                # 生成背景mask（在GPU上）
                background_mask = self.get_background_mask(data_i["image"])

                # 提取关键点并保持在GPU上
                keypoints = feats['keypoints'].squeeze(0)  # [N, 2]

                # 确保keypoints的坐标在图像范围内
                kpts_y = keypoints[:, 1].long().clamp(0, background_mask.shape[0] - 1)
                kpts_x = keypoints[:, 0].long().clamp(0, background_mask.shape[1] - 1)

                # 获取有效的背景点（在GPU上）
                valid_points = background_mask[kpts_y, kpts_x]

                # 更新特征字典，所有操作保持在GPU上
                pred_i = {
                    'keypoints': keypoints[valid_points].unsqueeze(0),
                    'keypoint_scores': feats['keypoint_scores'].squeeze(0)[valid_points].unsqueeze(0),
                    'descriptors': feats['descriptors'][:, valid_points]
                }

                # 打印调试信息
                print(f"View {i}:")
                print(f"Original keypoints: {len(keypoints)}")
                print(f"Valid background points: {valid_points.sum().item()}")
                print(f"Updated keypoints shape: {pred_i['keypoints'].shape}")
                print(f"Device: {pred_i['keypoints'].device}")

            elif self.conf.extractor.name and not self.conf.allow_no_extract:
                pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}

            return pred_i

        except Exception as e:
            print(f"Error in extract_view: {e}")
            raise

    def _forward(self, data):
        """前向传播处理"""
        try:
            # 处理两个视图
            pred0 = self.extract_view(data, "0")
            pred1 = self.extract_view(data, "1")

            # 组合两个视图的预测结果
            pred = {
                **{k + "0": v for k, v in pred0.items()},
                **{k + "1": v for k, v in pred1.items()},
            }

            # 特征匹配
            if self.conf.matcher.name:
                pred = {**pred, **self.matcher({**data, **pred})}
            # 过滤匹配
            if self.conf.filter.name:
                pred = {**pred, **self.filter({**data, **pred})}
            # 位姿求解
            if self.conf.solver.name:
                pred = {**pred, **self.solver({**data, **pred})}
            # 处理真值信息
            if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
                gt_pred = self.ground_truth({**data, **pred})
                pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

            return pred

        except Exception as e:
            print(f"Error in _forward: {e}")
            raise
        finally:
            # 清理DINOv2资源
            self.cleanup_dinov2()

    def loss(self, pred, data):
        """计算损失函数"""
        losses = {}
        metrics = {}
        total = 0

        try:
            if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
                gt_pred = self.ground_truth({**data, **pred})
                pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

            for k in self.components:
                apply = True
                if "apply_loss" in self.conf[k].keys():
                    apply = self.conf[k].apply_loss
                if self.conf[k].name and apply:
                    try:
                        losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                    except NotImplementedError:
                        continue
                    losses = {**losses, **losses_}
                    metrics = {**metrics, **metrics_}
                    total = losses_["total"] + total

            return {**losses, "total": total}, metrics

        except Exception as e:
            print(f"Error in loss: {e}")
            raise

    def cleanup_dinov2(self):
        """清理DINOv2相关资源"""
        if hasattr(self, 'keys_extractor') and self.keys_extractor is not None:
            self.keys_extractor.remove()
            self.keys_extractor = None
        if hasattr(self, 'dinov2_model') and self.dinov2_model is not None:
            if self.device.type == 'cuda':
                self.dinov2_model.cpu()
            self.dinov2_model = None
        if hasattr(self, 'processor'):
            self.processor = None

        # 清理 CUDA 缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
