from omegaconf import OmegaConf
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
import torch.nn.functional as F
from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container

class KeysExtractor:
    def __init__(self, model, n_layer=-1):
        self.keys = None
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys

class TwoViewPipeline(BaseModel):
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
        "dinov2": {
            "model_name": "facebook/dinov2-base",
            "n_clusters": 2,
            "background_label": 0
        }
    }
    
    required_data_keys = ["view0", "view1"]
    strict_conf = False
    components = ["extractor", "matcher", "filter", "solver", "ground_truth"]

    def _init_(self, conf):
        super()._init_(conf)
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))
        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))
        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))
        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))
        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(to_ctr(conf.ground_truth))
            
        # 初始化DINOv2相关组件
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(self.conf.dinov2.model_name)
        self.dinov2_model = AutoModel.from_pretrained(self.conf.dinov2.model_name).to(self.device)
        self.keys_extractor = KeysExtractor(self.dinov2_model)
        
    def get_background_mask(self, image):
        """使用DINOv2生成背景mask"""
        # 确保图像格式正确
        if isinstance(image, torch.Tensor):
            # 假设输入是[C, H, W]格式的张量
            image = image.permute(1, 2, 0).cpu().numpy()
            
        # 处理图像
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs, output_attentions=True)
            
        # 获取keys并进行聚类
        K = self.keys_extractor.keys[:, 1:, :].cpu().reshape(256, -1)
        labels = KMeans(n_clusters=self.conf.dinov2.n_clusters).fit_predict(K).reshape(16, 16)
        
        # 生成背景mask
        background_mask = (labels == self.conf.dinov2.background_label)
        
        # 上采样到原图大小
        background_mask = F.interpolate(
            torch.from_numpy(background_mask).float().unsqueeze(0).unsqueeze(0),
            size=(image.shape[0], image.shape[1]),
            mode='nearest'
        ).squeeze().bool().to(self.device)
        
        return background_mask

    def filter_background_features(self, features, mask):
        """根据背景mask过滤特征点"""
        # 去掉batch维度
        keypoints = features['keypoints'].squeeze(0)
        
        # 确保keypoints的坐标在图像范围内
        kpts_y = keypoints[:, 1].long().clamp(0, mask.shape[0] - 1)
        kpts_x = keypoints[:, 0].long().clamp(0, mask.shape[1] - 1)
        
        # 获取背景点mask
        valid_points = mask[kpts_y, kpts_x]
        
        # 更新特征字典，只保留背景点
        filtered_features = {
            'keypoints': keypoints[valid_points].unsqueeze(0),
            'keypoint_scores': features['keypoint_scores'].squeeze(0)[valid_points].unsqueeze(0),
            'descriptors': features['descriptors'][:, valid_points]
        }
        
        return filtered_features

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        
        if self.conf.extractor.name and not skip_extract:
            # 获取特征点和描述符
            features = self.extractor(data_i)
            
            # 生成背景mask
            background_mask = self.get_background_mask(data_i["image"])
            
            # 过滤前景特征点
            filtered_features = self.filter_background_features(features, background_mask)
            
            # 更新预测结果
            pred_i = {**pred_i, **filtered_features}
            
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
            
        return pred_i

    def forward(self, data):
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }
        
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
            
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0
        
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