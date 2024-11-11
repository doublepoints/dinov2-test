import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import DATA_PATH, EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.visualization.viz2d import plot_cumulative
from gluefactory.eval.eval_pipeline import EvalPipeline
from gluefactory.eval.file_io import get_eval_parser, load_model, parse_eval_args
from gluefactory.eval.utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust

# 导入您第一段代码中需要的库
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

logger = logging.getLogger(__name__)

# 禁用beta警告（如果需要）
import torchvision
torchvision.disable_beta_transforms_warning()

class MegaDepth1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "megadepth1500/pairs_calibrated.txt",
            "root": "megadepth1500/images/",
            "extra_data": "relative_pose",
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "name": "custom_model",  # 使用自定义模型
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results

# 新增的自定义模型，集成前景/背景特征点筛选逻辑
class CustomModel(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化Dinov2模型和处理器
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.ke = KeysExtractor(self.dino_model)

        # 初始化SuperPoint和LightGlue
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

        self.device = device

    def forward(self, data):
        # 从数据中获取图像路径
        img_path0 = data['image0']['path'][0]
        img_path1 = data['image1']['path'][0]

        # 加载并处理图像
        imgs = [Image.open(p).convert('RGB') for p in [img_path0, img_path1]]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)

        # 生成前景/背景掩码
        with torch.no_grad():
            outputs = self.dino_model(**inputs, output_attentions=True)

        n_clusters = 2
        K = self.ke.keys[:, 1:, :].cpu().reshape(len(imgs) * 256, -1)
        labels = KMeans(n_clusters=n_clusters).fit_predict(K).reshape(len(imgs), 16, 16)

        masks = []
        for i, img in enumerate(imgs):
            mask = labels[i]
            background_mask = (mask == 0)
            # 上采样到原图大小
            background_mask = torch.nn.functional.interpolate(
                torch.from_numpy(background_mask).float().unsqueeze(0).unsqueeze(0),
                size=img.size[::-1],
                mode='nearest'
            ).squeeze().bool()
            background_mask = background_mask.to(self.device)
            masks.append(background_mask)

        # 使用SuperPoint提取特征
        images = [load_image(p).to(self.device) for p in [img_path0, img_path1]]
        feats = []
        for img, mask in zip(images, masks):
            feat = self.extractor.extract(img)
            keypoints = feat['keypoints'].squeeze(0).to(self.device)
            # 确保keypoints的坐标在图像范围内
            kpts_y = keypoints[:, 1].long().clamp(0, mask.shape[0] - 1)
            kpts_x = keypoints[:, 0].long().clamp(0, mask.shape[1] - 1)
            # 获取这些点是否在背景中
            valid_points = mask[kpts_y, kpts_x]
            # 更新特征字典，只保留背景点
            feat['keypoints'] = keypoints[valid_points].unsqueeze(0)
            feat['keypoint_scores'] = feat['keypoint_scores'].squeeze(0)[valid_points].unsqueeze(0)
            feat['descriptors'] = feat['descriptors'][:, valid_points]
            feats.append(feat)

        # 检查是否有足够的特征点进行匹配
        if len(feats[0]['keypoints'].squeeze(0)) == 0 or len(feats[1]['keypoints'].squeeze(0)) == 0:
            # 如果没有足够的特征点，返回空的匹配结果
            pred = {
                'keypoints0': feats[0]['keypoints'].cpu().numpy(),
                'keypoints1': feats[1]['keypoints'].cpu().numpy(),
                'matches0': np.array([-1] * feats[0]['keypoints'].shape[1]),
                'matches1': np.array([-1] * feats[1]['keypoints'].shape[1]),
                'keypoint_scores0': feats[0]['keypoint_scores'].cpu().numpy(),
                'keypoint_scores1': feats[1]['keypoint_scores'].cpu().numpy(),
                'matching_scores0': np.array([]),
                'matching_scores1': np.array([]),
            }
            return pred

        # 使用LightGlue进行匹配
        data = {
            'image0': feats[0],
            'image1': feats[1],
        }
        matches = self.matcher(data)

        # 处理匹配结果
        feats0, feats1, matches = [rbd(x) for x in [feats[0], feats[1], matches]]
        matches_indices = matches['matches']
        matching_scores0 = matches['matching_scores0']
        matching_scores1 = matches['matching_scores1']

        pred = {
            'keypoints0': feats0['keypoints'].cpu().numpy(),
            'keypoints1': feats1['keypoints'].cpu().numpy(),
            'matches0': matches_indices[:, 1].cpu().numpy(),
            'matches1': matches_indices[:, 0].cpu().numpy(),
            'keypoint_scores0': feats0['keypoint_scores'].cpu().numpy(),
            'keypoint_scores1': feats1['keypoint_scores'].cpu().numpy(),
            'matching_scores0': matching_scores0.cpu().numpy(),
            'matching_scores1': matching_scores1.cpu().numpy(),
        }
        return pred

# 定义KeysExtractor类
class KeysExtractor:
    def __init__(self, model, n_layer=-1):
        self.keys = None
        key_module = model.encoder.layer[n_layer].attention.attention.key
        key_module.register_forward_hook(self.save_keys)

    def save_keys(self, layer, inputs, keys):
        self.keys = keys

# 加载自定义模型的函数
def load_model(model_conf, checkpoint=None):
    if model_conf.name == 'custom_model':
        model = CustomModel(model_conf).eval()
        return model
    else:
        # 如果不是自定义模型，使用原始的加载方式
        # 请根据您的实际情况修改
        pass

if __name__ == "__main__":
    from gluefactory import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500Pipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = MegaDepth1500Pipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
