from typing import Dict, Tuple
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module(force=True)
class GeneratePoseHeatmap(BaseTransform):
    """为 7 个关键点生成高斯热图"""
    
    def __init__(self, heatmap_size: Tuple[int, int] = (48, 48), sigma: float = 2.0):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = 7
    
    def transform(self, results: Dict) -> Dict:
        img_shape = results.get('img_shape', (results['height'], results['width']))
        h, w = img_shape[:2]
        hm_h, hm_w = self.heatmap_size
        
        heatmap = np.zeros((self.num_keypoints, hm_h, hm_w), dtype=np.float32)
        keypoints_tensor = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        
        # 从 instances 提取 keypoints
        if 'instances' in results and len(results['instances']) > 0:
            inst = results['instances'][0]
            if 'keypoints' in inst:
                kpts = inst['keypoints']
                for i in range(self.num_keypoints):
                    idx = i * 3
                    if idx + 2 < len(kpts):
                        x, y, v = float(kpts[idx]), float(kpts[idx+1]), float(kpts[idx+2])
                        keypoints_tensor[i] = [x, y, v]
                        if v > 0:
                            hm_x = int(x * hm_w / w)
                            hm_y = int(y * hm_h / h)
                            if 0 <= hm_x < hm_w and 0 <= hm_y < hm_h:
                                heatmap[i] = self._generate_gaussian(hm_h, hm_w, hm_x, hm_y)
        
        # 关键：存储到 results，让 PackDetInputsWithPose 读取
        results['gt_keypoints'] = keypoints_tensor
        results['gt_keypoints_heatmap'] = heatmap
        
        return results
    
    def _generate_gaussian(self, h: int, w: int, cx: int, cy: int) -> np.ndarray:
        x = np.arange(0, w, 1, np.float32)
        y = np.arange(0, h, 1, np.float32)[:, np.newaxis]
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.sigma ** 2))
@TRANSFORMS.register_module(force=True)
class CopyImgIdToId(BaseTransform):
    """Copy results['img_id'] to results['id'] for mmpose CocoMetric."""
    def __init__(self, src_key: str = 'img_id', dst_key: str = 'id'):
        super().__init__()
        self.src_key = src_key
        self.dst_key = dst_key

    def transform(self, results: Dict) -> Dict:
        if self.src_key in results and self.dst_key not in results:
            results[self.dst_key] = results[self.src_key]
        return results    
