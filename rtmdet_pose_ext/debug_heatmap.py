"""
调试脚本：可视化验证 GeneratePoseHeatmap 生成的热图是否正确。
用法: python debug_heatmap.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simulate_pipeline():
    """模拟 Resize + RandomFlip + GeneratePoseHeatmap 的流程"""

    # ---- 模拟一个标注样本 ----
    ori_h, ori_w = 480, 640
    target_h, target_w = 192, 192

    # 原始关键点 (x, y, v)，7个点
    original_kpts = [
        320, 100, 2,   # 头部中央偏上
        200, 200, 2,   # 左肩
        440, 200, 2,   # 右肩
        160, 350, 2,   # 左手
        480, 350, 2,   # 右手
        250, 450, 2,   # 左脚
        400, 450, 2,   # 右脚
    ]

    w_scale = target_w / ori_w   # 192/640 = 0.3
    h_scale = target_h / ori_h   # 192/480 = 0.4

    results = {
        'img_shape': (target_h, target_w),
        'scale_factor': (w_scale, h_scale),
        'flip': False,
        'flip_direction': 'horizontal',
        'instances': [{'keypoints': original_kpts}],
    }

    # ---- 用修复后的 transform ----
    from keypoint_transforms import GeneratePoseHeatmap
    transform = GeneratePoseHeatmap(heatmap_size=(48, 48), sigma=2.0, num_keypoints=7)
    results = transform.transform(results)

    heatmap = results['gt_keypoints_heatmap']   # (7, 48, 48)
    kpts = results['gt_keypoints']              # (7, 3)

    print("=== 无翻转 ===")
    for i in range(7):
        x, y, v = kpts[i]
        expected_x = original_kpts[i*3] * w_scale
        expected_y = original_kpts[i*3+1] * h_scale
        hm_max = heatmap[i].max()
        print(f"  kpt[{i}]: got ({x:.1f}, {y:.1f}), "
              f"expected ({expected_x:.1f}, {expected_y:.1f}), "
              f"heatmap_max={hm_max:.3f}")

    # ---- 测试翻转情况 ----
    results_flip = {
        'img_shape': (target_h, target_w),
        'scale_factor': (w_scale, h_scale),
        'flip': True,
        'flip_direction': 'horizontal',
        'instances': [{'keypoints': original_kpts}],
    }
    results_flip = transform.transform(results_flip)
    kpts_flip = results_flip['gt_keypoints']

    print("\n=== 水平翻转 ===")
    for i in range(7):
        x, y, v = kpts_flip[i]
        expected_x = target_w - 1.0 - original_kpts[i*3] * w_scale
        expected_y = original_kpts[i*3+1] * h_scale
        print(f"  kpt[{i}]: got ({x:.1f}, {y:.1f}), "
              f"expected ({expected_x:.1f}, {expected_y:.1f})")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 热图叠加（无翻转）
    combined = heatmap.max(axis=0)  # (48, 48)
    axes[0].imshow(combined, cmap='hot')
    for i in range(7):
        x, y, v = kpts[i]
        hm_x = x * 48 / target_w
        hm_y = y * 48 / target_h
        if v > 0:
            axes[0].plot(hm_x, hm_y, 'g+', markersize=10, markeredgewidth=2)
    axes[0].set_title('No Flip - Heatmap + Keypoints')

    # 热图叠加（翻转）
    heatmap_flip = results_flip['gt_keypoints_heatmap']
    combined_flip = heatmap_flip.max(axis=0)
    axes[1].imshow(combined_flip, cmap='hot')
    for i in range(7):
        x, y, v = kpts_flip[i]
        hm_x = x * 48 / target_w
        hm_y = y * 48 / target_h
        if v > 0:
            axes[1].plot(hm_x, hm_y, 'g+', markersize=10, markeredgewidth=2)
    axes[1].set_title('Flipped - Heatmap + Keypoints')

    plt.tight_layout()
    plt.savefig('debug_heatmap.png', dpi=150)
    print("\n✔ 已保存可视化到 debug_heatmap.png")


if __name__ == '__main__':
    simulate_pipeline()
