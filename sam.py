from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam = sam.cuda()

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=10, #调参
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
)

def score_mask(mask, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    region_rgb = image[mask]
    region_hsv = hsv[mask]
    region_lab = lab[mask]

    if len(region_rgb) == 0:
        return -1

    # RGB
    R, G, B = region_rgb[:,0], region_rgb[:,1], region_rgb[:,2]
    rgb_score = np.mean(R) + np.mean(G) - 1.5 * np.mean(B) #调参 关注棕色部分同时剔除正常细胞

    # HSV（棕色）
    H, S = region_hsv[:,0], region_hsv[:,1]
    hsv_score = np.sum((H > 10) & (H < 30) & (S > 50)) / len(H)

    # LAB
    L, A, B_lab = region_lab[:,0], region_lab[:,1], region_lab[:,2]
    lab_score = np.mean(A) + np.mean(B_lab) - np.mean(L)

    # 综合评分
    score = 0.4 * rgb_score + 0.3 * hsv_score + 0.3 * lab_score

    return score

def generate_mask(image):
    # image is RGB numpy array, uint8
    masks = mask_generator.generate(image)

    selected_masks = []

    for m in masks:
        seg = m['segmentation']
        area = m['area']

        # 面积过滤（去掉太小的）
        if area < 300:
            continue

        s = score_mask(seg, image)

        if s > 0.3:   # 这个阈值可以调
            selected_masks.append(seg)

    if not selected_masks:
        final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    else:
        final_mask = np.zeros_like(selected_masks[0], dtype=np.uint8)

        for m in selected_masks:
            final_mask = final_mask | m.astype(np.uint8)

    final_mask = final_mask.astype(np.float32)

    final_mask = cv2.GaussianBlur(final_mask, (7,7), 0)

    # 归一化到 0~1
    final_mask = final_mask / final_mask.max() if final_mask.max() > 0 else final_mask

    return final_mask