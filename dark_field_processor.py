# dark_field_processor.py
import cv2
import numpy as np


def process_dark_field(img_df_gray, thresh_df, show_mask=True,
                       ksize=3, iters=1):
    """
    暗場影像處理：
    - 輸入：灰階暗場影像、threshold、是否顯示 mask、膨脹 kernel / iterations
    - 輸出：
        view_df_bgr: 要在 DF viewer 顯示的 BGR 圖
        mask_df_raw: 原始 threshold 後的 DF mask
        mask_df_dilated: 膨脹後的 DF mask（iter=0 或 ksize<=1 則與 raw 相同）
    """
    if img_df_gray is None:
        return None, None, None

    # 1. 二值化
    _, mask_df_raw = cv2.threshold(img_df_gray, thresh_df, 255, cv2.THRESH_BINARY)

    # 2. 膨脹處理（iterations=0 表示不做膨脹）
    if iters > 0 and ksize > 1:
        kernel = np.ones((ksize, ksize), np.uint8)
        mask_df_dilated = cv2.dilate(mask_df_raw, kernel, iterations=iters)
    else:
        mask_df_dilated = mask_df_raw

    # 3. 顯示用 BGR（暗場 viewer 顯示的是膨脹後的結果）
    view_df_bgr = cv2.cvtColor(img_df_gray, cv2.COLOR_GRAY2BGR)
    if show_mask:
        # 用綠色標記 DF mask（膨脹後）
        view_df_bgr[mask_df_dilated == 255] = [0, 255, 0]

    return view_df_bgr, mask_df_raw, mask_df_dilated
