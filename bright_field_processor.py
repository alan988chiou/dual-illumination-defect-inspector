# bright_field_processor.py
import cv2
import numpy as np


def process_bright_field(img_bf_gray, thresh_bf, show_mask=True):
    """
    明場影像處理：
    - 輸入：灰階明場影像、threshold、是否顯示 mask
    - 輸出：
        view_bf_bgr: 要在 BF viewer 顯示的 BGR 圖
        mask_bf: 0/255 的二值 mask（瑕疵 + 灰塵）
    """
    if img_bf_gray is None:
        return None, None

    # 二值化
    _, mask_bf = cv2.threshold(img_bf_gray, thresh_bf, 255, cv2.THRESH_BINARY)

    # 顯示用 BGR
    view_bf_bgr = cv2.cvtColor(img_bf_gray, cv2.COLOR_GRAY2BGR)
    if show_mask:
        # 用紅色標記 BF mask
        view_bf_bgr[mask_bf == 255] = [0, 0, 255]

    return view_bf_bgr, mask_bf
