# dark_field_processor.py
import cv2
import numpy as np


def process_dark_field(img_df_gray, thresh_df, show_mask=True,
                       ksize=3, iters=1):
    """
    Dark field image processing:
    - Input: grayscale DF image, threshold, whether to show mask, dilation kernel size/iterations
    - Output:
        img_df_gray: grayscale image before binarization
        mask_df_raw: raw DF mask after thresholding
        mask_df_dilated: dilated DF mask (same as raw when iter=0 or ksize<=1)
        view_df_bgr: BGR image for the DF viewer
    """
    if img_df_gray is None:
        return None, None, None, None

    # 1. Thresholding
    _, mask_df_raw = cv2.threshold(img_df_gray, thresh_df, 255, cv2.THRESH_BINARY)

    # 2. Dilation (iterations=0 means skip dilation)
    if iters > 0 and ksize > 1:
        kernel = np.ones((ksize, ksize), np.uint8)
        mask_df_dilated = cv2.dilate(mask_df_raw, kernel, iterations=iters)
    else:
        mask_df_dilated = mask_df_raw

    # 3. BGR image for display (viewer uses the dilated result)
    view_df_bgr = cv2.cvtColor(img_df_gray, cv2.COLOR_GRAY2BGR)
    if show_mask:
        # Mark the dilated DF mask in green
        view_df_bgr[mask_df_dilated == 255] = [0, 255, 0]

    return img_df_gray, mask_df_raw, mask_df_dilated, view_df_bgr
