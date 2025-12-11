# bright_field_processor.py
import cv2
import numpy as np


def process_bright_field(
    img_bf_gray,
    thresh_bf,
    show_mask=True,
    blur_enabled=False,
    blur_ksize=3,
    inverse_threshold=False,
):
    """
    Bright field image processing:
    - Input: grayscale BF image, threshold, whether to show mask
    - Output:
        bf_for_process: grayscale image after preprocessing (e.g., blur)
        mask_bf: binary mask for defects and dust (0/255)
        view_bf_bgr: BGR image for the BF viewer
    """
    if img_bf_gray is None:
        return None, None, None

    bf_for_process = img_bf_gray
    if blur_enabled and blur_ksize > 1:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        bf_for_process = cv2.GaussianBlur(img_bf_gray, (ksize, ksize), 0)

    # Binary mask
    threshold_type = cv2.THRESH_BINARY_INV if inverse_threshold else cv2.THRESH_BINARY
    _, mask_bf = cv2.threshold(bf_for_process, thresh_bf, 255, threshold_type)

    # BGR image for display
    view_bf_bgr = cv2.cvtColor(bf_for_process, cv2.COLOR_GRAY2BGR)
    if show_mask:
        # Mark BF mask in red
        view_bf_bgr[mask_bf == 255] = [0, 0, 255]

    return bf_for_process, mask_bf, view_bf_bgr
