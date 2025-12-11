# bright_field_processor.py
import cv2
import numpy as np


def process_bright_field(img_bf_gray, thresh_bf, show_mask=True):
    """
    Bright field image processing:
    - Input: grayscale BF image, threshold, whether to show mask
    - Output:
        view_bf_bgr: BGR image for the BF viewer
        mask_bf: binary mask for defects and dust (0/255)
    """
    if img_bf_gray is None:
        return None, None

    # Binary mask
    _, mask_bf = cv2.threshold(img_bf_gray, thresh_bf, 255, cv2.THRESH_BINARY)

    # BGR image for display
    view_bf_bgr = cv2.cvtColor(img_bf_gray, cv2.COLOR_GRAY2BGR)
    if show_mask:
        # Mark BF mask in red
        view_bf_bgr[mask_bf == 255] = [0, 0, 255]

    return view_bf_bgr, mask_bf
