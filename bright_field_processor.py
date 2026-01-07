# bright_field_processor.py
import cv2
import numpy as np


def robust_thr_median_mad(
    img_u8: np.ndarray, k: float, min_thr: int = 5, max_thr: int = 255
) -> int:
    x = img_u8.reshape(-1).astype(np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-6)
    thr = med + k * 1.4826 * mad
    return int(np.clip(thr, min_thr, max_thr))


def dog_highpass_u8(img_u8: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    return cv2.subtract(g1, g2)


def process_bright_field(
    img_bf_gray,
    thresh_bf,
    show_mask=True,
    blur_enabled=False,
    blur_ksize=3,
    inverse_threshold=False,
    method="blur_threshold",
    dog_params=None,
):
    """
    Bright field image processing supporting two methods:
    - ``blur_threshold``: optional Gaussian blur then fixed thresholding
    - ``dog_highpass``: single Difference-of-Gaussian high-pass with
      robust median absolute deviation thresholding
    Returns:
        bf_for_process: grayscale image after preprocessing (e.g., blur / DoG)
        mask_bf: binary mask for defects and dust (0/255)
        view_bf_bgr: BGR image for the BF viewer
    """
    if img_bf_gray is None:
        return None, None, None

    if dog_params is None:
        dog_params = {}

    if method == "dog_highpass":
        sigma1 = dog_params.get("sigma1", 0.8)
        sigma2 = dog_params.get("sigma2", 2.4)
        thr_k = dog_params.get("k", 1.0)
        min_thr = dog_params.get("min_thr", 5)
    

        H, W = img_bf_gray.shape
        small = cv2.resize(img_bf_gray, (W//4, H//4), interpolation=cv2.INTER_AREA)
        bf_for_process = dog_highpass_u8(small, sigma1, sigma2)
        thr_val = robust_thr_median_mad(bf_for_process, thr_k, min_thr=min_thr)
        threshold_type = cv2.THRESH_BINARY_INV if inverse_threshold else cv2.THRESH_BINARY
        _, mask_bf = cv2.threshold(bf_for_process, thr_val, 255, threshold_type)
        bf_for_process = cv2.resize(bf_for_process, (W, H), interpolation=cv2.INTER_AREA)
        mask_bf = cv2.resize(mask_bf, (W, H), interpolation=cv2.INTER_AREA)

    else:
        bf_for_process = img_bf_gray
        if blur_enabled and blur_ksize > 1:
            ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            bf_for_process = cv2.GaussianBlur(img_bf_gray, (ksize, ksize), 0)

        threshold_type = cv2.THRESH_BINARY_INV if inverse_threshold else cv2.THRESH_BINARY
        _, mask_bf = cv2.threshold(bf_for_process, thresh_bf, 255, threshold_type)

    view_bf_bgr = cv2.cvtColor(bf_for_process, cv2.COLOR_GRAY2BGR)
    if show_mask:
        view_bf_bgr[mask_bf == 255] = [0, 0, 255]

    return bf_for_process, mask_bf, view_bf_bgr
