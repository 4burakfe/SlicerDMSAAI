"""
Image preprocessing and quantification ported from DMSAAI (dmsaai.py).

All arrays are 2D numpy arrays in raw DICOM pixel order (row, column).
Label convention of the segmentation networks: 0 = background,
1 = left kidney, 2 = right kidney (posterior view).
"""

import numpy as np

TARGET_SHAPE = (256, 256)
TARGET_SPACING_MM = 2.0
TARGET_TOTAL_COUNTS = 200000.0

LEFT = 1
RIGHT = 2
LABEL_NAMES = {LEFT: "Left", RIGHT: "Right"}

BACKGROUND_METHODS = ("Circumferential", "Center Rectangle", "Inferolateral Crescent")


def preprocess(image, spacing=None, resample=False, normalize=False, log=None):
    """Optional 2x2 mm resampling, optional 200k count normalization, then centre crop/pad to 256x256."""
    img = np.asarray(image).astype(np.float32)

    if resample:
        if spacing is not None:
            factor_y = spacing[0] / TARGET_SPACING_MM
            factor_x = spacing[1] / TARGET_SPACING_MM
            if factor_y != 1.0 or factor_x != 1.0:
                import cv2
                new_w = int(img.shape[1] * factor_x)
                new_h = int(img.shape[0] * factor_y)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        elif log:
            log("Warning: no PixelSpacing metadata. Skipping resample.")

    if normalize:
        total_counts = np.sum(img)
        if total_counts > 0:
            img = img * (TARGET_TOTAL_COUNTS / total_counts)

    th, tw = TARGET_SHAPE
    h, w = img.shape
    if h > th or w > tw:
        y_start = max((h - th) // 2, 0)
        y_end = min(y_start + th, h)
        x_start = max((w - tw) // 2, 0)
        x_end = min(x_start + tw, w)
        img = img[y_start:y_end, x_start:x_end]
        h, w = img.shape

    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        img = np.pad(img, ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
                     mode="constant", constant_values=0)
    return img


def output_spacing(spacing, resample):
    """Pixel spacing (row, col) of a preprocessed image."""
    if resample and spacing is not None:
        return (TARGET_SPACING_MM, TARGET_SPACING_MM)
    return spacing if spacing is not None else (1.0, 1.0)


def masks_from_probabilities(probs, thresh_left=0.5, thresh_right=0.5):
    """Threshold softmax output (3, H, W) into a label map, resolving overlaps by the higher probability."""
    mask = np.zeros_like(probs[0], dtype=np.uint8)
    left = probs[LEFT] >= thresh_left
    right = probs[RIGHT] >= thresh_right
    mask[left] = LEFT
    mask[right] = RIGHT
    overlap = left & right
    if np.any(overlap):
        left_higher = probs[LEFT] > probs[RIGHT]
        mask[overlap & left_higher] = LEFT
        mask[overlap & ~left_higher] = RIGHT
    return mask


def anterior_mask(posterior_mask):
    """The anterior view is the mirror image of the posterior view."""
    return np.fliplr(posterior_mask)


def background_roi(full_mask, image, label_idx, method):
    """Return (mean background count per pixel, background mask) for one kidney."""
    import scipy.ndimage as ndi

    kidney_mask = full_mask == label_idx
    if not np.any(kidney_mask):
        return 0.0, None

    bg_mask = np.zeros_like(full_mask, dtype=bool)

    if method == "Circumferential":
        dilated = ndi.binary_dilation(kidney_mask, iterations=6)
        gap = ndi.binary_dilation(kidney_mask, iterations=2)
        bg_mask = dilated ^ gap

    elif method == "Inferolateral Crescent":
        dilated = ndi.binary_dilation(kidney_mask, iterations=8)
        gap = ndi.binary_dilation(kidney_mask, iterations=3)
        ring_mask = dilated ^ gap

        coords = np.argwhere(full_mask > 0)
        if len(coords) > 0:
            overall_cx = (coords[:, 1].min() + coords[:, 1].max()) / 2
        else:
            overall_cx = image.shape[1] / 2

        cy, cx = ndi.center_of_mass(kidney_mask)
        y_coords, x_coords = np.indices(image.shape)
        if cx < overall_cx:
            directional = (y_coords > cy) & (x_coords < cx)
        else:
            directional = (y_coords > cy) & (x_coords > cx)
        bg_mask = ring_mask & directional
        if not np.any(bg_mask):
            bg_mask = ring_mask

    elif method == "Center Rectangle":
        coords = np.argwhere(full_mask > 0)
        if len(coords) > 0:
            max_y = coords[:, 0].max()
            min_x = coords[:, 1].min()
            max_x = coords[:, 1].max()
            center_x = (min_x + max_x) // 2

            rect_h, rect_w = 10, 40
            y_start = min(max_y + 10, image.shape[0] - rect_h - 1)
            y_start = max(y_start, 0)
            y_end = min(y_start + rect_h, image.shape[0])
            x_start = max(center_x - rect_w // 2, 0)
            x_end = min(x_start + rect_w, image.shape[1])

            if y_start < y_end and x_start < x_end:
                bg_mask[y_start:y_end, x_start:x_end] = True
            else:
                dilated = ndi.binary_dilation(kidney_mask, iterations=8)
                gap = ndi.binary_dilation(kidney_mask, iterations=2)
                bg_mask = dilated ^ gap

    else:
        raise ValueError(f"Unknown background method: {method}")

    bg_pixels = image[bg_mask]
    return (float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 0.0), bg_mask


def kidney_counts(image, mask, method):
    """Gross, background and net counts for both kidneys of one view."""
    result = {}
    for label_idx in LABEL_NAMES:
        kidney = mask == label_idx
        bg_mean, bg_mask = background_roi(mask, image, label_idx, method)
        gross = float(np.sum(image[kidney]))
        area = int(np.sum(kidney))
        net = gross - bg_mean * area
        result[label_idx] = {
            "gross": gross,
            "area": area,
            "bgMean": bg_mean,
            "net": max(0.0, net),
            "bgMask": bg_mask,
        }
    return result


def split_renal_function(post_image, post_mask, ant_image=None, ant_mask=None, method="Circumferential"):
    """Posterior-only and (if anterior is given) geometric-mean split renal function in percent."""
    post = kidney_counts(post_image, post_mask, method)
    total_post = post[LEFT]["net"] + post[RIGHT]["net"]
    result = {
        "method": method,
        "posterior": post,
        "postSRF": {
            LEFT: post[LEFT]["net"] / total_post * 100 if total_post > 0 else 0.0,
            RIGHT: post[RIGHT]["net"] / total_post * 100 if total_post > 0 else 0.0,
        },
        "postTotalCounts": float(np.sum(post_image)),
        "anterior": None,
        "gmSRF": None,
        "antTotalCounts": None,
    }
    if ant_image is not None:
        if ant_mask is None:
            ant_mask = anterior_mask(post_mask)
        ant = kidney_counts(ant_image, ant_mask, method)
        gm_l = np.sqrt(ant[LEFT]["net"] * post[LEFT]["net"])
        gm_r = np.sqrt(ant[RIGHT]["net"] * post[RIGHT]["net"])
        total_gm = gm_l + gm_r
        result["anterior"] = ant
        result["antTotalCounts"] = float(np.sum(ant_image))
        result["gmSRF"] = {
            LEFT: float(gm_l / total_gm * 100) if total_gm > 0 else 0.0,
            RIGHT: float(gm_r / total_gm * 100) if total_gm > 0 else 0.0,
        }
    return result


def format_srf(result):
    lines = ["=== Split Renal Function (SRF) ===", "",
             f"Background ROI: {result['method']}", "",
             "Posterior Only Method:",
             f"  Left Kidney:  {result['postSRF'][LEFT]:.1f} %",
             f"  Right Kidney: {result['postSRF'][RIGHT]:.1f} %", ""]
    if result["gmSRF"] is not None:
        lines += ["Geometric Mean Method:",
                  f"  Left Kidney:  {result['gmSRF'][LEFT]:.1f} %",
                  f"  Right Kidney: {result['gmSRF'][RIGHT]:.1f} %", ""]
    else:
        lines += ["Geometric Mean Method:", "  N/A (Anterior view not provided)", ""]
    return "\n".join(lines)
