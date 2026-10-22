"""
Network definitions and inference, matching the DMSAAI training code so the
published weights load without changes. torch and monai are imported lazily
because they are installed on first use inside Slicer.
"""

import os

import numpy as np

WEIGHTS_BASE_URL = "https://raw.githubusercontent.com/4burakfe/DMSAAI/main/"

SEGMENTATION_MODELS = {
    "SwinUNETR (2D)": "swinunetr_seg.pth",
    "SegResNet (2D)": "segresnet_seg.pth",
    "DynUNet (2D)": "dynunet_seg.pth",
}

CLASSIFIER_FILES = {
    "Anomaly": {"left": "cortical_anomaly_left.pth", "right": "cortical_anomaly_right.pth"},
    "Normal": {"left": "normal_kidney_left.pth", "right": "normal_kidney_right.pth"},
    "Afunc": {"left": "afunctional_left.pth", "right": "afunctional_right.pth"},
}
HORSESHOE_FILE = "horseshoe.pth"

# Probability cut-offs from DMSAAI.
THRESHOLDS = {
    "Anomaly": {"left": 0.677, "right": 0.353},
    "Normal": {"left": 0.713, "right": 0.810},
    "Afunc": {"left": 0.620, "right": 0.580},
    "Horseshoe": 0.710,
}

ALL_WEIGHT_FILES = (list(SEGMENTATION_MODELS.values())
                    + [f for files in CLASSIFIER_FILES.values() for f in files.values()]
                    + [HORSESHOE_FILE])


def _build_segmentation_network(name):
    import torch.nn as nn
    from monai.networks.nets import SwinUNETR, SegResNet, DynUNet

    # The wrappers keep the "model." prefix of the trained state dict keys.
    class SwinUNETRR(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SwinUNETR(in_channels=1, out_channels=3, feature_size=24, num_heads=(3, 6, 12, 24),
                                   depths=(2, 2, 2, 2), spatial_dims=2, drop_rate=0.1, use_checkpoint=True)

        def forward(self, x):
            return self.model(x)

    class SegResNett(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SegResNet(spatial_dims=2, in_channels=1, out_channels=3, init_filters=32,
                                   blocks_down=(1, 2, 2, 4))

        def forward(self, x):
            return self.model(x)

    class DynUNett(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DynUNet(spatial_dims=2, in_channels=1, out_channels=3,
                                 kernel_size=[[3, 3]] * 5,
                                 strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
                                 upsample_kernel_size=[[2, 2]] * 4,
                                 res_block=True)

        def forward(self, x):
            return self.model(x)

    if name.startswith("SwinUNETR"):
        return SwinUNETRR()
    if name.startswith("SegResNet"):
        return SegResNett()
    if name.startswith("DynUNet"):
        return DynUNett()
    raise ValueError(f"Unknown segmentation model: {name}")


def _build_classifier():
    from monai.networks.nets import DenseNet
    return DenseNet(spatial_dims=2, in_channels=1, out_channels=2, init_features=32, growth_rate=16,
                    block_config=(6, 12, 24, 16), dropout_prob=0.1)


def default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(model, path, device):
    import torch
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _to_tensor(image, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def segment(image, modelName, weightsDir, device=None):
    """Run a segmentation network on a preprocessed 256x256 posterior image. Returns softmax (3, H, W)."""
    import torch
    import torch.nn.functional as F
    device = device or default_device()
    model = _load(_build_segmentation_network(modelName),
                  os.path.join(weightsDir, SEGMENTATION_MODELS[modelName]), device)
    with torch.no_grad():
        probs = F.softmax(model(_to_tensor(image, device)), dim=1)
    return probs.cpu().numpy()[0]


def _positive_probability(model, tensor):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1)[0, 1].item()


def classify(image, weightsDir, device=None):
    """Run the horseshoe and per-kidney evaluation models on the preprocessed posterior image.

    Returns a dict with the raw probabilities and the findings text, using the
    DMSAAI thresholds and wording.
    """
    device = device or default_device()
    tensor = _to_tensor(image, device)
    result = {"horseshoe": None, "probabilities": {}, "findings": {}}

    hkPath = os.path.join(weightsDir, HORSESHOE_FILE)
    if os.path.exists(hkPath):
        p = _positive_probability(_load(_build_classifier(), hkPath, device), tensor)
        result["horseshoe"] = p

    for side in ("left", "right"):
        findings = []
        for typeName, files in CLASSIFIER_FILES.items():
            p = _positive_probability(_load(_build_classifier(), os.path.join(weightsDir, files[side]), device),
                                      tensor)
            result["probabilities"][(typeName, side)] = p
            if p >= THRESHOLDS[typeName][side]:
                if typeName == "Anomaly":
                    findings.append(f"Probable Abnormal Cortical Uptake (conf: {p:.2f})")
                elif typeName == "Normal":
                    findings.append(f"Probably Normal Kidney (conf: {p:.2f})")
                elif typeName == "Afunc":
                    findings.append(f"Probable Afunctional/Agenetic Kidney (conf: {p:.2f})")
            else:
                if typeName == "Anomaly":
                    findings.append(f"Probably Normal Cortical Uptake (conf: {1 - p:.2f})")
                elif typeName == "Normal":
                    findings.append(f"Probably Abnormal Kidney (conf: {1 - p:.2f})")
        result["findings"][side] = findings
    return result


def format_classification(result):
    lines = ["=== Clinical Evaluation Results ===", ""]
    hk = result.get("horseshoe")
    if hk is not None and hk >= THRESHOLDS["Horseshoe"]:
        lines += [f"Global Morphology: Probable Horseshoe Kidney Anomaly (conf: {hk:.2f})", ""]
    for side in ("left", "right"):
        findings = result["findings"].get(side) or []
        text = ", ".join(findings) if findings else "Indeterminate"
        lines.append(f"{side.capitalize()} Kidney: {text}")
    lines += ["", "Research use only. Not for standalone clinical decision making."]
    return "\n".join(lines)


def missing_weights(weightsDir, files=None):
    files = files or ALL_WEIGHT_FILES
    return [f for f in files if not os.path.exists(os.path.join(weightsDir, f))]


def download_weights(weightsDir, files=None, progress=None):
    """Download missing weight files from the DMSAAI repository."""
    import urllib.request
    os.makedirs(weightsDir, exist_ok=True)
    for name in missing_weights(weightsDir, files):
        if progress:
            progress(f"Downloading {name}...")
        target = os.path.join(weightsDir, name)
        tmp = target + ".part"
        urllib.request.urlretrieve(WEIGHTS_BASE_URL + name, tmp)
        os.replace(tmp, target)
