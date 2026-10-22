"""
Stand-alone tests of the DMSAAI helpers (no Slicer needed).

    pip install pydicom numpy scipy opencv-python-headless pytest
    # optional, for the network tests:
    pip install torch "monai>=1.3" einops

    DMSAAI_DIR=/path/to/DMSAAI pytest Testing

DMSAAI_DIR is a checkout of https://github.com/4burakfe/DMSAAI, which provides
the sample studies (images_to_test) and the model weights.
"""

import glob
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DMSAAI"))

from DMSAAILib import dicomviews, processing  # noqa: E402

DMSAAI_DIR = os.environ.get("DMSAAI_DIR", "")
SAMPLES = sorted(glob.glob(os.path.join(DMSAAI_DIR, "images_to_test", "*"))) if DMSAAI_DIR else []
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("monai") is not None

needs_samples = pytest.mark.skipif(not SAMPLES, reason="set DMSAAI_DIR to a DMSAAI checkout")


def study_files(study):
    return sorted(glob.glob(os.path.join(study, "*.dcm")))


def test_view_from_angle():
    assert dicomviews.view_from_angle(180) == "ant"
    assert dicomviews.view_from_angle(360) == "post"
    assert dicomviews.view_from_angle(0) == "post"
    assert dicomviews.view_from_angle(3) == "post"
    assert dicomviews.view_from_angle(135) == "lao"
    assert dicomviews.view_from_angle(315) == "rpo"
    assert dicomviews.view_from_angle(45) == "lpo"
    assert dicomviews.view_from_angle(225) == "rao"
    assert dicomviews.view_from_angle(90) is None


def test_view_from_text():
    assert dicomviews.view_from_text("DMSA ANT") == "ant"
    assert dicomviews.view_from_text("Posterior") == "post"
    assert dicomviews.view_from_text("ANT-POST") is None
    assert dicomviews.view_from_text("LPO - RAO") is None


@pytest.mark.parametrize("text", [
    "DMSA", "Static Kidney", "Statik Böbrek", "STATİK BÖBREK", "Kortikal Böbrek", "Cortical Kidney",
    "Renal Cortical", "1-DMSA", "Bobrek Parankim Sintigrafisi, Planar Tc-99m DMSA", "Tc99m-DMSA", "Succimer",
    "Renal cortex scan", "Kidney_Cortical", "static renal"])
def test_dmsa_keywords_match(text):
    from types import SimpleNamespace
    assert dicomviews.is_dmsa_study(SimpleNamespace(StudyDescription=text))


@pytest.mark.parametrize("text", ["Whole Body Bone", "MAG3 Renogram", "Thyroid", "DTPA", "Static Bone", ""])
def test_dmsa_keywords_reject(text):
    from types import SimpleNamespace
    assert not dicomviews.is_dmsa_study(SimpleNamespace(StudyDescription=text, SeriesDescription="ANT-POST"))


def test_dmsa_custom_keywords():
    from types import SimpleNamespace
    ds = SimpleNamespace(StudyDescription="Renal Sintigrafi")
    assert not dicomviews.is_dmsa_study(ds)
    assert dicomviews.is_dmsa_study(ds, ["renal sintigrafi"])
    assert not dicomviews.is_dmsa_study(SimpleNamespace(StudyDescription="DMSA"), [])


def test_preprocess_pads_and_crops():
    assert processing.preprocess(np.ones((128, 100))).shape == (256, 256)
    assert processing.preprocess(np.ones((300, 512))).shape == (256, 256)
    img = processing.preprocess(np.ones((64, 64)), normalize=True)
    assert np.isclose(img.sum(), processing.TARGET_TOTAL_COUNTS)


def test_srf_synthetic():
    image = np.zeros((256, 256), dtype=np.float32)
    image[100:140, 70:100] = 30.0
    image[100:140, 160:190] = 10.0
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:140, 70:100] = processing.LEFT
    mask[100:140, 160:190] = processing.RIGHT
    for method in processing.BACKGROUND_METHODS:
        result = processing.split_renal_function(image, mask, np.fliplr(image), method=method)
        assert result["postSRF"][processing.LEFT] == pytest.approx(75.0)
        assert result["gmSRF"][processing.LEFT] == pytest.approx(75.0)


@needs_samples
@pytest.mark.parametrize("study", SAMPLES, ids=lambda s: s[-8:])
def test_sample_views(study):
    files = study_files(study)
    for f in files:
        assert dicomviews.is_planar_nm(dicomviews.read_dataset(f, stop_before_pixels=True))
    assert any(dicomviews.is_dmsa_study(dicomviews.read_dataset(f, stop_before_pixels=True)) for f in files)
    views, patient = dicomviews.collect_views(files)
    assert "post" in views and "ant" in views
    assert views["post"].pixels.shape == (256, 256)
    assert set(views) <= set(dicomviews.VIEW_KEYS)


@needs_samples
@pytest.mark.skipif(not HAS_TORCH, reason="torch/monai not installed")
@pytest.mark.parametrize("study", SAMPLES[:3], ids=lambda s: s[-8:])
def test_sample_segmentation(study):
    from DMSAAILib import networks
    views, _ = dicomviews.collect_views(study_files(study))
    post = processing.preprocess(views["post"].pixels)
    ant = processing.preprocess(views["ant"].pixels)
    probs = networks.segment(post, "SegResNet (2D)", DMSAAI_DIR)
    mask = processing.masks_from_probabilities(probs)
    assert (mask > 0).sum() > 100
    result = processing.split_renal_function(post, mask, ant)
    assert 0 <= result["gmSRF"][processing.LEFT] <= 100
    assert result["gmSRF"][processing.LEFT] + result["gmSRF"][processing.RIGHT] == pytest.approx(100)
