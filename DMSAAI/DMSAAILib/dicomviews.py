"""
Pure pydicom/numpy helpers for planar Tc-99m DMSA nuclear medicine images.

Nothing in this file depends on Slicer, so it can be shared by the DICOM plugin,
the module logic and the stand-alone tests.

Orientation policy
------------------
The networks of DMSAAI were trained on the *raw* DICOM pixel array (rows/columns
exactly as stored in the file). Slicer, on the other hand, honours the
ImageOrientationPatient of every detector: for a posterior view this is
[-1,0,0, 0,0,-1], for an anterior view [1,0,0, 0,0,-1], so in the coronal slice
viewer one of the two always appears horizontally mirrored compared with the
nuclear medicine reading convention. Multi-frame files (ANT+POST in one file)
often carry no orientation at all, so Slicer guesses differently for them.

To make the module robust we therefore always work on the raw stored pixel
array and never trust the geometry of a scalar volume for quantification.
"""

import os
from dataclasses import dataclass, field

import numpy as np

NM_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.20"

VIEW_KEYS = ("ant", "post", "lao", "rpo", "lpo", "rao")

VIEW_NAMES = {
    "ant": "Anterior",
    "post": "Posterior",
    "lao": "Left Anterior Oblique",
    "rpo": "Right Posterior Oblique",
    "lpo": "Left Posterior Oblique",
    "rao": "Right Anterior Oblique",
    "unknown": "Unknown",
}

# Detector start angle (degrees) for each view, as used by DMSAAI.
VIEW_ANGLES = {
    "ant": (180,),
    "post": (0, 360),
    "lao": (135,),
    "rpo": (315,),
    "lpo": (45,),
    "rao": (225,),
}

# ImageType values that mean the object is not a planar static acquisition.
NON_PLANAR_IMAGE_TYPES = ("TOMO", "RECON TOMO", "GATED TOMO", "RECON GATED TOMO", "DYNAMIC", "GATED")


@dataclass
class PlanarView:
    """One planar frame extracted from a DICOM file."""
    view: str
    pixels: np.ndarray
    spacing: tuple = None  # (row spacing, column spacing) in mm, or None
    angle: float = None
    frame: int = 0
    detector: int = None
    imageOrientation: tuple = None  # ImageOrientationPatient of the detector, if present
    sourceFile: str = None
    warnings: list = field(default_factory=list)

    @property
    def name(self):
        return VIEW_NAMES.get(self.view, self.view)


def view_from_text(text):
    """Guess a view from free text (DatasetName, SeriesDescription, ProtocolName)."""
    desc = (text or "").lower()
    if "ant" in desc and "post" not in desc:
        return "ant"
    if "post" in desc and "ant" not in desc:
        return "post"
    if "lao" in desc and "rpo" not in desc:
        return "lao"
    if "rpo" in desc and "lao" not in desc:
        return "rpo"
    if "lpo" in desc and "rao" not in desc:
        return "lpo"
    if "rao" in desc and "lpo" not in desc:
        return "rao"
    return None


def view_from_angle(angle, tolerance=10.0):
    """Map a detector start angle to a view (same rule as DMSAAI)."""
    if angle is None:
        return None
    angle = float(angle) % 360.0
    for view, refs in VIEW_ANGLES.items():
        for ref in refs:
            if abs(angle - ref) < tolerance:
                return view
    return None


def frame_detector_index(ds, frame):
    """0-based index into DetectorInformationSequence for a frame."""
    vector = getattr(ds, "DetectorVector", None)
    if vector is not None:
        try:
            values = list(vector) if not isinstance(vector, (int, float)) else [vector]
            if frame < len(values):
                return int(values[frame]) - 1
        except (TypeError, ValueError):
            pass
    return frame


def detector_item(ds, frame):
    seq = getattr(ds, "DetectorInformationSequence", None)
    if not seq:
        return None
    index = frame_detector_index(ds, frame)
    if 0 <= index < len(seq):
        return seq[index]
    if frame < len(seq):
        return seq[frame]
    return None


def detector_angle(ds, frame=0):
    item = detector_item(ds, frame)
    try:
        if item is not None and hasattr(item, "StartAngle"):
            return float(item.StartAngle)
        if hasattr(ds, "StartAngle"):
            return float(ds.StartAngle)
    except (TypeError, ValueError):
        pass
    return None


def detector_orientation(ds, frame=0):
    item = detector_item(ds, frame)
    value = None
    if item is not None and hasattr(item, "ImageOrientationPatient"):
        value = item.ImageOrientationPatient
    elif hasattr(ds, "ImageOrientationPatient"):
        value = ds.ImageOrientationPatient
    if value is None:
        return None
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None


def description_text(ds):
    return " ".join(str(getattr(ds, key, "")) for key in ("DatasetName", "SeriesDescription", "ProtocolName"))


def determine_view(ds, frame=0, warnings=None):
    """Identify the view of a frame. The detector angle wins over the description text."""
    angleHint = view_from_angle(detector_angle(ds, frame))
    nameHint = view_from_text(description_text(ds))
    if angleHint and nameHint and angleHint != nameHint and warnings is not None:
        warnings.append(f"Metadata conflict: angle suggests {angleHint.upper()} but text suggests "
                        f"{nameHint.upper()}. Using the angle.")
    return angleHint or nameHint or "unknown"


# Default phrases that identify a DMSA / static renal cortical study. A phrase matches when its
# words appear consecutively in a description after normalize_text() (case, Turkish letters and
# punctuation are ignored), e.g. "Statik Böbrek" matches "STATIK_BOBREK" and "Tc99m-DMSA" matches "DMSA".
# Users can edit the list in the module (Advanced > DMSA study keywords).
DEFAULT_DMSA_KEYWORDS = (
    "DMSA",
    "Succimer",
    "Static Kidney",
    "Static Renal",
    "Statik Böbrek",
    "Kortikal Böbrek",
    "Böbrek Korteks",
    "Böbrek Parankim",
    "Cortical Kidney",
    "Kidney Cortical",
    "Kidney Cortex",
    "Renal Cortical",
    "Renal Cortex",
    "Cortical Renal",
    "Renal Parenchyma",
)

_TURKISH_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "İ": "i", "ö": "o", "ş": "s", "ü": "u",
                               "Ç": "c", "Ğ": "g", "Ö": "o", "Ş": "s", "Ü": "u"})


def normalize_text(text):
    import re
    text = str(text or "").translate(_TURKISH_FOLD).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def study_text(ds):
    """Descriptions and radiopharmaceutical names that can identify the study type."""
    parts = [str(getattr(ds, key, "") or "") for key in (
        "StudyDescription", "SeriesDescription", "ProtocolName", "DatasetName",
        "PerformedProcedureStepDescription", "RequestedProcedureDescription")]
    for item in getattr(ds, "RadiopharmaceuticalInformationSequence", None) or []:
        parts.append(str(getattr(item, "Radiopharmaceutical", "") or ""))
        for code in getattr(item, "RadiopharmaceuticalCodeSequence", None) or []:
            parts.append(str(getattr(code, "CodeMeaning", "") or ""))
    return " | ".join(p for p in parts if p)


def is_dmsa_study(ds, keywords=None):
    """True if a description or radiopharmaceutical name contains one of the DMSA keywords."""
    text = f" {normalize_text(study_text(ds))} "
    for keyword in keywords if keywords is not None else DEFAULT_DMSA_KEYWORDS:
        phrase = normalize_text(keyword)
        if phrase and f" {phrase} " in text:
            return True
    return False


def is_nm(ds):
    return getattr(ds, "Modality", "") == "NM" or NM_IMAGE_STORAGE in str(getattr(ds, "SOPClassUID", ""))


def is_planar_nm(ds):
    """True for NM static/whole-body planar images (not SPECT, gated or dynamic)."""
    if not is_nm(ds):
        return False
    imageType = [str(v).upper() for v in (getattr(ds, "ImageType", None) or [])]
    joined = " ".join(imageType)
    if any(t in imageType for t in NON_PLANAR_IMAGE_TYPES) or "TOMO" in joined:
        return False
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    # Planar DMSA files have one frame per view; SPECT projections have dozens.
    return frames <= 8


def pixel_spacing(ds):
    try:
        spacing = ds.PixelSpacing
        return (float(spacing[0]), float(spacing[1]))
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def extract_views(ds, sourceFile=None):
    """Split a dataset into PlanarView objects (one per frame) using the raw pixel array."""
    arr = ds.pixel_array
    spacing = pixel_spacing(ds)
    views = []
    if arr.ndim == 2:
        frames = [arr]
    elif arr.ndim == 3:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        return views
    for index, pixels in enumerate(frames):
        warnings = []
        item = detector_item(ds, index)
        detector = frame_detector_index(ds, index) + 1 if item is not None else None
        views.append(PlanarView(
            view=determine_view(ds, index, warnings),
            pixels=np.array(pixels),
            spacing=spacing,
            angle=detector_angle(ds, index),
            frame=index,
            detector=detector,
            imageOrientation=detector_orientation(ds, index),
            sourceFile=sourceFile,
            warnings=warnings,
        ))
    return views


def read_dataset(path, stop_before_pixels=False):
    import pydicom
    return pydicom.dcmread(path, force=True, stop_before_pixels=stop_before_pixels)


def collect_views(paths, log=None):
    """Read many files and fill one slot per view, like DMSAAI.process_files.

    Returns (views dict keyed by view, patient dict).
    """
    def _log(msg):
        if log:
            log(msg)

    views = {}
    patient = {}
    for path in paths:
        try:
            ds = read_dataset(path)
        except Exception:
            continue
        if not is_nm(ds):
            _log(f"Ignored {os.path.basename(path)}: not an NM image (Modality: {getattr(ds, 'Modality', '')}).")
            continue
        if not hasattr(ds, "PixelData"):
            _log(f"Ignored {os.path.basename(path)}: no pixel data.")
            continue
        pid = str(getattr(ds, "PatientID", "Unknown_ID"))
        if not patient:
            patient = {"PatientID": pid, "PatientName": str(getattr(ds, "PatientName", "Unknown_Name"))}
        elif pid != patient["PatientID"]:
            _log(f"Ignored {os.path.basename(path)}: patient ID mismatch.")
            continue
        for planar in extract_views(ds, sourceFile=path):
            for warning in planar.warnings:
                _log(warning)
            if planar.view in VIEW_KEYS and planar.view not in views:
                views[planar.view] = planar
                _log(f"Assigned {os.path.basename(path)} frame {planar.frame} to {planar.view.upper()}.")
    return views, patient
