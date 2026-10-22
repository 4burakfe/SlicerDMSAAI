"""Persistent DMSAAI settings (Qt settings of the Slicer application)."""

import qt

from . import dicomviews

DMSA_KEYWORDS_KEY = "DMSAAI/DMSAKeywords"


def dmsa_keywords():
    """Keywords that mark a study as DMSA: the user's list, or the defaults if none was saved."""
    value = qt.QSettings().value(DMSA_KEYWORDS_KEY)
    if value is None:
        return list(dicomviews.DEFAULT_DMSA_KEYWORDS)
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def set_dmsa_keywords(keywords):
    qt.QSettings().setValue(DMSA_KEYWORDS_KEY, "\n".join(k.strip() for k in keywords if k.strip()))


def reset_dmsa_keywords():
    qt.QSettings().remove(DMSA_KEYWORDS_KEY)


# Analysis and display options that the user can save with "Save settings". Combo boxes are stored by item
# text, so reordering the lists does not change a saved choice.
ANALYSIS_SETTINGS_GROUP = "DMSAAI/Analysis/"
DEFAULT_ANALYSIS_SETTINGS = {
    "SegmentationArchitecture": "SwinUNETR (2D)",
    "ClassifierArchitecture": "DenseNet121",
    "Normalize": True,
    "Resample": True,
    "LeftThreshold": 50.0,
    "RightThreshold": 50.0,
    "BackgroundMethod": "Circumferential",
    "Colormap": "Grey",
}


def _convert(value, default):
    # QSettings may hand values back as strings (e.g. "true", "50"), depending on the platform backend.
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes")
    if isinstance(default, float):
        return float(value)
    return str(value)


def analysis_settings():
    """Saved analysis settings, with the defaults for anything not saved or unreadable."""
    store = qt.QSettings()
    result = dict(DEFAULT_ANALYSIS_SETTINGS)
    for key, default in DEFAULT_ANALYSIS_SETTINGS.items():
        value = store.value(ANALYSIS_SETTINGS_GROUP + key)
        if value is None:
            continue
        try:
            result[key] = _convert(value, default)
        except (TypeError, ValueError):
            pass
    return result


def save_analysis_settings(values):
    store = qt.QSettings()
    for key in DEFAULT_ANALYSIS_SETTINGS:
        if key in values:
            store.setValue(ANALYSIS_SETTINGS_GROUP + key, values[key])


def reset_analysis_settings():
    store = qt.QSettings()
    for key in DEFAULT_ANALYSIS_SETTINGS:
        store.remove(ANALYSIS_SETTINGS_GROUP + key)
