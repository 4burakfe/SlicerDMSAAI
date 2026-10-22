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
