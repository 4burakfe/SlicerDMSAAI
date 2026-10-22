import logging
import os

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from DICOMLib import DICOMLoadable, DICOMPlugin

from DMSAAILib import dicomviews, volumes


#
# DICOMDmsaaiPluginClass
#

class DICOMDmsaaiPluginClass(DICOMPlugin):
    """Loads planar nuclear medicine (Tc-99m DMSA) images.

    Each view (ANT, POST and the obliques) becomes its own scalar volume, split
    out of multi-frame files, with the fixed display geometry described in
    DMSAAILib.volumes and the DICOM metadata the DMSAAI module needs stored as
    node attributes.
    """

    def __init__(self):
        super().__init__()
        self.loadType = "DMSA Planar NM"
        self.tags["modality"] = "0008,0060"
        self.tags["sopClassUID"] = "0008,0016"
        self.tags["seriesDescription"] = "0008,103e"
        self.tags["patientName"] = "0010,0010"

    def examine(self, fileLists):
        loadables = []
        for files in fileLists:
            loadables += self.examineFiles(files)
        return loadables

    def examineFiles(self, files):
        candidates = []
        for f in files:
            modality = slicer.dicomDatabase.fileValue(f, self.tags["modality"]) if slicer.dicomDatabase else ""
            sopClass = slicer.dicomDatabase.fileValue(f, self.tags["sopClassUID"]) if slicer.dicomDatabase else ""
            if modality and modality != "NM" and dicomviews.NM_IMAGE_STORAGE not in (sopClass or ""):
                continue
            candidates.append(f)
        if not candidates:
            return []

        planarFiles = []
        viewNames = []
        for f in candidates:
            try:
                ds = dicomviews.read_dataset(f, stop_before_pixels=True)
            except Exception:
                continue
            if not dicomviews.is_planar_nm(ds):
                continue
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            planarFiles.append(f)
            viewNames += [dicomviews.determine_view(ds, i).upper() for i in range(frames)]
        if not planarFiles:
            return []

        known = [v for v in viewNames if v != "UNKNOWN"]
        loadable = DICOMLoadable()
        loadable.files = planarFiles
        description = slicer.dicomDatabase.fileValue(planarFiles[0], self.tags["seriesDescription"]) or "DMSA"
        loadable.name = f"{description} ({', '.join(known) if known else 'planar NM'})"
        loadable.tooltip = (f"Planar NM image(s) loaded per view for DMSAAI: {', '.join(viewNames)}. "
                            "Images are displayed as stored, without mirroring.")
        loadable.selected = True
        # Higher than the generic scalar volume plugin (0.5) so this is the default for DMSA series.
        loadable.confidence = 0.95 if known else 0.6
        return [loadable]

    def load(self, loadable):
        firstNode = None
        posteriorNode = None
        for path in loadable.files:
            try:
                ds = dicomviews.read_dataset(path)
                views = dicomviews.extract_views(ds, sourceFile=path)
            except Exception as e:
                logging.error(f"DMSAAI: could not read {path}: {e}")
                continue
            patientName = str(getattr(ds, "PatientName", "")) or "DMSA"
            for planar in views:
                label = planar.view.upper() if planar.view != "unknown" else f"frame {planar.frame + 1}"
                node = volumes.create_planar_volume(planar.pixels, planar.spacing, f"{patientName} {label}",
                                                    volumes.planar_attributes(planar, ds))
                # One subject hierarchy series item per view, under the patient/study of the file.
                single = DICOMLoadable()
                single.files = [path]
                single.name = node.GetName()
                single.uids = [getattr(ds, "SOPInstanceUID", "")]
                self.addSeriesInSubjectHierarchy(single, node)
                for warning in planar.warnings:
                    logging.warning(f"DMSAAI {os.path.basename(path)}: {warning}")
                firstNode = firstNode or node
                if planar.view == "post" and posteriorNode is None:
                    posteriorNode = node
        return posteriorNode or firstNode


#
# DICOMDmsaaiPlugin
#

class DICOMDmsaaiPlugin(ScriptedLoadableModule):
    """Registers the DMSA planar NM DICOM plugin with the DICOM module."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DICOM DMSA Planar Plugin"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.contributors = ["Burak Demir (Mehmet Akif Inan Education and Research Hospital)"]
        parent.helpText = ("Loads planar Tc-99m DMSA nuclear medicine images, one volume per view, "
                           "displayed as stored and with the DICOM metadata kept on each node.")
        parent.acknowledgementText = "Part of the SlicerDMSAAI extension."
        parent.hidden = True

        try:
            slicer.modules.dicomPlugins
        except AttributeError:
            slicer.modules.dicomPlugins = {}
        slicer.modules.dicomPlugins["DICOMDmsaaiPlugin"] = DICOMDmsaaiPluginClass
