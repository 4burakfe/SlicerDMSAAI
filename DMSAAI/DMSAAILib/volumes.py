"""
Slicer helpers that put planar DMSA images into the scene with a fixed,
documented geometry, and read the raw pixels back from any volume node.

Display geometry
----------------
Every planar image is placed in the coronal plane so that the slice viewer
shows the raw DICOM pixel array as stored: column 0 on the screen left and
row 0 at the top. This is the usual nuclear medicine display (posterior
view: patient left on the screen left; anterior view: patient right on the
screen left). The true acquisition orientation is kept in node attributes
instead of being applied to the geometry, which is what made Slicer show
one of the two views mirrored.
"""

import os

import numpy as np
import vtk

import slicer

from . import dicomviews

ATTR_PREFIX = "DMSAAI."
ATTR_VIEW = ATTR_PREFIX + "View"
ATTR_ANGLE = ATTR_PREFIX + "StartAngle"
ATTR_FRAME = ATTR_PREFIX + "Frame"
ATTR_DETECTOR = ATTR_PREFIX + "Detector"
ATTR_SOURCE = ATTR_PREFIX + "SourceFile"
ATTR_SPACING = ATTR_PREFIX + "PixelSpacing"
ATTR_IOP = ATTR_PREFIX + "ImageOrientationPatient"
ATTR_PATIENT_ID = ATTR_PREFIX + "PatientID"
ATTR_PATIENT_NAME = ATTR_PREFIX + "PatientName"
ATTR_STUDY_DATE = ATTR_PREFIX + "StudyDate"
ATTR_ROLE = ATTR_PREFIX + "Role"


def set_planar_geometry(volumeNode, spacing):
    """Coronal placement that displays the IJK array like the raw DICOM image."""
    rowSpacing, colSpacing = spacing if spacing else (1.0, 1.0)
    # Columns of the direction matrix are the RAS directions of I, J, K.
    # I (column index) -> patient left (screen right in Slicer's coronal view)
    # J (row index)    -> inferior (screen down)
    # K                -> anterior (keeps a right-handed frame)
    directions = vtk.vtkMatrix4x4()
    for row, values in enumerate(((-1, 0, 0), (0, 0, -1), (0, -1, 0))):
        for col, value in enumerate(values):
            directions.SetElement(row, col, value)
    volumeNode.SetIJKToRASDirectionMatrix(directions)
    volumeNode.SetSpacing(colSpacing, rowSpacing, 1.0)
    dims = volumeNode.GetImageData().GetDimensions() if volumeNode.GetImageData() else (1, 1, 1)
    # Centre the image on the origin of the coronal plane.
    volumeNode.SetOrigin((dims[0] - 1) * colSpacing / 2.0, 0.0, (dims[1] - 1) * rowSpacing / 2.0)


def create_planar_volume(pixels, spacing, name, attributes=None, nodeClass="vtkMRMLScalarVolumeNode"):
    """Create a volume node from a 2D array (rows, cols) with the fixed display geometry."""
    volumeNode = slicer.mrmlScene.AddNewNodeByClass(nodeClass, slicer.mrmlScene.GenerateUniqueName(name))
    update_planar_volume(volumeNode, pixels, spacing)
    for key, value in (attributes or {}).items():
        if value is not None:
            volumeNode.SetAttribute(key, str(value))
    volumeNode.CreateDefaultDisplayNodes()
    return volumeNode


def update_planar_volume(volumeNode, pixels, spacing):
    slicer.util.updateVolumeFromArray(volumeNode, np.ascontiguousarray(pixels)[np.newaxis, :, :])
    set_planar_geometry(volumeNode, spacing)


def planar_attributes(planar, ds=None):
    attributes = {
        ATTR_VIEW: planar.view,
        ATTR_ANGLE: planar.angle,
        ATTR_FRAME: planar.frame,
        ATTR_DETECTOR: planar.detector,
        ATTR_SOURCE: planar.sourceFile,
        ATTR_SPACING: "\\".join(str(v) for v in planar.spacing) if planar.spacing else None,
        ATTR_IOP: "\\".join(str(v) for v in planar.imageOrientation) if planar.imageOrientation else None,
    }
    if ds is not None:
        attributes[ATTR_PATIENT_ID] = str(getattr(ds, "PatientID", ""))
        attributes[ATTR_PATIENT_NAME] = str(getattr(ds, "PatientName", ""))
        attributes[ATTR_STUDY_DATE] = str(getattr(ds, "StudyDate", ""))
    return attributes


def _source_files(volumeNode):
    """DICOM files a volume node was loaded from, if known."""
    path = volumeNode.GetAttribute(ATTR_SOURCE)
    if path and os.path.exists(path):
        return [path]
    uids = volumeNode.GetAttribute("DICOM.instanceUIDs")
    files = []
    if uids and slicer.dicomDatabase:
        for uid in uids.split():
            f = slicer.dicomDatabase.fileForInstance(uid)
            if f and os.path.exists(f):
                files.append(f)
    return files


def raw_view_from_node(volumeNode, log=None):
    """Return a PlanarView with the raw stored pixels of a volume node.

    Order of preference:
      1. the DICOM file recorded by the DMSAAI DICOM plugin (exact frame),
      2. the DICOM files behind a volume loaded by any other DICOM plugin
         (the frame is matched by its pixel values, so orientation and frame
         order of the loader do not matter),
      3. the voxel array of the node, in IJK order (Slicer keeps the stored
         pixel order and puts the orientation into the IJK-to-RAS matrix).
    """
    def _log(msg):
        if log:
            log(msg)

    nodeArray = slicer.util.arrayFromVolume(volumeNode)
    nodeImage = nodeArray[0] if nodeArray.ndim == 3 else nodeArray

    files = _source_files(volumeNode)
    frameAttr = volumeNode.GetAttribute(ATTR_FRAME)
    if files and frameAttr is not None and volumeNode.GetAttribute(ATTR_SOURCE):
        ds = dicomviews.read_dataset(files[0])
        views = dicomviews.extract_views(ds, sourceFile=files[0])
        frame = int(frameAttr)
        if frame < len(views):
            return views[frame], ds

    if files:
        nodeSorted = np.sort(nodeImage, axis=None)
        for path in files:
            try:
                ds = dicomviews.read_dataset(path)
                candidates = dicomviews.extract_views(ds, sourceFile=path)
            except Exception:
                continue
            for planar in candidates:
                if planar.pixels.shape in (nodeImage.shape, nodeImage.shape[::-1]) and \
                        np.array_equal(np.sort(planar.pixels, axis=None), nodeSorted):
                    return planar, ds
        _log(f"Could not match '{volumeNode.GetName()}' to its DICOM frame. Using the voxel array.")

    # Fall back to the node itself.
    spacing = volumeNode.GetSpacing()
    view = volumeNode.GetAttribute(ATTR_VIEW) or "unknown"
    spacingAttr = volumeNode.GetAttribute(ATTR_SPACING)
    rowCol = tuple(float(v) for v in spacingAttr.split("\\")) if spacingAttr else (spacing[1], spacing[0])
    _log(f"'{volumeNode.GetName()}' has no DICOM source. Assuming its voxel order is the stored pixel order.")
    return dicomviews.PlanarView(view=view, pixels=np.array(nodeImage), spacing=rowCol), None


def guess_view_of_node(volumeNode):
    """View of a node from DMSAAI attributes, its DICOM source or its name."""
    view = volumeNode.GetAttribute(ATTR_VIEW)
    if view:
        return view
    files = _source_files(volumeNode)
    if len(files) == 1:
        try:
            ds = dicomviews.read_dataset(files[0], stop_before_pixels=True)
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            if frames == 1:
                return dicomviews.determine_view(ds, 0)
        except Exception:
            pass
    return dicomviews.view_from_text(volumeNode.GetName())
