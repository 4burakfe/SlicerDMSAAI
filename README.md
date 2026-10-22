# SlicerDMSAAI

3D Slicer extension for **DMSAAI**, the AI-powered renal cortical scintigraphy analysis tool for Tc-99m DMSA studies:
automated kidney segmentation, split renal function (SRF) and AI-assisted kidney evaluation from planar
nuclear medicine DICOM images.

This is the Slicer port of the stand-alone [DMSAAI](https://github.com/4burakfe/DMSAAI) application. It uses the
same networks, weights, preprocessing and quantification, and gives the same results.

![banner](DMSAAI/Resources/banner3.jpg)

## Authors

Dr. Burak Demir - Mehmet Akif İnan Education and Research Hospital - Department of Nuclear Medicine - Şanlıurfa, Turkey

Dr. Demet Nak - Recep Tayyip Erdogan University - Department of Nuclear Medicine - Rize, Turkey

## Contents

| Module | Description |
|---|---|
| **DMSAAI** (Nuclear Medicine category) | Segmentation, SRF (posterior-only and geometric mean), background ROIs and kidney evaluation |
| **DICOM DMSA Planar Plugin** (hidden) | DICOM plugin that loads planar NM DMSA studies one volume per view, without mirroring, and keeps the DICOM metadata on each volume |

## Installation (development)

1. Clone this repository.
2. In Slicer: *Edit → Application Settings → Modules → Additional module paths*, add the `DMSAAI` folder of the
   clone, and restart Slicer.
3. On the first segmentation the module offers to install PyTorch, MONAI, einops and OpenCV into Slicer's Python.
   If the [SlicerPyTorch](https://github.com/fepegar/SlicerPyTorch) extension is installed, it is used to pick the
   right CUDA build of PyTorch. This can also be done in advance with *Advanced → Install Python dependencies*.
4. The model weights (about 140 MB) are downloaded from the DMSAAI repository on first use into Slicer's cache
   folder. You can point *Advanced → Model weights* at an existing DMSAAI folder instead.

## Usage

1. Load the DMSA study, either
   - with the **DICOM** module: import the files and load the series. The default loader for planar NM is
     "DMSA Planar NM", which creates one volume per view (ANT, POST, LAO, RPO, LPO, RAO). Its confidence (1.1) is
     above MultiVolumeImporter's (1.0), so the series is not also loaded as a multivolume/sequence. After loading,
     Slicer switches to the DMSAAI module with the new POST/ANT volumes selected (set the
     `DMSAAI/SwitchToModuleAfterDICOMLoad` setting to `false` to turn this off), or
   - with **Load DICOM files...** / **Load folder...** in the DMSAAI module, which reads the files directly.
2. Check the *Posterior* and *Anterior* selectors. They are filled automatically; use **Detect views in scene** for
   volumes loaded another way.
3. Choose the architecture (SwinUNETR, SegResNet or DynUNet) and optional preprocessing.
4. Click **Segment & Quantify SRF**. Adjust the kidney thresholds and background ROI method; the results update live.
5. Click **Clinical Evaluation** for the cortical abnormality, normal kidney, afunctional kidney and horseshoe
   kidney models.

The layout follows DMSAAI: the anterior view on the left and the posterior view on the right, each with only its own
kidney (red = left, blue = right) and background (orange / cyan) contours and the SRF under each kidney (geometric
mean when the anterior view is available). Oblique views (LAO, RAO, LPO, RPO) of the same study are shown in a row
below when loaded; they are not segmented.

## Orientation and DICOM metadata

The networks were trained on the raw DICOM pixel array. Slicer normally applies each detector's
ImageOrientationPatient (`[-1,0,0, 0,0,-1]` for a posterior view, `[1,0,0, 0,0,-1]` for an anterior view), so in the
coronal view one of the two always looks mirrored, and multi-frame ANT+POST files without orientation tags are
placed differently again. To avoid this:

- **All quantification uses the raw stored pixels.** For any input volume the module reads the source DICOM file:
  the exact frame recorded by the DMSA plugin, or, for volumes loaded by another DICOM plugin, the frame whose pixel
  values match the volume. Only volumes without any DICOM source fall back to the voxel array.
- **Display uses one fixed geometry.** The DMSA plugin and the module place every planar image in the coronal plane
  so that it is shown as stored (posterior view: patient left on the screen left; anterior view: patient right on the
  screen left), which is the usual nuclear medicine display. This geometry is for display only and is not an
  anatomical patient coordinate system.
- **Metadata is kept on the volume.** The plugin stores `DMSAAI.View`, `DMSAAI.StartAngle`, `DMSAAI.Frame`,
  `DMSAAI.Detector`, `DMSAAI.PixelSpacing`, `DMSAAI.ImageOrientationPatient`, `DMSAAI.SourceFile`, patient name/ID
  and study date as node attributes.

Views are identified like DMSAAI: from the detector start angle (180° ANT, 0/360° POST, 135° LAO, 315° RPO,
45° LPO, 225° RAO) and, if there is no angle, from DatasetName / SeriesDescription / ProtocolName.

## Tests

Stand-alone tests of the processing code (no Slicer needed), using the sample studies and weights from a DMSAAI
checkout:

```bash
pip install pydicom numpy scipy opencv-python-headless pytest torch "monai>=1.3" einops
DMSAAI_DIR=/path/to/DMSAAI pytest Testing
```

The module test (`DMSAAITest`) runs inside Slicer from the module's *Reload & Test* button.

## Research use

This software is intended for research, education and algorithm development. It is **not approved for standalone
clinical decision making**.

## License

MIT license
