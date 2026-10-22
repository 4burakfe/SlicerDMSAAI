import logging
import os
import random

import numpy as np
import qt
import ctk
import vtk

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from DMSAAILib import dicomviews, networks, processing, settings, volumes


#
# DMSAAI
#

class DMSAAI(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DMSAAI"
        self.parent.categories = ["Nuclear Medicine"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Burak Demir (Mehmet Akif Inan Education and Research Hospital, Sanliurfa, Turkey)",
            "Demet Nak (Recep Tayyip Erdogan University, Rize, Turkey)",
        ]
        self.parent.helpText = """
AI-powered renal cortical scintigraphy analysis for Tc-99m DMSA studies: automated kidney segmentation,
split renal function (posterior-only and geometric mean) and AI-assisted kidney evaluation.
Load the planar ANT/POST images with the DICOM module (they are recognised as "DMSA Planar NM") or with
"Load DICOM files" below, then run "Segment &amp; Quantify SRF".
See the <a href="https://github.com/4burakfe/SlicerDMSAAI">project page</a> for details.
<br><b>Research use only. Not approved for standalone clinical decision making.</b>
"""
        self.parent.acknowledgementText = """
Slicer port of <a href="https://github.com/4burakfe/DMSAAI">DMSAAI</a>.
Segmentation and classification networks are built with MONAI and PyTorch.
"""


#
# DMSAAIWidget
#

# Built-in Slicer colour tables (by node ID) followed by the custom DMSAAI colour maps,
# which are read from Resources/ColorMaps/<name>.csv (256 rows of r,g,b in 0..1).
COLORMAPS = (
    ("Inverted grey", "vtkMRMLColorTableNodeInvertedGrey"),
    ("Grey", "vtkMRMLColorTableNodeGrey"),
    ("DICOM-Custom", None),
    ("Hel", None),
    ("TruGold", None),
    ("Valhalla", None),
    ("Jotun", None),
    ("ViridisCustom", None),
    ("PET-HotIron", None),
    ("Inferno", "vtkMRMLColorTableNodeFileInferno.txt"),
)


class DMSAAIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._updatingGUI = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = DMSAAILogic()
        self.logic.logCallback = self.log

        # Banner
        bannerPath = self.resourcePath(f"banner{random.randint(1, 4)}.jpg")
        self.bannerPixmap = qt.QPixmap(bannerPath)
        self.bannerLabel = qt.QLabel()
        self.bannerLabel.setAlignment(qt.Qt.AlignCenter)
        self.bannerLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.bannerLabel.setToolTip("DMSAAI: Deep Learning-Based Cortical Renal Scintigraphy Analysis")
        self.layout.addWidget(self.bannerLabel)
        self._updateBanner()

        # Inputs
        inputsButton = ctk.ctkCollapsibleButton()
        inputsButton.text = "Input images"
        self.layout.addWidget(inputsButton)
        inputsLayout = qt.QFormLayout(inputsButton)

        loadRow = qt.QHBoxLayout()
        self.loadFilesButton = qt.QPushButton("Load DICOM files...")
        self.loadFilesButton.toolTip = ("Read DMSA DICOM files or a folder directly (without importing into the "
                                        "DICOM database). Views are detected from the detector angle and description.")
        self.loadFolderButton = qt.QPushButton("Load folder...")
        self.dicomBrowserButton = qt.QPushButton("DICOM browser")
        self.dicomBrowserButton.toolTip = "Open the DICOM database. DMSA series load with the 'DMSA Planar NM' plugin."
        loadRow.addWidget(self.loadFilesButton)
        loadRow.addWidget(self.loadFolderButton)
        loadRow.addWidget(self.dicomBrowserButton)
        inputsLayout.addRow(loadRow)

        self.postSelector = self._volumeSelector("Posterior view (required for segmentation)")
        inputsLayout.addRow("Posterior:", self.postSelector)
        self.antSelector = self._volumeSelector("Anterior view (optional, enables geometric mean SRF)")
        inputsLayout.addRow("Anterior:", self.antSelector)
        self.detectButton = qt.QPushButton("Detect views in scene")
        self.detectButton.toolTip = "Pick the posterior and anterior volumes from DICOM metadata of loaded volumes."
        inputsLayout.addRow(self.detectButton)
        self.patientLabel = qt.QLabel("Patient: N/A")
        self.patientLabel.wordWrap = True
        inputsLayout.addRow(self.patientLabel)

        # Preprocessing and segmentation
        segButton = ctk.ctkCollapsibleButton()
        segButton.text = "Segmentation and SRF"
        self.layout.addWidget(segButton)
        segLayout = qt.QFormLayout(segButton)

        self.modelCombo = qt.QComboBox()
        self.modelCombo.addItems(list(networks.SEGMENTATION_MODELS.keys()))
        segLayout.addRow("Architecture:", self.modelCombo)
        self.normalizeCheck = qt.QCheckBox("Normalize total counts to 200k")
        self.resampleCheck = qt.QCheckBox("Resample to 2x2 mm pixel size")
        self.normalizeCheck.checked = True
        self.resampleCheck.checked = True
        segLayout.addRow(self.normalizeCheck)
        segLayout.addRow(self.resampleCheck)

        self.segmentButton = qt.QPushButton("Segment && Quantify SRF")
        self.segmentButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.segmentButton.enabled = False
        segLayout.addRow(self.segmentButton)

        self.leftThresholdSlider = self._thresholdSlider("Probability threshold for the left kidney ROI")
        segLayout.addRow("Left kidney threshold (%):", self.leftThresholdSlider)
        self.rightThresholdSlider = self._thresholdSlider("Probability threshold for the right kidney ROI")
        segLayout.addRow("Right kidney threshold (%):", self.rightThresholdSlider)
        self.bgCombo = qt.QComboBox()
        self.bgCombo.addItems(list(processing.BACKGROUND_METHODS))
        segLayout.addRow("Background ROI:", self.bgCombo)
        self.showBackgroundCheck = qt.QCheckBox("Show background ROIs")
        self.showBackgroundCheck.checked = True
        segLayout.addRow(self.showBackgroundCheck)

        # Evaluation
        evalButton = ctk.ctkCollapsibleButton()
        evalButton.text = "Kidney evaluation"
        self.layout.addWidget(evalButton)
        evalLayout = qt.QFormLayout(evalButton)
        self.classArchCombo = qt.QComboBox()
        self.classArchCombo.addItems(["DenseNet121"])
        evalLayout.addRow("Architecture:", self.classArchCombo)
        self.classifyButton = qt.QPushButton("Clinical Evaluation")
        self.classifyButton.enabled = False
        evalLayout.addRow(self.classifyButton)

        # Display
        displayButton = ctk.ctkCollapsibleButton()
        displayButton.text = "Display"
        self.layout.addWidget(displayButton)
        displayLayout = qt.QFormLayout(displayButton)
        self.colormapCombo = qt.QComboBox()
        for name, _ in COLORMAPS:
            self.colormapCombo.addItem(name)
        displayLayout.addRow("Colormap:", self.colormapCombo)
        self.windowSlider = ctk.ctkSliderWidget()
        self.windowSlider.minimum = 1
        self.windowSlider.maximum = 100
        self.windowSlider.value = 100
        self.windowSlider.decimals = 0
        self.windowSlider.toolTip = "Upper display limit as a percentage of the maximum count"
        displayLayout.addRow("Window level (%):", self.windowSlider)
        self.overlayButton = qt.QPushButton("Hide segmentations / annotations")
        self.overlayButton.checkable = True
        self.overlayButton.checked = True
        self.overlayButton.toolTip = "Show or hide the kidney and background ROIs and the SRF labels"
        displayLayout.addRow(self.overlayButton)
        self.layoutButton = qt.QPushButton("Show DMSAAI layout")
        displayLayout.addRow(self.layoutButton)

        # Results
        resultsButton = ctk.ctkCollapsibleButton()
        resultsButton.text = "Results"
        self.layout.addWidget(resultsButton)
        resultsLayout = qt.QVBoxLayout(resultsButton)
        self.resultsText = qt.QPlainTextEdit()
        self.resultsText.readOnly = True
        self.resultsText.setMinimumHeight(220)
        font = qt.QFont("Courier")
        font.setStyleHint(qt.QFont.Monospace)
        self.resultsText.setFont(font)
        resultsLayout.addWidget(self.resultsText)
        self.copyResultsButton = qt.QPushButton("Copy results")
        resultsLayout.addWidget(self.copyResultsButton)

        # Advanced
        advancedButton = ctk.ctkCollapsibleButton()
        advancedButton.text = "Advanced"
        advancedButton.collapsed = True
        self.layout.addWidget(advancedButton)
        advancedLayout = qt.QFormLayout(advancedButton)
        self.weightsPathEdit = ctk.ctkPathLineEdit()
        self.weightsPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.weightsPathEdit.currentPath = self.logic.weightsDir
        self.weightsPathEdit.toolTip = "Folder with the DMSAAI .pth model weights. Missing files are downloaded here."
        advancedLayout.addRow("Model weights:", self.weightsPathEdit)
        self.downloadButton = qt.QPushButton("Download model weights")
        advancedLayout.addRow(self.downloadButton)
        self.installButton = qt.QPushButton("Check / install Python dependencies")
        self.installButton.toolTip = ("Checks that PyTorch is installed (install it with the PyTorch Utils module) "
                                      "and installs MONAI, einops and OpenCV into Slicer's Python")
        advancedLayout.addRow(self.installButton)
        self.deviceLabel = qt.QLabel("Compute device: (detected on first run)")
        advancedLayout.addRow(self.deviceLabel)
        self.keywordsEdit = qt.QPlainTextEdit()
        self.keywordsEdit.setMaximumHeight(120)
        self.keywordsEdit.toolTip = (
            "One keyword or phrase per line. A DICOM series whose study/series/protocol description or "
            "radiopharmaceutical contains one of them is loaded by the DMSA plugin by default. Case, Turkish "
            "letters and punctuation are ignored. Changes apply to series examined from now on.")
        self.keywordsEdit.setPlainText("\n".join(settings.dmsa_keywords()))
        advancedLayout.addRow("DMSA study keywords:", self.keywordsEdit)
        self.restoreKeywordsButton = qt.QPushButton("Restore default keywords")
        advancedLayout.addRow(self.restoreKeywordsButton)
        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.setMaximumHeight(140)
        advancedLayout.addRow("Log:", self.logText)

        self.layout.addStretch(1)

        # Connections
        self.loadFilesButton.connect("clicked(bool)", self.onLoadFiles)
        self.loadFolderButton.connect("clicked(bool)", self.onLoadFolder)
        self.dicomBrowserButton.connect("clicked(bool)", lambda: slicer.util.selectModule("DICOM"))
        self.detectButton.connect("clicked(bool)", self.onDetectViews)
        self.postSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.antSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.normalizeCheck.connect("toggled(bool)", self.onPreprocessingChanged)
        self.resampleCheck.connect("toggled(bool)", self.onPreprocessingChanged)
        self.segmentButton.connect("clicked(bool)", self.onSegment)
        self.leftThresholdSlider.connect("valueChanged(double)", self.onMaskParametersChanged)
        self.rightThresholdSlider.connect("valueChanged(double)", self.onMaskParametersChanged)
        self.bgCombo.connect("currentIndexChanged(int)", self.onMaskParametersChanged)
        self.showBackgroundCheck.connect("toggled(bool)", self.onShowBackground)
        self.classifyButton.connect("clicked(bool)", self.onClassify)
        self.colormapCombo.connect("currentIndexChanged(int)", self.onDisplayChanged)
        self.windowSlider.connect("valueChanged(double)", self.onDisplayChanged)
        self.layoutButton.connect("clicked(bool)", self.onShowLayout)
        self.overlayButton.connect("toggled(bool)", self.onToggleOverlays)
        self.copyResultsButton.connect("clicked(bool)", self.onCopyResults)
        self.weightsPathEdit.connect("currentPathChanged(QString)", self.onWeightsPathChanged)
        self.downloadButton.connect("clicked(bool)", self.onDownloadWeights)
        self.installButton.connect("clicked(bool)", self.onInstallDependencies)
        self.keywordsEdit.connect("textChanged()", self.onKeywordsChanged)
        self.restoreKeywordsButton.connect("clicked(bool)", self.onRestoreKeywords)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.onInputChanged()

    # -- helpers --------------------------------------------------------

    def _volumeSelector(self, toolTip):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.showHidden = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.toolTip = toolTip
        return selector

    def _thresholdSlider(self, toolTip):
        slider = ctk.ctkSliderWidget()
        slider.minimum = 1
        slider.maximum = 99
        slider.singleStep = 1
        slider.decimals = 0
        slider.value = 50
        slider.tracking = False
        slider.toolTip = toolTip
        return slider

    def _updateBanner(self):
        if self.bannerPixmap.isNull():
            self.bannerLabel.hide()
            return
        width = max(200, min(self.parent.width - 20, 520)) if self.parent.width > 0 else 400
        self.bannerLabel.setPixmap(self.bannerPixmap.scaledToWidth(width, qt.Qt.SmoothTransformation))

    def log(self, message):
        logging.info(f"DMSAAI: {message}")
        if hasattr(self, "logText"):
            self.logText.appendPlainText(message)
        slicer.app.processEvents()

    def enter(self):
        self._updateBanner()
        if self.postSelector.currentNode() is None and self.antSelector.currentNode() is None:
            self.selectInputs()

    def cleanup(self):
        self.removeObservers()

    def onSceneEndClose(self, caller, event):
        self.logic.reset()
        self.resultsText.setPlainText("")
        self.patientLabel.text = "Patient: N/A"
        self.onInputChanged()

    # -- input ----------------------------------------------------------

    def onLoadFiles(self):
        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), "Select DMSA DICOM files", "",
                                                "DICOM files (*.dcm *.DCM *.ima *.IMA *);;All files (*)")
        if paths:
            self._loadPaths(list(paths))

    def onLoadFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Select a folder with DMSA DICOM files")
        if folder:
            self._loadPaths([folder])

    def _loadPaths(self, paths):
        with slicer.util.tryWithErrorDisplay("Failed to load DICOM files.", waitCursor=True):
            nodes = self.logic.loadFiles(paths)
            if not nodes:
                slicer.util.warningDisplay("No planar NM DMSA images were found in the selection.")
                return
            self.selectInputs(list(nodes.values()))

    def onDetectViews(self):
        if not self.selectInputs():
            slicer.util.warningDisplay("No volume with a recognisable ANT or POST view was found in the scene.")

    def selectInputs(self, nodes=None):
        """Fill the Posterior/Anterior selectors from the given nodes (e.g. just loaded) or the scene.

        Returns True if a posterior or anterior view was found.
        """
        found = self.logic.findViews(nodes) if nodes else {}
        if not found:
            found = self.logic.findViewsInScene()
        if not found:
            return False
        # Set both selectors first, then prepare the images once.
        for selector in (self.postSelector, self.antSelector):
            selector.blockSignals(True)
        self.postSelector.setCurrentNode(found.get("post"))
        self.antSelector.setCurrentNode(found.get("ant"))
        for selector in (self.postSelector, self.antSelector):
            selector.blockSignals(False)
        self.onInputChanged()
        return True

    def onInputChanged(self, node=None):
        postNode = self.postSelector.currentNode()
        self.segmentButton.enabled = postNode is not None
        self.classifyButton.enabled = False
        self.logic.reset(keepInputs=True)
        self.resultsText.setPlainText("")
        info = self.logic.patientInfo(postNode or self.antSelector.currentNode())
        self.patientLabel.text = info
        if postNode is not None:
            with slicer.util.tryWithErrorDisplay("Failed to prepare the input images.", waitCursor=True):
                self._prepare()

    def onPreprocessingChanged(self, checked=None):
        if self.postSelector.currentNode() is None:
            return
        with slicer.util.tryWithErrorDisplay("Failed to preprocess the images.", waitCursor=True):
            self.classifyButton.enabled = False
            self.resultsText.setPlainText("")
            self._prepare()

    def _prepare(self):
        self.logic.prepare(self.postSelector.currentNode(), self.antSelector.currentNode(),
                           resample=self.resampleCheck.checked, normalize=self.normalizeCheck.checked)
        self.onShowLayout()

    # -- analysis -------------------------------------------------------

    def onSegment(self):
        with slicer.util.tryWithErrorDisplay("Segmentation failed.", waitCursor=True):
            if not self.logic.ensureRequirements(askConfirmation=True):
                return
            if not self.logic.ensureWeights([networks.SEGMENTATION_MODELS[self.modelCombo.currentText]],
                                            askConfirmation=True):
                return
            if self.logic.postProcessedNode is None:
                self._prepare()
            self.deviceLabel.text = f"Compute device: {self.logic.device()}"
            self.logic.segment(self.modelCombo.currentText)
            self.onMaskParametersChanged()
            self.classifyButton.enabled = True

    def onMaskParametersChanged(self, value=None):
        if self.logic.postProbabilities is None:
            return
        with slicer.util.tryWithErrorDisplay("Failed to update the kidney ROIs.", waitCursor=True):
            self.logic.updateMasks(self.leftThresholdSlider.value / 100.0,
                                            self.rightThresholdSlider.value / 100.0,
                                            self.bgCombo.currentText)
            self.logic.showBackground(self.showBackgroundCheck.checked)
            self._showResults()

    def onShowBackground(self, checked):
        self.logic.showBackground(checked)

    def onClassify(self):
        with slicer.util.tryWithErrorDisplay("Clinical evaluation failed.", waitCursor=True):
            if not self.logic.ensureRequirements(askConfirmation=True):
                return
            files = [f for f in networks.ALL_WEIGHT_FILES if f not in networks.SEGMENTATION_MODELS.values()]
            if not self.logic.ensureWeights(files, askConfirmation=True):
                return
            self.logic.classify()
            self._showResults()

    def _showResults(self):
        self.resultsText.setPlainText(self.logic.resultsText())

    # -- display --------------------------------------------------------

    def onDisplayChanged(self, value=None):
        name, colorNodeID = COLORMAPS[self.colormapCombo.currentIndex]
        if colorNodeID is None:
            colorNodeID = self.logic.customColorNode(name, self.resourcePath(f"ColorMaps/{name}.csv")).GetID()
        self.logic.setDisplay(colorNodeID, self.windowSlider.value / 100.0)

    def onToggleOverlays(self, show):
        self.overlayButton.text = "Hide segmentations / annotations" if show else "Show segmentations / annotations"
        self.logic.setOverlaysVisible(show)

    def onShowLayout(self):
        self.logic.showLayout()
        self.onDisplayChanged()

    def onCopyResults(self):
        qt.QApplication.clipboard().setText(self.resultsText.toPlainText())

    # -- setup ----------------------------------------------------------

    def onWeightsPathChanged(self, path):
        self.logic.weightsDir = path

    def onKeywordsChanged(self):
        settings.set_dmsa_keywords(self.keywordsEdit.toPlainText().splitlines())

    def onRestoreKeywords(self):
        settings.reset_dmsa_keywords()
        self.keywordsEdit.blockSignals(True)
        self.keywordsEdit.setPlainText("\n".join(dicomviews.DEFAULT_DMSA_KEYWORDS))
        self.keywordsEdit.blockSignals(False)

    def onDownloadWeights(self):
        with slicer.util.tryWithErrorDisplay("Failed to download model weights.", waitCursor=True):
            if self.logic.ensureWeights(networks.ALL_WEIGHT_FILES, askConfirmation=False):
                slicer.util.infoDisplay(f"All model weights are available in:\n{self.logic.weightsDir}")

    def onInstallDependencies(self):
        with slicer.util.tryWithErrorDisplay("Failed to install Python dependencies.", waitCursor=True):
            if self.logic.ensureRequirements(askConfirmation=False):
                self.deviceLabel.text = f"Compute device: {self.logic.device()}"
                slicer.util.infoDisplay("PyTorch, MONAI and the other dependencies are installed.")


#
# DMSAAILogic
#

class DMSAAILogic(ScriptedLoadableModuleLogic):
    """Analysis of planar DMSA images. All computation is done on the raw DICOM pixel arrays.

    Display nodes created by the logic:
      - "DMSAAI POST" / "DMSAAI ANT": preprocessed 256x256 images that the networks see
      - "DMSAAI LAO" / "RAO" / "LPO" / "RPO": preprocessed obliques, display only
      - "DMSAAI POST ROIs" / "DMSAAI ANT ROIs": kidney and background ROIs, each shown only in its own view
      - SRF labels under each kidney, like the DMSAAI overlay
    """

    SETTINGS_WEIGHTS_DIR = "DMSAAI/WeightsDirectory"
    # Same colours as the DMSAAI overlay: left kidney red, right kidney blue,
    # left background orange, right background cyan.
    SEGMENT_COLORS = {
        "LeftKidney": (1.0, 0.0, 0.0),
        "RightKidney": (0.0, 0.0, 1.0),
        "LeftBackground": (1.0, 0.65, 0.0),
        "RightBackground": (0.0, 1.0, 1.0),
    }
    LABEL_COLORS = {processing.LEFT: (1.0, 0.0, 0.0), processing.RIGHT: (0.0, 0.0, 1.0)}
    OBLIQUE_KEYS = ("lao", "rao", "lpo", "rpo")
    # Slice view of each image. The layout is 2x3: posterior views (POST, LPO, RPO) on the top row and
    # anterior views (ANT, LAO, RAO) below, so each oblique sits in the row of the view it is closest to.
    LAYOUT_ROWS = (("post", "lpo", "rpo"), ("ant", "lao", "rao"))
    VIEW_SLICE = {"post": "Red", "ant": "Yellow",
                  "lao": "DMSAAILAO", "rao": "DMSAAIRAO", "lpo": "DMSAAILPO", "rpo": "DMSAAIRPO"}
    VIEW_TITLES = {"ant": "Anterior View", "post": "Posterior View",
                   "lao": "LAO View", "rao": "RAO View", "lpo": "LPO View", "rpo": "RPO View"}
    LAYOUT_ID = 75302
    SEGMENT_NAMES = {
        "LeftKidney": "Left kidney",
        "RightKidney": "Right kidney",
        "LeftBackground": "Left kidney background",
        "RightBackground": "Right kidney background",
    }
    REQUIREMENTS = (
        ("monai", "monai>=1.3"),
        ("einops", "einops"),
        ("cv2", "opencv-python-headless"),
    )

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.logCallback = None
        self._device = None
        self.postProcessedNode = None
        self.antProcessedNode = None
        self.postSegmentationNode = None
        self.antSegmentationNode = None
        self.obliqueProcessedNodes = {}
        self.srfLabelNodes = {}
        self.overlaysVisible = True
        self.reset()

    # -- general --------------------------------------------------------

    def log(self, message):
        if self.logCallback:
            self.logCallback(message)
        else:
            logging.info(f"DMSAAI: {message}")

    def reset(self, keepInputs=False):
        self.postView = None
        self.antView = None
        self.postImage = None
        self.antImage = None
        self.postProbabilities = None
        self.postMask = None
        self.antMask = None
        self.obliqueImages = {}
        self.srf = None
        self.classification = None
        self.modelName = None
        if not keepInputs:
            self.postProcessedNode = None
            self.antProcessedNode = None
            self.postSegmentationNode = None
            self.antSegmentationNode = None
            self.obliqueProcessedNodes = {}
            self.srfLabelNodes = {}

    @property
    def weightsDir(self):
        default = os.path.join(slicer.app.cachePath, "DMSAAI", "weights")
        return qt.QSettings().value(self.SETTINGS_WEIGHTS_DIR, default) or default

    @weightsDir.setter
    def weightsDir(self, path):
        qt.QSettings().setValue(self.SETTINGS_WEIGHTS_DIR, path)

    # -- dependencies and weights ----------------------------------------

    def torchInstalled(self):
        import importlib.util
        return importlib.util.find_spec("torch") is not None

    def missingRequirements(self):
        """pip requirements (other than PyTorch) that are not installed yet."""
        import importlib.util
        return [requirement for module, requirement in self.REQUIREMENTS if importlib.util.find_spec(module) is None]

    def requirementsInstalled(self):
        return self.torchInstalled() and not self.missingRequirements()

    def ensureRequirements(self, askConfirmation=True):
        """Check PyTorch first, then offer to install MONAI, einops and OpenCV.

        PyTorch is never installed here: the right build depends on the GPU and CUDA version, so the
        user installs it with the PyTorch Utils module of the PyTorch extension.
        """
        if not self.torchInstalled():
            self._promptTorchInstall()
            return False
        missing = self.missingRequirements()
        if not missing:
            return True
        if askConfirmation and not slicer.util.confirmOkCancelDisplay(
                f"DMSAAI also needs {', '.join(missing)}, which are not installed in Slicer's Python yet.\n\n"
                "Install them now?"):
            return False
        import importlib
        for requirement in missing:
            self.log(f"Installing {requirement}...")
            slicer.util.pip_install(requirement)
        importlib.invalidate_caches()
        self.log("Python dependencies installed.")
        return self.requirementsInstalled()

    def _promptTorchInstall(self):
        if hasattr(slicer.modules, "pytorchutils"):
            if slicer.util.confirmOkCancelDisplay(
                    "PyTorch is not installed in Slicer's Python.\n\n"
                    "Install it with the PyTorch Utils module, which picks the right build (CPU or CUDA) for this "
                    "computer, then run DMSAAI again.\n\nOpen PyTorch Utils now?"):
                slicer.util.selectModule("PyTorchUtils")
        else:
            slicer.util.warningDisplay(
                "PyTorch is not installed in Slicer's Python.\n\n"
                "Install the PyTorch extension from the Extensions Manager, restart Slicer, then install PyTorch "
                "with the PyTorch Utils module and run DMSAAI again.")

    def ensureWeights(self, files=None, askConfirmation=True):
        missing = networks.missing_weights(self.weightsDir, files)
        if not missing:
            return True
        if askConfirmation and not slicer.util.confirmOkCancelDisplay(
                f"{len(missing)} model weight file(s) are missing from:\n{self.weightsDir}\n\n"
                "Download them from the DMSAAI GitHub repository now?"):
            return False
        networks.download_weights(self.weightsDir, missing, progress=self.log)
        self.log("Model weights are ready.")
        return not networks.missing_weights(self.weightsDir, files)

    def device(self):
        if self._device is None:
            self._device = networks.default_device()
        return self._device

    # -- loading --------------------------------------------------------

    def loadFiles(self, paths):
        """Read DICOM files/folders directly and create one volume per view. Returns {view: node}."""
        files = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, names in os.walk(path):
                    files += [os.path.join(root, n) for n in sorted(names)]
            else:
                files.append(path)
        self.log(f"Scanning {len(files)} file(s)...")
        views, patient = dicomviews.collect_views(files, log=self.log)
        if "post" not in views:
            self.log("Warning: no posterior image detected. Segmentation requires a posterior image.")
        nodes = {}
        for key in dicomviews.VIEW_KEYS:
            planar = views.get(key)
            if planar is None:
                continue
            ds = dicomviews.read_dataset(planar.sourceFile, stop_before_pixels=True)
            name = f"{patient.get('PatientName', 'DMSA')} {key.upper()}"
            nodes[key] = volumes.create_planar_volume(planar.pixels, planar.spacing, name,
                                                      volumes.planar_attributes(planar, ds))
        return nodes

    def findViews(self, nodes):
        """First volume of each view (ANT, POST, obliques) among the nodes. Analysis nodes are skipped."""
        found = {}
        for node in nodes:
            if node is None or not node.IsA("vtkMRMLScalarVolumeNode") or node.GetAttribute(volumes.ATTR_ROLE):
                continue
            view = volumes.guess_view_of_node(node)
            if view in dicomviews.VIEW_KEYS and view not in found:
                found[view] = node
        return found

    def findViewsInScene(self):
        """Volumes of each view in the scene, preferring the most recently added ones."""
        return self.findViews(reversed(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")))

    def findObliques(self, postNode):
        """Oblique views in the scene that belong to the same patient and study as the posterior view."""
        if postNode is None:
            return {}
        keys = (volumes.ATTR_PATIENT_ID, volumes.ATTR_STUDY_DATE)
        reference = [postNode.GetAttribute(key) for key in keys]
        found = {}
        for view, node in self.findViewsInScene().items():
            if view not in self.OBLIQUE_KEYS:
                continue
            values = [node.GetAttribute(key) for key in keys]
            # Only compare when both nodes carry the DICOM attributes of the DMSA loader.
            if all(reference) and all(values) and values != reference:
                continue
            found[view] = node
        return found

    def patientInfo(self, node):
        if node is None:
            return "Patient: N/A"
        name = node.GetAttribute(volumes.ATTR_PATIENT_NAME)
        pid = node.GetAttribute(volumes.ATTR_PATIENT_ID)
        if name is None:
            try:
                planar, ds = volumes.raw_view_from_node(node)
                if ds is not None:
                    name = str(getattr(ds, "PatientName", ""))
                    pid = str(getattr(ds, "PatientID", ""))
            except Exception:
                pass
        return f"Patient name: {name or 'N/A'}\nPatient ID: {pid or 'N/A'}"

    # -- processing -----------------------------------------------------

    def prepare(self, postNode, antNode=None, resample=False, normalize=False):
        """Extract raw pixels of the inputs, preprocess them and update the display volumes."""
        self.reset(keepInputs=True)
        self.postView, postDs = volumes.raw_view_from_node(postNode, log=self.log)
        self.postImage = processing.preprocess(self.postView.pixels, self.postView.spacing, resample, normalize,
                                               log=self.log)
        self.postProcessedNode = self._updateProcessedNode(self.postProcessedNode, self.postImage,
                                                           processing.output_spacing(self.postView.spacing, resample),
                                                           "DMSAAI POST", "post")
        if postDs is not None and self.postView.view not in ("post", "unknown"):
            self.log(f"Warning: the selected posterior volume looks like a {self.postView.view.upper()} view.")
        if antNode is not None:
            self.antView, antDs = volumes.raw_view_from_node(antNode, log=self.log)
            self.antImage = processing.preprocess(self.antView.pixels, self.antView.spacing, resample, normalize,
                                                  log=self.log)
            self.antProcessedNode = self._updateProcessedNode(self.antProcessedNode, self.antImage,
                                                              processing.output_spacing(self.antView.spacing, resample),
                                                              "DMSAAI ANT", "ant")
            if antDs is not None and self.antView.view not in ("ant", "unknown"):
                self.log(f"Warning: the selected anterior volume looks like a {self.antView.view.upper()} view.")
        else:
            self._removeNode(self.antProcessedNode)
            self._removeNode(self.antSegmentationNode)
            self.antProcessedNode = None
            self.antSegmentationNode = None
        # Obliques are preprocessed like in DMSAAI and only displayed.
        obliqueNodes = self.findObliques(postNode)
        for view in self.OBLIQUE_KEYS:
            node = obliqueNodes.get(view)
            if node is None:
                self._removeNode(self.obliqueProcessedNodes.pop(view, None))
                continue
            planar, _ = volumes.raw_view_from_node(node, log=self.log)
            image = processing.preprocess(planar.pixels, planar.spacing, resample, normalize, log=self.log)
            self.obliqueImages[view] = image
            self.obliqueProcessedNodes[view] = self._updateProcessedNode(
                self.obliqueProcessedNodes.get(view), image, processing.output_spacing(planar.spacing, resample),
                f"DMSAAI {view.upper()}", view)
        self._hideSegmentations()
        self.log("Images prepared (raw DICOM pixels, 256x256).")

    def _updateProcessedNode(self, node, image, spacing, name, view):
        if node is None or slicer.mrmlScene.GetNodeByID(node.GetID()) is None:
            node = volumes.create_planar_volume(image, spacing, name, {volumes.ATTR_ROLE: "processed",
                                                                       volumes.ATTR_VIEW: view})
            # Keep the analysis images out of the input selectors.
            node.SetHideFromEditors(True)
        else:
            volumes.update_planar_volume(node, image, spacing)
        return node

    def _removeNode(self, node):
        if node is not None and slicer.mrmlScene.GetNodeByID(node.GetID()) is not None:
            slicer.mrmlScene.RemoveNode(node)

    def segment(self, modelName):
        if self.postImage is None:
            raise ValueError("Select a posterior image first.")
        self.log(f"Running {modelName} on the posterior image ({self.device()})...")
        self.modelName = modelName
        self.classification = None
        self.postProbabilities = networks.segment(self.postImage, modelName, self.weightsDir, self.device())
        self.log("Segmentation complete.")

    def updateMasks(self, thresholdLeft=0.5, thresholdRight=0.5, backgroundMethod="Circumferential"):
        if self.postProbabilities is None:
            raise ValueError("Run the segmentation first.")
        self.postMask = processing.masks_from_probabilities(self.postProbabilities, thresholdLeft, thresholdRight)
        self.antMask = processing.anterior_mask(self.postMask) if self.antImage is not None else None
        self.srf = processing.split_renal_function(self.postImage, self.postMask, self.antImage, self.antMask,
                                                   backgroundMethod)
        self.postSegmentationNode = self._updateSegmentation(self.postSegmentationNode, "DMSAAI POST ROIs",
                                                             self.postProcessedNode, self.postMask,
                                                             self.srf["posterior"], "post")
        if self.antImage is not None:
            self.antSegmentationNode = self._updateSegmentation(self.antSegmentationNode, "DMSAAI ANT ROIs",
                                                                self.antProcessedNode, self.antMask,
                                                                self.srf["anterior"], "ant")
        self._updateSrfLabels()
        self._updateViewAnnotations()
        return self.srf

    def displayedSRF(self):
        """SRF shown on the images: geometric mean when the anterior view is available, like DMSAAI."""
        if self.srf is None:
            return None
        return self.srf["gmSRF"] if self.srf["gmSRF"] is not None else self.srf["postSRF"]

    def sliceNodeID(self, view):
        return "vtkMRMLSliceNode" + self.VIEW_SLICE[view]

    def _updateSegmentation(self, segNode, name, referenceNode, mask, counts, view):
        if segNode is None or slicer.mrmlScene.GetNodeByID(segNode.GetID()) is None:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
            segNode.SetAttribute(volumes.ATTR_ROLE, "rois")
            segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceNode)
        segmentation = segNode.GetSegmentation()
        arrays = {
            "LeftKidney": mask == processing.LEFT,
            "RightKidney": mask == processing.RIGHT,
            "LeftBackground": counts[processing.LEFT]["bgMask"],
            "RightBackground": counts[processing.RIGHT]["bgMask"],
        }
        for segmentId, array in arrays.items():
            if segmentation.GetSegment(segmentId) is None:
                segmentation.AddEmptySegment(segmentId, self.SEGMENT_NAMES[segmentId], self.SEGMENT_COLORS[segmentId])
            if array is None:
                array = np.zeros_like(mask, dtype=bool)
            slicer.util.updateSegmentBinaryLabelmapFromArray(array.astype(np.uint8)[np.newaxis, :, :], segNode,
                                                             segmentId, referenceNode)
        display = segNode.GetDisplayNode()
        # Contours only, as in DMSAAI, and only in this view's slice viewer.
        display.SetOpacity2DFill(0.0)
        display.SetOpacity2DOutline(1.0)
        display.SetSliceIntersectionThickness(2)
        display.SetViewNodeIDs([self.sliceNodeID(view)])
        display.SetVisibility(self.overlaysVisible)
        return segNode

    def setOverlaysVisible(self, show):
        """Show or hide all ROIs and SRF labels."""
        self.overlaysVisible = bool(show)
        if self.srf is not None:
            for segNode in (self.postSegmentationNode, self.antSegmentationNode):
                if segNode is not None and segNode.GetDisplayNode() is not None:
                    segNode.GetDisplayNode().SetVisibility(self.overlaysVisible)
        for node in self.srfLabelNodes.values():
            if slicer.mrmlScene.GetNodeByID(node.GetID()) is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(self.overlaysVisible)

    def customColorNode(self, name, csvPath):
        """Colour table node for a DMSAAI colour map CSV, created once per scene."""
        nodeName = f"DMSAAI {name}"
        node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if node is not None and node.IsA("vtkMRMLColorTableNode"):
            return node
        colors = np.loadtxt(csvPath, delimiter=",", ndmin=2)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", nodeName)
        node.SetTypeToUser()
        node.SetNumberOfColors(len(colors))
        node.GetLookupTable().SetRange(0, len(colors) - 1)
        for index, (r, g, b) in enumerate(colors):
            node.SetColor(index, f"{name} {index}", r, g, b, 1.0)
        node.SetAttribute("Category", "DMSAAI")
        return node

    def _updateSrfLabels(self):
        """Percentage under each kidney in its colour, like the DMSAAI overlay."""
        values = self.displayedSRF()
        for view, volumeNode, mask in (("post", self.postProcessedNode, self.postMask),
                                       ("ant", self.antProcessedNode, self.antMask)):
            for label in (processing.LEFT, processing.RIGHT):
                key = (view, label)
                node = self.srfLabelNodes.get(key)
                if node is not None and slicer.mrmlScene.GetNodeByID(node.GetID()) is None:
                    node = None
                if node is None:
                    if volumeNode is None:
                        continue
                    node = self._createLabelNode(f"DMSAAI {view.upper()} {processing.LABEL_NAMES[label]} SRF",
                                                 self.LABEL_COLORS[label], view)
                    self.srfLabelNodes[key] = node
                node.RemoveAllControlPoints()
                node.GetDisplayNode().SetVisibility(self.overlaysVisible)
                if volumeNode is None or mask is None or values is None or values[label] <= 0.0:
                    continue
                coords = np.argwhere(mask == label)
                if len(coords) == 0:
                    continue
                row = coords[:, 0].max() + 8
                col = coords[:, 1].mean()
                ijkToRas = vtk.vtkMatrix4x4()
                volumeNode.GetIJKToRASMatrix(ijkToRas)
                ras = ijkToRas.MultiplyPoint((col, row, 0, 1))[:3]
                node.AddControlPoint(list(ras), f"{values[label]:.1f}%")

    def _createLabelNode(self, name, color, view):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.SetAttribute(volumes.ATTR_ROLE, "label")
        node.SetHideFromEditors(True)
        node.SetLocked(True)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetColor(color)
        display.SetSelectedColor(color)
        if hasattr(display, "SetActiveColor"):
            display.SetActiveColor(color)
        display.SetUseGlyphScale(False)
        display.SetGlyphSize(0.01)
        display.SetTextScale(4.0)
        display.SetPointLabelsVisibility(True)
        if hasattr(display, "SetPropertiesLabelVisibility"):
            display.SetPropertiesLabelVisibility(False)
        if hasattr(display, "GetTextProperty"):
            textProperty = display.GetTextProperty()
            textProperty.SetBold(True)
            textProperty.SetBackgroundColor(1.0, 1.0, 1.0)
            textProperty.SetBackgroundOpacity(0.6)
        display.SetViewNodeIDs([self.sliceNodeID(view)])
        return node

    def showBackground(self, show):
        for segNode in (self.postSegmentationNode, self.antSegmentationNode):
            if segNode is None or segNode.GetDisplayNode() is None:
                continue
            for segmentId in ("LeftBackground", "RightBackground"):
                segNode.GetDisplayNode().SetSegmentVisibility(segmentId, bool(show))

    def _hideSegmentations(self):
        for segNode in (self.postSegmentationNode, self.antSegmentationNode):
            if segNode is not None and segNode.GetDisplayNode() is not None:
                segNode.GetDisplayNode().SetVisibility(False)
        for node in self.srfLabelNodes.values():
            if slicer.mrmlScene.GetNodeByID(node.GetID()) is not None:
                node.RemoveAllControlPoints()

    def classify(self):
        if self.postImage is None:
            raise ValueError("Select a posterior image first.")
        self.log("Running the kidney evaluation models...")
        self.classification = networks.classify(self.postImage, self.weightsDir, self.device())
        self.log("Clinical evaluation complete.")
        return self.classification

    def resultsText(self):
        parts = []
        if self.modelName:
            parts.append(f"Segmentation model: {self.modelName}\n")
        if self.srf is not None:
            parts.append(processing.format_srf(self.srf))
            lines = ["=== Counts ==="]
            for label, side in ((processing.LEFT, "Left"), (processing.RIGHT, "Right")):
                post = self.srf["posterior"][label]
                lines.append(f"POST {side}: gross {post['gross']:,.0f}  bg/px {post['bgMean']:.2f}  "
                             f"net {post['net']:,.0f}  area {post['area']} px")
                if self.srf["anterior"] is not None:
                    ant = self.srf["anterior"][label]
                    lines.append(f"ANT  {side}: gross {ant['gross']:,.0f}  bg/px {ant['bgMean']:.2f}  "
                                 f"net {ant['net']:,.0f}  area {ant['area']} px")
            lines.append(f"Posterior total counts: {self.srf['postTotalCounts']:,.0f}")
            if self.srf["antTotalCounts"] is not None:
                lines.append(f"Anterior total counts: {self.srf['antTotalCounts']:,.0f}")
            parts.append("\n".join(lines) + "\n")
        if self.classification is not None:
            parts.append(networks.format_classification(self.classification))
        return "\n".join(parts)

    # -- display --------------------------------------------------------

    def _layoutXML(self):
        def sliceView(view, color):
            return (f'<item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="{self.VIEW_SLICE[view]}">'
                    '<property name="orientation" action="default">Coronal</property>'
                    f'<property name="viewlabel" action="default">{view.upper()}</property>'
                    f'<property name="viewcolor" action="default">{color}</property></view></item>')
        colors = {"post": "#F34A33", "ant": "#EDD54C"}
        rows = "".join(
            # Equal split sizes so all six views get the same size.
            '<item splitSize="500"><layout type="horizontal" split="true">'
            + "".join(sliceView(view, colors.get(view, "#8C8C8C")) for view in row)
            + "</layout></item>"
            for row in self.LAYOUT_ROWS)
        return f'<layout type="vertical" split="true">{rows}</layout>'

    def showLayout(self):
        """2x3 coronal views: POST, LPO, RPO on top and ANT, LAO, RAO below. Views without an image stay empty."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        if layoutNode.IsLayoutDescription(self.LAYOUT_ID):
            layoutNode.SetLayoutDescription(self.LAYOUT_ID, self._layoutXML())
        else:
            layoutNode.AddLayoutDescription(self.LAYOUT_ID, self._layoutXML())
        layoutManager.setLayout(self.LAYOUT_ID)
        for view, volumeNode in self._displayedVolumes().items():
            sliceWidget = layoutManager.sliceWidget(self.VIEW_SLICE[view])
            if sliceWidget is None:
                continue
            sliceNode = sliceWidget.mrmlSliceNode()
            sliceNode.SetOrientation("Coronal")
            composite = sliceWidget.mrmlSliceCompositeNode()
            composite.SetBackgroundVolumeID(volumeNode.GetID() if volumeNode else None)
            composite.SetForegroundVolumeID(None)
            composite.SetLabelVolumeID(None)
            sliceWidget.sliceLogic().FitSliceToAll()
            sliceNode.SetSliceOffset(0)
        self._updateViewAnnotations()

    def _displayedVolumes(self):
        volumesByView = {"ant": self.antProcessedNode, "post": self.postProcessedNode}
        for view in self.OBLIQUE_KEYS:
            volumesByView[view] = self.obliqueProcessedNodes.get(view)
        return volumesByView

    def _updateViewAnnotations(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        for view, node in self._displayedVolumes().items():
            sliceWidget = layoutManager.sliceWidget(self.VIEW_SLICE[view])
            if sliceWidget is None:
                continue
            title = self.VIEW_TITLES[view]
            if node is None:
                title += " (Not Loaded)"
            elif view in ("ant", "post") and self.srf is not None:
                title = title.replace(" View", " + Seg Overlay")
            corner = sliceWidget.sliceView().cornerAnnotation()
            corner.SetText(vtk.vtkCornerAnnotation.UpperEdge, title)
            corner.SetText(vtk.vtkCornerAnnotation.UpperRight, "")
            corner.SetText(vtk.vtkCornerAnnotation.LowerRight, "")
            if node is not None:
                # Patient side markers, as the image is displayed as stored: anterior views (ANT, LAO, RAO)
                # have the patient's right on the screen left, posterior views (POST, LPO, RPO) the left.
                left, right = ("RIGHT", "LEFT") if view in ("ant", "lao", "rao") else ("LEFT", "RIGHT")
                corner.SetText(vtk.vtkCornerAnnotation.LeftEdge, left)
                corner.SetText(vtk.vtkCornerAnnotation.RightEdge, right)
            else:
                corner.SetText(vtk.vtkCornerAnnotation.LeftEdge, "")
                corner.SetText(vtk.vtkCornerAnnotation.RightEdge, "")
            sliceWidget.sliceView().scheduleRender()

    def setDisplay(self, colorNodeID, windowFraction=1.0):
        pairs = [(self.postProcessedNode, self.postImage), (self.antProcessedNode, self.antImage)]
        pairs += [(self.obliqueProcessedNodes.get(view), self.obliqueImages.get(view)) for view in self.OBLIQUE_KEYS]
        for node, image in pairs:
            if node is None or image is None or node.GetDisplayNode() is None:
                continue
            display = node.GetDisplayNode()
            colorNode = slicer.mrmlScene.GetNodeByID(colorNodeID)
            if colorNode is not None:
                display.SetAndObserveColorNodeID(colorNodeID)
            maximum = float(np.max(image)) * windowFraction
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(0.0, max(maximum, 1e-6))


#
# DMSAAITest
#

class DMSAAITest(ScriptedLoadableModuleTest):
    """Tests that run inside Slicer without the networks (no PyTorch needed)."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_PlanarGeometry()
        self.setUp()
        self.test_QuantificationOnSyntheticImage()

    def test_PlanarGeometry(self):
        """Raw pixel (row 0, col 0) must be displayed at the top left of a coronal view."""
        self.delayDisplay("Testing planar geometry")
        pixels = np.zeros((4, 6), dtype=np.uint16)
        pixels[0, 0] = 100
        node = volumes.create_planar_volume(pixels, (2.0, 3.0), "geometry test")
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        first = ijkToRas.MultiplyPoint((0, 0, 0, 1))
        lastCol = ijkToRas.MultiplyPoint((5, 0, 0, 1))
        lastRow = ijkToRas.MultiplyPoint((0, 3, 0, 1))
        # Increasing column -> patient left (smaller R), i.e. screen right in the coronal view
        self.assertLess(lastCol[0], first[0])
        self.assertAlmostEqual(first[0] - lastCol[0], 15.0)
        # Increasing row -> inferior (screen down)
        self.assertLess(lastRow[2], first[2])
        self.assertAlmostEqual(first[2] - lastRow[2], 6.0)
        planar, ds = volumes.raw_view_from_node(node)
        self.assertTrue(np.array_equal(planar.pixels, pixels))
        self.delayDisplay("Planar geometry test passed")

    def test_QuantificationOnSyntheticImage(self):
        self.delayDisplay("Testing SRF on a synthetic image")
        image = np.zeros((256, 256), dtype=np.float32)
        image[100:140, 70:100] = 30.0   # left kidney (posterior view: screen left)
        image[100:140, 160:190] = 10.0  # right kidney
        probs = np.zeros((3, 256, 256), dtype=np.float32)
        probs[0] = 1.0
        probs[1, 100:140, 70:100] = 1.0
        probs[2, 100:140, 160:190] = 1.0
        logic = DMSAAILogic()
        node = volumes.create_planar_volume(image, (2.0, 2.0), "synthetic POST", {volumes.ATTR_VIEW: "post"})
        logic.prepare(node)
        logic.postProbabilities = probs
        srf = logic.updateMasks(0.5, 0.5, "Circumferential")
        self.assertAlmostEqual(srf["postSRF"][processing.LEFT], 75.0, places=3)
        self.assertAlmostEqual(srf["postSRF"][processing.RIGHT], 25.0, places=3)
        self.assertIsNotNone(logic.postSegmentationNode.GetSegmentation().GetSegment("LeftKidney"))
        # ROIs of the posterior view are shown only in the posterior slice view.
        display = logic.postSegmentationNode.GetDisplayNode()
        self.assertEqual(display.GetNumberOfViewNodeIDs(), 1)
        self.assertEqual(display.GetNthViewNodeID(0), logic.sliceNodeID("post"))
        self.assertEqual(logic.srfLabelNodes[("post", processing.LEFT)].GetNthControlPointLabel(0), "75.0%")
        self.delayDisplay("SRF test passed")
