#-----------------------------------------------------------------------------
set(MODULE_NAME DMSAAI)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  DICOMDmsaaiPlugin.py
  DMSAAILib/__init__.py
  DMSAAILib/dicomviews.py
  DMSAAILib/networks.py
  DMSAAILib/processing.py
  DMSAAILib/settings.py
  DMSAAILib/volumes.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/banner1.jpg
  Resources/banner2.jpg
  Resources/banner3.jpg
  Resources/banner4.jpg
  Resources/ColorMaps/DICOM-Custom.csv
  Resources/ColorMaps/Hel.csv
  Resources/ColorMaps/HotIron.csv
  Resources/ColorMaps/Jotun.csv
  Resources/ColorMaps/TruGold.csv
  Resources/ColorMaps/Valhalla.csv
  Resources/ColorMaps/Vanaheim.csv
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
