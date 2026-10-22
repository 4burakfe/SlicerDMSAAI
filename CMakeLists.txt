cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DMSAAI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerDMSAAI")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir (Kocaeli City Hospital), Demet Nak (Recep Tayyip Erdogan University)")
set(EXTENSION_DESCRIPTION "AI-powered renal cortical scintigraphy analysis for Tc-99m DMSA studies: kidney segmentation, split renal function and AI-assisted kidney evaluation from planar NM DICOM images.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerDMSAAI/main/DMSAAI.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/DMSAAI/main/scr.jpg")
set(EXTENSION_DEPENDS "PyTorch") # PyTorch Utils installs the right torch build for the GPU/CUDA

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DMSAAI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
