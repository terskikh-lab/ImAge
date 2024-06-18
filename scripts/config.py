import re
from pathlib import Path
import numpy as np
from imagepubautomation.feature_modeling.modules import (
    s1_o1_image_cenvec_bootstrap_traintestsplit,
    s1_o1_image_cenvec_bootstrap_accuracy_curve,
    s1_o1_image_svm_bootstrap_traintestsplit,
    s1_o1_image_svm_bootstrap_accuracy_curve,
)

# Path To Image Directory
# NOTE: MAKE SURE IMAGE DIRECTORY CONTAINS SHORT NAME AT THE END IN BRACKETS
folder = "MIEL72_MOUSE-WBC_CLACK_V3_20191210131745[MIEL72_PBMC_clock]"
experiment_name = (
    re.search(re.compile("[[]{1}[A-Za-z0-9_]+[]]{1}"), folder)[0]
    .replace("[", "")
    .replace("]", "")
)

# DO NOT CHANGE dataPathRoot UNLESS NECESSARY
dataPathRoot = Path("../data/images")
# Path to save directory
resultsPathRoot = Path("../data/results")


############################################
# Image Segmentation Parameters
############################################
# Segment Images in full 3D or calculate maximum projections and run in 2D
run_3d = False

# Run Basic Illumination Correction on images? (must have more than 25 wells)
run_illumination_correction = False

# Don't mess with these unless you know what you're doing

# # PHENIX PRESETS
# # Segmentation Channel
# # NOTE: should appear as seen in image filnames
# # NOTE: CAPS SENSITIVE
# segmentation_channel = "ch1"
# search_pattern = ".tif"
# metadata_pattern = re.compile(
#     "r([0-9]{2})c([0-9]{2})f([0-9]{2})p([0-9]{2})-([a-zA-Z0-9]{3})"
# )
# channelIndex = 4
# rowIndex = 0
# colIndex = 1
# zIndex = 3
# FOVIndex = 2
# tIndex = None
# imagedims = (0.6, 0.6) # um per pixel

# IC200 PRESETS
# Segmentation Channel
# NOTE: should appear as seen in image filnames
# NOTE: CAPS SENSITIVE
segmentation_channel = "DAPI"
search_pattern = ".tif"
metadata_pattern = re.compile(
    "([A-Za-z0-9]+)__([A-Za-z]{1})_([0-9]{3})_r_([0-9]{4})_c_([0-9]{4})_t_([a-zA-Z0-9]+)_z_"
)
channelIndex = 0
rowIndex = 1
colIndex = 2
zIndex = 5
FOVIndex = (3, 4)
tIndex = None
imagedims = (0.3, 0.3)  # um per pixel


############################################
# Feature Extraction Parameters
############################################

# Label Channels
# NOTE: TAS features will not be extracted on these
# NOTE: should appear as seen in image filnames
# NOTE: CAPS SENSITIVE
# NOTE: Should be in quotes, separated by commas, surrounded by brackets
# EX: ["CD3", "LEF1"]
label_channels = ["CD3"]


############################################
# ImAge Axis / Bootstrapping / Analysis Parameters
############################################

# Parameters
platemap_path = Path("../data/platemaps/platmap.xlsx")
sheet_name = "platemap"
seed = 425
num_iters = 100
area_feature = "MOR_object_pixel_count"
tas_fstring_regex = "%s_TXT_TAS"
group_col = "ExperimentalCondition"
sample_col = "Sample"
group_A = "Young"
group_B = "Old"
subset = re.compile("TXT")
num_cells = 200
num_bootstraps = 1000
cd3_threshold = 150

high_size_threshold = np.inf
if any(i in folder for i in ["behavior"]):
    low_size_threshold = 300  # ~5um
elif any(i in folder for i in ["MIEL86", "MIEL83"]):
    low_size_threshold = 600  # ~ 7um
elif any(i in folder for i in ["liver_3ages", "cancer"]):
    low_size_threshold = 75  # ~3um
elif any(i in folder for i in ["3ages", "MIEL72"]):
    low_size_threshold = 100  # ~3um
elif any(i in folder for i in ["OSKM"]):
    low_size_threshold = 75
else:
    low_size_threshold = 0

# For Centroid Vector Classifier
image_script = s1_o1_image_cenvec_bootstrap_traintestsplit
accuracy_curve_script = s1_o1_image_cenvec_bootstrap_accuracy_curve

# For SVM Classifier
# script = s1_o1_image_svm_bootstrap_traintestsplit
# accuracy_curve_script = s1_o1_image_svm_bootstrap_accuracy_curve
