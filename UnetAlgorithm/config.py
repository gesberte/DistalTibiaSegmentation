"""
    Parameters to configure the program
"""
import numpy as np

# Define directories for dcm ================================
BASE_PATH = "cheminVersLaBaseDeDonnees"
DCM_PATH = BASE_PATH + "/NomDuFichierDeLaFracture.nrrd"
SAVE_MASKS_FILE = "./NomDuFichierDuLabel.nrrd"

# Define files to save training model, etc. ===================
MODEL_PATH = "./segm_shoulder_model.pth"

# Parameters of the model ================================
INPUT_IMAGE_HEIGHT = 512    # 336    # 256
INPUT_IMAGE_WIDTH = 512     # 336     # 256
NUM_CHANNELS = 1
NUM_CLASSES = 2         # 10
PADDING = 'same'
BATCH_NORM = True
DROPOUT = True
NUM_FEATURES = 16
WEIGHT_ENTROPY = False
SCHEDULER = True
NETWORK = 'Unet'

# Parameters of the image dataset
SLOPE = 1.0
INTERCEPT = -1024.0

VERBOSE = True

# Hyper parameters ===============================
BATCH_SIZE = 1

# Dictionary to define mapping between pixel values in a mask image and index of each class ============
mapping_mask_to_class = {
    0: 0,       # Background
    255: 1      # Tibia
}

# Dictionary to define mapping between color and pixel values of mask image ============
color_map_mask = {0: np.array([0, 0, 0]),  # black for background
                  1: np.array([255, 255, 0])}  # yellow for tibia dist
