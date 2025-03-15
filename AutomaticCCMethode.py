#exec(open(r"C:\Users\Enora\OneDrive - ETS\Documents\Seuillage\AutomaticMethod\test2_Optimisation_finale2.py").read())
import numpy as np
import slicer
from scipy.ndimage import binary_opening
import vtk

# Chargement du volume dans Slicer
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
volumeArray = slicer.util.arrayFromVolume(volumeNode)

z_dim, y_dim, x_dim = volumeArray.shape
segmentedpixels = np.zeros_like(volumeArray, dtype=np.uint8)
structure = np.ones((3, 3)) 

high_threshold = 220  # Seuil pour l'os cortical
low_threshold = 70    # Seuil pour l'os trabéculaire

previous_segment_size = 0

for y in range(y_dim):  # Parcourt chaque couche
    current_slice = volumeArray[:, y, :]
    for z in range(1, z_dim - 1):  # Limiter les indices pour éviter des débordements
        for x in range(1, x_dim - 1):
            if current_slice[z, x] > high_threshold:
                # Définir la fenêtre 3x3 autour du pixel central
                neighborhood = current_slice[z - 1:z + 2, x - 1:x + 2]
                if np.all(neighborhood > low_threshold):
                    segmentedpixels[z, y, x] = 1  # Marque le pixel (binaire)

    # Application de l'ouverture morphologique si la segmentation actuelle est plus grande
    current_segment_size = np.sum(segmentedpixels[:, y, :])
    if current_segment_size > previous_segment_size:
        segmentedpixels[:, y, :] = binary_opening(segmentedpixels[:, y, :], structure=structure)

    previous_segment_size = current_segment_size

# Conversion des pixels segmentés en intensités de 255
segmentedpixels *= 255

# Création du volume segmenté dans Slicer
segmentedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SegmentedVolume")
slicer.util.updateVolumeFromArray(segmentedNode, segmentedpixels)

# Création du noeud de segmentation
seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segmentedNode, seg)

# Affichage dans Slicer
segmentedNode.CopyOrientation(volumeNode)
slicer.util.setSliceViewerLayers(label=segmentedNode, foreground=volumeNode, foregroundOpacity=0.5)
