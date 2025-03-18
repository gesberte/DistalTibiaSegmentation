import numpy as np
import slicer
from scipy.ndimage import binary_opening
import vtk

# Load the volume in Slicer
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
volumeArray = slicer.util.arrayFromVolume(volumeNode)

z_dim, y_dim, x_dim = volumeArray.shape
segmentedpixels = np.zeros_like(volumeArray, dtype=np.uint8)
structure = np.ones((3, 3)) 

high_threshold = 220  # Threshold for cortical bone
low_threshold = 70    # Threshold for trabecular bone

previous_segment_size = 0

for y in range(y_dim):  # Iterate through each slice
    current_slice = volumeArray[:, y, :]
    for z in range(1, z_dim - 1):  # Limit indices to avoid overflow
        for x in range(1, x_dim - 1):
            if current_slice[z, x] > high_threshold:
                # Define the 3x3 window around the central pixel
                neighborhood = current_slice[z - 1:z + 2, x - 1:x + 2]
                if np.all(neighborhood > low_threshold):
                    segmentedpixels[z, y, x] = 1  # Mark the pixel (binary)

    # Apply morphological opening if the current segmentation is larger
    current_segment_size = np.sum(segmentedpixels[:, y, :])
    if current_segment_size > previous_segment_size:
        segmentedpixels[:, y, :] = binary_opening(segmentedpixels[:, y, :], structure=structure)

    previous_segment_size = current_segment_size

# Convert segmented pixels to intensity values of 255
segmentedpixels *= 255

# Create the segmented volume in Slicer
segmentedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SegmentedVolume")
slicer.util.updateVolumeFromArray(segmentedNode, segmentedpixels)

# Create the segmentation node
seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segmentedNode, seg)

# Display in Slicer
segmentedNode.CopyOrientation(volumeNode)
slicer.util.setSliceViewerLayers(label=segmentedNode, foreground=volumeNode, foregroundOpacity=0.5)
