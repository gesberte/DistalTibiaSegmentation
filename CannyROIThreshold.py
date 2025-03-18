import numpy as np
import SimpleITK as sitk
import slicer
import matplotlib.pyplot as plt
import vtk

# -------------------------------- OBTAINING THE MAIN VOLUME --------------------------------

# Get the first scalar volume node in the Slicer scene
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

# Convert the volume to a numpy array
volumeArray = slicer.util.arrayFromVolume(volumeNode)

# Get the voxel spacing of the volume (in mm)
spacing = volumeNode.GetSpacing()

# -------------------------------- OBTAINING THE ROI (REGION OF INTEREST) --------------------------------

# Retrieve the ROI node corresponding to the distal region of interest
distal_region = slicer.util.getNode('distal_region')  

def get_roi_data(roi_node, volumeNode, volumeArray):
    """
    This function extracts data from the Region of Interest (ROI) defined in Slicer.
    It converts the ROI coordinates from the RAS system to the IJK system to index the volume.

    Inputs:
    - roi_node: ROI node
    - volumeNode: volume node containing the data
    - volumeArray: numpy array containing the volume image

    Outputs:
    - roi_data: extracted sub-volume corresponding to the ROI
    - Min and max coordinates in x, y, and z in IJK indices
    """
    
    # Retrieve ROI bounds in RAS coordinates
    bounds = [0] * 6
    roi_node.GetBounds(bounds)
    
    # Transform RAS coordinates to IJK
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)

    def convert_ras_to_ijk(point_ras):
        """Convert an RAS point to IJK indices"""
        point_ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(np.append(point_ras, 1.0), point_ijk)
        return [int(round(c)) for c in point_ijk[:3]]

    # Convert ROI bounds to IJK indices
    x_min, y_min, z_min = convert_ras_to_ijk([bounds[0], bounds[2], bounds[4]])
    x_max, y_max, z_max = convert_ras_to_ijk([bounds[1], bounds[3], bounds[5]])

    # Extract the sub-volume corresponding to the ROI
    roi_data = volumeArray[z_min:z_max, y_max:y_min, x_max:x_min]

    return roi_data, x_min, x_max, y_min, y_max, z_min, z_max

# Retrieve ROI coordinates and its content
distal_region_array, x_min, x_max, y_min, y_max, z_min, z_max = get_roi_data(distal_region, volumeNode, volumeArray)

print(x_min, x_max, y_min, y_max, z_min, z_max)

# Define the coordinates of the cropped image
coordinates_x = [x_max, x_min]
coordinates_y = [y_max, y_min]
coordinates_z = [z_min, z_max]

# Create a volume node to display the cropped image
output_image_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
slicer.util.updateVolumeFromArray(output_image_node, distal_region_array)
output_image_node.SetName('ImageDistal')

# -------------------------------- EDGE DETECTION WITH CANNY --------------------------------

# Convert the volume to a SimpleITK Image and cast it to float
image_distal = sitk.GetImageFromArray(distal_region_array)
image_distal = sitk.Cast(image_distal, sitk.sitkFloat32)

# Apply the Canny edge detection filter
canny_filter = sitk.CannyEdgeDetectionImageFilter()
edges_distal = canny_filter.Execute(image_distal)

# Convert the result to a numpy array
edges_array_distal = sitk.GetArrayFromImage(edges_distal)

# Create a binary mask from the Canny result
mask_distal = edges_array_distal > 0

# Extract intensity values of pixels detected by Canny in the original image
detected_intensities_Distal = distal_region_array[mask_distal]
detected_intensities_Distal = detected_intensities_Distal[detected_intensities_Distal > 0]

# Compute statistics on detected pixels
mean_intensity_Distal = np.mean(detected_intensities_Distal) if detected_intensities_Distal.size > 0 else 0
std_intensity_Distal = np.std(detected_intensities_Distal) if detected_intensities_Distal.size > 0 else 0

print(f'Average intensity of detected pixels in the distal region: {mean_intensity_Distal}')

# Create a volume node to visualize the edge detection result
output_image_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
slicer.util.updateVolumeFromArray(output_image_node, distal_region_array * mask_distal)
output_image_node.SetName('FilteredCannyImageDistal')

# -------------------------------- SEGMENTATION BY THRESHOLDING --------------------------------

# Create a segmentation node
segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
segmentationNode.CreateDefaultDisplayNodes()  # Add default display nodes
segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

# Create a segmentation editor widget
segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setSourceVolumeNode(volumeNode)

# Apply thresholding based on the mean intensity detected by Canny
segmentEditorWidget.setActiveEffectByName("Threshold")
effect = segmentEditorWidget.activeEffect()
effect.setParameter("MinimumThreshold", mean_intensity_Distal)
effect.self().onApply()
