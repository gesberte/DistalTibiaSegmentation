#exec(open(r"C:\Users\Enora\OneDrive - ETS\Documents\Seuillage\SeuillageSimple\TestCannyThresholdDistal_Final.py").read())
import numpy as np
import SimpleITK as sitk
import slicer
import matplotlib.pyplot as plt
import vtk

# -------------------------------- OBTENTION DU VOLUME PRINCIPAL --------------------------------

# Récupérer le premier nœud de volume scalaire dans la scène Slicer
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

# Convertir le volume en tableau numpy
volumeArray = slicer.util.arrayFromVolume(volumeNode)

# Récupérer l'espacement des voxels du volume (en mm)
spacing = volumeNode.GetSpacing()

# -------------------------------- OBTENTION DE LA ROI (REGION OF INTEREST) --------------------------------

# Récupérer le nœud ROI correspondant à la région d'intérêt distale
distal_region = slicer.util.getNode('distal_region')  

def get_roi_data(roi_node, volumeNode, volumeArray):
    """
    Cette fonction extrait les données de la région d'intérêt (ROI) définie dans Slicer.
    Elle convertit les coordonnées de la ROI du système RAS au système IJK pour indexer le volume.

    Entrées :
    - roi_node : nœud de la ROI
    - volumeNode : nœud du volume contenant les données
    - volumeArray : tableau numpy contenant l'image du volume

    Sorties :
    - roi_data : sous-volume extrait correspondant à la ROI
    - Coordonnées min et max en x, y et z en indices IJK
    """
    
    # Récupération des bornes de la ROI en coordonnées RAS
    bounds = [0] * 6
    roi_node.GetBounds(bounds)
    
    # Transformation des coordonnées RAS en IJK
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)

    def convert_ras_to_ijk(point_ras):
        """Convertit un point RAS en indice IJK"""
        point_ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(np.append(point_ras, 1.0), point_ijk)
        return [int(round(c)) for c in point_ijk[:3]]

    # Conversion des bornes de la ROI en indices IJK
    x_min, y_min, z_min = convert_ras_to_ijk([bounds[0], bounds[2], bounds[4]])
    x_max, y_max, z_max = convert_ras_to_ijk([bounds[1], bounds[3], bounds[5]])

    # Extraction du sous-volume correspondant à la ROI
    roi_data = volumeArray[z_min:z_max, y_max:y_min, x_max:x_min]

    return roi_data, x_min, x_max, y_min, y_max, z_min, z_max

# Récupérer les coordonnées de la ROI et son contenu
distal_region_array, x_min, x_max, y_min, y_max, z_min, z_max = get_roi_data(distal_region, volumeNode, volumeArray)

print(x_min, x_max, y_min, y_max, z_min, z_max)

# Définir les coordonnées de l'image cropée
coordonnesx = [x_max, x_min]
coordonneesy = [y_max, y_min]
coordonnesz = [z_min, z_max]

# Création d'un nœud volume pour afficher l'image cropée
output_image_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
slicer.util.updateVolumeFromArray(output_image_node, distal_region_array)
output_image_node.SetName('ImageDistal')

# -------------------------------- DÉTECTION DES CONTOURS AVEC CANNY --------------------------------

# Conversion du volume en SimpleITK Image et mise au format float
image_distal = sitk.GetImageFromArray(distal_region_array)
image_distal = sitk.Cast(image_distal, sitk.sitkFloat32)

# Application du filtre de détection des contours de Canny
canny_filter = sitk.CannyEdgeDetectionImageFilter()
edges_distal = canny_filter.Execute(image_distal)

# Conversion du résultat en tableau numpy
edges_array_distal = sitk.GetArrayFromImage(edges_distal)

# Création d'un masque binaire à partir du résultat de Canny
mask_distal = edges_array_distal > 0

# Extraction des intensités des pixels détectés par Canny dans l'image originale
detected_intensities_Distal = distal_region_array[mask_distal]
detected_intensities_Distal = detected_intensities_Distal[detected_intensities_Distal > 0]

# Calcul des statistiques sur les pixels détectés
mean_intensity_Distal = np.mean(detected_intensities_Distal) if detected_intensities_Distal.size > 0 else 0
std_intensity_Distal = np.std(detected_intensities_Distal) if detected_intensities_Distal.size > 0 else 0

print(f'Intensité moyenne des pixels détectés dans la région distale : {mean_intensity_Distal}')

# Création d'un nœud volume pour visualiser le résultat de la détection des contours
output_image_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
slicer.util.updateVolumeFromArray(output_image_node, distal_region_array * mask_distal)
output_image_node.SetName('FilteredCannyImageDistal')

# -------------------------------- SEGMENTATION PAR SEUILLAGE --------------------------------

# Création d'un nœud de segmentation
segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
segmentationNode.CreateDefaultDisplayNodes()  # Ajoute des nœuds d'affichage par défaut
segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

# Création d'un widget d'édition de segmentation
segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setSourceVolumeNode(volumeNode)

# Application d'un seuillage basé sur l'intensité moyenne détectée par Canny
segmentEditorWidget.setActiveEffectByName("Threshold")
effect = segmentEditorWidget.activeEffect()
effect.setParameter("MinimumThreshold", mean_intensity_Distal)
effect.self().onApply()
