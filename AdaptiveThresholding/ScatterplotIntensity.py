###################### IMPORT LIBRARIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

###################### CONFIGURABLE PARAMETERS #############################
# Define paths to your NRRD files
base_path_ct = "path/to/your/CT_scans"  # Replace with the folder containing CT scans
base_path_seg = "path/to/your/Segmentations"  # Replace with the folder containing segmentations
save_path = "path/to/output"  # Replace with the folder where the graph will be saved

# List of fracture and segmentation files
fractures = [
    {"path": os.path.join(base_path_ct, f"Fracture{i+1}.nrrd")} for i in range(8)
]

segmentations = [
    {"path": os.path.join(base_path_seg, f"Segmentation{i+1}.nrrd")} for i in range(8)
]

# Define indices of the layers of interest for each fracture
# To be manually entered according to the CT frontal slices where the tibia is visible
couches_interet = []  # Example: [(y_min1, y_max1), (y_min2, y_max2), ...]

# Output file name
file_name = "Scatter_plot_donnees.png"
output_file = os.path.join(save_path, file_name)

###################### INITIALIZE GRAPH #############################
plt.figure(figsize=(12, 8))
colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]

###################### CREATE SCATTER PLOT #############################
for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):
    volumeNode = slicer.util.loadVolume(fracture["path"])
    labelmapNode = slicer.util.loadLabelVolume(segmentation["path"])
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)
    
    y_min, y_max = couches_interet[i]  # Select layers of interest
    all_z_normalized, all_mean_intensities = [], []
    
    for y_target in range(y_min, y_max + 1):
        indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :]))
        if indicesCoucheY.size == 0:
            continue
        
        x_min, x_max = np.min(indicesCoucheY[1]), np.max(indicesCoucheY[1])
        z_min, z_max = np.min(indicesCoucheY[0]), np.max(indicesCoucheY[0])
        mean_intensities, z_range = [], []
        
        for z in range(z_min, z_max + 1):
            xlinez = [
                volumeArray[z, y_target, x]
                for x in range(x_min, x_max + 1)
                if labelmapArray[z, y_target, x] == 1
            ]
            
            if xlinez:
                mean_intensities.append(np.mean(xlinez))
                z_range.append(z)
        
        if mean_intensities:
            z_range_normalized = (np.array(z_range) - z_min) / (z_max - z_min)
            all_z_normalized.extend(z_range_normalized)
            all_mean_intensities.extend(mean_intensities)
    
    if all_z_normalized and all_mean_intensities:
        plt.scatter(all_z_normalized, all_mean_intensities, color=colors[i], alpha=0.5, label=f"Fracture {i+1}")

plt.xlabel("Normalized Positions (z)")
plt.ylabel("Mean Intensities")
plt.title("Scatter Plot of Intensities by Fracture")
plt.legend(title="Fractures")
plt.grid(True)
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Graph saved at: {output_file}")