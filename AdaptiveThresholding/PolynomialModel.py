###################### IMPORT LIBRARIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

###################### DEFINE GENERIC PATHS #############################
# Replace these paths with those corresponding to your project
base_data_path = "/path/to/data"
base_output_path = "/path/to/output"

fractures = [os.path.join(base_data_path, f"Fracture{i+1}.nrrd") for i in range(8)]
segmentations = [os.path.join(base_data_path, f"Label{i+1}.nrrd") for i in range(8)]

###################### INITIALIZE PLOT #############################
output_file = os.path.join(base_output_path, "Polynomials_degree3_normalized_z.png")
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
all_mean_intensities = []
all_predicted_intensities = []

###################### DATA EXTRACTION AND ANALYSIS #############################
# Define y-ranges for each fracture
# Define indices of the relevant slices for each fracture
# To be entered manually based on the CT frontal slices where the tibia is visible
y_ranges = []  # Example: [(y_min1, y_max1), (y_min2, y_max2), ...]

for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):
    volumeNode = slicer.util.loadVolume(fracture)
    labelmapNode = slicer.util.loadLabelVolume(segmentation)
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)
    
    y_min, y_max = y_ranges[i]  # Retrieve the associated y-range
    first_legend = True
    
    for y_target in range(y_min, y_max + 1):
        layerIndicesY = np.array(np.nonzero(labelmapArray[:, y_target, :]))
        if layerIndicesY.size == 0:
            continue
        
        x_min, x_max = np.min(layerIndicesY[1]), np.max(layerIndicesY[1])
        z_min, z_max = np.min(layerIndicesY[0]), np.max(layerIndicesY[0])

        mean_intensities = np.zeros(z_max + 1)
        
        for z in range(z_min, z_max + 1):
            xlinez = [volumeArray[z, y_target, x] for x in range(x_min, x_max + 1) if labelmapArray[z, y_target, x] == 1]
            mean_intensities[z] = np.mean(xlinez) if xlinez else 0

        z_range = np.array(range(z_min, z_max + 1))
        mean_intensities = mean_intensities[mean_intensities > 0]
        z_range = z_range[: len(mean_intensities)]
        z_range_normalized = (z_range - z_min) / (z_max - z_min)

        if len(mean_intensities) > 1:
            coefficients = np.polyfit(z_range_normalized, mean_intensities, 3)
            poly_func = np.poly1d(coefficients)
            
            all_mean_intensities.extend(mean_intensities)
            all_predicted_intensities.extend(poly_func(z_range_normalized))
            
            plt.plot(z_range_normalized, poly_func(z_range_normalized),
                     label=f'Fracture {i+1}' if first_legend else "",
                     color=colors[i % len(colors)])
            first_legend = False

###################### COMPUTE METRICS AND SAVE #############################
mse = mean_squared_error(all_mean_intensities, all_predicted_intensities)
rmse = np.sqrt(mse)

plt.xlabel("Positions (z)")
plt.ylabel("Mean Intensities")
plt.title("Polynomial Degree 3 Modeling")
plt.legend()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Graph saved at: {output_file}")
