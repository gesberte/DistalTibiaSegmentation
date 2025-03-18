import slicer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

# Configure paths (modify according to file locations)
data_dir = "path/to/data"
save_path = "path/to/save"
fractures = [os.path.join(data_dir, f"Fracture{i+1}.nrrd") for i in range(8)]
segmentations = [os.path.join(data_dir, f"Label{i+1}.nrrd") for i in range(8)]
output_file = os.path.join(save_path, "Linear_curves_normalized_z.png")

# Define the index ranges for the regions of interest (ROI) for each fracture
# To be entered manually based on the CT frontal slices where the tibia is visible
roi_ranges = []  # Example: [(y_min1, y_max1), (y_min2, y_max2), ...]

# Initialize variables
plt.figure(figsize=(12, 8))
total_mse = []
total_r2 = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

# Process images
for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):
    volumeNode = slicer.util.loadVolume(fracture)
    labelmapNode = slicer.util.loadLabelVolume(segmentation)
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)

    y_min, y_max = roi_ranges[i]
    first_legend = True

    for y_target in range(y_min, y_max + 1):
        indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :])) if i != 2 else np.array(np.nonzero(labelmapArray[:, :, y_target]))
        if indicesCoucheY.size == 0:
            continue

        x_min, x_max = np.min(indicesCoucheY[1]), np.max(indicesCoucheY[1])
        z_min, z_max = np.min(indicesCoucheY[0]), np.max(indicesCoucheY[0])
        mean_intensities = np.zeros(z_max + 1)

        for z in range(z_min, z_max + 1):
            xlinez = [volumeArray[z, y_target, x] for x in range(x_min, x_max + 1) if labelmapArray[z, y_target, x] == 1]
            mean_intensities[z] = np.mean(xlinez) if xlinez else 0

        z_range = np.array(range(z_min, z_max + 1))
        mean_intensities = mean_intensities[mean_intensities > 0]
        z_range = z_range[: len(mean_intensities)]
        z_range_normalized = (z_range - z_min) / (z_max - z_min)

        if len(mean_intensities) > 1:
            lowess_result = lowess(mean_intensities, z_range_normalized, frac=0.3)
            y_pred = lowess_result[:, 1]
            mse = mean_squared_error(mean_intensities, y_pred)
            total_mse.append(mse)
            r2 = 1 - (np.sum((mean_intensities - y_pred) ** 2) / np.sum((mean_intensities - np.mean(mean_intensities)) ** 2))
            total_r2.append(r2)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], label=f'Fracture {i+1}' if first_legend else "", color=colors[i % len(colors)])
            first_legend = False

# Plot settings
plt.xlabel("Positions (z)")
plt.ylabel("Mean intensities")
plt.title("Linear curve modeling with LOWESS")
plt.legend()
plt.savefig(output_file, dpi=300)
plt.close()

# Print statistics
print(f"MSE : {np.mean(total_mse)}")
print(f"R² : {np.mean(total_r2)}")
print(f"RMSE: {np.sqrt(np.mean(total_mse))}")
print(f"Graph saved at: {output_file}")
