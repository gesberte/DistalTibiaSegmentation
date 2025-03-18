###################### IMPORT LIBRARIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

###################### IMPORT IMAGES AND THEIR LABELS #############################
# Generic paths for adaptability
base_path = "/path/to/data"
fractures = [os.path.join(base_path, f"Fracture{i}.nrrd") for i in range(1, 9)]
segmentations = [os.path.join(base_path, f"Label{i}.nrrd") for i in range(1, 9)]

###################### SCRIPT INITIALIZATION #############################
save_path = "/path/to/save"

# Sigmoid function definition
def sigmoid(z, a, b, c):
    return a / (1 + np.exp(-b * (z - c)))

plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
mse_values, r2_values = [], []

###################### GRAPH CREATION #############################
# Define the y-layer indices of interest for each fracture
# Must be entered manually based on the CT frontal slices where the tibia is visible
y_ranges = []  # Example: [(y_min1, y_max1), (y_min2, y_max2), ...]

for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):
    volumeNode = slicer.util.loadVolume(fracture)
    labelmapNode = slicer.util.loadLabelVolume(segmentation)
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)
    
    y_min, y_max = y_ranges[i]
    first_legend = True
    
    for y_target in range(y_min, y_max + 1):
        indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :]))
        if indicesCoucheY.size == 0:
            continue
        
        x_min, x_max = np.min(indicesCoucheY[1]), np.max(indicesCoucheY[1])
        z_min, z_max = np.min(indicesCoucheY[0]), np.max(indicesCoucheY[0])
        
        mean_intensities = np.zeros(z_max + 1)
        for z in range(z_min, z_max + 1):
            xlinez = [volumeArray[z, y_target, x] for x in range(x_min, x_max + 1) 
                      if labelmapArray[z, y_target, x] == 1]
            mean_intensities[z] = np.mean(xlinez) if xlinez else 0
        
        mean_intensities = mean_intensities[mean_intensities > 0]
        if len(mean_intensities) < 2:
            continue
        
        z_range = np.arange(z_min, z_min + len(mean_intensities))
        z_range_normalized = (z_range - z_min) / (z_max - z_min)
        
        try:
            popt, _ = curve_fit(sigmoid, z_range_normalized, mean_intensities, 
                                p0=[np.max(mean_intensities), 1, 0.5], 
                                bounds=([0, 0, 0], [np.max(mean_intensities), 10, 1]))
            fitted_curve = sigmoid(z_range_normalized, *popt)
            
            mean_intensities_normalized = (mean_intensities - np.min(mean_intensities)) / (np.max(mean_intensities) - np.min(mean_intensities))
            fitted_curve_normalized = (fitted_curve - np.min(fitted_curve)) / (np.max(fitted_curve) - np.min(fitted_curve))
            
            mse = mean_squared_error(mean_intensities_normalized, fitted_curve_normalized)
            mse_values.append(mse)
            
            ss_total = np.sum((mean_intensities_normalized - np.mean(mean_intensities_normalized)) ** 2)
            ss_residual = np.sum((mean_intensities_normalized - fitted_curve_normalized) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            r2_values.append(r2)
            
            if first_legend:
                plt.plot(z_range_normalized, fitted_curve, label=f'Fracture {i+1}', color=colors[i % len(colors)])
                first_legend = False
            else:
                plt.plot(z_range_normalized, fitted_curve, color=colors[i % len(colors)])
        except RuntimeError:
            print(f"Fit failed for fracture {i+1}")

###################### ANALYSIS AND SAVING #############################
mse = np.mean(mse_values)
r2 = np.mean(r2_values)
rmse = np.sqrt(mse)
print(f"MSE: {mse}\nR²: {r2}\nRMSE: {rmse}")

output_file = os.path.join(save_path, "segmentation_intensity_plot.png")
plt.savefig(output_file, dpi=300)
plt.close()
