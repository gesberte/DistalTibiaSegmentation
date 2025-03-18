###################### IMPORT LIBRARIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.linear_model import LinearRegression

###################### PARAMETERS #############################
# Define file paths (modify according to your environment)
data_path = "path/to/data"  # Update with the correct path
output_path = "path/to/output"  # Update with the save path

# List of fracture and label files
fractures = [os.path.join(data_path, f"Fracture{i+1}.nrrd") for i in range(8)]
segmentations = [os.path.join(data_path, f"Label{i+1}.nrrd") for i in range(8)]

# Output file name
graph_file = os.path.join(output_path, "Linear_fits_normalized_z.png")

###################### INITIALIZE PLOT #############################
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
all_X, all_y, all_y_pred = [], [], []

###################### LOOP THROUGH EACH CASE #############################
# Define the indices of the relevant layers for each fracture
# To be manually entered according to the frontal CT slices where the tibia is visible
y_ranges = []  # Example: [(y_min1, y_max1), (y_min2, y_max2), ...]

for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):
    volumeNode = slicer.util.loadVolume(fracture)
    labelmapNode = slicer.util.loadLabelVolume(segmentation)
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)

    y_min, y_max = y_ranges[i]
    first_legend = True

    for y_target in range(y_min, y_max + 1):
        indicesCoucheY = np.array(np.nonzero(labelmapArray[:, :, y_target])) if i == 2 else np.array(np.nonzero(labelmapArray[:, y_target, :]))
        if indicesCoucheY.size == 0:
            continue

        x_min, x_max = np.min(indicesCoucheY[1]), np.max(indicesCoucheY[1])
        z_min, z_max = np.min(indicesCoucheY[0]), np.max(indicesCoucheY[0])

        mean_intensities = np.zeros(z_max + 1)
        xlinez = []
        
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                if labelmapArray[z, :, y_target][x] == 1:
                    xlinez.append(volumeArray[z, :, y_target][x])
            mean_intensities[z] = np.mean(xlinez) if xlinez else 0
            xlinez = []

        z_range = np.array(range(z_min, z_max + 1))
        mean_intensities = mean_intensities[mean_intensities > 0]
        z_range = z_range[: len(mean_intensities)]
        z_range_normalized = (z_range - z_min) / (z_max - z_min)

        if len(mean_intensities) > 1:
            X = z_range_normalized.reshape(-1, 1)
            y = mean_intensities
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            if first_legend:
                plt.plot(X, y_pred, label=f'Fracture {i+1}', color=colors[i % len(colors)])
                first_legend = False
            else:
                plt.plot(X, y_pred, color=colors[i % len(colors)])

            all_X.extend(X.flatten())
            all_y.extend(y)
            all_y_pred.extend(y_pred)

###################### COMPUTE METRICS AND SAVE #############################
all_X, all_y, all_y_pred = map(np.array, [all_X, all_y, all_y_pred])

mse = mean_squared_error(all_y, all_y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(all_y, all_y_pred)

plt.xlabel("Positions (z)")
plt.ylabel("Mean intensities")
plt.title("Linear model fits")
plt.legend()
plt.savefig(graph_file, dpi=300)
plt.close()

print(f"MSE: {mse}")
print(f"R²: {r2}")
print(f"RMSE: {rmse}")
print(f"Graph saved at: {graph_file}")
