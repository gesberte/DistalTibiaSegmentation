#with open(r"/Users/enoragesbert/Library/CloudStorage/OneDrive-ETS/Documents/Algorithmes rapport/sigmoide_modelisation_MSE.py", encoding="utf-8") as f:
    #exec(f.read())
###################### IMPORT DES LIBRAIRIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

###################### AFFICHAGE DU NUAGE DE POINTS DES INTENSITÉS DES PIXELS EN FONCTION DE LEUR POSITION #############################

###################### IMPORT DES IMAGES ET DE LEUR LABEL #############################
fractures = [
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture1.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture2.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture3.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture4.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture5.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture6.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture7.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Fracture8.nrrd",
]

segmentations = [
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label1.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label2.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label3.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label4.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label5.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label6.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label7.nrrd",
    "/Users/enoragesbert/Documents/ETS/Projet/RdN/Data/Label8.nrrd",
]

###################### INITIALISATION DU SCRIPT #############################
save_path = r"/Users/enoragesbert/Library/CloudStorage/OneDrive-ETS/Documents/IntensiteSegmentation"
def sigmoid(z, a, b, c): #création méthode sigmoide
    return a / (1 + np.exp(-b * (z - c)))
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
mse_values = []
r2_values = []

###################### CRÉATION DU GRAPHIQUE #############################
for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):#Pour chaque fracture et son label
    volumeNode = slicer.util.loadVolume(fracture)#On importe dans slicer le volume
    labelmapNode = slicer.util.loadLabelVolume(segmentation) #Et son label
    #Conversion des données en Numpy Array
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)
    #On importe que les couches d'intérêt, c'est-à-dire celles où on voit le tibia
    y_ranges = [(180, 246), (153, 189), (348, 371), (185, 210), 
                (223, 255), (307, 319), (230, 277), (246, 285)]
    y_min, y_max = y_ranges[i]

    first_legend = True  # Pour ne créer qu'une légende par fracture
    #Pour chaque couche de chaque CT-scan
    for y_target in range(y_min, y_max + 1):
        indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :]))#On regarde le CT selon la coupe frontale, on extrait donc ces couches uniquement
        if indicesCoucheY.size == 0:
            continue
        #On récupère les coordonnées des premières et dernières couches selon les deux autre axes
        x_min, x_max = np.min(indicesCoucheY[1]), np.max(indicesCoucheY[1])
        z_min, z_max = np.min(indicesCoucheY[0]), np.max(indicesCoucheY[0])
        
        mean_intensities = np.zeros(z_max + 1)#On initialise une liste nulle, dans laquelle on placera les moyennes d'intensité

        for z in range(z_min, z_max + 1):#On parcourt le CT sur la longueur de l'os
            xlinez = [volumeArray[z, y_target, x] for x in range(x_min, x_max + 1) #Puis pour chaque position sur la longueur de l'os, on parcourt la largeur de l'os sur cette position et on récupère l'intensité
                      if labelmapArray[z, y_target, x] == 1] #seulement si le pixel fait parti de l'os
            mean_intensities[z] = np.mean(xlinez) if xlinez else 0#On fait la moyenne des intensités extraites
        
        mean_intensities = mean_intensities[mean_intensities > 0]
        if len(mean_intensities) < 2:
            continue  # On ignore si pas assez de données
        
        # Normalisation de z_range entre 0 et 1
        z_range = np.arange(z_min, z_min + len(mean_intensities))
        z_range_normalized = (z_range - z_min) / (z_max - z_min)

        try:
            #Modélisation sigmoide
            popt, _ = curve_fit(sigmoid, z_range_normalized, mean_intensities, 
                    p0=[np.max(mean_intensities), 1, 0.5], 
                    bounds=([0, 0, 0], [np.max(mean_intensities), 10, 1]))

            fitted_curve = sigmoid(z_range_normalized, *popt)

            # Normalisation des intensités pour comparaison
            mean_intensities_normalized = (mean_intensities - np.min(mean_intensities)) / (np.max(mean_intensities) - np.min(mean_intensities))
            fitted_curve_normalized = (fitted_curve - np.min(fitted_curve)) / (np.max(fitted_curve) - np.min(fitted_curve))

            # Calcul du MSE
            mse = mean_squared_error(mean_intensities_normalized, fitted_curve_normalized)
            mse_values.append(mse)

            # Calcul du coefficient de détermination R²
            ss_total = np.sum((mean_intensities_normalized - np.mean(mean_intensities_normalized)) ** 2)
            ss_residual = np.sum((mean_intensities_normalized - fitted_curve_normalized) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            r2_values.append(r2)

            # Affichage
            if first_legend:
                plt.plot(z_range_normalized, fitted_curve, label=f'Fracture {i+1}', color=colors[i % len(colors)])
                first_legend = False
            else:
                plt.plot(z_range_normalized, fitted_curve, color=colors[i % len(colors)])
        except RuntimeError:
            print(f"Ajustement impossible pour la fracture {i+1}")

# On calcule l'écart entre les valeurs réelles et ajustées
intensites_diff = mean_intensities - fitted_curve



# On affiche les valeurs d'écart
print(f"Différence maximale : {np.max(np.abs(intensites_diff))}")
print(f"Différence minimale : {np.min(np.abs(intensites_diff))}")

# Sauvegarder le graphe
plt.savefig(output_file, dpi=300) 
plt.close()  # Fermer la figure pour libérer la mémoire


# On affiche mse et r2
mse = np.mean(mse_values)
r2 = np.mean(r2_values)
rmse = np.sqrt(mse)
print(f"MSE : {mse}")
print(f"R² : {r2}")
print("RMSE:", rmse)
