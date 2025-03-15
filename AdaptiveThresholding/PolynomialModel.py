#with open(r"/Users/enoragesbert/Library/CloudStorage/OneDrive-ETS/Documents/Algorithmes rapport/regressionpolynome3.py", encoding="utf-8") as f:
    #exec(f.read())

###################### IMPORT DES LIBRAIRIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os


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

###################### INITIALISATION DU GRAPHE #############################
save_path = r"/Users/enoragesbert/Library/CloudStorage/OneDrive-ETS/Documents/IntensiteSegmentation"
file_name = "Polynomes_degre3_normalise_z.png"
output_file = os.path.join(save_path, file_name)
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
# Initialiser les variables pour stocker MSE et R²
all_mean_intensities = []
all_predicted_intensities = []

###################### CRÉATION DU GRAPHIQUE #############################
for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)):#Pour chaque fracture et son label
    volumeNode = slicer.util.loadVolume(fracture)#On importe dans slicer le volume
    labelmapNode = slicer.util.loadLabelVolume(segmentation) #Et son label
    #Conversion des données en Numpy Array
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    labelmapArray = slicer.util.arrayFromVolume(labelmapNode)

    #On importe que les couches d'intérêt, c'est-à-dire celles où on voit le tibia
    if i == 0:  # Fracture1
        y_min, y_max = 180, 246
    elif i == 1:  # Fracture2
        y_min, y_max = 153, 189
    elif i == 2:  # Fracture3
        y_min, y_max = 348, 371
    elif i == 3:  # Fracture4
        y_min, y_max = 185, 210
    elif i == 4:  # Fracture5
        y_min, y_max = 223, 255
    elif i == 5:  # Fracture6
        y_min, y_max = 307, 319
    elif i == 6:  # Fracture7
        y_min, y_max = 230, 277
    elif i == 7:  # Fracture8
        y_min, y_max = 246, 285

    first_legend = True  # Pour ne créer qu'une légende par fracture
    #Pour chaque couche de chaque CT-scan
    for y_target in range(y_min, y_max + 1):
        if i == 2:#la fracture 3 constitue un cas particulier
            indicesCoucheY = np.array(np.nonzero(labelmapArray[:, :, y_target]))
        else:
            indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :]))#On regarde le CT selon la coupe frontale, on extrait donc ces couches uniquement

        if indicesCoucheY.size == 0:
            continue
        #On récupère les coordonnées des premières et dernières couches selon les deux autre axes
        x_min = np.min(indicesCoucheY[1])
        x_max = np.max(indicesCoucheY[1])
        z_min = np.min(indicesCoucheY[0])
        z_max = np.max(indicesCoucheY[0])

        mean_intensities = np.zeros(z_max + 1)#On initialise une liste nulle, dans laquelle on placera les moyennes d'intensité
        xlinez = []#On initialise une liste vide, où on placera les intensités des pixels d'intérêt
        if i == 2:#la fracture 3 constitue un cas particulier
            for z in range(z_min, z_max + 1):
                for x in range(x_min, x_max + 1):
                    if labelmapArray[z, x, y_target] == 1:
                        xlinez.append(volumeArray[z, x, y_target])
                mean_intensities[z] = np.mean(xlinez) if len(xlinez) > 0 else 0
                xlinez = []
        else:
            for z in range(z_min, z_max + 1):#On parcourt le CT sur la longueur de l'os
                for x in range(x_min, x_max + 1):#Puis pour chaque position sur la longueur de l'os, on parcourt la largeur de l'os sur cette position
                    if labelmapArray[z, y_target, x] == 1:#Si le pixel est détecté comme faisant parti de l'os
                        xlinez.append(volumeArray[z, y_target, x])#On récupère l'intensité de ce pixel dans la liste xlinez
                mean_intensities[z] = np.mean(xlinez) if len(xlinez) > 0 else 0#On fait la moyenne des intensités extraites
                xlinez = []#On vide la liste des intensités avant de passer à la position z+1 suivante

        # Normalisation de z_range entre 0 et 1
        z_range = np.array(range(z_min, z_max + 1))
        mean_intensities = mean_intensities[mean_intensities > 0]
        z_range = z_range[: len(mean_intensities)]
        z_range_normalized = (z_range - z_min) / (z_max - z_min)

        if len(mean_intensities) > 1:
            coefficients = np.polyfit(z_range_normalized, mean_intensities, 3)  # Degré 2 ou 3 de la régression polynomiale en fonction de ce qu'on veut
            poly_func = np.poly1d(coefficients)
            # Ajouter aux listes pour calculer le MSE et R²
            all_mean_intensities.extend(mean_intensities)
            all_predicted_intensities.extend(poly_func(z_range_normalized))
            # On trace la courbe ajustée avec la régression polynomiale
            if first_legend:
                plt.plot(
                    z_range_normalized, poly_func(z_range_normalized),
                    label=f'Fracture {i+1}',
                    color=colors[i % len(colors)]
                )
                first_legend = False
            else:
                plt.plot(
                    z_range_normalized, poly_func(z_range_normalized),
                    color=colors[i % len(colors)]
                )


# Calcul du MSE et du R² globaux
mse = mean_squared_error(all_mean_intensities, all_predicted_intensities)
rmse = np.sqrt(mse)
r_squared = 1 - (np.sum((np.array(all_mean_intensities) - np.array(all_predicted_intensities)) ** 2) / 
                 np.sum((np.array(all_mean_intensities) - np.mean(all_mean_intensities)) ** 2))


plt.xlabel("Positions (z)")
plt.ylabel("Intensités moyennes")
plt.title("Modélisations polynomiales de degré 3")
plt.legend()

# Sauvegarder le graphe
plt.savefig(output_file, dpi=300) 
plt.close()  # Fermer la figure pour libérer la mémoire

print(f"MSE : {mse}")
print(f"R² : {r_squared}")
print("RMSE:", rmse)
print(f"Graphe enregistré à : {output_file}")