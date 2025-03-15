#exec(open(r"C:\Users\Enora\OneDrive - ETS\Documents\IntensiteSegmentation\DonneesIntensites_legende.py").read())

###################### IMPORT DES LIBRAIRIES #############################
import slicer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
from sklearn.linear_model import LinearRegression


###################### AFFICHAGE DU NUAGE DE POINTS DES INTENSITÉS DES PIXELS EN FONCTION DE LEUR POSITION #############################

###################### IMPORT DES IMAGES ET DE LEUR LABEL #############################

fractures = [
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture1.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture2.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture3.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture4.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture5.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture6.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture7.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Data_Unet\Fracture8.nrrd"},
]

segmentations = [
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture1\Segmentation.Label.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture2\Segmentation.corrige-SegmentComplet-label.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture3\Segmentation.Label.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture4\Segmentation.Label.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture5\Segmentation.labels2.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture6\Segmentation.Label.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture7\Label7.nrrd"},
    {"path": r"C:\Users\Enora\OneDrive - ETS\Documents\Data\Fracture8\Label8.nrrd"},
]

###################### INITIALISATION DU GRAPH #############################

save_path = r"C:\Users\Enora\OneDrive - ETS\Documents\IntensiteSegmentation" #emplacement du graphique de sortie
file_name = "Scatter_plot_donnees.png" #nom du graphique de sortie
output_file = os.path.join(save_path, file_name) 
plt.figure(figsize=(12, 8))

###################### CRÉATION DU GRAPHIQUE #############################

for i, (fracture, segmentation) in enumerate(zip(fractures, segmentations)): #Pour chaque fracture et son label
    volumeNode = slicer.util.loadVolume(fracture["path"]) #On importe dans slicer le volume CT
    labelmapNode = slicer.util.loadLabelVolume(segmentation["path"]) #on importe dans Slicer sa segmentation
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

    #On initialise deux listes vides
    all_z_normalized = []
    all_mean_intensities = []
    
    #Pour chaque couche de chaque CT-scan 
    for y_target in range(y_min, y_max + 1):
        if i == 2:#la fracture 3 constitue un cas particulier
            indicesCoucheY = np.array(np.nonzero(labelmapArray[:, :, y_target]))
        else:
            indicesCoucheY = np.array(np.nonzero(labelmapArray[:, y_target, :]))#On regarde le CT selon la coupe frontale, on extrait donc ces couches uniquement

        if indicesCoucheY.size == 0:
            continue

        #On récupère la première et dernière couche selon les deux autre axes
        x_min = np.min(indicesCoucheY[1])
        x_max = np.max(indicesCoucheY[1])
        z_min = np.min(indicesCoucheY[0])
        z_max = np.max(indicesCoucheY[0])

        mean_intensities = []#On initialise une liste vide, dans laquelle on placera les moyennes d'intensité
        z_range = []#On initialise une liste vide, dans laquelle on placera les z

        #On parcourt chaque CT sur l'axe des z, ie. le long de l'os
        for z in range(z_min, z_max + 1):
            xlinez = []#On initialise une liste vide, où on placera les intensités des pixels d'intérêt
            for x in range(x_min, x_max + 1): #on parcourt toute la largeur de l'os sur la position z, pixel par pixel
                if (i == 2 and labelmapArray[z, x, y_target] == 1) or (i != 2 and labelmapArray[z, y_target, x] == 1): #Si on détecte de l'os sur la labelmap de segmentation (binaire, 1=os et 0=fond)
                    xlinez.append(volumeArray[z, x, y_target] if i == 2 else volumeArray[z, y_target, x])#On vient enregistrer l'intensité du pixel détecté et on le place dans la liste xlinez
            if xlinez: #si cette liste n'est pas vide
                mean_intensities.append(np.mean(xlinez))#On moyenne les intensités détectées une fois toute la ligne étudiée
                z_range.append(z)#On enregistre la position z qu'on vient d'étudier

        if mean_intensities:#Si il y a des moyennes d'intensités de pixels
            z_range_normalized = (np.array(z_range) - z_min) / (z_max - z_min) #On normalise la position z
            all_z_normalized.extend(z_range_normalized) #On enregistre ces positions z normalisées
            all_mean_intensities.extend(mean_intensities)

    # On créé seule légende par fracture, à chacune sa couleur et on affiche ça dans un nuage de points
    if all_z_normalized and all_mean_intensities:
        plt.scatter(all_z_normalized, all_mean_intensities, color=colors[i % len(colors)], alpha=0.5, label=f"Fracture {i+1}")
#Titres
plt.xlabel("Positions (z) normalisées")
plt.ylabel("Intensités moyennes")
plt.title("Graphique de dispersion des intensités par fracture")
plt.legend(title="Fractures")
plt.grid(True)

# On sauvegarde le graphe
plt.savefig(output_file, dpi=300)
plt.close()
#Indication sur console Slicer
print(f"Graphe enregistré à : {output_file}")
