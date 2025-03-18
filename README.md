# DistalTibiaSegmentation

# Automatic segmentation of distal tibia fractures from CT images : 

This project focuses on developing an automatic or semi-automatic segmentation technique for distal tibia fractures from CT images. The goal is to generate a 3D segmentation model to improve preoperative visualization and understanding of fractures for surgeons.

The segmentation methods are primarily based on pixel intensity analysis, including:

- Mathematical modeling and advanced thresholding
- Region of Interest (ROI) segmentation with edge detection (Canny filter)
- Connected component analysis for automatic segmentation
- Deep learning-based segmentation using a 2D U-Net model

This repository contains Python scripts and methodologies for preprocessing, segmentation, and visualization of tibia fractures.

---

## 📂 Project Structure :

```
📂 segmentation-fractures-tibia/
│── 📜 README.md (Project documentation)
│── 📜 requirements.txt (Dependencies for Python environment)
│── 📂 scripts/ (Python scripts for segmentation)
│ │── 📂 AdaptiveThresholding/ (Adaptive thresholding method)
    │── ScatterplotIntensity.py 
    │── LinearRegression.py
    │── PolynomialModel.py
    │── LowessModel.py
    │── SigmoidModel.py 
│ │── CannyROIThreshold.py (ROI-based segmentation with edge detection)
│ │── AutomaticCCMethode.py (Connected component segmentation)
│ │── 📂 UnetAlgorithm/ (Deep Learning-based segmentation (U-Net))
    │── config.py 
    │── main.py
    │── tibiadataset.py
    │── transformations.py
    │── unetmodel.py 
```
---

🛠 Installation  

1️⃣ Clone the repository 
```bash
git clone https://github.com/gesberte/segmentation-fractures-tibia.git
cd segmentation-fractures-tibia
```

2️⃣ Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## Methods :  
These methods are independent of each other and can be used individually for segmentation.

### First Method : Adaptive thresholding based on position 

This approach explores the relationship between pixel position in the image (along the bone) and its intensity. The goal is to mathematically model this relationship to enable adaptive thresholding based on the pixel's position.
Distal fractures of long bones, such as the tibia, are challenging to segment due to variations in pixel intensity caused by the thinning of cortical bone at the epiphysis. While this intensity variation is well-documented in the literature, few studies have attempted to quantify this variation to adapt the thresholding of pixels based on their position within the image.
Thus, this method involves:
- Extracting pixel intensities along the bone in each image from the available fracture cases.
- Analyzing the intensity variation along the bone from these positions, visualizing this relationship as a scatter plot using the **ScatterplotIntensity.py** script.
- Modeling this variation using several mathematical approaches to predict an adaptive threshold. The different modeling methods include:
  - Linear regression: A simple modeling of intensity variation with a straight line (script **LinearRegression.py**).
  - Polynomial modeling: Using degree 2 or 3 polynomials for more flexible modeling (script **PolynomialModel.py**).
  - Lowess function: Local regression to adjust the intensities non-parametrically (script **LowessModel.py**).
  - Sigmoid modeling: Applying a sigmoid function to capture the gradual transitions in intensity (script **SigmoidModel.py**).

### Second Method : ROI-based segmentation with Canny filter 

This script implements a segmentation method based on the Canny filter, applied to a region of interest (ROI) manually defined around the tibia in a CT scan. The process begins with the user selecting a bounding box around the tibia in 3D Slicer. The Canny filter is then applied within this ROI to detect bone contours, and the intensity values of the detected pixels are used to compute an adaptive threshold. This threshold is dynamically adjusted based on the mean intensity of the detected contours to improve segmentation accuracy. The final step involves applying this threshold to the image, generating a segmented volume that is displayed in 3D Slicer. The method is particularly useful for segmenting fractured bones, where standard thresholding techniques often fail due to variations in pixel intensity. The script requires CT scan data in NRRD format and utilizes tools such as 3D Slicer for visualization and Python for execution. Key dependencies include NumPy and OpenCV. To use the script, it can be executed from 3D Slicer’s Python console with the following command:

```bash
python CannyROI_Threshold.py --input data/sample_ct.nii.gz --output results/roi_segmented.nii.gz
```

This approach enhances segmentation accuracy by focusing on relevant bone structures and adjusting the threshold dynamically based on the image characteristics. The output provides a refined segmentation of the tibia, which can be visualized and analyzed in 3D Slicer.

### Third Method : Automated Connected Component segmentation 

The Automated Connected Component Segmentation Method is an advanced bone fracture segmentation technique based on the 8-connected component method, as described by Ruikar, Santosh, and Hegadi (2019) (cf Reference). With a segmentation accuracy of 95.45%, this fully automated method was originally designed for bone fractures and validated on CT scans. Given its effectiveness, it has been implemented in this project to improve the segmentation of bone fragments. The algorithm analyzes each pixel and its eight neighboring pixels, making it particularly suitable for fractured bones, where intensity discontinuities are common due to multiple bone fragments. The method applies double thresholding, setting a high threshold for cortical bone (220 HU) and a low threshold for trabecular bone (70 HU), ensuring that only relevant bone structures are segmented. A 4×4 window verification further refines the selection by ensuring that a pixel identified as cortical bone is surrounded by pixels with at least trabecular bone intensity. To improve segmentation accuracy and reduce noise, morphological opening operations are applied, comparing the segmentation results layer by layer to remove artifacts and correctly separate bone fragments. The final segmented volume is saved and visualized in 3D Slicer.  

To use this method, run the following command in the Python console within 3D Slicer:  

```bash
python scripts/AutomaticCCMethode.py --input data/sample_ct.nii.gz --output results/connected_segmented.nii.gz
```

This script loads the CT scan, processes it using the connected component segmentation method, and saves the final segmentation result for further analysis.

### Fourth Method : Deep Learning-based segmentation (U-Net)

U-Net is used for segmenting distal tibia fractures due to its ability to preserve fine contours and perform well with limited data. A 2D model was chosen over a 3D one to expand the training dataset by leveraging CT scan slices.

**Model Development**
The model is implemented in Python using PyTorch and SimpleITK, with Compute Canada handling high-performance computing. Results are saved in NRRD format to preserve spatial information and analyzed in 3D Slicer.
- Retaining color images to avoid information loss.
- Normalization using the z-score method.
- Implementing a full U-Net with 23 convolutional layers.
- Data split: 80% training (2400 images), 10% validation, 10% testing.
- Data augmentation (rotation, translation, flipping).

**Protocol**
Labels were manually created in 3D Slicer and refined using its "Surface Wrap Solidify" module. After segmentation, post-processing is applied in 3D Slicer:
- Smoothing using morphological opening.
- Separating bone fragments with the "split islands" tool.

**Scripts used:** 
```bash
main.py, config.py, tibiadataset.py, transformations.py, unet_model.py.
```
---

Reference :  
1. [Automated Fractured Bone Segmentation and Labeling from CT Images](https://doi.org/10.1007/s10916-019-1176-x)  

---


## 📩 Contact :   
For any questions, feel free to reach out:  
📧 **enora.gesbert01@gmail.com**  
📍 [LinkedIn Profile](https://www.linkedin.com/in/enora-gesbert/)  

---

© 2025 Enora Gesbert - Licensed under MIT

