"""
 Implement custom transforms to apply transformations on images and masks for semantic segmentation.

"""

# Import Libraries ====================================
import numpy as np
import torch
from torchvision.transforms import functional as F


class Compose:
    """
    Baseclass - composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Normalize:
    """
    Normalize image and mask with z-score function.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class ToTensor:
    """
    Transfer image and mask to tensor.
    """
    def __call__(self, image):
        # Passe dans un tensor
        if torch.is_tensor(image):
            image = image.to(torch.float32)
        else:
            image = np.array(image, dtype=np.float32)
            image = torch.from_numpy(image)
            # Ajout du parametre Channel = 1 : C x H x W
            image = torch.unsqueeze(image, 0)
        return image

