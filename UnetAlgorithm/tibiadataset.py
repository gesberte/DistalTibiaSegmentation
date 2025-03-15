"""
    TibiaDataset is a custom Dataset that subclass torch.utils.data.Dataset and implement functions specific
    to tibia bone. It stores the samples and their corresponding labels for shoulder bones semantic segmentation.

"""

# Import Libraries ====================================
import numpy as np
import config
import torch
from torch.utils.data import Dataset


class TibiaDataset(Dataset):
    """
    A custom Dataset class that subclass Dataset for tibia bone dataset.

    :param Dataset: torch.utils.data.Dataset
    """
    def __init__(self, np_dcm_volume, transform=None):
        """
        :param np_dcm_volume: volume d'images.
        :param transform: specify the image and mask transformations.
        """
        self.np_images = np_dcm_volume
        self.transform = transform

    def __len__(self):
        """
        The __len__ function returns the number of samples in the dataset.

        :return: the number of samples in the dataset.
        """
        return self.np_images.shape[0]

    def __getitem__(self, idx):
        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index,
        it identifies the image and ths mask locations and converts them to tensor.

        :param idx: index idx given to return a sample from the dataset.
        :return: the tensors for image at the given index idx.
        """
        # Read image
        image_slice = self.np_images[idx, :, :]
        np_img = np.squeeze(image_slice, axis=None)
        image = torch.as_tensor(np_img.astype(np.int64))

        # Rescale pixel values
        if torch.min(image) > 0:
            intercept = int(config.INTERCEPT)
            slope = int(config.SLOPE)
            image = (slope * image) + intercept
        image = torch.unsqueeze(image, 0)

        # Call the transform functions on image and mask if applicable
        if self.transform is not None:
            image_t = self.transform(image)

        # Return tensors for image and corresponding label
        return image_t
