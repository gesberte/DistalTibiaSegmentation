import config
import numpy as np
import transformations as TF
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tibiadataset import TibiaDataset
from unet_model import UNet


def read_dcm(dcm_file_name):
    image_volume = sitk.ReadImage(dcm_file_name)
    np_volume = sitk.GetArrayFromImage(image_volume)

    dimension = image_volume.GetDimension()
    width = image_volume.GetWidth()
    height = image_volume.GetHeight()
    depth = image_volume.GetDepth()
    spacing = image_volume.GetSpacing()
    origin = image_volume.GetOrigin()
    direction = image_volume.GetDirection()

    if config.VERBOSE:
        print(f"Available fields: {image_volume.GetMetaDataKeys()}")
        print(image_volume.GetSize())
        print(np_volume.shape)
        print(np_volume.dtype)
        print(dimension)
        print(width)
        print(height)
        print(depth)
        print(spacing)
        print(origin)
        print(direction)

    return np_volume, image_volume, dimension, width, height, depth, spacing, origin, direction


def save_as_masks_nrrd_file(np_masks_segm, file_name, info_dcm):
    # np_labels_segm = np.transpose(np_masks_segm, (1, 2, 0))
    sitk_volume_masks = sitk.GetImageFromArray(np_masks_segm)
    sitk_volume_masks.CopyInformation(info_dcm)

    if config.VERBOSE:
        print(sitk_volume_masks.GetSize())
        print(sitk_volume_masks.GetOrigin())
        print(sitk_volume_masks.GetDimension())
        print(sitk_volume_masks.GetSpacing())
        print(sitk_volume_masks.GetDirection())

    # write the image
    sitk.WriteImage(sitk_volume_masks, file_name)


#def get_model(model_file: str):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # load the model from disk and flash it to the current device
    #model = torch.load(model_file, weights_only=False).to(device)

    #return model
def get_model(model_file: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Charger le modèle en mappant sur le bon appareil
    model = torch.load(model_file, map_location=device, weights_only=False)
    model.to(device)
    return model

def get_dataloader(np_volume_dcm, batch_size=config.BATCH_SIZE):
    images_transforms = TF.Compose([
        TF.ToTensor(),
        TF.Normalize([-1154.55], [1030.22])
    ])

    dcm_dataset = TibiaDataset(np_dcm_volume=np_volume_dcm, transform=images_transforms)
    dcm_data_loader = DataLoader(dcm_dataset, batch_size, shuffle=False)
    return dcm_data_loader


def segmentation_tibia(model: UNet, data_loader: DataLoader, volume_size):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    np_volume_masks = np.zeros(volume_size, dtype=np.uint8)

    print(np_volume_masks.shape)

    # switch off autograd
    model.eval()
    with torch.no_grad():
        for i, image in enumerate(data_loader):
            print(i)

            # send the input to the device
            image = image.to(device)

            # make the predictions and compute the validation loss
            predicted = model(image)
            out_softmax = torch.softmax(predicted, dim=1)  # normalise les valeurs entre 0 et 1 avec sum  = 1
            predicted_masks = torch.argmax(out_softmax, dim=1)  # perform argmax to generate 1 channel
            predicted_mask = predicted_masks[0].cpu().numpy()

            np_volume_masks[i] = predicted_mask

    return np_volume_masks


def segm_dcm(file_name):
    # load volume images from dcm file
    np_vol, info_dcm_volume, dim_vol, width_vol, height_vol, depth_vol, spacing_vol, origin_vol, direction_vol = read_dcm(file_name)

    # Prepare the Dataloader to segment the volume
    dcm_data_loader = get_dataloader(np_vol)

    # load the model from disk and flash it to the current device
    print("[INFO] load up model ...")
    model = get_model(config.MODEL_PATH)

    # Segmentation de l'humerus et de la scapula par Unet : retourne volume avec masques
    np_segm_masks = segmentation_tibia(model, dcm_data_loader, np_vol.shape)
    np_segm_tibia = np.array(np.where(np_segm_masks == 1, 1, 0), dtype=np.uint8)  # label 1 = humerus

    if config.VERBOSE:
        print()
        print("==== Debug - Volume Segm Tibia ====")
        print(np_segm_masks.shape)
        print(np_segm_masks.dtype)
        u, count = np.unique(np_segm_masks, return_counts=True)
        print(u)
        print(count)
        print("==== Debug - Volume Segm Tibia ====")
        print(np_segm_tibia.shape)
        print(np_segm_tibia.dtype)
        u, count = np.unique(np_segm_tibia, return_counts=True)
        print(u)
        print(count)

    # Save segmentation results in nrrd file
    save_as_masks_nrrd_file(np_segm_masks, config.SAVE_MASKS_FILE, info_dcm_volume)


if __name__ == '__main__':
    segm_dcm(config.DCM_PATH)
