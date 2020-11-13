from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from skimage import io, transform
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import sys
np.set_printoptions(threshold=sys.maxsize)

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

# some utility functions
def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    mask = mask.transpose((2,3,1,0))
    mask = np.squeeze(mask)
    #mask = mask.transpose((1,2,0))
    mask = np.squeeze(mask)
    return mask

# converting tensor to image
def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    std = np.array((0.5,0.5,0.5))
    mean = np.array((0.5,0.5,0.5))
    image  = std * image + mean
    image = image.clip(0,1)
    image = (image * 255).astype(np.uint8)
    return image



class NuclieDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.folders = os.listdir(path)
        #self.transforms = get_transforms(0.5, 0.5)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (256, 256))
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

        ## 0-1 mask range ###
        mask = self.get_mask(mask_folder, 256, 256).astype('float32')
        mask *= 1.0 / mask.max()
        #augmented = self.transforms(image=img)

        #img = augmented['image']
        #mask = augmented['mask']

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        img = img.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        return (img, mask)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask

class NuclieTestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_transforms(0.5, 0.5)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (256, 256))

        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)

        return img

if __name__ == '__main__':
    base_dir = './data/'
    data = NuclieDataset(base_dir)
    # print out some sample data
    print(data.__len__())
    for d in data:
        visualize(image_convert(d[0]))
        visualize(mask_convert(d[1]))
        print(d[1].numpy())
        break
