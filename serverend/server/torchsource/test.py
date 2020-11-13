import torch
from model import Unet
from dataset import NuclieTestDataset
from torch.utils.data import SequentialSampler, DataLoader
from matplotlib import pyplot as plt
import cv2 as cv
from dataset import mask_convert, visualize
from model import resnet18
import numpy as np

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

model = resnet18()
best_model_path = './bestmodel/bestmodel_255_resnet.pth'
test_data_dir = './data/'
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print(f'loading bestmodel UNET....')
model, optimizer, epochnum, valid_loss_min = load_ckp(best_model_path, model, optimizer)
print(f'EPOCH: {epochnum}, valid_loss_min: {valid_loss_min}')

testset = NuclieTestDataset(test_data_dir)
test_loader = DataLoader(dataset=testset,
                          drop_last=False,
                          sampler=SequentialSampler(testset),
                          batch_size=1)

model = model.cuda()
with torch.no_grad():
    for step,(image) in enumerate(test_loader):
        print(image.shape)
        imgshow = image.clone()
        imgshow = imgshow.squeeze()
        imgshow = imgshow.permute(1,2,0).numpy()
        imgshow = imgshow.astype(int)
        visualize(imgshow)
        image = image.cuda()
        pred_mask_1 = model.forward(image)
        print(pred_mask_1.shape)

        mask_img_1 = mask_convert(pred_mask_1)
        ### round values ###
        #mask_img_1 = np.round(mask_img_1)
        #mask_img_2 = np.round(mask_img_2)
        mask_img_1 = cv.normalize(mask_img_1, None, 0, 1, cv.NORM_MINMAX)
        ### normalize values to 0-1 ###
        #print(mask_img_1)
        mask_img_1 = np.round(mask_img_1)
        visualize(mask_img_1)
        break

