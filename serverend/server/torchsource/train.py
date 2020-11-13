import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
from dataset import NuclieDataset
import torch
from model import Unet
from model import DiceBCELoss
from model import iou_batch
import numpy as np
import shutil
from model import resnet18, resnet50


# ref https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)

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


base_dir = './data/'
checkpoint_path = './checkpoint'
initial_checkpoint = None
best_model_path = './bestmodel/bestmodel.pth'
data = NuclieDataset(base_dir)
rng = 205322607
torch.manual_seed(rng)
training_set, valid_set = random_split(data, [590, 80])


batch_size = 10
#### create dataloader ####
train_loader = DataLoader(dataset=training_set,
                          drop_last=True,
                          sampler=RandomSampler(training_set),
                          batch_size=batch_size)

valid_loader = DataLoader(dataset=valid_set,
                          drop_last=False,
                          sampler=SequentialSampler(valid_set),
                          batch_size=batch_size)

epochs = 25
model = Unet().cuda()
criterion = DiceBCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate, weight_decay=1e-5)

train_loss,val_loss = [],[]
train_iou,val_iou = [],[]
valid_loss_min = 10000
i = 1
if initial_checkpoint is not None:
    model, optimizer, epochnum, valid_loss_min = load_ckp(checkpoint_path+initial_checkpoint, model, optimizer)
    print(f'initial ckp: {epochnum}')
    i = i + epochnum


for epoch in range(epochs):

    running_train_loss = []
    running_train_score = []
    model.train()
    for step,(image, mask) in enumerate(train_loader):

        image = image.cuda()
        mask = mask.cuda()
        pred_mask = model.forward(image)  # forward propogation
        loss = criterion(pred_mask, mask)
        score = iou_batch(pred_mask, mask)
        optimizer.zero_grad()  # setting gradient to zero
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())
        running_train_score.append(score)
        print(f'batch DiceBCELoss: {loss.item()}')
        print(f'batch iou: {score}')

    else:
        ### do valid after epoch ###
        running_val_loss = []
        running_val_score = []
        with torch.no_grad():
            for image, mask in valid_loader:
                image = image.cuda()
                mask = mask.cuda()
                pred_mask = model.forward(image)
                loss = criterion(pred_mask, mask)
                score = iou_batch(pred_mask, mask)
                running_val_loss.append(loss.item())
                running_val_score.append(score)

    ### then compute mean loss for current epoch and save ckp ###
    epoch_train_loss, epoch_train_score = np.mean(running_train_loss), np.mean(running_train_score)
    print('Train loss : {} iou : {}'.format(epoch_train_loss, epoch_train_score))
    train_loss.append(epoch_train_loss)
    train_iou.append(epoch_train_score)

    epoch_val_loss, epoch_val_score = np.mean(running_val_loss), np.mean(running_val_score)
    print('Validation loss : {} iou : {}'.format(epoch_val_loss, epoch_val_score))
    val_loss.append(epoch_val_loss)
    val_iou.append(epoch_val_score)

    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': i,
        'valid_loss_min': epoch_val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path + f'/model_{i}.pth', best_model_path)
    ## TODO: save the model if validation loss has decreased
    if epoch_val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, epoch_val_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path + f'/model_{i}.pth', best_model_path)
        valid_loss_min = epoch_val_loss

    i = i + 1

