import os
import numpy as np
from skimage import io, transform
import random
import shutil
###### processing the rawdata ######

image_height = 256
image_width = 256
data_dir = './rawdata/data/'

sample_img = np.load('./dataset/demo_brain/nuc_input.npy')
sample_label = np.load('./dataset/demo_brain/nuc_label.npy')
print(sample_img.shape)
print(sample_label.shape)
sava_dir = './dataset/nucleus'


def remake_raw_dataset(save_dir):
    print('start remaking kaggle dataset...')
    for imageid in os.listdir(data_dir):
        print(imageid)
        image_dir = data_dir + imageid +'/images'
        mask_dir = data_dir + imageid + '/masks'
        ### combine binary mask ###
        mask = get_mask(mask_dir, image_height, image_width)
        ### load and resize image then without processing ###
        image = get_image(image_dir, image_height, image_width)
        print(image.shape)
        np.save(save_dir+f'/{imageid}_input.npy', image)
        np.save(save_dir+f'/{imageid}_label.npy', mask)
    pass


def get_mask(mask_folder, IMG_HEIGHT, IMG_WIDTH):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
    for mask_ in os.listdir(mask_folder):
        mask_ = io.imread(os.path.join(mask_folder, mask_))
        mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
        #mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)
    return mask

def get_image(image_folder, IMG_HEIGHT, IMG_WIDTH):
    img_dir= os.listdir(image_folder)[0]
    img = io.imread(os.path.join(image_folder, img_dir))[:, :, :3].astype('float32')
    img = transform.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    return img

def remove_existing_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def split_valid(log_dir, dataset_dir):
    seed = 205322607
    random.seed(seed)
    files = os.listdir(dataset_dir)
    IDs = list()
    for file in files:
        #print(file)
        split = file.split('_')
        if split[1][0:5] == 'label':
            print(split[0])
            IDs.append(split[0])
    random.shuffle(IDs)
    # [590, 80]
    training_set = IDs[0:590]
    valid_set = IDs[590:]
    print(training_set)
    print(valid_set)
    train_dir = os.path.join(log_dir, 'train')
    valid_dir = os.path.join(log_dir, 'valid')
    ### remove and copy ###
    remove_existing_files(train_dir)
    remove_existing_files(valid_dir)
    for id in training_set:
        input = id + '_input.npy'
        label = id + '_label.npy'
        print(id)
        shutil.copy(os.path.join(dataset_dir,input), train_dir)
        shutil.copy(os.path.join(dataset_dir,label), train_dir)

    for id in valid_set:
        input = id + '_input.npy'
        label = id + '_label.npy'
        print(id)
        shutil.copy(os.path.join(dataset_dir,input), valid_dir)
        shutil.copy(os.path.join(dataset_dir,label), valid_dir)

    print('Done')




if __name__ == "__main__":
    #remake_raw_dataset(sava_dir)
    #split_valid('./network/','./dataset/nucleus/')
    #image_dir = './dataset/demo_brain/test.png'
    #image = get_image(image_dir, image_height, image_width)

    pass