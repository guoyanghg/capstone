import os
import numpy as np
import tensorflow as tf
import DataGen
import glob
import matplotlib.pyplot as plt
import architectures
from sklearn.metrics import f1_score
from utils import load_image,dice_hard,dice_soft,my_func,resolve_status,contoured_image

TRAINING_STATUS = 3 ### inference
restore,is_training = resolve_status(TRAINING_STATUS)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
iter_limit = 300

#model_dir = './network/model.ckpt'
### gy use my dataset ###
INPUT_CHANNEL = 3
IMAGE_SIZE = 256
BATCH_SIZE = 1
### gy use my dataset ###
config = tf.ConfigProto(allow_soft_placement=True)
input_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNEL]

x = tf.placeholder(shape=input_shape, dtype=tf.float32, name="x")
#y = tf.placeholder(dtype=tf.float32, name="y")
#phase = tf.placeholder(dtype = tf.bool, name='phase')
global_step = tf.Variable(0, name='global_step', trainable=False)
out_seg = architectures.ddunet_single(x,is_training)
y_out_dl = tf.round(out_seg)

#Dice = dice_soft(out_seg, y)
#seg_loss = 1 - Dice
#l2_loss = tf.losses.get_regularization_loss()
#seg_loss += l2_loss
#total_loss = seg_loss
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#Dice_hard = dice_hard(out_seg, y)
saver = tf.train.Saver(tf.global_variables())
save_dir = './network/list/unet'
with tf.Session(config=config) as sess:
    print("Converting model to SavedModel")
    # saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, tf.train.latest_checkpoint('./network/model.ckpt'))

    image_add = './dataset/demo_brain/nuc_input.npy'
    image = load_image(image_add, 1, False)

    seg_out = sess.run(y_out_dl, {x: image})

    seg_out = seg_out[0, :, :, 0]


    seg_out_coutour = contoured_image(seg_out, image[0, :, :, 0])
    plt.imshow(seg_out_coutour)
    plt.show()


    # Export to SavedModel
    '''
    tf.compat.v1.saved_model.simple_save(
        sess,
        save_dir,
        inputs={x.name: x},
        outputs={'unet_output': y_out_dl}
    )
    '''

