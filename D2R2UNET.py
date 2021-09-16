# Md Yousuf Harun
# Tensorflow implementation
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook #, tnrange
from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
import time
t_start = time.time()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Set some parameters
im_width = 256
im_height = 256
border = 5
path_train = "C:/Users/yousu/Downloads/BlastsOnline"

# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "/imagesResized_256_1368_updated"))[2]
    ids2= next(os.walk(path + "/labelResized_256_1368_updated"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids2), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/imagesResized_256_1368_updated/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)
        X[n] = x_img / 255.0

    for n, id_ in tqdm_notebook(enumerate(ids2), total=len(ids2)):
        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/labelResized_256_1368_updated/' + id_, grayscale=True))
            mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)
            y[n] = mask / 255.0

        # Save images
        #X[n] = x_img / 255
        #if train:
            #y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X

X, y = get_data(path_train, train=True)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2018)
print(X_valid.shape)

def BatchActivate(x):
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = Activation('elu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def recurrent_block(x, filters, size, strides=(1,1), padding='same', activation=True, t=2):
    for i in range(t):
        if i==0:
            x1 = Conv2D(filters, size, strides=strides, padding=padding)(x)
            if activation == True:
                 x1 = BatchActivate(x1)

        x1 = Add()([x, x1])
        x1 = Conv2D(filters, size, strides=strides, padding=padding)(x1)
        if activation == True:
             x1 = BatchActivate(x1)
        #x1 = Add()([x, x1])
    return x1

def residual_block(x, num_filters=16, batch_activate = False):
    x = BatchActivate(x)
    x1 = recurrent_block(x, num_filters, (3,3))
    x1 = recurrent_block(x1, num_filters, (3,3))

    x = Conv2D(num_filters, (1,1), strides=(1,1), padding='same')(x)
    x = Add()([x1, x])
    if batch_activate:
        x = BatchActivate(x)
    return x

def bottleneck(x, filters_bottleneck, mode='cascade', depth=4,
               kernel_size=(3, 3), activation='elu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)

def res_path(x, num_filters, length):
    shortcut = x
    shortcut = Conv2D(num_filters, (1,1), strides=(1,1), padding='same')(shortcut)
    out = Conv2D(num_filters, (3,3), strides=(1,1), padding='same')(x)
    out = Add()([shortcut, out])
    out = BatchActivate(out)
    for i in range(length-1):
        shortcut = out
        shortcut = Conv2D(num_filters, (1,1), strides=(1,1), padding='same')(shortcut)
        out = Conv2D(num_filters, (3,3), strides=(1,1), padding='same')(out)
        out = Add()([shortcut, out])
        out = BatchActivate(out)
    return out

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    #conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    #conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    #conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    #conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    #convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    #convm = residual_block(convm,start_neurons * 16)
    #convm = residual_block(convm,start_neurons * 16, True)

    #Dilated Middle
    convm = bottleneck(pool4, filters_bottleneck=start_neurons * 16, mode='cascade', depth=5,
               kernel_size=(3, 3), activation='elu')
    #convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    #conv4 = recurrent_block(conv4, start_neurons * 8, (3,3), activation=True, t=0)
    conv4 = res_path(conv4, start_neurons * 8, 1)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    #uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    #conv3 = recurrent_block(conv3, start_neurons * 4, (3,3), activation=True, t=1)
    conv3 = res_path(conv3, start_neurons * 4, 2)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    #uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    #conv2 = recurrent_block(conv2, start_neurons * 2, (3,3), activation=True, t=2)
    conv2 = res_path(conv2, start_neurons * 2, 3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = BatchNormalization()(uconv2)

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    #uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    #conv1 = recurrent_block(conv1, start_neurons * 1, (3,3), activation=True, t=3)
    conv1 = res_path(conv1, start_neurons * 1, 4)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = BatchNormalization()(uconv1)

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    #uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)

    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)

    return output_layer

from segmentation_models import Unet
#from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.losses import bce_dice_loss
from segmentation_models.metrics import iou_score


# model
img_size_target = 256
input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16,0.05)
lr=.0001
# del model
model = Model(input_layer, output_layer)
model.compile(optimizer=Nadam(lr), loss=bce_jaccard_loss, metrics=[iou_score])
model.summary()

data_gen_args = dict(horizontal_flip=True,
                    vertical_flip=True,
                    rotation_range=270,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 2018
bs = 4 #batch size for gpu

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)

# Just zip the two generators to get a generator that provides augmented images and masks at the same time
train_generator = zip(image_generator, mask_generator)

NAME = "D2R2UNET_30-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.05, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('model-D2R2Unet_30.h5', verbose=1, save_best_only=True, save_weights_only=True),
    tensorboard
]

results = model.fit_generator(train_generator, steps_per_epoch=(len(X_train) // bs), epochs=100, callbacks=callbacks,
                              validation_data=(X_valid, y_valid))

history = model.fit(X_train, y_train,
                    validation_data=[X_valid, y_valid],
                    epochs=50,
                    batch_size=4,
                    callbacks=callbacks,
                    verbose=1)

# Load best model
model.load_weights('model-D2R2Unet_23.h5')
# Set some parameters
im_width = 256
im_height = 256
border = 5
path_train = "D:\EmbryoSoft_v1"

# Get and resize train images and masks
def get_data(path):
    ids = next(os.walk(path + "/D2020.01.24_DFaLi_13197tB_100.0_wellsAA-5_video"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/D2020.01.24_DFaLi_13197tB_100.0_wellsAA-5_video/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)
        X[n] = x_img / 255.0

    print('Done!')
    return X

X = get_data(path_train)

# Predict on train, val and test
preds_val = model.predict(X, verbose=1)

# Threshold predictions
preds_val_t = (preds_val > 0.5).astype(np.float32)

for x in range (len(X)):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(X[x, ...,0],cmap='gray')
    ax.contour(preds_val_t[x].squeeze(), colors='r', levels=[8])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('DFaLi_13197tB_100.0_wellsAA-5_video' + '_' + str(x+1) + '.png')

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)
# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.float32)
preds_val_t = (preds_val > 0.5).astype(np.float32)

from keras import backend as K

def accuracy(y_true, y_pred):
     #test accuracy
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
     numerator = (tp + tn)
     denominator = (tp+tn+fp+fn)
     return numerator / (denominator + K.epsilon())

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
#history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
y_valid = tf.convert_to_tensor(y_valid, np.float32)
preds_val_t = tf.convert_to_tensor(preds_val_t, np.float32)
#preds_val = tf.convert_to_tensor(preds_val, np.float32)

with tf.Session() as sess:
    print(sess.run(recall_m(y_valid,preds_val_t)))
    print(sess.run(precision_m(y_valid,preds_val_t)))
    print(sess.run(f1_m(y_valid,preds_val_t)))

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    epsilon = 1e-15
    #intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    #union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f)+tf.reduce_sum(y_pred_f)

    return (tf.reduce_mean(intersection + epsilon)/ (union - intersection + epsilon))

from keras import backend as K

def specificity(y_true, y_pred):
     #test accuracy
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
     numerator = tn
     denominator = (tn+fp)
     return numerator / (denominator + K.epsilon())

with tf.Session() as sess:
    print(sess.run(dice_coef(y_valid,preds_val_t)))
    print(sess.run(jaccard(y_valid,preds_val_t)))
    print(sess.run(accuracy(y_valid, preds_val_t)))
    print(sess.run(specificity(y_valid, preds_val_t)))

# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix,...,0], cmap='gray', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Embryo')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Label');
#print(ix)

import cv2
from PIL import Image
import imageio

#for x in range (len(X_valid)):
x = 14
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(X_valid[x, ...,0],cmap='gray')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('Org' + '_' + str(x+1) + '.png')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(preds_val_t[x].squeeze(), vmin=-1, vmax=1)
ax.contour(y_valid[x].squeeze(), colors='r', levels=[7])
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('Prd' + '_' + str(x+1) + '.png')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(y_valid[x].squeeze(), vmin=-2, vmax=2)
#ax.contour(y_valid[x].squeeze(), colors='r', levels=[0.8])
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('GT' + '_' + str(x+1) + '.png')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(preds_val_t[x].squeeze(), vmin=-1, vmax=1)
#ax.contour(y_valid[x].squeeze(), colors='r', levels=[1])
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('Prdwithout' + '_' + str(x+1) + '.png')

image1 = Image.open('Org' + '_' + str(x+1) + '.png')
image2 = Image.open('Prd' + '_' + str(x+1) + '.png')
#mage1 = image1.resize((500,500), Image.NEAREST)
#mage2 = image2.resize((500,500), Image.NEAREST)
alphaBlended1 = Image.blend(image1, image2, alpha=0.7)
imageio.imwrite('Ov' + '_' + str(x+1) + '.png', alphaBlended1)

fig, ax = plt.subplots(1, 4, figsize=(20,10))
im1=Image.open('Org' + '_' + str(x+1) + '.png')
#mm1=im1.resize((500,500),Image.NEAREST)
ax[0].imshow(im1)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Input Image')

im2 = Image.open('GT' + '_' + str(x+1) + '.png')
#mm2=im2.resize((500,500),Image.NEAREST)
ax[1].imshow(im2)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Ground Truth')

im3 = Image.open('Prdwithout' + '_' + str(x+1) + '.png')
#mm3= im3.resize((500,500),Image.NEAREST)
ax[2].imshow(im3)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('Prediction')

im4 = Image.open('Ov' + '_' + str(x+1) + '.png')
#mm4= im4.resize((500,500),Image.NEAREST)

ax[3].imshow(im4)
ax[3].grid(False)
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[3].set_title('Comparison')
fig.savefig('Final_' + str(x+1) + '.png')

import cv2
from PIL import Image
import imageio

for x in range (len(X_valid)):
#x = 14
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(X_valid[x, ...,0],cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('Org' + '_' + str(x+1) + '.png')

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(preds_val_t[x].squeeze(), cmap='gray')
    #ax.imshow(preds_val_t[x].squeeze(), vmin=-1, vmax=1)
    #ax.contour(y_valid[x].squeeze(), colors='r', levels=[7])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('Prd' + '_' + str(x+1) + '.png')

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(y_valid[x].squeeze(), cmap='gray')
    #ax.contour(y_valid[x].squeeze(), colors='r', levels=[0.8])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('GT' + '_' + str(x+1) + '.png')

def jaccard(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    epsilon = 1e-15
    #intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    #union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f)+tf.reduce_sum(y_pred_f)

    return (tf.reduce_mean(intersection + epsilon)/ (union - intersection + epsilon))

JI = []
for x in range (len(X_valid)):
#for x in range 2:
    with tf.Session() as sess:
         iou = sess.run(jaccard(y_valid[x],preds_val_t[x]))
    #iou = jaccard(y_valid[x], preds_val_t[x])
    iou = round(iou*100,2)
    JI.append(iou)

import numpy
a = numpy.asarray(JI)
numpy.savetxt("JI.csv", a, delimiter=",")
#x=38
#image1 = Image.open('Seg' + '_' + str(x+1) + '.png')
for x in range (len(X_valid)):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(X_valid[x, ...,0],cmap='gray')
    ax.contour(preds_val_t[x].squeeze(), colors='b', levels=[7])
    ax.contour(y_valid[x].squeeze(), colors='r', levels=[7])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('Ovp' + '_' + str(x+1) + '.png')
x=1
image1 = Image.open('Cont' + '_' + str(x+1) + '.png')
image2 = Image.open('Seg' + '_' + str(x+1) + '.png')
alphaBlended1 = Image.blend(image1, image2, alpha=0.6)
imageio.imwrite('Ov' + '_' + str(x+1) + '.png', alphaBlended1)
