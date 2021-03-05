from keras import layers as KL
from keras import backend as K
import numpy as np
#import tensorflow as tf


def soft_erode(img):
    p1 = -KL.MaxPool3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p2 = -KL.MaxPool3D(pool_size=(3, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p3 = -KL.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    return tf.math.minimum(tf.math.minimum(p1, p2), p3)

def soft_dilate(img):
    return KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)
#     p2 = KL.MaxPool3D(pool_size=(1, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(img)
#     p3 = KL.MaxPool3D(pool_size=(1, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)
#     return tf.math.maximum(tf.math.maximum(p1, p2), p3)

def soft_open(img):
    img = soft_erode(img)
    img = soft_dilate(img)
    return img
    
def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = tf.nn.relu(img-img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel

def soft_clDice_loss(iters = 50):
    def loss(y_true, y_pred):
        smooth = 1.
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:,1:,:,:,:])+smooth)/(K.sum(skel_pred[:,1:,:,:,:])+smooth)    
        rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:,1:,:,:,:])+smooth)/(K.sum(skel_true[:,1:,:,:,:])+smooth)    
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        return cl_dice
    return loss
	
	
	
	
## combination with regular soft-Dice	
	
def soft_combined_dice_loss(iters = 15):
    def loss(y_true, y_pred):
        smooth = 1.
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:,1:,:,:,:])+smooth)/(K.sum(skel_pred[:,1:,:,:,:])+smooth)    
        rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:,1:,:,:,:])+smooth)/(K.sum(skel_true[:,1:,:,:,:])+smooth)    
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        cl_dice = soft_dice(skel_true, skel_pred)
        dice = soft_dice(y_true, y_pred)
        return 0.5*dice+0.5*cl_dice
    return loss

def soft_dice(y_true, y_pred):
    smooth = 1
    intersection = K.sum((y_true * y_pred)[:,1:,:,:,:])
    coeff = (2. *  intersection + smooth) / (K.sum(y_true[:,1:,:,:,:]) + K.sum(y_pred[:,1:,:,:,:]) + smooth)

    return (1. - coeff)

	
