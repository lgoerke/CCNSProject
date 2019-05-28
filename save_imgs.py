from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook
import sys
import os
from chainer import serializers
import chainer.functions as F
import scipy.misc
# from resizeimage import resizeimage
from PIL import Image

'''
Saves images to disk for later reading before training loop
'''


# Load dataset
dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

imgs = dataset_zip['imgs']

# squares = pkl.load(open('squares.pkl','r'))
ellipses = pkl.load(open('ellipses10000.pkl','r'))

# for i in tqdm(squares,desc='Squares Loop'):
# 	save = np.zeros((imgs[i,:,:].shape[0],imgs[i,:,:].shape[1],3))
# 	save[:,:,0] = imgs[i,:,:]
# 	save[:,:,1] = imgs[i,:,:]
# 	save[:,:,2] = imgs[i,:,:]
# 	scipy.misc.toimage(save).save('saved_imgs/square_{:}.png'.format(i))

for i in tqdm(ellipses,desc='Ellipses Loop'):
	save = np.zeros((imgs[i,:,:].shape[0],imgs[i,:,:].shape[1],3))
	save[:,:,0] = imgs[i,:,:]
	save[:,:,1] = imgs[i,:,:]
	save[:,:,2] = imgs[i,:,:]
	scipy.misc.toimage(save).save('saved_ellipses/ellipse_{:}.png'.format(i))

