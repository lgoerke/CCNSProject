import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import h5py
from tqdm import tqdm
import scipy.misc
from tqdm import tnrange, tqdm_notebook

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import random
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
import data_utils
from data_utils import give_stats
import sys
import os
import argparse
from vae_2stream import OurClassifier, OurAutoencoder

def main(args):
	'''
	Plot the mean and std vec when encoding the images from idx_list.
	Can be used to find regularities in mean/std for different shapes, sizes, ...

	savedir: which network to use
	idx_list: list of indices of test data images (because script is seeded, use same indices for same imgs)
	'''

	savedir = args.savedir
	idx_list = args.idx
	if not idx_list:
		# idx_list = [0,1,2,3,4,5,6,7,8,9]
		idx_list = [0,1,2,3]

	## Data loading
	np.random.seed(4517)
	data = data_utils.assemble_data()

	autoencoder = OurClassifier(OurAutoencoder())
	serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	data_idx = np.random.permutation(len(data))
	cutoff = int (len(data)*0.8)

	test = data[data_idx[cutoff:],:,:,:]
	test = np.reshape(test,(test.shape[0],3,32,32))


	if len(idx_list) == 4:
		f,ax = plt.subplots(2,4)
	no_imgs = len(idx_list)
	mean_img = np.zeros((32,no_imgs))
	mean_img2 = np.zeros((32,no_imgs))
	std_img = np.zeros((32,no_imgs))
	std_img2 = np.zeros((32,no_imgs))
	cnt = 0
	# Show input
	for i,index in enumerate(idx_list):
		img = test[index,0,:,:]
		img2 = test[index,1,:,:]
		
		if len(idx_list) == 4:
			ax[0,i].imshow(np.reshape(img,(32,32)),interpolation="nearest")
			ax[0,i].set_title('Image 1')
			ax[1,i].imshow(np.reshape(img2,(32,32)),interpolation="nearest")
			ax[1,i].set_title('Image 2')

		m,s,m2,s2 = autoencoder.encode(np.reshape(img,(1,1,32,32)),np.reshape(img2,(1,1,32,32)))
		mean_img[:,cnt] = m.data[0].T
		mean_img2[:,cnt] = m2.data[0].T
		std_img[:,cnt] = np.exp(s.data[0].T)
		std_img2[:,cnt] = np.exp(s2.data[0].T)
		cnt += 1

		## Uncomment for printing of values
		# print('-------------------')
		# print('-- Image 1       --')
		# print('-- Mean          --')
		# for val in m.data[0]:
		# 	print('{:0.4f}'.format(val))
		# print('-- Std           --')
		# for val in s.data[0]:
		# 	print('{:0.4f}'.format(np.exp(val)))
		# print('-- Image 2       --')
		# print('-- Mean          --')
		# for val in m2.data[0]:
		# 	print('{:0.4f}'.format(val))
		# print('-- Std           --')
		# for val in s2.data[0]:
		# 	print('{:0.4f}'.format(np.exp(val)))


	f, ax = plt.subplots(1,2)
	im = ax[0].imshow(np.concatenate((mean_img, std_img), axis=1), cmap = data_utils.shiftedColorMap(matplotlib.cm.jet, midpoint=data_utils.calcMidpointForCM(np.concatenate((mean_img, std_img), axis=1)), name='shifted'), interpolation="nearest")
	ax[0].set_title('Squares')
	data_utils.colorbar(im)
	im = ax[1].imshow(np.concatenate((mean_img2, std_img2), axis=1), cmap = data_utils.shiftedColorMap(matplotlib.cm.jet, midpoint=data_utils.calcMidpointForCM(np.concatenate((mean_img2, std_img2), axis=1)), name='shifted'), interpolation="nearest")
	data_utils.colorbar(im)
	ax[1].set_title('Ellipses')
	plt.show()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', action='store',
											help='Directory where model was stored')
	parser.add_argument('--idx', nargs='*', help='Image indices',type=int)
	args = parser.parse_args()

	main(args)
