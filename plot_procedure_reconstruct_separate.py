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
import data_utils
from data_utils import give_stats
import sys
import os
import argparse
from vae_2stream import OurClassifier, OurAutoencoder

class Plotting():
	'''
	Plot reconstructions class. Holds data and model.
	'''
	def __init__(self, autoencoder, data):
		self.autoencoder = autoencoder
		self.data = data

	def plot_all_imgs(self,index=0):
		f,ax = plt.subplots(2,2)

		img = self.data[index,0,:,:]
		img2 = self.data[index,1,:,:]

		ax[0,0].imshow(np.reshape(img,(32,32)))
		ax[0,0].set_title('Original')
		ax[1,0].imshow(np.reshape(img2,(32,32)))
		ax[1,0].set_title('Original2')

		m,s,m2,s2 = self.autoencoder.encode(np.reshape(img,(1,1,32,32)),np.reshape(img2,(1,1,32,32)))
		sample1 = F.gaussian(m, s)
		sample2 = F.gaussian(m2,s2)
		sample = F.concat((sample1,sample2))
		mean = F.concat((m,m2))

		# Reconstruct using sample given m,s
		decoding = np.reshape(self.autoencoder.decode(sample,for_plot=False).data,(32,32))
		give_stats(decoding,'Decoding')
		im = ax[0,2].imshow(decoding)
		ax[0,2].set_title('Reconstruct with sampling')
		data_utils.colorbar(im)

		decoding = np.reshape(self.autoencoder.decode(sample,for_plot=True).data,(32,32))
		give_stats(decoding,'Decoding Sig')
		im = ax[1,2].imshow(decoding)
		ax[1,2].set_title('Sig(Reconstruct with sampling)')
		data_utils.colorbar(im)

def main(args):
	'''
	Plots reconstructions for several images. Plots one figure per image.
	Plot shows separate input images.
	savedir: where the network is saved
	idx_list: list of indices for images to be plotted. Default: [0,1,2]
	'''
	savedir = args.savedir
	idx_list = args.idx
	if not idx_list:
		idx_list = [0,1,2]

	## Data loading
	np.random.seed(4517)
	data = data_utils.assemble_data()

	autoencoder = OurClassifier(OurAutoencoder())
	serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	data_idx = np.random.permutation(len(data))
	cutoff = int (len(data)*0.8)

	test = data[data_idx[cutoff:],:,:,:]
	test = np.reshape(test,(test.shape[0],3,32,32))

	p = Plotting(autoencoder,test)

	for i in idx_list:
		p.plot_all_imgs(i)
		print('  ')
		print('=================')
		print('  ')

	plt.show()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', action='store',
											help='Directory where model was stored')
	parser.add_argument('--idx', nargs='*', help='Image indices',type=int)
	args = parser.parse_args()

	main(args)
