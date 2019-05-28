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
import argparse
import data_utils
from data_utils import give_stats
from vae_2stream import OurClassifier, OurAutoencoder
import matplotlib
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

def main(args):
	'''
	After looking at the average of the Gaussian sampled vec of some images, 
	some salient numbers in the average vec were determined. Using those, 
	we explore how important the value at the salient positions are
	(original avg value, original sign of value but all set to 2,
	all set to 2). The images were divided into images with two separate shapes
	and images with one big shape. The values are for the model
	e200_b16_lam0.01_lr0.001
	'''

	np.random.seed(4517)

	savedir = args.savedir

	autoencoder = OurClassifier(OurAutoencoder())
	serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	##### Separate shapes #####

	f,ax = plt.subplots(1,3)
	
	sample = np.zeros((1,64)).astype(np.float32)

	# Handpicked indices as determined through plot_procedure_sampling
	minusonetwo = [2,14,19,26,31,33,44,47,48,59]
	one = [0,11,20,34,45,50,57,61]
	oneeight = [21]
	sample[0,minusonetwo] = -1.2
	sample[0,one] = 1
	sample[0,oneeight] = 1.8

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[0].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)

	sample = np.zeros((1,64)).astype(np.float32)

	sample[0,minusonetwo] = -2
	sample[0,one] = 2
	sample[0,oneeight] = 2

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[1].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)

	sample = np.zeros((1,64)).astype(np.float32)

	sample[0,minusonetwo] = 2
	sample[0,one] = 2
	sample[0,oneeight] = 2

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[2].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)


	##### One big shape #####

	f,ax = plt.subplots(1,3)

	sample = np.zeros((1,64)).astype(np.float32)

	# Handpicked indices as determined through plot_procedure_sampling
	minuestwofive = [21,53,54]
	minusonetwo = [4,9,10,14,19,23,29,31,32,39,47,58]
	one = [0,2,20,25,45,46,49,62,63]
	oneeight = [12,16,40,55,56,57]
	sample[0,minuestwofive] = -2.5
	sample[0,minusonetwo] = -1.2
	sample[0,one] = 1
	sample[0,oneeight] = 1.8

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[0].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)

	sample = np.zeros((1,64)).astype(np.float32)

	sample[0,minuestwofive] = -2
	sample[0,minusonetwo] = -2
	sample[0,one] = 2
	sample[0,oneeight] = 2

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[1].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)

	sample = np.zeros((1,64)).astype(np.float32)

	sample[0,minuestwofive] = 2
	sample[0,minusonetwo] = 2
	sample[0,one] = 2
	sample[0,oneeight] = 2

	decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
	give_stats(decoding_sample,'Decoding_sample')
	im = ax[2].imshow(decoding_sample,interpolation="nearest")
	data_utils.colorbar(im)

	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', action='store',
											help='Directory where model was stored')
	args = parser.parse_args()

	main(args)