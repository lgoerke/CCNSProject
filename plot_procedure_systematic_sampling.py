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
	Play around with latent vector
	savedir: model to be used
	'''

	np.random.seed(4517)

	savedir = args.savedir

	autoencoder = OurClassifier(OurAutoencoder())

	serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	## Handpicked indices as determined through plot_procedure_ms e200_b16_lam0.01_lr0.001
	squares_one = [8,10,11,12,15,17,18,19,23,24,25,28,30,31]
	squares_zerosix = [1,4,5,22]
	square_zerofifteen = [0,2,3,6,7,9,13,14,16,20,21,26,27,29]

	ellipses_one = [0,2,3,4,7,8,10,11,16,23,24,26,28,31]
	ellipses_zerosix = [1,6,19,22,25,29]
	ellipses_zerofifteen = [5,9,11,13,14,15,17,18,20,21,27,30]

	# sample = np.random.uniform(size=(1,64)).astype(np.float32)
	sample = (np.ones((1,64)).astype(np.float32))*2

	f,ax = plt.subplots(2,3)

	# sys_sampling01
	# zerosix = [-1,-1,-1,-1,-0.1,-0.1,-0.1,-0.1]
	# zerofifteen = [-0.5,-1.8,-5,-10,-0.5,-1.8,-5,-10]
	
	# sys_sampling02
	# zerosix = [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]
	# zerofifteen = np.arange(0,-2,-0.25)


	# sys_sampling03
	# zerosix = np.arange(0,1,0.125)
	zerosix = np.ones((8,1))
	zerofifteen = np.linspace(0.1,1,6)


	i = j = 0
	for index in range(6):
		sample[0,squares_zerosix] = zerosix[index]
		sample[0,square_zerofifteen] = zerofifteen[index]
		sample[0,[val + 32 for val in ellipses_zerosix]] = zerosix[index]
		sample[0,[val + 32 for val in ellipses_zerofifteen]] = zerofifteen[index]
		decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
		give_stats(decoding_sample,'Decoding_sample')
		im = ax[i,j].imshow(decoding_sample,interpolation="nearest")
		data_utils.colorbar(im)
		j += 1
		if j > 2:
			i+=1
			j = 0

	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', action='store',
											help='Directory where model was stored')
	args = parser.parse_args()

	main(args)