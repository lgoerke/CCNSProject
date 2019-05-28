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
	Plot decoded image using random samples from unit Gaussian.
	savedir: model to be used
	no_samples: optional. How many samples should be displayed (one per figure)
	'''

	np.random.seed(4517)

	savedir = args.savedir
	no_samples = args.no_samples
	square = args.square

	data = data_utils.assemble_data()

	autoencoder = OurClassifier(OurAutoencoder())
	serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	data_idx = np.random.permutation(len(data))
	cutoff = int (len(data)*0.8)

	test = data[data_idx[cutoff:],:,:,:]
	test = np.reshape(test,(test.shape[0],3,32,32))

	sample_img_enc1 = np.zeros((32,no_samples))
	sample_img_enc2 = np.zeros((32,no_samples))

	# Sampling and plotting procedure
	for i in range(no_samples):
		f, ax = plt.subplots(1,3)

		if square:
			img = test[i,0,:,:]
		else:
			img = test[i,1,:,:]
		ax[0].imshow(np.reshape(img,(32,32)),interpolation="nearest")

		# Encode square
		if square:
			m,s,_,_ = autoencoder.encode(np.reshape(img,(1,1,32,32)),np.reshape(img,(1,1,32,32)))
			sample1 = F.gaussian(m, s)
		else:
			_,_,m,s = autoencoder.encode(np.reshape(img,(1,1,32,32)),np.reshape(img,(1,1,32,32)))
			sample2 = F.gaussian(m, s)

		# Reconstruct using random sample instead of ellipse
		size_sample = (1,32)
		if square:
			sample2 = F.gaussian(np.zeros(size_sample).astype(np.float32), np.zeros(size_sample).astype(np.float32))
		else:
			sample1 = F.gaussian(np.zeros(size_sample).astype(np.float32), np.zeros(size_sample).astype(np.float32))

		sample = F.concat((sample1,sample2))

		sample_img_enc1[:,i] = sample.data[0][:32]
		sample_img_enc2[:,i] = sample.data[0][32:]

		decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=False).data,(32,32))
		give_stats(decoding_sample,'Decoding_sample')
		im = ax[1].imshow(decoding_sample, cmap = data_utils.shiftedColorMap(matplotlib.cm.jet, midpoint=data_utils.calcMidpointForCM(decoding_sample), name='shifted'),interpolation="nearest")
		data_utils.colorbar(im)

		decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))
		give_stats(decoding_sample,'Decoding_sample')
		im = ax[2].imshow(decoding_sample,interpolation="nearest")
		data_utils.colorbar(im)


	# Get mean of sampled vector for images with separate shapes and for those with one big shape
	# Indices handpicked for default idx_list and e200_b16_lam0.01_lr0.001
	f, ax = plt.subplots(1,2)
	m1_enc1 = np.reshape(np.mean(sample_img_enc1[:,[4,6,9]],axis=1),(32,1))
	m2_enc1 = np.reshape(np.mean(sample_img_enc1[:,[0,1,2,3,5,7,8]],axis=1),(32,1))

	m1_enc2 = np.reshape(np.mean(sample_img_enc2[:,[4,6,9]],axis=1),(32,1))
	m2_enc2 = np.reshape(np.mean(sample_img_enc2[:,[0,1,2,3,5,7,8]],axis=1),(32,1))

	im = ax[0].imshow(np.concatenate((m1_enc1,m2_enc1),axis=1), cmap = data_utils.shiftedColorMap(matplotlib.cm.jet, midpoint=data_utils.calcMidpointForCM(sample_img_enc1), name='shifted'), interpolation="nearest")
	data_utils.colorbar(im,size="20%")
	im = ax[1].imshow(np.concatenate((m1_enc2,m2_enc2),axis=1),cmap = data_utils.shiftedColorMap(matplotlib.cm.jet, midpoint=data_utils.calcMidpointForCM(sample_img_enc2), name='shifted'), interpolation="nearest")
	data_utils.colorbar(im,size="20%")
	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', action='store',
											help='Directory where model was stored')
	parser.add_argument('--no_samples', action='store', help='How many samples should be taken',type=int,default=1)
	parser.add_argument('--square', help='Boolean for taking square or ellipse as img to be encoded',action='store_true')
	args = parser.parse_args()

	main(args)