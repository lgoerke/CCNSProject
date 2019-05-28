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
from scipy import stats
from vae_2stream import OurClassifier, OurAutoencoder

def main(args):

	s_list = args.savedir
	np.random.seed(4517)

	# Get data
	data = data_utils.assemble_data()

	data_idx = np.random.permutation(len(data))
	cutoff = int (len(data)*0.8)

	test = data[data_idx[cutoff:],:,:,:]
	test = np.reshape(test,(test.shape[0],3,32,32))

	d_list = []
	kl1_list = []
	kl2_list = []
	for i,savedir in enumerate(s_list):

		autoencoder = OurClassifier(OurAutoencoder())
		serializers.load_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

		mse_decoding_sample = np.zeros((test.shape[0],1))
		kl1_decoding_sample = np.zeros((test.shape[0],1))
		kl2_decoding_sample = np.zeros((test.shape[0],1))
		# Compute MSE for all test images
		for index in tqdm(range(test.shape[0]),desc='All test images'):

			img = test[index,0,:,:]
			img2 = test[index,1,:,:]
			target = test[index,2,:,:]

			m,s,m2,s2 = autoencoder.encode(np.reshape(img,(1,1,32,32)),np.reshape(img2,(1,1,32,32)))
			kl1_decoding_sample[index] = F.gaussian_kl_divergence(m, s).data/32
			kl2_decoding_sample[index] = F.gaussian_kl_divergence(m2, s2).data/32

			sample1 = F.gaussian(m, s)
			sample2 = F.gaussian(m2,s2)
			sample = F.concat((sample1,sample2))

			# Reconstruct using sample given m,s
			decoding_sample = np.reshape(autoencoder.decode(sample,for_plot=True).data,(32,32))

			mse_decoding_sample[index] = chainer.functions.mean_squared_error(target, decoding_sample).data

		d_list.append(mse_decoding_sample)
		kl1_list.append(kl1_decoding_sample)
		kl2_list.append(kl2_decoding_sample)

	# Use paired sample t-test to find out if performance is significantly different
	d_significance = []
	kl1_significance = []
	kl2_significance = []
	if (len(s_list)>1):
		for k in range(1,len(s_list)):
			# binwidth = 0.02
			# plt.figure()
			# plt.hist(kl2_list[k],bins=np.arange(np.min(kl2_list[k]), np.max(kl2_list[k]) + binwidth, binwidth),label='No Log')
			# plt.hist(np.log(kl2_list[k]),bins=np.arange(np.min(np.log(kl2_list[k])), np.max(np.log(kl2_list[k])) + binwidth, binwidth))
			# plt.legend()
			# plt.hist(np.log(d_list[k-1]),bins=np.arange(np.min(np.log(d_list[k-1])), np.max(np.log(d_list[k-1])) + binwidth, binwidth))
			# plt.title('Log of Reconstruction MSE (with sampling)')
			p = stats.ttest_rel(np.log(d_list[k-1]),np.log(d_list[k]))[1]
			if p < 0.05:
				d_significance.append(k-0.5)
			p = stats.ttest_rel(np.log(kl1_list[k-1]),np.log(kl1_list[k]))[1]
			if p < 0.05:
				kl1_significance.append(k-0.5)
			p = stats.ttest_rel(np.log(kl2_list[k-1]),np.log(kl2_list[k]))[1]
			if p < 0.05:
				kl2_significance.append(k-0.5)
				

	d_means = [np.mean(ary) for ary in d_list]
	d_stds = [np.std(ary) for ary in d_list]
	kl1_means = [np.mean(ary) for ary in kl1_list]
	kl1_std = [np.std(ary) for ary in kl1_list]
	kl2_means = [np.mean(ary) for ary in kl2_list]
	kl2_std = [np.std(ary) for ary in kl2_list]
	# Print for table in paper
	print('MSE')
	for index,m in enumerate(d_means):
		print s_list[index] + " " + "{:.06f}".format(m) + " & " + "{:.04f}".format(d_stds[index]) + " \\\\"

	print('KL1')
	for index,m in enumerate(kl1_means):
		print s_list[index] + " " + "{:.06f}".format(m) + " & " + "{:.04f}".format(kl1_std[index]) + " \\\\"
	print('KL2')
	for index,m in enumerate(kl2_means):
		print s_list[index] + " " + "{:.06f}".format(m) + " & " + "{:.04f}".format(kl2_std[index]) + " \\\\"
	# Plot MSE
	plt.figure()
	for j in range(len(d_means)):
		plt.errorbar(j, d_means[j], d_stds[j],label="Model {}".format(j+1),marker='^')

	if (len(s_list)>1):
		plt.scatter(d_significance,np.zeros((len(d_significance),1)),marker='*')
	plt.xlim([-0.5,len(d_means)-0.5])
	tenth = (np.max(d_means) - np.min(d_means))/10
	# plt.ylim([-tenth,np.max(d_means)+tenth])
	plt.title('Reconstruction MSE')
	plt.legend(loc=2)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])

	# Plot KL1
	plt.figure()
	for j in range(len(kl1_means)):
		plt.errorbar(j, kl1_means[j], kl1_std[j],label="Model {}".format(j+1),marker='^')

	if (len(s_list)>1):
		plt.scatter(kl1_significance,np.ones((len(kl1_significance),1))*0.2,marker='*')
	plt.xlim([-0.5,len(kl1_means)-0.5])
	tenth = (np.max(kl1_means) - np.min(kl1_means))/10
	# plt.ylim([-tenth,np.max(d_means)+tenth])
	plt.title('Encoding KL divergence Enc1')
	plt.legend(loc=1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])

	# Plot KL2
	plt.figure()
	for j in range(len(kl2_means)):
		plt.errorbar(j, kl2_means[j], kl2_std[j],label="Model {}".format(j+1),marker='^')

	if (len(s_list)>1):
		plt.scatter(kl2_significance,np.ones((len(kl2_significance),1))*0.2,marker='*')
	plt.xlim([-0.5,len(kl2_means)-0.5])
	tenth = (np.max(kl2_means) - np.min(kl2_means))/10
	# plt.ylim([-tenth,np.max(d_means)+tenth])
	plt.title('Encoding KL divergence Enc2')
	plt.legend(loc=1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])

	plt.show()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot reconstructions')
	parser.add_argument('savedir', nargs='+', action='store',
											help='Directory where model was stored')
	args = parser.parse_args()

	main(args)
