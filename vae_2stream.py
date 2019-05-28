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
import data_utils
import sys
import os

class OurAutoencoder(Chain):
	'''
	Variational Autoencoder Class
	'''

	def __init__(self):
		'''
		Initialization
		Change architecture here
		'''
		super(OurAutoencoder, self).__init__()
		with self.init_scope():
			self.enc1_l1 = L.Linear(None, 512)
			self.enc1_mu = L.Linear(None, 32)
			self.enc1_ln_std = L.Linear(None, 32)

			self.enc2_l1 = L.Linear(None, 512)
			self.enc2_mu = L.Linear(None, 32)
			self.enc2_ln_std = L.Linear(None, 32)

			self.dec_l1 = L.Linear(None, 512)
			self.dec_output = L.Linear(None, 1024)

	def encode(self, x1, x2):
		'''
		Encode two input images
		Input: input images
		Output: mean and std vec for each encoder (=2)
		'''
		# encode first image
		h1 = x1
		h1 = F.relu(self.enc1_l1(h1))
		# encode second image
		h2 = x2
		h2 = F.relu(self.enc1_l1(h2))
		return self.enc1_mu(h1), self.enc1_ln_std(h1), self.enc2_mu(h2), self.enc2_ln_std(h2)

	def decode(self, sample, for_plot=False):
		'''
		Decode a vector sampled from unit Gaussian
		Input: the vector to be decoded, flag to apply sigmoid on output
		Output: decoded vector image (32,32)
		'''
		h = sample
		h = F.relu(self.dec_l1(h))
		h = self.dec_output(h)
		if for_plot:
			h = F.sigmoid(h)
		h = F.reshape(h, (h.shape[0], 1, 32, 32))
		return h

	def compute_loss(self, x1, x2, t):
		'''
		Compute both encoders losses and decoder loss
		Use KL divergence for encoder and Bernoulli loss for decoder
		Input: two images and target (composition of both images)
		Output: decoder loss, encoder1 loss, encoder2 loss
		'''
		mu1, ln_std1, mu2, ln_std2 = self.encode(x1, x2)
		kl1 = F.gaussian_kl_divergence(mu1, ln_std1)
		kl2 = F.gaussian_kl_divergence(mu2, ln_std2)
		sample1 = F.gaussian(mu1, ln_std1)
		sample2 = F.gaussian(mu2, ln_std2)

		sample = F.concat((sample1, sample2))
		output = self.decode(sample)
		nll = F.bernoulli_nll(F.reshape(t, (t.shape[0], 1, 32, 32)),output)
		return nll / (t.shape[0] * 32 * 32), kl1 / (x1.shape[0] * 32), kl2 / (x2.shape[0] * 32)

	def reconstruct(self, x1, x2):
		'''
		Reconstructions of input images without sampling step (taking means produced by encoder directly)
		Output: reconstructed image
		'''
		mu1, ln_std1, mu2, ln_std2 = self.encode(x1, x2)
		output = F.sigmoid(self.decode(F.concat((mu1, mu2))))
		return output



class OurClassifier(Chain):
	'''
	Classifier Class
	'''

	def __init__(self, predictor):
		super(OurClassifier, self).__init__()
		with self.init_scope():
			self.predictor = predictor

	def decode(self,sample,for_plot=False):
		out = self.predictor.decode(sample,for_plot)
		return out

	def encode(self,x1,x2):
		m,s,m2,s2 = self.predictor.encode(x1,x2)
		return m,s,m2,s2

	def compute_loss(self,x1,x2,t):
		l = self.predictor.compute_loss(x1,x2,t)
		return l

	def reconstruct(self,x1,x2):
		out = self.predictor.reconstruct(x1,x2)
		return out