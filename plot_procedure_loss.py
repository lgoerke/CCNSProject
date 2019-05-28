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
from vae_2stream import OurClassifier, OurAutoencoder

def main(args):
	'''
	Plot training and test loss of a network
	savedir: where the network was saved
	'''

	savedir = args.savedir

	auto_dec_loss = pkl.load(open(os.path.join(savedir,'auto_dec_loss.pkl'),'rb'))
	auto_loss = pkl.load(open(os.path.join(savedir,'auto_loss.pkl'),'rb'))
	test_auto_dec_loss = pkl.load(open(os.path.join(savedir,'test_auto_dec_loss.pkl'),'rb'))
	test_auto_loss = pkl.load(open(os.path.join(savedir,'test_auto_loss.pkl'),'rb'))

	auto_enc1_loss = pkl.load(open(os.path.join(savedir,'auto_enc1_loss.pkl'),'rb'))
	auto_enc2_loss = pkl.load(open(os.path.join(savedir,'auto_enc2_loss.pkl'),'rb'))
	test_auto_enc1_loss = pkl.load(open(os.path.join(savedir,'test_auto_enc1_loss.pkl'),'rb'))
	test_auto_enc2_loss = pkl.load(open(os.path.join(savedir,'test_auto_enc2_loss.pkl'),'rb'))
	f,ax = plt.subplots(1,3)

	# Plot losses (separately, as there magnitude is different) and accuracy of discriminator
	ax[0].plot(test_auto_dec_loss,'g',label='Test')
	ax[0].plot(auto_dec_loss,'b',label='Decoder')
	ax[0].set_ylabel('Loss')
	ax[0].set_xlabel('Epochs')
	ax[0].legend()

	ax[1].plot(test_auto_enc1_loss,'g',label='Test')
	ax[1].plot(auto_enc1_loss,'b',label='Encoder1')
	ax[1].set_ylabel('Loss')
	ax[1].set_xlabel('Epochs')
	ax[1].legend()

	ax[2].plot(test_auto_enc2_loss,'g',label='Test')
	ax[2].plot(auto_enc2_loss,'b',label='Encoder2')
	ax[2].set_ylabel('Loss')
	ax[2].set_xlabel('Epochs')
	ax[2].legend()

	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot losses')
	parser.add_argument('savedir', action='store',help='Directory where model was stored')
	args = parser.parse_args()

	main(args)