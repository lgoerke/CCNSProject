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
	Plots losses of several networks for comparison
	savedir: possibly multiple directories with saved networks
	title: optional, will be plotted as figure title
	'''
	s_list = args.savedir
	title = args.title

	f_test,ax_test = plt.subplots(1,3)
	plt.suptitle(title + ", Training losses")
	f_train,ax_train = plt.subplots(1,3)
	plt.suptitle(title + ", Testing losses")

	for i,savedir in enumerate(s_list):

		auto_dec_loss = pkl.load(open(os.path.join(savedir,'auto_dec_loss.pkl'),'rb'))
		auto_loss = pkl.load(open(os.path.join(savedir,'auto_loss.pkl'),'rb'))
		test_auto_dec_loss = pkl.load(open(os.path.join(savedir,'test_auto_dec_loss.pkl'),'rb'))
		test_auto_loss = pkl.load(open(os.path.join(savedir,'test_auto_loss.pkl'),'rb'))

		auto_enc1_loss = pkl.load(open(os.path.join(savedir,'auto_enc1_loss.pkl'),'rb'))
		auto_enc2_loss = pkl.load(open(os.path.join(savedir,'auto_enc2_loss.pkl'),'rb'))
		test_auto_enc1_loss = pkl.load(open(os.path.join(savedir,'test_auto_enc1_loss.pkl'),'rb'))
		test_auto_enc2_loss = pkl.load(open(os.path.join(savedir,'test_auto_enc2_loss.pkl'),'rb'))

		# Plot losses (separately, as there magnitude is different)
		ax_train[0].plot(auto_dec_loss,label='Decoder, Model {}'.format(i + 1))
		ax_train[0].set_ylabel('Loss')
		ax_train[0].set_xlabel('Epochs')
		ax_train[0].legend()

		ax_train[1].plot(auto_enc1_loss,label='Encoder1, Model {}'.format(i + 1))
		ax_train[1].set_ylabel('Loss')
		ax_train[1].set_xlabel('Epochs')
		ax_train[1].set_ylim([0,1.3])
		ax_train[1].legend()

		ax_train[2].plot(auto_enc2_loss,label='Encoder2, Model {}'.format(i + 1))
		ax_train[2].set_ylabel('Loss')
		ax_train[2].set_xlabel('Epochs')
		ax_train[2].set_ylim([0,1.3])
		ax_train[2].legend()
		
		ax_test[0].plot(test_auto_dec_loss,label='Decoder, Model {}'.format(i + 1))
		ax_test[0].set_ylabel('Loss')
		ax_test[0].set_xlabel('Epochs')
		ax_test[0].legend()

		ax_test[1].plot(test_auto_enc1_loss,label='Encoder1, Model {}'.format(i + 1))
		ax_test[1].set_ylabel('Loss')
		ax_test[1].set_xlabel('Epochs')
		ax_test[1].set_ylim([0,1.3])
		ax_test[1].legend()

		ax_test[2].plot(test_auto_enc2_loss,label='Encoder2, Model {}'.format(i + 1))
		ax_test[2].set_ylabel('Loss')
		ax_test[2].set_xlabel('Epochs')
		ax_test[2].set_ylim([0,1.3])
		ax_test[2].legend()

	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Plot losses')
	parser.add_argument('savedir', nargs='+', action='store',help='Directory where model was stored')
	parser.add_argument('--title',action='store',help='Title of comparison',default="")
	args = parser.parse_args()

	main(args)