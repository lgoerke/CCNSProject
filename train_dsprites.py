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
import argparse

from vae_2stream import OurClassifier, OurAutoencoder

def our_training(autoencoder, optimizer, train, test, epochs, batch_size, plotting=False, verbosity=1, lam = 1, savedir='.'):
	"""
	Our training procedure
	"""
	auto_dec_loss = []
	test_auto_dec_loss = []
	auto_loss = []
	test_auto_loss = []

	auto_enc1_loss = []
	auto_enc2_loss = []
	test_auto_enc1_loss = []
	test_auto_enc2_loss = []
	
	# Training loop over all epochs. One epoch includes one complete run through all training images
	for e in tqdm(range(epochs),desc='Epochs'):
		
		dec_losses = []
		a_losses = []

		enc1_losses = []
		enc2_losses = []
		
		# Go through all training images in batches of batch_size
		for current_batch in data_utils.TrainIterator(train, batch_size=batch_size):
			# Calculate the prediction of the network
			dec_loss, enc1_loss, enc2_loss = autoencoder.compute_loss(current_batch[:,0,:,:],current_batch[:,1,:,:],current_batch[:,2,:,:])        
			loss = dec_loss + lam*(enc1_loss + enc2_loss)

			dec_losses.append(dec_loss.data)
			a_losses.append(loss.data)

			enc1_losses.append(enc1_loss.data)
			enc2_losses.append(enc2_loss.data)

			# Calculate the gradients in the network
			autoencoder.cleargrads()
			loss.backward()

			# Update all the trainable parameters
			optimizer.update()        
		
		auto_dec_loss.append(np.mean(dec_losses))
		auto_loss.append(np.mean(a_losses)) 

		auto_enc1_loss.append(np.mean(enc1_losses))
		auto_enc2_loss.append(np.mean(enc2_losses))

		dec_losses = []
		a_losses = []
		enc1_losses = []
		enc2_losses = []


		if plotting:
			# Prepare plotting of some reconstructions
			rows = 3
			total = 15
			plotted = 0
			if ((e+1)%verbosity==0) or e ==0:
				f,ax = plt.subplots(rows,5)
				i = 0
				j = 0
		# Go through all testing images in batches of batch_size           
		for current_batch in data_utils.TestIterator(test, batch_size=batch_size):
			# Calculate the prediction of the network
			dec_loss,enc1_loss,enc2_loss = autoencoder.compute_loss(current_batch[:,0,:,:],current_batch[:,1,:,:],current_batch[:,2,:,:])        
			loss = dec_loss + lam*(enc1_loss + enc2_loss)
			
			# Only save loss
			dec_losses.append(dec_loss.data)
			a_losses.append(loss.data)

			enc1_losses.append(enc1_loss.data)
			enc2_losses.append(enc2_loss.data)
			
			if plotting:
				# Plot some reconstructions for later progress overview
				if ((e+1)%verbosity==0) or e ==0:
					reconstruction = np.reshape(autoencoder.reconstruct(current_batch[:,0,:,:],current_batch[:,1,:,:]).data[0,:],(32,32))
					if plotted < total:
						ax[i,j].imshow(reconstruction,'binary')
						j += 1
						plotted += 1
						if j > 4:
							i += 1
							j = 0

		test_auto_dec_loss.append(np.mean(dec_losses))
		test_auto_loss.append(np.mean(a_losses))
		test_auto_enc1_loss.append(np.mean(enc1_losses))
		test_auto_enc2_loss.append(np.mean(enc2_losses))
		
		
		if ((e+1)%verbosity==0) or e ==0:
			# Display the training loss and accuracy
			sys.stdout.write('\n')
			print('+++++++++++++++++')
			print('+++ Epoch: {:02d} +++'.format(e+1))
			print('+++++++++++++++++')
			print('+++ Training ++++')
			print('Encoder1 Loss:{:.04f} and Encoder2 Loss:{:.04f} and Decoder Loss:{:.04f}'.format(auto_enc1_loss[-1],auto_enc2_loss[-1], auto_dec_loss[-1]))
			print('+++ Test ++++++++')
			print('Encoder1 Loss:{:.04f} amd Encoder2 Loss:{:.04f} and Decoder Loss:{:.04f}'.format(test_auto_enc1_loss[-1],test_auto_enc2_loss[-1], test_auto_dec_loss[-1]))
		
		if plotting:
			# Save plotted reconstructions on disk
			if ((e+1)%verbosity==0) or e ==0:
				plt.savefig(os.path.join(savedir, 'epoch_{}.png'.format(e+1)))
	  
	return auto_dec_loss,auto_enc1_loss,auto_enc2_loss,auto_loss,test_auto_dec_loss,test_auto_enc1_loss,test_auto_enc2_loss,test_auto_loss

def main(args):
	'''
	Main training procedure, create two networks including optimizers and call training function
	'''

	# Get arguments from argparse
	epoch_no = args.epochs
	batch_size = args.batch_size
	plotting = args.plot
	verbosity = args.verbosity
	lam = args.lam
	lr = args.lr

	# Save all relevant files in directory made of parameter string
	savedir = "e" + str(epoch_no) + "_b" + str(batch_size) + "_lam" + str(lam) + "_lr" +str(lr)

	if not os.path.exists(savedir):
			os.makedirs(savedir)

	sys.stdout = data_utils.Logger(os.path.join(savedir, 'stdout.txt'))
	sys.stderr = data_utils.Logger(os.path.join(savedir, 'stderr.txt'))

	np.random.seed(4517)

	print('########################')
	print(args)    # print all args in the log file so we know what we were running
	print('########################')

	# Initialize network
	autoencoder = OurClassifier(OurAutoencoder())
	optimizer = optimizers.Adam(alpha=lr)
	optimizer.setup(autoencoder)

	# Get data
	data = data_utils.assemble_data()

	data_idx = np.random.permutation(len(data))
	cutoff = int (len(data)*0.8)

	train = data[data_idx[:cutoff],:,:,:]
	train = np.reshape(train,(train.shape[0],3,32,32))
	test = data[data_idx[cutoff:],:,:,:]
	test = np.reshape(test,(test.shape[0],3,32,32))
			
	# Train network and save relevant quantities afterwards
	auto_dec_loss,auto_enc1_loss,auto_enc2_loss,auto_loss,test_auto_dec_loss,test_auto_enc1_loss,test_auto_enc2_loss,test_auto_loss  = our_training(autoencoder,optimizer, train, test, epochs=epoch_no, batch_size=batch_size, plotting = plotting, verbosity=verbosity,lam = lam,savedir=savedir)

	serializers.save_npz(os.path.join(savedir,'autoencoder.model'), autoencoder)

	pkl.dump(auto_dec_loss,open(os.path.join(savedir,'auto_dec_loss.pkl'),'wb'))
	pkl.dump(auto_loss,open(os.path.join(savedir,'auto_loss.pkl'),'wb'))
	pkl.dump(test_auto_dec_loss,open(os.path.join(savedir,'test_auto_dec_loss.pkl'),'wb'))
	pkl.dump(test_auto_loss,open(os.path.join(savedir,'test_auto_loss.pkl'),'wb'))

	pkl.dump(auto_enc1_loss,open(os.path.join(savedir,'auto_enc1_loss.pkl'),'wb'))
	pkl.dump(auto_enc2_loss,open(os.path.join(savedir,'auto_enc2_loss.pkl'),'wb'))
	pkl.dump(test_auto_enc1_loss,open(os.path.join(savedir,'test_auto_enc1_loss.pkl'),'wb'))
	pkl.dump(test_auto_enc2_loss,open(os.path.join(savedir,'test_auto_enc2_loss.pkl'),'wb'))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train VAE')
	parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=10,
											help='Number of training epochs')
	parser.add_argument('-bs', action='store', dest='batch_size', type=int, default=32,
											help='Size of training mini batches')
	parser.add_argument('-v', action='store', dest='verbosity', type=int, default=1,
											help='When to write to console')
	parser.add_argument('-lam', action='store', dest='lam', type=float, default=1,
											help='Weight of encoder loss')
	parser.add_argument('-lr', action='store', dest='lr', type=float, default=0.001,
											help='Learning rate ')
	parser.add_argument('--plot', action='store_true', dest='plot', default=False,
											help='Plot results')
	args = parser.parse_args()

	main(args)

