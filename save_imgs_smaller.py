from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import h5py
from tqdm import tqdm
import scipy.misc
from tqdm import tnrange, tqdm_notebook
from scipy import ndimage

import random
from matplotlib.ticker import MaxNLocator
import data_utils
import sys
import os
import argparse
from PIL import Image
import PIL

'''
Loads saved images and saves resized version to disk
'''

ellipses = pkl.load(open('ellipses10000.pkl','r'))

cnt = 0

for i in tqdm(ellipses,desc='Ellipses Loop'):
	basewidth = 32
	img = Image.open('saved_ellipses64/ellipse_{:}.png'.format(i))
	wpercent = (basewidth / float(img.size[0]))
	hsize = int((float(img.size[1]) * float(wpercent)))
	img = img.resize((basewidth, hsize),PIL.Image.ANTIALIAS)
	img.save('saved_ellipses/ellipse_{:}.png'.format(i))