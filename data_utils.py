import numpy as np
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
from tqdm import tqdm
import pickle as pkl
import scipy.misc

def give_stats(value,title=''):
    '''
    Print minimum, mean and maximum value of an array (for debugging/monitoring)
    '''
    print('---',title,'---')
    print('Min',np.min(value))
    print('Mean',np.mean(value))
    print('Max',np.max(value))

def assemble_data():
    '''
    As saving this data matrix takes forever, we will assemble before training using the saved images
    Output: data matrix with data[:,0,:,:] => Squares, data[:,1,:,:] => Ellipses, data[:,2,:,:] => Combined img
    '''
    new_data_no = 10000
    data = np.zeros((new_data_no,3,32,32),dtype=np.float32)

    ## These pkl files contain the indices of 10.000 squares/ellipses.
    ## They were created with the randint fct in a jupyter notebook in the beginning of the project
    squares = pkl.load(open('squares.pkl','r'))
    ellipses = pkl.load(open('ellipses.pkl','r'))

    cnt = 0

    for i in tqdm(squares,desc='Squares Loop'):
        for j in ellipses:
            img = scipy.misc.imread('saved_imgs_shape/square_{:}.png'.format(i))
            new_img = np.zeros(img[:,:,0].shape)
            new_img[img[:,:,0] >= 128] = 1
            new_img[img[:,:,0] < 128] = 0
            data[cnt,0,:,:] = new_img
            data[cnt,0,:,:] = data[cnt,0,:,:].astype(np.float32)
            cnt += 1 

    cnt = 0
        
    for i in tqdm(squares,desc='Ellipses Loop'):
        for j in ellipses:
            img = scipy.misc.imread('saved_imgs_shape/ellipse_{:}.png'.format(j))
            new_img = np.zeros(img[:,:,0].shape)
            new_img[img[:,:,0] >= 128] = 1
            new_img[img[:,:,0] < 128] = 0
            data[cnt,1,:,:] = new_img 
            data[cnt,1,:,:] = data[cnt,1,:,:].astype(np.float32)
            cnt += 1 

    for i in tqdm(range(len(data)),desc='Combine images'):
        new_img = data[i,1,:,:] + data[i,0,:,:]
        new_img[new_img > 1] = 1
        data[i,2,:,:] = new_img
        data[i,2,:,:] = data[i,2,:,:].astype(np.float32)

    return data

class TrainIterator(object):
    """
    Generates random subsets of data
    Adapted from utils.py
    """

    def __init__(self, data, batch_size=1):
        """

        Args:
            data (TupleDataset):
            batch_size (int):

        Returns:
            list of batches consisting of (input, output) pairs
        """

        self.data = data.astype(np.float32)

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            return self.data[self._order[i:(i + self.batch_size)]]
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]])

class TestIterator(object):
    """
    Generates subsets of data (sequential)
    Adapted from utils.py
    """

    def __init__(self, data, batch_size=1):
        """

        Args:
            data (TupleDataset):
            batch_size (int):

        Returns:
            list of batches consisting of (input, output) pairs
        """

        self.data = data.astype(np.float32)

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = range(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            return self.data[self._order[i:(i + self.batch_size)]]
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]])

class Logger(object):
    '''
    Logger class
    From: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    '''
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except Exception as e:
            print('[logger] ', e)
        
    def flush(self):
        self.log.flush()
        self.terminal.flush()
        
    def __getattr__(self, attr): 
        return getattr(self.terminal, attr)

def colorbar(mappable,size="5%"):
    # From: http://joseph-long.com/writing/colorbars/
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def calcMidpointForCM(image):
    return 1 - np.max(image)/(np.max(image) + abs(np.min(image)))

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    From: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap