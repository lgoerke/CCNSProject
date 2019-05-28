from argparse import Namespace
import train_dsprites
import time

# Settings
epochs = 123
batch_size = 16
plot = True
verbosity = 1
lam = 0.02
lr = 0.001

args = Namespace(epochs=epochs,batch_size=batch_size,plot=plot,
	verbosity=verbosity,lam=lam,lr=lr)

print('########################')
print(args)    # print all args in the log file so we know what we were running
print('########################')

start = time.time()
train_dsprites.main(args)
print("The training took {:.1f} minutes".format((time.time()-start)/60)) # measure time
