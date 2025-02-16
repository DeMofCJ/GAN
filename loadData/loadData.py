from __future__ import division
import numpy as np
import os
from glob import glob



def load_mnist(datasetName):
	data_dir = os.path.join("../loadData/data", datasetName)
	fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(np.float)

	fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(np.float) 

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i][y[i]] = 1.0


	return X/255., y_vec
	
