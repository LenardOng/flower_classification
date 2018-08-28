import tensorflow as tf
import numpy as np
import os

processed_data_dir = os.path.join('..', 'data', 'preprocessed_flowers')

def load_data(mode = 'train'):
	if mode=='train':
		x=np.load(os.path.join(processed_data_dir, 'x_train.npy'))
		y=np.load(os.path.join(processed_data_dir, 'y_train.npy'))
	elif mode=='test':
		x=np.load(os.path.join(processed_data_dir, 'x_test.npy'))
		y=np.load(os.path.join(processed_data_dir, 'y_test.npy'))
	return x, y
	
	