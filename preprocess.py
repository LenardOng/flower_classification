import tensorflow as tf
import os 
import skimage
import numpy as np
from skimage import io, transform
import random

#   Preprocess the dataset



dataset_dir = os.path.join('..', 'data', 'flowers')
out_dir = os.path.join('..', 'data', 'preprocessed_flowers')
labels = os.listdir(dataset_dir)
img_size = (128, 128)
img_store = []
gt = []

# Create out_dir path if necessary
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
	
print('Loading images')
# Pulls labels from the folder

for i, label in enumerate(labels):
    print('Processing images for ' + label)
    filenames = os.listdir(os.path.join(dataset_dir, label))
    for training_example in filenames:
        file = os.path.join(dataset_dir, label, training_example)
        img = io.imread(file)
        img_resize = transform.resize(img, img_size, mode='reflect', anti_aliasing=True)
        img_rescale = skimage.img_as_float(img_resize)		
        img_store.append(img_resize)
        gt.append(i)

n_dat = len(gt)
print('Shuffling data')
#Data shuffle
data = list(zip(img_store, gt))
random.shuffle(data)
imgs, labels = zip(*data)

img_set = np.array(imgs)
label_set = np.array(labels)
print('Splitting data')
#Train, dev, test split
n_lower = int(n_dat*0.2)
n_higher = int(n_dat*0.8)

x_train = img_set[0:n_higher]
x_test = img_set[n_higher:n_dat-1]

y_train = label_set[0:n_higher]
y_test = label_set[n_higher:n_dat-1]
print('Saving data')
np.save(os.path.join(out_dir, 'x_train'), x_train)
np.save(os.path.join(out_dir, 'x_test'), x_test)
np.save(os.path.join(out_dir, 'y_train'), y_train)
np.save(os.path.join(out_dir, 'y_test'), y_test)
print('Data saved')

