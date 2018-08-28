import tensorflow as tf
import os
from tensorflow.keras.applications import VGG16
from load_data import load_data
import numpy as np

data_dir = os.path.join('..', 'data', 'flowers', 'preprocessed')
model_dir = os.path.join('..', 'data', 'VGG16')
model_name = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# NN hidden unit number
params=[128, 128]

# Input image shape
input_shape = [128, 128, 3]

# Training length
iterations = 100
test_iterations = 10
epochs = 1000
#   Learning Rate
lr = 1e-2

with tf.Session() as sess:
    # Define place holders for inputs
    x = tf.placeholder(tf.float32, [None, 128, 128, 3], name='x')
    y_gt = tf.placeholder(tf.uint8, [None], name='y_gt')
    # Learning rate placeholder allows annealing
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # Input preprocessing
    # Converting to one hot
    # Reshape input
    y_onehot = tf.one_hot(y_gt, 5)
    input = tf.reshape(x, [-1, 128, 128, 3])
    
    #   Preload VGG16 model and weights
    vgg16 = VGG16(include_top = False, 
            input_shape = input_shape,
            weights=os.path.join(model_dir, model_name))
    vgg_output = vgg16(input)
    shape = vgg_output.get_shape().as_list()
    net = tf.reshape(vgg_output, [-1, shape[1]*shape[2]*shape[3]])
    print('VGG16 loaded')
    
    #   Insert decision layers at the end
    with tf.variable_scope('DNN_layers'):
        for layer in range(len(params)):
            net = tf.layers.dense(net, units=params[layer], activation=tf.nn.relu)
        logits = tf.layers.dense(net, units=5, activation=None)
    
    #   Cost function
    cost_calc = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits))
    
    
    #   OPTIMIZER - 3 methods possible:
    #   1) Minimize and leave it to tensorflow to decide
    #   2) Choose labelled layers to apply gradients on
    #   3) Apply gradient clipping if required

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"DNN_layers")
    grads_and_vars = optimizer.compute_gradients(cost_calc, tvr)
    #clipped_grads_and_vars = [(tf.clip_by_norm(grad,5.), var) for grad, var in grads_and_vars]
    #train_op = optimizer.apply_gradients(clipped_grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)
    #train_op = optimizer.minimize(cost_calc) 
    predicted_classes = tf.argmax(logits, 1)
    accuracy, acc_update = tf.metrics.accuracy(labels=y_gt,
                                    predictions=predicted_classes)
                           
    #   Initialise variables
    sess.run(tf.local_variables_initializer()) 
    sess.run(tf.global_variables_initializer())
    print('Variables initialised')
    
    #   Load data
    train_x, train_y = load_data('train')
    test_x, test_y = load_data('test')
    
    #   Calculate batch splitting parameters
    n_samples = train_x.shape[0]
    batch_size = int(n_samples/iterations)
    n_test = test_x.shape[0]
    test_batch_size = int(n_test/test_iterations)
    
    #   Training
    for e in range(epochs):
        epoch_acc = []
        print('Starting Epoch number: ', e)
        #   Begin iterations
        #   Currently biased as the training order is not shuffled
        for i in range(iterations):
            #   Split batches
            train_x_slice = train_x[i*batch_size:(i+1)*batch_size]
            train_y_slice = train_y[i*batch_size:(i+1)*batch_size]
            _, cost = sess.run([train_op, cost_calc],
                                    feed_dict={x: train_x_slice, y_gt: train_y_slice, learning_rate: lr})
        #   Calc test accuracy
        for i in range(test_iterations):
            #   Split batches
            test_x_slice = train_x[i*test_batch_size:(i+1)*test_batch_size]
            test_y_slice = train_y[i*test_batch_size:(i+1)*test_batch_size]
            _, acc = sess.run([acc_update, accuracy], feed_dict={x: test_x_slice, y_gt: test_y_slice})
            epoch_acc.append(acc)
        #   Output average accuracy, simple mean is ok as test batch size is constant
        epoch_acc = np.mean(np.array(epoch_acc))       
        print('Test set accuracy for this epoch = '+str(epoch_acc))