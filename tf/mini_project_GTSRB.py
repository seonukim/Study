## GTSRB ; (German Traffic Sign Recognition Benchmark)

# image preprocessing
N_CLASSES = 43
RESIZED_IMAGE = (32, 32)

import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import namedtuple          ## for onehotencoding
path = 'C:/Users/dnsrl/Downloads/data/GTSRB/Final_Training/'
os.chdir(path)
SEED = 101
np.random.seed(SEED)

Dataset = namedtuple('Dataset', ['X', 'y'])

def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis = 0).astype(np.float32)

def read_dataset_ppm(rootpath, n_labels, resize_to):
    images = []
    labels = []
    path = 'C:/Users/dnsrl/Downloads/data/GTSRB/Final_Training/Images/'

    for c in range(n_labels):
        full_path = path + format(c, '05d') + '/'
        for img_name in glob.glob(full_path + '*.ppm'):
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img / 255.0)[:, :, 0]

            if resize_to:
                img = resize(img, resize_to, mode = 'reflect')

            label = np.zeros((n_labels, ), dtype = np.float32)
            label[c] = 1.0

            images.append(img.astype(np.float32))
            labels.append(label)
    
    return Dataset(X = to_tf_format(images).astype(np.float32),
                   y = np.matrix(labels).astype(np.float32))

dataset = read_dataset_ppm('GTSRB/Final_Training/Images', N_CLASSES, RESIZED_IMAGE)
print(dataset.X.shape)                  # (39209, 32, 32, 1)
print(dataset.y.shape)                  # (39209, 43)

## Printing the first image
plt.imshow(dataset.X[0, :, :, :].reshape(RESIZED_IMAGE))    # first image
plt.show()
print(dataset.y[0, :])      # label

## Printing the last image
plt.imshow(dataset.X[-1, :, :, :].reshape(RESIZED_IMAGE))    # last image
print(dataset.y[-1, :])    # label

## Split train and test data
idx_train, idx_test = train_test_split(range(dataset.X.shape[0]),
                                       test_size = 0.25, random_state = SEED)
X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, :, :]
y_train = dataset.y[idx_train, :]
y_test = dataset.y[idx_test, ]

print(f'X_train : {X_train.shape} \ny_train : {y_train.shape}')
print(f'X_test : {X_test.shape} \ny_test : {y_test.shape}')

## Create function of minibatch of models
def minibatcher(X, y, batch_size, shuffle):
    assert X.shape[0] == y.shape[0]
    n_samples = X.shape[0]
    
    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        edx = list(range(n_samples))
        
    for k in range(int(np.ceil(n_samples / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]
    
for mb in minibatcher(X_train, y_train, 10000, True):
    print(mb[0].shape, mb[1].shape)

## Modeling
def fc_no_activation_layer(in_tensors, n_units):
    w = tf.compat.v1.get_variable('fc_W',
                                  [in_tensors.get_shape()[1], n_units],
                                  tf.float32, tf.contrib.layers.xavier_initializer())
    b = tf.compat.v1.get_variable('fc_B',
                                  [n_units, ],
                                  tf.float32,
                                  tf.compat.v1.constant_initializer(0.0))
    return tf.compat.v1.matmul(in_tensors, w) + b

# Activation function
def fc_layer(in_tensors, n_units):
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))

# Convolutional Layer
def conv_layer(in_tensors, kernel_size, n_units):
    w = tf.compat.v1.get_variable('conv_W',
                                  [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
                                  tf.float32, tf.contrib.layers.xavier_initializer())
    b = tf.compat.v1.get_variable('conv_B',
                                  [n_units, ],
                                  tf.float32,
                                 tf.compat.v1.constant_initializer(0.0))
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)

# Max_pool layer
def maxpool_layer(in_tensors, sampling):
    return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1], [1, sampling, sampling, 1], 'SAME')

# Dropout layer
def dropout(in_tensors, keep_proba, is_training):
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)

# Final Model
def model(in_tensors, is_training):
    # first layer: 5 x 5, 2 dim conv, 32 filters, 2x maxpool, 0.2 dropout
    with tf.compat.v1.variable_scope('l1'):
        l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
        l1_out = dropout(l1, 0.8, is_training)
    
    # second layer : 5 x 5, 2 dim conv, 64 filters, 2x maxpool, 0.2 dropout
    with tf.compat.v1.variable_scope('l2'):
        l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
        l2_out = dropout(l2, 0.8, is_training)
    
    with tf.compat.v1.variable_scope('flatten'):
        l2_out_flat = tf.layers.flatten(l2_out)
    
    # fully connected layer, 1024 of nodes, 0.4 dropout
    with tf.compat.v1.variable_scope('l3'):
        l3 = fc_layer(l2_out_flat, 1024)
        l3_out = dropout(l3, 0.6, is_training)
    
    # output layer
    with tf.compat.v1.variable_scope('out'):
        out_tensors = fc_no_activation_layer(l3_out, N_CLASSES)
    
    return out_tensors

## Definite function of train and test of model's perfomance
# This function gives train_set, test_set, label, learning_speed, epoch, batch_size.
def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size):
    # Most important : mb of image, mb of label, is training?
    in_X_tensors_batch = tf.compat.v1.placeholder(tf.float32,
                                                  shape = [None, RESIZED_IMAGE[0], RESIZED_IMAGE[1], 1])
    in_y_tensors_batch = tf.compat.v1.placeholder(tf.float32, shape = (None, N_CLASSES))
    is_training = tf.compat.v1.placeholder(tf.bool)
    
    # Definite result, evaluate score, optimized model
    logits = model(in_X_tensors_batch, is_training)
    out_y_pred = tf.nn.softmax(logits)
    loss_score = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = in_y_tensors_batch)
    loss = tf.compat.v1.reduce_mean(loss_score)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # Last, printing to training by using the minibatch
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        for epoch in range(max_epochs):
            print(f'Epoch = {epoch}')
            
            tf_score = []
            for mb in minibatcher(X_train, y_train, batch_size, shuffle = True):
                tf_output = sess.run([optimizer, loss],
                                     feed_dict = {in_X_tensors_batch: mb[0],
                                                  in_y_tensors_batch: mb[1],
                                                  is_training: True})
                tf_score.append(tf_output[1])
            print(f'Train_loss_score : {np.mean(tf_score)}')
            
        print('TEST SET PERFOMANCE')
        y_test_pred, test_loss = sess.run([out_y_pred, loss],
                                          feed_dict = {in_X_tensors_batch: X_test,
                                          in_y_tensors_batch: y_test,
                                          is_training: False})
        print(f'Test_loss_score : {test_loss}')
        y_test_pred_classified = np.argmax(y_test_pred, axis = 1).astype(np.int32)
        y_test_true_classified = np.argmax(y_test, axis = 1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))
        
        cm = confusion_matrix(y_test_true_classified, y_test_pred_classified)
        
        plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
        plt.imshow(np.log2(cm + 1), interpolation = 'nearest', cmap = plt.get_cmap('tab20'))
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    tf.reset_default_graph()

train_model(X_train, y_train, X_test, y_test, 1e-4, 10, 256)