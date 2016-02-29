import os
import gzip
import numpy as np
import six
from six.moves.urllib import request
from common import Network, Dimensions


# MNIST source URLs
base_url= 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'

# MNIST image properties. Total number of images = 70000
num_channels = 1
num_train = 60000 
num_test = 10000


def download_mnist():
    print('Downloading MNIST...')
    download_file('{}/{}'.format(base_url, train_images), train_images)
    download_file('{}/{}'.format(base_url, train_labels), train_labels)
    download_file('{}/{}'.format(base_url, test_images), test_images)
    download_file('{}/{}'.format(base_url, test_labels), test_labels)

def download_file(source_file, target_file):
    # TODO Use os.path.join
    if not os.path.exists(target_file):
        print('Downloading {}...'.format(source_file))
        request.urlretrieve(source_file, target_file)
    else:
        print('Found {}'.format(target_file))


def create_from_mnist(filename, shape, size):
    if not os.path.exists(filename):
        print('Processing MNIST to create {}, this might take a while...'.format(filename))
        data_train, target_train = load_reshape(train_images, train_labels, num_train, shape, size)
        data_test, target_test = load_reshape(test_images, test_labels, num_test, shape, size)
        mnist = {}
        mnist['data'] = np.append(data_train, data_test, axis=0)
        mnist['target'] = np.append(target_train, target_test, axis=0)
        with open(filename, 'wb') as output:
            six.moves.cPickle.dump(mnist, output, -1)
    else:
        print('Found {}'.format(filename))

#     # Load the cached data
#     print('Loading file {}'.format(filename))
#     with open(filename, 'rb') as mnist_pickle:
#         mnist = six.moves.cPickle.load(mnist_pickle)
#         print('Successfully loaded file')
# 
#     return mnist
# 

def load_reshape(images, labels, num, shape, size):
    shape = tuple([num, *shape])   
    data = np.zeros(num * size, dtype=np.uint8).reshape(shape)
    target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    with gzip.open(images, 'rb') as f_images,\
            gzip.open(labels, 'rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in six.moves.range(num):
            target[i] = ord(f_labels.read(1))
            if len(shape) == 2: 
                # One dimensional representation of MNIST
                for j in six.moves.range(shape[1]):
                    data[i, j] = ord(f_images.read(1))
            elif len(shape) == 4:
                # Two dimensional representation of MNIST
                for j in six.moves.range(shape[1]): # For each pixel in width
                    for k in six.moves.range(shape[2]): # For each pixel in height
                        data[i, 0, j, k] = ord(f_images.read(1))
            else: 
                print('Unknown target shape: {}'.format(shape))
    
    return data, target


if __name__ == '__main__':
    print('Preparing the MNIST data...')
    
    download_mnist()

    width = 28
    height = 28
    size = width * height
    num_channels = 1
    mlp_dim = (size,)
    cnn_dim = (num_channels, width, height)
    create_from_mnist('mnist_1dim.pkl', mlp_dim, size)
    create_from_mnist('mnist_2dim.pkl', cnn_dim, size)
    print('Done preparing the MNIST dataset')

