import os
import sys
import gzip
import numpy as np
import six
from six.moves.urllib import request

# MNIST source URLs
base_url= 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'


def download_mnist():
    """ Download the MNIST dataset from Yann LeCun's homepage.

    """
    download_file('{}/{}'.format(base_url, train_images), train_images)
    download_file('{}/{}'.format(base_url, train_labels), train_labels)
    download_file('{}/{}'.format(base_url, test_images), test_images)
    download_file('{}/{}'.format(base_url, test_labels), test_labels)


def download_file(source_file, target_file):
    if not os.path.exists(target_file):
        print('Downloading {}...'.format(source_file))
        request.urlretrieve(source_file, target_file)
    else:
        print('Found {}'.format(target_file))


def create_from_mnist(filename, shape, size):
    """ Save an MNIST dataset representational file to disk with the name
    specified by the parameter.

    """
    num_train = 60000 
    num_test = 10000
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


def load_reshape(images, labels, num, shape, size):
    """ Load the MNIST gzip files from disk and reshape the data.

    """
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
    """ Download the MNIST dataset from Yann LeCun's homepage, reshape the data 
    to 1-dimensional data for MLP models and 2-dimensional data for CNN models.
    A .pkl file is created in this directory for each of the two types of models.

    The dataset contains a total of 70000 images, consisting of  60000 training 
    samples and 10000 test samples.

    Note that the data only contains single grayscale channels and not three 
    as in RGB images.

    """ 

    mlp_filename = 'mnist_1dim.pkl'
    cnn_filename = 'mnist_2dim.pkl'

    if os.path.exists(mlp_filename) and os.path.exists(cnn_filename):
        print('The MNIST dataset is already downloaded and reshaped: {}, {}'.format(mlp_filename, cnn_filename))
        print('To download them again, delete the files and rerun this script')
        sys.exit()
    
    print('Preparing the MNIST dataset...')
    
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

