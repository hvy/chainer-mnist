import os
import sys
import argparse
import numpy as np
import six
from six.moves.urllib import request
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import serializers
from models.simplecnn import SimpleCNN
from models.simplemlp import SimpleMLP

# Use numpy or chainer.cuda.cupy depending on the --gpu flag but default to numpy
xp = np


def parse_args():
    parser = argparse.ArgumentParser('Trains a simple model. The model may be saved when the training is finished or resumed if there is a pretrained model.')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=-1)  # Negative values to use the CPU
    return parser.parse_args()


def train(model, x_train, x_test, y_train, y_test, epochs, batchsize):
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    trainsize = y_train.size
    
    for epoch in range(epochs):
        print('Epoch: {epoch}'.format(epoch=epoch))
        # For each epoch, radomize the order of the data samples
        indexes = np.random.permutation(trainsize)
        for i in range(0, trainsize, batchsize):  # (start, stop, step)
            x = Variable(xp.asarray(x_train[indexes[i : i + batchsize]]))
            t = Variable(xp.asarray(y_train[indexes[i : i + batchsize]]))
            update_auto(optimizer, model, x, t)
        mean_loss, mean_acc = evaluate_model(model, x_test, y_test, batchsize)
        print('Mean loss: {loss}'.format(loss=mean_loss))
        print('Mean accuracy: {acc}'.format(acc=mean_acc))

    return model


def update_auto(optimizer, model, x, t):
    """ Passing the loss function (model) so that we don't have to call the model.zerograds() explicitly to reset the gradients in each iteration
    """
    optimizer.update(model, x, t)


def update_manual(optimizer, model, x, t):
    model.zerograds()  # Reset the gradients from the previous iteration, since they are accumulated otherwise
    loss = model(x, t)  # Call the ___call__, i.e. forward pass method
    loss.backward()  # Compute the gradient, x will be updated with it holding the gradient value
    optimizer.update()


def evaluate_model(model, x_test, y_test, batchsize):
    sum_loss = 0
    sum_acc = 0
    testsize = y_test.size

    for i in range(0, testsize, batchsize):
        x = Variable(xp.asarray(x_test[i : i + batchsize]))
        t = Variable(xp.asarray(y_test[i : i + batchsize]))
        loss = model(x, t)
        sum_loss += loss.data * batchsize
        sum_acc += model.accuracy.data * batchsize
    mean_loss = sum_loss / testsize
    mean_acc = sum_acc / testsize

    return mean_loss, mean_acc


def load_data(filename):
    with open(filename, 'rb') as dataset_pickle:
        dataset = six.moves.cPickle.load(dataset_pickle)
        data = dataset['data']
        target = dataset['target']
    return data, target


def split_mnist(data, target, trainsize=60000):
    x_all = data.astype(np.float32) / 255 # Scale the data to [0, 1]
    y_all = target.astype(np.int32)
    x_train, x_test = np.split(x_all, [trainsize])
    y_train, y_test = np.split(y_all, [trainsize])
    return x_train, x_test, y_train, y_test


def set_mode(gpu):
    """ Set the global xp variable based on the GPU settings and the CUDA availability.

    """
    global xp
    gpu_mode = False
    print('aaaa')
    if gpu >= 0:
        # Set to GPU, CuPy
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        xp = cuda.cupy
        gpu_mode = True
    else:
        # Set to CPU, numpy
        xp = np
        gpu_mode = False

    print('xp: {}'.format(xp))
    return gpu_mode


def load_model(filename, model):
    print('Loading pretrained model...')
    try:
        serializers.load_hdf5(filename, model)
        print('Loaded pretrained model')
    except OSError as err:
        print('OS error: {}'.format(err))
        print('Could not find a pretrained model. Proceeding with a randomly initialized model.')


def save_model(filename, model):
    print('Saving trained model...')

    if os.path.exists(filename):
        print('Overwriting existing file {}'.format(filename))

    serializers.save_hdf5(filename, model)
    print('Saved trained model {}'.format(filename))


if __name__ == '__main__':
    args = parse_args()

    model = None
    data_file = None
    model_name = args.model
    if model_name == 'mlp':
        model = SimpleMLP()
        data_file = 'mnist_1dim.pkl'
    elif model_name == 'cnn':
        model = SimpleCNN()
        data_file = 'mnist_2dim.pkl'
    else:
        print('Unknown model {}'.format(model_name))
        sys.exit()
    
    print('Training model: {}'.format(model))
    data, target = load_data(data_file)
    x_train, x_test, y_train, y_test = split_mnist(data, target, trainsize=60000)
   
    # Make sure that the model updates the weights on update
    model.train = True
    
    # The filename of the model to load, or save
    filename = 'mlp.model'  
    load = args.load_model
    save = args.save_model
    if load:
        load_model(model, filename)
    
    # Set the mode to either GPU or CPU
    gpu = args.gpu
    if gpu >= 0:
        gpu_success = set_mode(args.gpu) 
        if gpu_success:
            model.to_gpu()

    print('Load pretrained mode: {}'.format(load))
    print('Save model when finished: {}'.format(save))
    
    epochs = args.epochs
    batchsize = args.batchsize
    print('Starting with configuration:')
    print('Epochs: {}'.format(epochs))
    print('Batch size: {}'.format(batchsize))
    
    model = train(model, x_train, x_test, y_train, y_test, epochs=epochs, batchsize=batchsize)
    
    if save:
        save_model(model, filename)

    print('Done')

