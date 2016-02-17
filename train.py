import os
import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import serializers

import data
from models.mlp import MLP, MLPClassifier



def parse_args():
    parser = argparse.ArgumentParser('Trains a simple model. The model may be saved when the training is finished or resumed if there is a pretrained model.')
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
            x = Variable(cuda.cupy.asarray(x_train[indexes[i : i + batchsize]]))
            t = Variable(cuda.cupy.asarray(y_train[indexes[i : i + batchsize]]))
            update_auto(optimizer, model, x, t)
        mean_loss, mean_acc = evaluate_model(model, x_test, y_test, batchsize)
        print('Mean loss: {loss}'.format(loss=mean_loss))
        print('Mean accuracy: {acc}'.format(acc=mean_acc))
    return model
           
def update_auto(optimizer, model, x, t):
    # Passing the loss function (model) so that we don't have to call the model.zerograds() explicitly to reset the gradients in each iteration
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
        x = Variable(cuda.to_gpu(x_test[i : i + batchsize]))
        t = Variable(cuda.to_gpu(y_test[i : i + batchsize]))
        loss = model(x, t)
        sum_loss += loss.data * batchsize
        sum_acc += model.accuracy.data * batchsize
    mean_loss = sum_loss / testsize
    mean_acc = sum_acc / testsize
    return mean_loss, mean_acc

def load_mnist():
    mnist = data.load_mnist_data()
    mnist_data = mnist['data']
    mnist_target = mnist['target']
    return mnist_data, mnist_target

xp = cuda.cupy

def split_mnist(data, target, trainsize=60000):
    print('xp: {}'.format(xp))
    x_all = data.astype(np.float32) / 255
    y_all = target.astype(np.int32)
    x_train, x_test = np.split(x_all, [trainsize])
    y_train, y_test = np.split(y_all, [trainsize])
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batchsize = args.batchsize
    save = args.save_model
    load = args.load_model
    gpu = args.gpu
    print('Starting with:')
    print('Epochs: {}'.format(epochs))
    print('Batch size: {}'.format(batchsize))
    print('Load pretrained mode: {}'.format(load))
    print('Save model when finished: {}'.format(save))
    print('Loading MNIST data...')
    data, target = load_mnist()
    print('Loaded MNIST')
    x_train, x_test, y_train, y_test = split_mnist(data, target, trainsize=60000)
    model = MLP()
    model.to_gpu()
    model = MLPClassifier(model)  # Alternatively, use the build-in classifier with L.Classifier(MLP()))
    if gpu >= 0:
        print('Using GPU {}'.format(gpu))
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model.to_gpu()

    filename = 'mlp.model'  # The filename of the model to load, or save
    if load:
        print('Loading pretrained model...')
        try:
            serializers.load_hdf5(filename, model)
            print('Loaded pretrained model')
        except OSError as err:
            print('OS error: {}'.format(err))
            print('Could not find a pretrained model. Starting from a new and randomly initialized model.')
    model = train(model, x_train, x_test, y_train, y_test, epochs=epochs, batchsize=batchsize)
    if save:
        print('Saving trained model...')
        serializers.save_hdf5(filename, model)
        print('Saved trained model {}'.format(filename))
    print('Done')

