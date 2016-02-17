import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import data
from models.mlp import MLP, MLPClassifier

def train(x_train, x_test, y_train, y_test, epochs, batchsize):
    model = MLPClassifier(MLP())  # Alternatively, use the build-in classifier with L.Classifier(MLP()))
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    trainsize = y_train.size
    for epoch in range(epochs):
        print('Epoch: {epoch}'.format(epoch=epoch))
        # For each epoch, radomize the ordser of the data samples
        indexes = np.random.permutation(trainsize)
        for i in range(0, trainsize, batchsize):  # (start, stop, step)
            x = Variable(x_train[indexes[i : i + batchsize]])
            t = Variable(y_train[indexes[i : i + batchsize]])
            update_auto(optimizer, model, x, t)
        mean_loss, mean_acc = evaluate_model(model, x_test, y_test, batchsize)
        print('Mean loss: {loss}'.format(loss=mean_loss))
        print('Mean accuracy: {acc}'.format(acc=mean_acc))
           
def update_auto(optimizer, model, x, t):
    # Passing the loss function (model) so that we don't have to call the model.zerograds() explicitly to reset the gradients in each iteration
    optimizer.update(model, x, t)

def update_manual(optimizer, model, x, y):
    model.zerograds()  # Reset the gradients from the previous iteration, since they are accumulated otherwise
    loss = model(x, t)  # Call the ___call__, i.e. forward pass method
    loss.backward()  # Compute the gradient, x will be updated with it holding the gradient value
    optimizer.update()

def evaluate_model(model, x_test, y_test, batchsize):
    sum_loss = 0
    sum_acc = 0
    testsize = y_test.size
    for i in range(0, testsize, batchsize):
        x = Variable(x_test[i : i + batchsize])
        t = Variable(y_test[i : i + batchsize])
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

def split_mnist(data, target, trainsize=60000):
    x_all = data.astype(np.float32) / 255
    y_all = target.astype(np.int32)
    x_train, x_test = np.split(x_all, [trainsize])
    y_train, y_test = np.split(y_all, [trainsize])
    print('MNIST data (Number of pixels): {datasize}'.format(datasize=data.size))
    print('MNIST size: {targetsize}'.format(targetsize=target.size))
    print('Train data: {trainsize}'.format(trainsize=x_train.size))
    print('Test data: {testsize}'.format(testsize=x_test.size))
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    print('Starting...')
    data, target = load_mnist()
    x_train, x_test, y_train, y_test = split_mnist(data, target, trainsize=60000)
    train(x_train, x_test, y_train, y_test, epochs=20, batchsize=100)
    print('Done!')

