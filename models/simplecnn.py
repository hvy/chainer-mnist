import chainer
import chainer.links as L
import chainer.functions as F

class SimpleCNN(chainer.Chain):

    """ Simple Convolutional Neural Network for training the MNIST dataset.
    Input dimensions are 28, 28, 1 where 1 represent the single grayscale channel.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__(
                conv1 = L.Convolution2D(1, 8, 5, stride=1, pad=2),
                conv2 = L.Convolution2D(8, 16, 3, stride=1, pad=2),
                fc1 = L.Linear(576, 10)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=3)
        h = self.fc1(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

