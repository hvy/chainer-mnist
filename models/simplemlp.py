from chainer import Chain
import chainer.functions as F
import chainer.links as L


class SimpleMLP(Chain):
    """ A simple 3 layer Multi Layered Perceptron.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__(
                l1 = L.Linear(784, 100),  # MNIST images are 28 * 28 = 784 pixels large
                l2 = L.Linear(100, 100),
                l3 = L.Linear(100, 10)  # 10 classes in MNIST, i.e. the numbers [0, 9], to return the scores of the ten numbers
        )
        self.train = False

    def __call__(self, x, t):
        """ Method that is called directly on the instance e.g. mlp = MLP() and then mlp()
        that computes the output of the network.
        """
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            self.prediction = F.softmax(h)
            return self.prediction

