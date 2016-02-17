from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MLP(Chain):
    """ A simple 3 layer Multi Layered Perceptron.
    """
    def __init__(self):
        super(MLP, self).__init__(
                l1 = L.Linear(784, 100),  # MNIST images are 28 * 28 = 784 pixels large
                l2 = L.Linear(100, 100),
                l3 = L.Linear(100, 10)  # 10 classes in MNIST, i.e. the numbers [0, 9], to return the scores of the ten numbers
        )

    def __call__(self, x):
        """ Method that is called directly on the instance e.g. mlp = MLP() and then mlp()
        that computes the output of the network.
        """
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


class MLPClassifier(Chain):
    """ A custom softmax entropy classifier. Note that there is an implementation ready 
    classifier provided by the Chainer framework, i.e. chainer.links.Classiier.
    """
    def __init__(self, predictor):
        """ Any arbitrary predictor link may be passed as an argument.
        """
        super(MLPClassifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        """ Compute the loss given the prediction and the ground truth labels.
        The prediction accuracy is also computed.
        """
        y = self.predictor(x)  # Forward pass
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

