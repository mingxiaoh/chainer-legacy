from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01


class SGDRule(optimizer.UpdateRule):

    """Update rule of vanilla stochastic gradient descent.

    See :class:`~chainer.optimizers.SGD` for the default values of the
    hyperparameters.

    Args:
        lr (float): Learning rate.

    """
    def __init__(self, lr=None):
        super(SGDRule, self).__init__()
        self.hyperparam = optimizer.Hyperparameter(_default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr

    def update_core_cpu(self, param):
        param.data -= self.hyperparam.lr * param.grad

    def update_core_gpu(self, param):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(param.grad, self.hyperparam.lr, param.data)


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent.

    Args:
        lr (float): Learning rate.

    """
    def __init__(self, lr=_default_hyperparam.lr):
        super(SGD, self).__init__()
        self.hyperparam = optimizer.Hyperparameter()
        self.hyperparam.lr = lr

    def setup_update_rule(self, param):
        param.update_rule = SGDRule()
        param.update_rule.hyperparam.set_parent(self.hyperparam)
