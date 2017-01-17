from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class MomentumSGDRule(optimizer.UpdateRule):

    """Update rule for the classical momentum SGD.

    See :class:`~chainer.optimizers.MomentumSGD` for the default values of the
    hyperparameters.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    def __init__(self, lr=None, momentum=None):
        super(MomentumSGDRule, self).__init__()
        self.hyperparam = optimizer.Hyperparameter(_default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        v = self.state['v']
        v *= self.hyperparam.momentum
        v -= self.hyperparam.lr * param.grad
        param.data += v

    def update_core_gpu(self, param):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(
                param.grad, self.hyperparam.lr, self.hyperparam.momentum,
                param.data, self.state['v'])


class MomentumSGD(optimizer.GradientMethod):

    """Momentum SGD optimizer.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(MomentumSGD, self).__init__()
        self.hyperparam = optimizer.Hyperparameter()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = Momentum

    def setup_update_rule(self, param):
        param.update_rule = MomentumSGDRule()
        param.update_rule.hyperparam.set_parent(self.hyperparam)
