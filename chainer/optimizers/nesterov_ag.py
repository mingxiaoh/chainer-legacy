from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class NesterovAGRule(optimizer.UpdateRule):

    """Update rule for Nesterov's Accelerated Gradient.

    See :class:`~chainer.optimizers.NesterovAG` for the default values of the
    hyperparameters.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    def __init__(self, lr=None, momentum=None):
        super(NesterovAGRule, self).__init__()
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
        lr, momentum = self.hyperparam.lr, self.hyperparam.momentum

        v *= momentum
        v -= lr * param.grad
        param.data += momentum * momentum * v
        param.data -= (1 + momentum) * lr * param.grad

    def update_core_gpu(self, param):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''
               v = v * momentum - lr * grad;
               param += momentum * momentum * v - (1 + momentum) * lr * grad;
            ''',
            'nesterov_ag')(
                param.grad, self.hyperparam.lr, self.hyperparam.momentum,
                param.data, self.state['v'])


class NesterovAG(optimizer.GradientMethod):

    """Nesterov's Accelerated Gradient.

    See: http://arxiv.org/abs/1212.0901

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(NesterovAG, self).__init__()
        self.hyperparam = optimizer.Hyperparameter()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    def setup_update_rule(self, param):
        param.update_rule = NesterovAGRule()
        param.update_rule.hyperparam.set_parent(self.hyperparam)
