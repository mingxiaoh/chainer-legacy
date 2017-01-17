import numpy

from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.001
_default_hyperparam.eps = 1e-8


class AdaGradRule(optimizer.UpdateRule):

    """Update rule of AdaGrad.

    See :class:`~chainer.optimizers.AdaGrad` for the default values of the
    hyperparameters.

    Args:
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """
    def __init__(self, lr=None, eps=None):
        super(AdaGradRule, self).__init__()
        self.hyperparam = optimizer.Hyperparameter(_default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            self.state['h'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        lr = self.hyperparam.lr
        eps = self.hyperparam.eps

        h = self.state['h']
        grad = param.grad

        h += grad * grad
        param.data -= lr * grad / (numpy.sqrt(h) + eps)

    def update_core_gpu(self, param):
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(param.grad, self.hyperparam.lr, self.hyperparam.eps,
                       param.data, self.state['h'])


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad optimizer.

    See: http://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr (float): Learning rate.
        eps (float): Small value for the numerical stability.

    """
    def __init__(self, lr=_default_hyperparam.lr, eps=_default_hyperparam.eps):
        super(AdaGrad, self).__init__()
        self.hyperparam = optimizer.Hyperparameter()
        self.hyperparam.rho = rho
        self.hyperparam.eps = eps

    def setup_update_rule(self, param):
        param.update_rule = AdaGradRule()
        param.update_rule.hyperparam.set_parent(self.hyperparam)
