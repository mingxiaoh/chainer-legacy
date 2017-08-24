import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import type_check

from chainer.functions.math import identity


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if not hasattr(self, 'mask'):
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            if xp == numpy:
                flag = xp.random.rand(*x[0].shape) >= self.dropout_ratio
            else:
                flag = (xp.random.rand(*x[0].shape, dtype=numpy.float32) >=
                        self.dropout_ratio)
            self.mask = scale * flag
        # FIXME: return x[0] * self.mask,
        return self.mask * x[0],

    def backward(self, x, gy):
        # FIXME: return gy[0] * self.mask,
        return self.mask * gy[0],


def dropout(x, ratio=.5):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_.

    """
    if configuration.config.train:
        return Dropout(ratio)(x)
    elif chainer.should_use_mkldnn('>=auto'):
        return identity.Identity()(x)
