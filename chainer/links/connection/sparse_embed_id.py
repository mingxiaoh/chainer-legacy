import numpy

import chainer
from chainer import cuda
from chainer import initializers
from chainer import link
from chainer.functions.array import stack


class SparseEmbedID(link.Link):

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super(SparseEmbedID, self).__init__()
        self._in_size = in_size
        self.ignore_label = ignore_label

        for i in range(in_size):
            name = str(i)
            self.add_param(name, (out_size,))

        if initialW is None:
            initialW = initializers.Normal(1.0)
        for param in self.params():
            initializers.init_weight(param.data, initialW)

        if ignore_label is not None:
            name = str(ignore_label)
            self.add_persistent(name, numpy.zeros(out_size, dtype='f'))

    def __call__(self, x):
        x = x.data
        if chainer.is_debug():
            xp = cuda.get_array_module(x)
            valid_x = xp.logical_and(0 <= x, x < self._in_size)
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy'
                                 '`0 <= x < len(W)`')

        x = cuda.to_cpu(x)
        return stack.stack([getattr(self, str(xi)) for xi in x])
