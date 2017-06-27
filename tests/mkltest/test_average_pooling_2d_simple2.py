import unittest

# import mock
import numpy
# import six

import chainer
# from chainer import cuda
# from chainer import functions
# from chainer import gradient_check
# from chainer import testing
# from chainer.testing import attr
# from chainer.testing import condition
from chainer.functions.math import identity
from mkldnn.chainer import avg_pooling_2d


class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.cover_all = False
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            1 * 1024 * 7 * 7, dtype=self.dtype).reshape(1, 1024, 7, 7)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        if self.cover_all:
            self.gy = numpy.random.uniform(
                -1, 1, (1, 1024, 1, 1)).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (1, 1024, 1, 1)).astype(self.dtype)
        self.check_backward_options = {'eps': 2.0 ** -8}

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        print('-------------:', x_data.shape, y_grad.shape)
        func = avg_pooling_2d.AvgPooling2DMKLDNN(7.0, stride=1.0, pad=0, cover_all=self.cover_all)
        x = chainer.Variable(x_data)
        y = func(x)
        y = identity.Identity()(*y)
        # for iy, igy in six.moves.zip(y, y_grad):
        #    print('11-----------------:', igy.shape)
        #    iy.grad = igy
        y.grad = y_grad[0]
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

test = TestMaxPooling2D()
test.setUp()
# test.test_forward_cpu()
# test.test_forward_cpu_wide()
test.test_backward_cpu()
