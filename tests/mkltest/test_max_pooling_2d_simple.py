import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.functions.math import identity
from mkldnn.chainer import max_pooling_2d

class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.cover_all = False
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        if self.cover_all:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 3, 2)).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 2, 2)).astype(self.dtype)
        self.check_backward_options = {'eps': 2.0 ** -8}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                         cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                if self.cover_all:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()],
                        [x[3:4, 0:2].max(), x[3:4, 1:3].max()]])
                else:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()]])
                testing.assert_allclose(expect, y_data[k, c])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0, cover_all=True)

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        func = max_pooling_2d.MaxPooling2DMKLDNN(3, stride=2, pad=1, cover_all=self.cover_all)
        x = chainer.Variable(x_data)
        y = func(x)
        y = identity.Identity()(*y)
        for iy, igy in six.moves.zip(y, y_grad):
            iy.grad = igy
        y[0].backward()


    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

test = TestMaxPooling2D()
test.setUp()
#test.test_forward_cpu()
#test.test_forward_cpu_wide()
test.test_backward_cpu()
