import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from mkldnn.runtime import Engine
from mkldnn.mdarray import *

@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float32],
}))
class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = mdarray(numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
                memory.nc, Engine())
        for i in numpy.ndindex(self.shape):
            if -0.1 < self.x[i] < 0.1:
                self.x[i] = 0.5
        self.gy = mdarray(numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
                memory.nc, Engine())
        self.check_backward_options = {}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        y = functions.relu(x)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] = 0

        testing.assert_allclose(
            expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
                functions.ReLUMKLDNN(), x_data, y_grad,
                **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

testing.run_module(__name__, __file__)
