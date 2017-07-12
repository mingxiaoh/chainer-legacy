import unittest

# import mock
import numpy

import chainer
# from chainer import cuda
from chainer import functions
from chainer.functions.activation import relu
from chainer import gradient_check
from chainer import testing
# from chainer.testing import attr
# from chainer.testing import condition

from mkldnn.chainer.runtime import Engine
import mkldnn.api.memory as m
import mkldnn.chainer.fanout
# import mkldnn.mdarray
from mkldnn.mdarray import mdarray


@testing.parameterize(*testing.product({
    'shape': [(3, 2)],
    'dtype': [numpy.float32],
}))
class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        # fanout.clear()
        mkldnn.chainer.fanout.FanoutRecorder.clear()
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.1 < x[i] < 0.1:
                x[i] = 0.5

        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        try:
            self.x.setbuffer(x)
            self.gy.setbuffer(gy)
        except AttributeError:
            self.x = mkldnn.mdarray(x, m.memory.nc, Engine())
            self.gy = mkldnn.mdarray(gy, m.memory.nc, Engine())

        self.check_backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.relu(x)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] = 0

        testing.assert_allclose(
            expected, y.data)

#    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
                relu.ReLUMKLDNN(), x_data, y_grad,
                **self.check_backward_options)

#    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

testing.run_module(__name__, __file__)
