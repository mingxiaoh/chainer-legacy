import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import dnn._dnn
from dnn._dnn import mdarray


#@testing.parameterize(*testing.product({
#    'shape': [(3, 2),],
#    'dtype': [numpy.float32,],
#}))
@testing.fix_random()
class TestReLU(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 2)
        self.dtype = numpy.float32
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.1 < self.x) & (self.x < 0.1)] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.relu(x)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = self.x.copy()
        expected[expected < 0] = 0

        testing.assert_allclose(expected, y.data)

    #def test_forward_cpu(self):
    #    self.check_forward(mdarray(self.x))

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                functions.relu, x_data, y_grad, dtype=None,
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(mdarray(self.x), mdarray(self.gy), 'never')

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            x = functions.relu(x)
            return x * x

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad, dtype=numpy.float32,
                **self.check_double_backward_options)

    #def test_double_backward_cpu(self):
    #    self.check_double_backward(mdarray(self.x), mdarray(self.gy), mdarray(self.ggx))



testing.run_module(__name__, __file__)
