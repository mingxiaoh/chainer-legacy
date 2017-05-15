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
from mkldnn.chainer import avg_pooling_2d


class TestAveragePooling2D(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 2, 2)).astype(self.dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float32}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float32, 'atol': 5e-4, 'rtol': 5e-3}

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                avg_pooling_2d.AvgPooling2DMKLDNN(3, 2, 1, False),
                x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

test = TestAveragePooling2D()
test.setUp()
test.test_backward_cpu()
