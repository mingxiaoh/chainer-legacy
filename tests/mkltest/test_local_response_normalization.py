import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
# from chainer.testing import attr
from chainer.testing import condition
from mkldnn.chainer import lrn


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    # 'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64]
}))
class TestLocalResponseNormalization(unittest.TestCase):

    def setUp(self):
        # n = 5
        # k = 1
        # alpha = 1e-4
        # beta = .75
        # self.x = numpy.random.uniform(
        #     -1, 1, (2, self.channel, 3, 2)).astype(self.dtype)
        # self.gy = numpy.random.uniform(
        #     -1, 1, (2, self.channel, 3, 2)).astype(self.dtype)
        self.x = numpy.random.uniform(
            -1, 1, (2, 7, 3, 2)).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (2, 7, 3, 2)).astype(self.dtype)
        self.check_forward_optionss = {}
        self.check_backward_optionss = {}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_optionss = {'atol': 5e-3, 'rtol': 5e-3}
        # self.lrn = lrn.LrnMKLDNN(n, k, alpha, beta)
        # self.lrn = functions.LocalResponseNormalization(n, k, alpha, beta)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        # y = self.lrn.forward_cpu(x_data)
        y = functions.local_response_normalization(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)
        # Naive implementation
        y_expect = numpy.zeros_like(self.x)
        with chainer.using_config('use_mkldnn', 'never'):
            self.assertFalse(chainer.should_use_mkldnn('==always'))
            self.assertFalse(chainer.should_use_mkldnn('>=auto'))
            y_expect = functions.local_response_normalization(x)
            y_expect = cuda.to_cpu(y_expect.data)
        # for n, c, h, w in numpy.ndindex(self.x.shape):
        #     s = 0
        #     for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
        #         s += self.x[n, i, h, w] ** 2
        #     denom = (2 + 1e-4 * s) ** .75
        #     y_expect[n, c, h, w] = self.x[n, c, h, w] / denom
        self.assertTrue(chainer.should_use_cudnn('>=auto'))
        testing.assert_allclose(
            y_expect, y_data, **self.check_forward_optionss)

    def check_forward1(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.local_response_normalization(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)
        # Naive implementation
        y_expect = numpy.zeros_like(self.x)
        for n, c, h, w in numpy.ndindex(self.x.shape):
            s = 0
            for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
                s += self.x[n, i, h, w] ** 2
            denom = (2 + 1e-4 * s) ** .75
            y_expect[n, c, h, w] = self.x[n, c, h, w] / denom
        testing.assert_allclose(
            y_expect, y_data, **self.check_forward_optionss)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu1(self):
        self.check_forward1(self.x)
    # @attr.gpu
    # @condition.retry(3)
    # def test_forward_gpu(self):
    #     self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lrn.LrnMKLDNN(), x_data, y_grad,
            eps=1, dtype=self.dtype, **self.check_backward_optionss)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    # @attr.gpu
    # @condition.retry(3)
    # def test_backward_gpu(self):
    #     self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
