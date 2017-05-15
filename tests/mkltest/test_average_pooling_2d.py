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


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32,],
}))
class TestAveragePooling2D(unittest.TestCase):

    def setUp(self):
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

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.average_pooling_2d(x, 3, stride=2, pad=1)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                expect = numpy.array([
                    [x[0:2, 0:2].sum(), x[0:2, 1:3].sum()],
                    [x[1:4, 0:2].sum(), x[1:4, 1:3].sum()]]) / 9
                testing.assert_allclose(
                    expect, y_data[k, c], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                avg_pooling_2d.AvgPooling2DMKLDNN(3, 2, 1, False),
                x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')
