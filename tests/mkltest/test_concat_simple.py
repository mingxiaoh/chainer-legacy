import unittest

import numpy

import chainer
from mkldnn import config as mkld_config
# from chainer import cuda
from chainer import functions
from chainer import testing
# from chainer.testing import attr


class TestConcat(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.shape = (2, 7, 3, 3)
        self.axis = 1
        self.slices = [
                [
                    slice(None), slice(None, 2)], [slice(None), slice(2, 5)], [slice(None), slice(5, None)], ]
        self.y = numpy.arange(
            numpy.prod(self.shape), dtype=self.dtype).reshape(self.shape)
        print('----------y.shape--------', self.y.shape)
        self.xs = [self.y[s] for s in self.slices]
        print('------------xs.len--------', len(self.xs))

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(y_data, y.data, atol=0, rtol=0)
        self.assertIsInstance(y.data.shape, tuple)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.y, axis=self.axis)

    def check_backward(self, xs_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        y.grad = y.data
        with mkld_config.using_config('gx_opt', False):
            y.backward()

        for x in xs:
            testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        # print('ingore backward test')
        self.check_backward(self.xs, axis=self.axis)

test = TestConcat()
test.setUp()
# test.test_forward_cpu()
test.test_backward_cpu()
