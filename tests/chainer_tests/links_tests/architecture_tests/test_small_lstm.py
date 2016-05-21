import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import links
from chainer.testing import attr


class TestSmallLSTM(unittest.TestCase):

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (1, 128)).astype(numpy.float32)
                   for _ in six.moves.range(10)]
        self.l = links.SmallLSTM()

    def check_forward(self, xp):
        xs = [chainer.Variable(xp.asarray(x)) for x in self.xs]
        self.l(xs)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)
