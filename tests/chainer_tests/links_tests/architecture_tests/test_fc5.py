import unittest

import numpy

import chainer
from chainer import cuda
from chainer import links
from chainer.testing import attr


class TestFC5(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 512)).astype(numpy.float32)
        self.l = links.FC5()

    def check_forward(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        self.l(x)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)
