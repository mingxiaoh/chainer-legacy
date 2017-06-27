import unittest

import numpy
# from mkldnn.api.support import *
from mkldnn.mdarray import mdarray
import mkldnn.api.memory as m
from mkldnn.chainer.runtime import Engine
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64]
}))
class TestSum(unittest.TestCase):
    def setUp(self):
        self.e = Engine()
        self.x = numpy.random.uniform(
            -1, 1, (2, self.channel, 3, 2)).astype(self.dtype)
        self.y = numpy.random.uniform(
            -1, 1, (2, self.channel, 3, 2)).astype(self.dtype)
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}

    # @staticmethod
    def test_check1(self):
        x = mdarray(self.x, m.memory.nchw, self.e)
        z = x + self.y
        # print(type(z))
        z_expect = self.x + self.y
        # print(type(z_expect))
        testing.assert_allclose(z_expect, z, **self.check_forward_optionss)

    # @staticmethod
    def test_check2(self):
        y = mdarray(self.y, m.memory.nchw, self.e)
        x = mdarray(self.x, m.memory.nchw, self.e)
        z = x + y
        # print(self.x * 1)
        z_expect = self.x + self.y
        # print(z * 1)
        testing.assert_allclose(z_expect, z, **self.check_forward_optionss)

    def test_check3(self):
        y = mdarray(self.y * 1, m.memory.nchw, self.e)
        x = mdarray(self.x * 1, m.memory.nchw, self.e)
        z = x + y
        # print(self.x * 1)
        z_expect = self.x + self.y
        # print(z * 1)
        testing.assert_allclose(z_expect, z, **self.check_forward_optionss)

    def test_check4(self):
        y = mdarray(self.y * 1, m.memory.nchw, self.e)
        x = mdarray(self.x * 1, m.memory.nchw, self.e)
        x += y
        z = x
        # print(type(z))
        # print(self.x * 1)
        z_expect = self.x + self.y
        # print(z * 1)
        testing.assert_allclose(z_expect, z, **self.check_forward_optionss)

testing.run_module(__name__, __file__)
