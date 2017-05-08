import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
from chainer import mkld


@testing.parameterize(*testing.product({
    'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64]
}))
class TestMaxpool3(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.c = self.channel
        self.h = 4
        self.w = 3
        self.x = np.random.rand(self.n, self.c, self.h, self.w).astype('f')
        self.x = 2 * self.x / self.x.size - 1

    def tearDown(self):
        self.x = None

    def check_maxpool3(self):
        mkld.enable_max_pooling = True
        y1 = F.max_pooling_2d(self.x, 3, stride=2, pad=1, cover_all=False)
        mkld.enable_max_pooling = False
        y2 = F.max_pooling_2d(self.x, 3, stride=2, pad=1, cover_all=False)
        # self.assertTrue(np.array_equal(y1.data, y2.data))
        testing.assert_allclose(y1.data, y2.data)

        mkld.enable_max_pooling = True
        y1 = F.max_pooling_2d(self.x, 3, stride=2, pad=1, cover_all=True)
        mkld.enable_max_pooling = False
        y2 = F.max_pooling_2d(self.x, 3, stride=2, pad=1, cover_all=True)
        # self.assertTrue(np.array_equal(y1.data, y2.data))
        testing.assert_allclose(y1.data, y2.data)

    @condition.retry(3)
    def test_cpu(self):
        self.check_maxpool3()


testing.run_module(__name__, __file__)
