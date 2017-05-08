import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
from chainer import mkld


@testing.parameterize(*testing.product({
    'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64]
}))
class TestMaxpool2(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(1, self.channel, 4, 4).astype('f'),
        self.gy = np.random.rand(1, self.channel, 4, 4).astype('f'),

    def tearDown(self):
        self.x = None
        self.gy = None

    def check_maxpool2(self):
        for _ in range(2):
            mkld.enable_max_pooling = True
            f = F.MaxPooling2D(3, stride=1, pad=1, use_cudnn=False)
            y = f.forward(self.x)
            gx = f.backward(self.x, self.gy)
            mkld.enable_max_pooling = False
            f = F.MaxPooling2D(3, stride=1, pad=1, use_cudnn=False)
            y_expect = f.forward_cpu(self.x)
            gx_expect = f.backward_cpu(self.x, self.gy)
            testing.assert_allclose(np.asarray(y), np.asarray(y_expect))
            testing.assert_allclose(np.asarray(gx), np.asarray(gx_expect))

    @condition.retry(3)
    def test_cpu(self):
        self.check_maxpool2()


testing.run_module(__name__, __file__)
