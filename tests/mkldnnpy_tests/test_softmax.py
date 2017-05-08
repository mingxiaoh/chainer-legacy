import numpy as np
import unittest
import chainer.testing as testing
import chainer.testing.condition as condition
from chainer import functions as F
from chainer import mkld


@testing.parameterize(*testing.product({
    'dims': [(2, 3), (1, 6), (3, 2), (3, 4), (4, 3), (2, 6), (6, 2)]
}))
class TestSoftmax(unittest.TestCase):
    def check_softmax(self):
        x_2d = np.random.rand(self.dims[0], self.dims[1]).astype('f')
        mkld.enable_softmax = True
        y_2d = F.softmax(x_2d, use_cudnn=False)
        mkld.enable_softmax = False
        y_2d_expect = F.softmax(x_2d, use_cudnn=False)
        testing.assert_allclose(y_2d.data, y_2d_expect.data)

    @condition.retry(3)
    def test_cpu(self):
        self.check_softmax()


testing.run_module(__name__, __file__)
