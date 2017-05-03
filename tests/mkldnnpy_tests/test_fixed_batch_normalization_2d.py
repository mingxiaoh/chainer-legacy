import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
import time
from mkldnn import switch


@testing.parameterize(*testing.product({
    'dtype': [np.float32],
    'channel': [2 * 32 * 32, 4 * 32 * 32, 8 * 32 * 32, 16 * 32 * 32, 24 * 32 * 32]
}))
class TestBatchNormalizationValidation(unittest.TestCase):
    def setUp(self):
        self.eps = 2e-5;
        self.decay = 0.9;
        self.x = np.random.uniform(-1, 1, (32, self.channel )).astype(self.dtype)
        self.mean = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.var = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gamma = np.random.uniform(-1, 1, (self.channel )).astype(self.dtype)
        self.beta = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gy = np.random.uniform(-1, 1, (32, self.channel)).astype(self.dtype)

        self.check_forward_optionss = {}
        self.check_backward_optionss = {}
        if self.channel >= 8:
            self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_optionss = {'atol': 5e-3, 'rtol': 5e-3}

    def check_forward(self, x):
        switch.enable_batch_normalization = True
        start = time.time()
        y = F.fixed_batch_normalization(
            x, self.gamma, self.beta, self.mean,
            self.var, self.eps, False)
        end = time.time()
        mkldnn_timing = end - start
        self.assertEqual(y[0].dtype, self.dtype)
        switch.enable_batch_normalization = False
        start = time.time()
        y_expect = F.fixed_batch_normalization(
            x, self.gamma, self.beta, self.mean,
            self.var, self.eps, False)
        end = time.time()
        cpu_timing = end - start

        print("mkldnn timing:", mkldnn_timing)
        print("cpu timing:", cpu_timing)

        testing.assert_allclose(y_expect.data, y.data,
                                **self.check_forward_optionss)


    def check_backward(self, x_data, y_grad):
        pass

    @condition.retry(3)
    def test_cpu(self):
        self.check_forward(self.x)
        self.check_backward(self.x, self.gy)


testing.run_module(__name__, __file__)
