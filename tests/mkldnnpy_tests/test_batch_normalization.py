import numpy as np
import unittest
from chainer.functions.normalization import batch_normalization
import chainer.testing as testing
import chainer.testing.condition as condition
import time
from chainer import mkld


@testing.parameterize(*testing.product({
    'dtype': [np.float32],
    'channel': [2, 4, 8, 16, 24]
}))
class TestBatchNormalizationValidation(unittest.TestCase):
    def setUp(self):
        self.eps = 2e-5
        self.decay = 0.9
        self.x = np.random.uniform(-1, 1, (32, self.channel, 224, 224)).astype(self.dtype)
        self.mean = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.var = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gamma = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.beta = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gy = np.random.uniform(-1, 1, (32, self.channel, 224, 224)).astype(self.dtype)

        self.check_forward_optionss = {}
        self.check_backward_optionss = {}
        self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-4}
        self.check_backward_optionss = {'atol': 1e-3, 'rtol': 1e-3}

    def check_forward(self, x):
        mkld.enable_batch_normalization = True
        self.func1 = batch_normalization.BatchNormalizationFunction(
            self.eps, self.mean, self.var, self.decay, False)
        start = time.time()
        y = self.func1.forward((x, self.gamma, self.beta))
        end = time.time()
        self.assertEqual(y[0].dtype, self.dtype)
        print("mkldnn timing:", end - start)

        mkld.enable_batch_normalization = False
        self.func2 = batch_normalization.BatchNormalizationFunction(
            self.eps, self.mean, self.var, self.decay, False)
        start = time.time()
        y_expect = self.func2.forward((x, self.gamma, self.beta))
        end = time.time()
        print("numpy timing:", end - start)

        testing.assert_allclose(self.func1.running_mean, self.func2.running_mean,
                                **self.check_forward_optionss)
        testing.assert_allclose(self.func1.running_var, self.func2.running_var,
                                **self.check_forward_optionss)
        testing.assert_allclose(y_expect[0], y[0],
                                **self.check_forward_optionss)

    def check_backward(self, x_data, y_grad):
        mkld.enable_batch_normalization = True
        start = time.time()
        (gx, ggamma, gbeta) = self.func1.backward(
            (x_data, self.gamma, self.beta), (y_grad,))
        end = time.time()
        print("mkldnn timing:", end - start)

        mkld.enable_batch_normalization = False
        start = time.time()
        (gx_expect, ggamma_expect, gbeta_expect) = self.func2.backward(
            (x_data, self.gamma, self.beta), (y_grad,))
        end = time.time()
        print("numpy timing:", end - start)

        testing.assert_allclose(gx_expect, gx, **self.check_backward_optionss)
        testing.assert_allclose(ggamma_expect, ggamma, **self.check_backward_optionss)
        testing.assert_allclose(gbeta_expect, gbeta, **self.check_backward_optionss)

    @condition.retry(3)
    def test_cpu(self):
        self.check_forward(self.x)
        self.check_backward(self.x, self.gy)


testing.run_module(__name__, __file__)
