import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
import time
from mkldnn import switch


@testing.parameterize(*testing.product({
    'dtype': [np.float32],
    'channel': [2, 4, 8, 16, 24]
}))
class TestBatchNormalizationValidation(unittest.TestCase):
    def setUp(self):
        self.eps = 2e-5;
        self.decay = 0.9;
        self.x = np.random.uniform(-1, 1, (32, self.channel, 224, 224)).astype(self.dtype)
        self.mean = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.var = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gamma = np.random.uniform(-1, 1, (self.channel )).astype(self.dtype)
        self.beta = np.random.uniform(-1, 1, (self.channel)).astype(self.dtype)
        self.gy = np.random.uniform(-1, 1, (2, self.channel, 3, 2)).astype(self.dtype)

        self.check_forward_optionss = {}
        self.check_backward_optionss = {}
        if self.channel >= 8:
            self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_optionss = {'atol': 5e-3, 'rtol': 5e-3}

    def check_forward(self, x):
        # !use_global_stats
        switch.enable_batch_normalization = True
        y1 = F.batch_normalization(
            x, self.gamma, self.beta, self.eps, self.mean,
            self.var, self.decay, False)
        start = time.time()
        y1 = F.batch_normalization(
            x, self.gamma, self.beta, self.eps, self.mean,
            self.var, self.decay, False)
        end = time.time()
        mkldnn_timing = end - start
        self.assertEqual(y1[0].dtype, self.dtype)
        switch.enable_batch_normalization = False
        start = time.time()
        y1_expect = F.batch_normalization(
            x, self.gamma, self.beta, self.eps, self.mean,
            self.var, self.decay, False)
        end = time.time()
        cpu_timing = end - start
        print("mkldnn timing:", mkldnn_timing)
        print("cpu timing:", cpu_timing)

        testing.assert_allclose(y1_expect[0].data, y1[0].data,
                                **self.check_forward_optionss)

        ## use_global_stats
        #switch.enable_batch_normalization = True
        ## dry run
        #y2 = F.fixed_batch_normalization(
        #    x, self.gamma, self.beta, self.mean,
        #    self.var, self.eps, False)
        #start = time.time()
        #y2 = F.fixed_batch_normalization(
        #    x, self.gamma, self.beta, self.mean,
        #    self.var, self.eps, False)
        #end = time.time()
        #mkldnn_timing = end - start
        #self.assertEqual(y2[0].dtype, self.dtype)
        #switch.enable_batch_normalization = False
        #start = time.time()
        #y2_expect = F.fixed_batch_normalization(
        #    x, self.gamma, self.beta, self.mean,
        #    self.var, self.eps, False)
        #end = time.time()
        #cpu_timing = end - start

        #print("mkldnn timing:", mkldnn_timing)
        #print("cpu timing:", cpu_timing)

        #testing.assert_allclose(y2_expect[0].data, y2[0].data,
        #                        **self.check_forward_optionss)


    def check_backward(self, x_data, y_grad):
        return
        #switch.enable_batch_normalization = True
        #gx = self.batch_normalization.backward((x_data,), (y_grad,))
        #switch.enable_batch_normalization = False
        #gx_expect = self.batch_normalization.backward((x_data,), (y_grad,))
        #testing.assert_allclose(gx_expect[0], gx[0], **self.check_backward_optionss)

    @condition.retry(3)
    def test_cpu(self):
        self.check_forward(self.x)
        self.check_backward(self.x, self.gy)

    @testing.attr.xeon
    @condition.retry(3)
    def test_xeon_cpu(self):
        print("test xeon")
        pass

    @testing.attr.xeon_phi
    @condition.retry(3)
    def test_xeon_phi_cpu(self):
        print("test xeon phi")
        pass


testing.run_module(__name__, __file__)
