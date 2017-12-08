import sys
import unittest

import numpy
import six

import dnn._dnn
from dnn._dnn import Pooling2D_Py_F32, pooling_param_t

try:
    import testing
    from testing import condition
    from testing.conv import col2im_cpu
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'channel': [1, 2, 4, 8, 10, 16, 24, 32, 64],
    'bs': [0, 1, 2, 4, 6, 8, 10, 16, 24, 32, 64],
    'stride': [2, ],
}))
class TestPooling2DPyF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (self.bs, self.channel, 4, 3)).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (self.bs, self.channel, 2, 2)).astype(self.dtype)

        self.pp = pooling_param_t()
        self.pp.src_d1, self.pp.src_d2 = self.bs, self.channel
        self.pp.src_d3, self.pp.src_d4 = 4, 3
        self.pp.dst_d1, self.pp.dst_d2 = self.gy.shape[0], self.gy.shape[1]
        self.pp.dst_d3, self.pp.dst_d4 = self.gy.shape[2], self.gy.shape[3]
        self.pp.kh, self.pp.kw = 3, 3
        self.pp.sy, self.pp.sx = self.stride, self.stride
        self.pp.pad_lh, self.pp.pad_lw = 1, 1
        self.pp.pad_rh, self.pp.pad_rw = 1, 1
        self.pp.algo_kind = pooling_param_t.pooling_avg_include_padding

        self.check_forward_options = {'atol': 1e-5, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-5, 'rtol': 1e-4}

    def check_forward(self, x, pp):
        x_mdarray = dnn._dnn.mdarray(x)
        (y_act,) = Pooling2D_Py_F32.Forward(x_mdarray, pp)
        y_act = numpy.array(y_act, dtype=self.dtype)

        for k in six.moves.range(self.bs):
            for c in six.moves.range(self.channel):
                x = self.x[k, c]
                expect = numpy.array([
                    [x[0:2, 0:2].sum(), x[0:2, 1:3].sum()],
                    [x[1:4, 0:2].sum(), x[1:4, 1:3].sum()]]) / 9
                numpy.testing.assert_allclose(
                    expect, y_act[k, c], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.pp)

    def check_backward(self, x, gy, pp):
        # self.shape[2:]
        h, w = 4, 3
        gcol = numpy.tile(gy[:, :, None, None],
                          (1, 1, 3, 3, 1, 1))
        gx_expect = col2im_cpu(gcol, 2, 2, 1, 1, h, w)
        gx_expect /= 3 * 3
        gy_mdarray = dnn._dnn.mdarray(gy)
        gx_act = Pooling2D_Py_F32.Backward(gy_mdarray, None, pp)
        gx_act = numpy.array(gx_act, dtype=self.dtype)

        numpy.testing.assert_allclose(
            gx_expect, gx_act, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.pp)


testing.run_module(__name__, __file__)
