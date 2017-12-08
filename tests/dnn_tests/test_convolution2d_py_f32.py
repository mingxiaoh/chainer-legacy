import sys
import unittest
import numpy
import dnn._dnn
from dnn._dnn import conv_param_t, Convolution2D_Py_F32

try:
    import testing
    from testing import condition
    from testing.conv import im2col_cpu
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, ],
    'cover_all': [False, True],
    'channel': [1, 2, 4, 8, 10, ],
    'bs': [1, 2, 4, 8, 10, 16, 32, 64, ],
    'with_bias': [True, ],
}))
@testing.fix_random()
class Convolution2DPyF32(unittest.TestCase):

    def setUp(self):
        self.x_shape = (self.bs, self.channel, 224, 224)
        self.w_shape = (self.channel, self.channel, 3, 3)
        self.b_shape = self.channel

        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.x = dnn._dnn.mdarray(self.x)
        self.w = numpy.random.uniform(-1, 1, self.w_shape).astype(self.dtype)
        self.w = dnn._dnn.mdarray(self.w)
        self.b = numpy.random.uniform(-1, 1, self.b_shape).astype(self.dtype)
        self.b = dnn._dnn.mdarray(self.b)

        self.cp = conv_param_t()
        self.cp.src_d1 = self.x_shape[0]
        self.cp.src_d2 = self.x_shape[1]
        self.cp.src_d3 = self.x_shape[2]
        self.cp.src_d4 = self.x_shape[3]
        self.cp.weights_d1 = self.w_shape[0]
        self.cp.weights_d2 = self.w_shape[1]
        self.cp.weights_d3 = self.w_shape[2]
        self.cp.weights_d4 = self.w_shape[3]
        self.cp.bias_d1 = self.x_shape[1]
        self.cp.dst_d1 = self.x_shape[0]
        self.cp.dst_d2 = self.x_shape[1]
        self.cp.dst_d3 = self.x_shape[2]
        self.cp.dst_d4 = self.x_shape[3]
        self.cp.sy = self.cp.sx = 1
        self.cp.pad_lh = self.cp.pad_lw = self.cp.pad_rh = self.cp.pad_rw = 1
        self.cp.with_bias = self.with_bias

        stride = 1
        pad = 1
        dilate = 1
        self.sy, self.sx = stride, stride
        self.ph, self.pw = pad, pad
        self.n = self.x_shape[0]
        self.outc = self.w_shape[0]
        self.outh = self.x_shape[2]
        self.outw = self.x_shape[3]
        self.cover_all = self.cover_all
        self.dy, self.dx = dilate, dilate

        self.gy = numpy.random.uniform(
             -1, 1,
             (self.n, self.outc, self.outh, self.outw)).astype(self.dtype)
        self.gy = dnn._dnn.mdarray(self.gy)

        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x, w, b, cp):

        if cp.with_bias:
            y_act = Convolution2D_Py_F32.Forward(x, w, b, cp)
        else:
            y_act = Convolution2D_Py_F32.Forward(x, w, None, cp)
        y_act = numpy.array(y_act, dtype=self.dtype)

        x = numpy.array(x, dtype=self.dtype)
        w = numpy.array(w, dtype=self.dtype)
        b = numpy.array(b, dtype=self.dtype)
        kh, kw = w.shape[2:]
        col = im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(
            col, w, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        y_expect = numpy.rollaxis(y, 3, 1)
        numpy.testing.assert_allclose(
            y_act, y_expect, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.w, self.b, self.cp)

    def check_backward_weights(self, x, w, b, cp, gy):
        gW_act, gB_act = Convolution2D_Py_F32.BackwardWeights(x, gy, cp)
        gW_act = numpy.array(gW_act, dtype=self.dtype)

        x = numpy.array(x, dtype=self.dtype)
        w = numpy.array(w, dtype=self.dtype)
        b = numpy.array(b, dtype=self.dtype)
        gy = numpy.array(gy, dtype=self.dtype)
        kh, kw = w.shape[2:]
        col = im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)

        gW_expect = numpy.tensordot(
            gy, col, ((0, 2, 3), (0, 4, 5))).astype(self.dtype, copy=False)
        numpy.testing.assert_allclose(
            gW_act, gW_expect, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu_weights(self):
        print("test_backward_cpu")
        self.check_backward_weights(self.x, self.w, self.b, self.cp, self.gy)


testing.run_module(__name__, __file__)
