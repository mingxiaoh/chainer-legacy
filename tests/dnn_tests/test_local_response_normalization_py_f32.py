import sys
import unittest

import numpy
import six
import dnn._dnn
from dnn._dnn import lrn_param_t, LocalResponseNormalization_Py_F32

try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(2, 7, 1, 1), (2, 7, 3, 2), ],
}))
class TestLocalResponseNormalizationPyF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)
        self.pp = lrn_param_t()
        self.pp.n = 5
        self.pp.k = 2
        self.pp.alpha = 1e-4
        self.pp.beta = .75
        self.pp.algo_kind = dnn._dnn.lrn_param_t.lrn_across_channels
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x, pp):
        x_mdarray = dnn._dnn.mdarray(x)
        (y_act, ws) = LocalResponseNormalization_Py_F32.Forward(x_mdarray, pp)
        y_act = numpy.array(y_act, dtype=self.dtype)

        y_expect = numpy.zeros_like(self.x)
        for n, c, h, w in numpy.ndindex(self.x.shape):
            s = 0
            for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
                s += self.x[n, i, h, w] ** 2
            denom = (2 + 1e-4 * s) ** .75
            y_expect[n, c, h, w] = self.x[n, c, h, w] / denom

        numpy.testing.assert_allclose(
            y_expect, y_act, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.pp)

    def check_backward(self, x, gy, pp):
        x_mdarray = dnn._dnn.mdarray(x)
        gy_mdarray = dnn._dnn.mdarray(gy)
        (y_act, ws) = LocalResponseNormalization_Py_F32.Forward(x_mdarray, pp)
        gx_act = LocalResponseNormalization_Py_F32.Backward(
            x_mdarray, gy_mdarray, ws, pp)
        gx_act = numpy.array(gx_act, dtype=self.dtype)

        half_n = self.pp.n // 2
        x2 = numpy.square(x)
        sum_part = x2.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += x2[:, :-i]
            sum_part[:, :-i] += x2[:, i:]
        self.unit_scale = pp.k + pp.alpha * sum_part
        self.scale = self.unit_scale ** -pp.beta
        self.y = x_mdarray * self.scale

        summand = self.y * gy / self.unit_scale
        sum_p = summand.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_p[:, i:] += summand[:, :-i]
            sum_p[:, :-i] += summand[:, i:]

        gx_expect = gy * self.scale - 2 * pp.alpha * pp.beta * x * sum_p
        numpy.testing.assert_allclose(
            gx_expect, gx_act, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.pp)


testing.run_module(__name__, __file__)
