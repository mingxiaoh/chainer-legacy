import sys
import unittest

import numpy

import dnn._dnn
from dnn._dnn import Relu_Py_F32

try:
    import testing
except Exception as ex:
    print('*** testing directory is missing: %s' % ex)
    sys.exit(-1)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (224, 224)],
    'dtype': [numpy.float32, ],
}))
@testing.fix_random()
class TestReluPyF32(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y = numpy.maximum(self.x, 0, dtype=(self.x).dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gx = (self.x > 0) * self.gy

    def check_forward(self, x, y):
        mx = dnn._dnn.mdarray(x)
        x2 = numpy.array(mx)
        numpy.testing.assert_allclose(x, x2)
        my = Relu_Py_F32.Forward(mx)
        y2 = numpy.array(my)
        numpy.testing.assert_allclose(y, y2)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    def check_double_forward(self, x, y):
        mx = dnn._dnn.mdarray(x)
        x2 = numpy.array(mx)
        numpy.testing.assert_allclose(x, x2)
        my = Relu_Py_F32.Forward(mx)
        y2 = numpy.array(my)
        numpy.testing.assert_allclose(y, y2)
        my = Relu_Py_F32.Forward(my)
        y2 = numpy.array(my)
        numpy.testing.assert_allclose(y, y2)

    def test_double_forward_cpu(self):
        self.check_double_forward(self.x, self.y)

    def check_backward(self, x, gy, gx):
        mx = dnn._dnn.mdarray(x)
        mgy = dnn._dnn.mdarray(gy)
        mgx = Relu_Py_F32.Backward(mx, mgy)
        gx1 = numpy.array(mgx)
        numpy.testing.assert_allclose(gx1, gx)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.gx)


testing.run_module(__name__, __file__)
