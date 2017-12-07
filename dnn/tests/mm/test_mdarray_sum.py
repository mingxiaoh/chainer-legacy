import dnn._dnn
import numpy
from chainer import testing
from dnn._dnn import Relu_Py_F32, mdarray

print('mdarray sum [larg shape routine]')
print('shape (256, 384, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((256, 384, 13, 13), dtype=numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = mdarray(x)
my = Relu_Py_F32.Forward(mx)

testing.assert_allclose(my.sum((0,2,3)), y.sum((0,2,3)))
print('pass ...\n')


print('mdarray sum [small shape routine]')
print('shape (39, 32, 13, 13) along (0, 2, 3)')
x = numpy.ndarray((39, 32, 13, 13), dtype=numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = mdarray(x)
my = Relu_Py_F32.Forward(mx)

testing.assert_allclose(my.sum((0,2,3)), y.sum((0,2,3)))
print('pass ...\n')


print('mdarray sum [numpy routine]')
print('shape (2, 2, 3, 3) along (0, 2, 3)')
x = numpy.ndarray((2, 2, 3, 3), dtype=numpy.float32)

mx = mdarray(x)

testing.assert_allclose(my.sum((0,2,3)), y.sum((0,2,3)))
print('pass ...\n')
