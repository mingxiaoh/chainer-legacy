import numpy
from chainer import testing
from chainer import utils
import dnn._dnn

x1 = numpy.ndarray(shape=(2,2,2,2), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x1)
print(x)
print("ndims=", x.ndim)
print("shape=", x.shape)
print("size=", x.size)
print("dtype=", x.dtype)
print("is_mdarry=", x.is_mdarray)

x1 += x
x += x
x2 = numpy.array(x)
testing.assert_allclose(x1, x2)


x1 = numpy.ones(shape=(2,2,2,2), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x1)
y = x + x1
y2 = numpy.array(y)
testing.assert_allclose(y2, x1+x1)

y = x * x1
y2 = numpy.array(y)
testing.assert_allclose(y2, x1*x1)

print(type(y))
y.ravel()
