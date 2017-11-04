import numpy
from chainer import testing
import dnn._dnn

x1 = numpy.ndarray(shape=(2,2,2,2), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x1)
print(type(x))
print("ndims=", x.ndim)
print("shape=", x.shape)
print("size=", x.size)
print("dtype=", x.dtype)
print("is_mdarry=", x.is_mdarray)
# print("will crash ", x.crash)

#y = numpy.zeros_like(x)
#print(y)

x2 = numpy.array(x)
testing.assert_allclose(x1, x2)
