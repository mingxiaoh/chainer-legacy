import numpy
import dnn._dnn

x = numpy.ndarray(shape=(2,2,2,2), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x)
print(type(x))
print("ndims=", x.ndim)
print("shape=", x.shape)
print("size=", x.size)
print("dtype=", x.dtype)
print("is_mdarry=", x.is_mdarray)
# print("will crash ", x.crash)
