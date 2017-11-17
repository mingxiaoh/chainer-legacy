import numpy
import dnn._dnn

#from dnn._dnn import conv_param_t, conv_test
from dnn._dnn import MdarrayVector, Concat_Py_F32

x1 = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x1 = dnn._dnn.mdarray(x1)

x2 = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x2 = dnn._dnn.mdarray(x2)

x3 = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x3 = dnn._dnn.mdarray(x3)

xs = MdarrayVector()
xs.push_back(x1)
xs.push_back(x2)
xs.push_back(x3)

print("fwd")
y = Concat_Py_F32.Forward(xs, 1)
print("==============")
y = Concat_Py_F32.Forward(xs, 1)
xs.push_back(y)
print("==============")
y = Concat_Py_F32.Forward(xs, 1)
print("y.shape=", y.shape)
