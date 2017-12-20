import numpy
import ideep4py._ideep4py

# from dnn._dnn import conv_param_t, conv_test
from ideep4py._ideep4py import IntVector, MdarrayVector, Concat_Py_F32

x1 = numpy.ndarray(shape=(1, 16, 224, 224), dtype=numpy.float32, order='C')
x2 = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x3 = numpy.ndarray(shape=(1, 64, 224, 224), dtype=numpy.float32, order='C')
inputs = (x1, x2, x3)
sizes = numpy.array(
    [v.shape[1] for v in inputs[:-1]]
).cumsum()
print("sizes=", sizes)
print("type=", type(sizes))

x1 = ideep4py._ideep4py.mdarray(x1)
x2 = ideep4py._ideep4py.mdarray(x2)
x3 = ideep4py._ideep4py.mdarray(x3)

xs = MdarrayVector()
xs.push_back(x1)
xs.push_back(x2)
xs.push_back(x3)

print("fwd")
y = Concat_Py_F32.Forward(xs, 1)
print("==============")
y = Concat_Py_F32.Forward(xs, 1)
print("y.shape=", y.shape)

print("backward")

int_sizes = IntVector()

for i in sizes:
    print("i=", i)
    int_sizes.push_back(i)

gxs = Concat_Py_F32.Backward(y, int_sizes, 1)

for gx in gxs:
    print("gx.type=", type(gx))
    print("gx.shape=", gx.shape)
print("after backward")
