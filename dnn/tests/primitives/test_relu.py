import numpy
from chainer import testing
import dnn._dnn
from dnn._dnn import Relu_Py_F32

# x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
y = numpy.maximum(x, 0, dtype=x.dtype)

mx = dnn._dnn.mdarray(x)
x2 = numpy.array(mx)
testing.assert_allclose(x, x2)

print("Relu fwd")
my = Relu_Py_F32.Forward(mx)
y2 = numpy.array(my)
testing.assert_allclose(y, y2)
my = Relu_Py_F32.Forward(my)
y2 = numpy.array(my)
testing.assert_allclose(y, y2)


# Test backward
print("Relu bwd")
x = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gy = numpy.random.uniform(-1, 1, (1, 32, 224, 224)).astype(numpy.float32)
gx = (x > 0) * gy


mx = dnn._dnn.mdarray(x)
mgy = dnn._dnn.mdarray(gy)
mgx = Relu_Py_F32.Backward(mx, mgy)


gx1 = numpy.array(mgx)
testing.assert_allclose(gx1, gx)
