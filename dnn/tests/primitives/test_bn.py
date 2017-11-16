import numpy
import dnn._dnn

from dnn._dnn import batchNormalizationF32

src = numpy.arange(3 * 2 * 2 * 2, dtype=numpy.float32)
src = src.reshape((3, 2, 2, 2))
src = dnn._dnn.mdarray(src)

gamma = numpy.ones(2, dtype=numpy.float32)
beta = numpy.zeros(2, dtype=numpy.float32)
w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
w = dnn._dnn.mdarray(w)

eps = 2e-5

print("fwd")
y = batchNormalizationF32.Forward(src, w, None, None, eps)
print(y)
print(-y[0])
print(-y[1])
print(-y[2])
print("==============")
y = batchNormalizationF32.Forward(src, w, None, None, eps)
print(y)
print(-y[0])
print(-y[1])
print(-y[2])
print("==============")
mean = y[1]
var = y[2]
y = batchNormalizationF32.Forward(src, w, mean, var, eps)
print(y)
print(-y[0])
print("==============")
