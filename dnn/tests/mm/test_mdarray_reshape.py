import numpy
import dnn._dnn

x1 = numpy.ndarray(shape=(3,2,6,6), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x1)
print(type(x))
y = x.reshape(len(x), -1)
x[0,0,0,0] = 3.333
assert(x[0,0,0,0] == y[0,0])

y = x.reshape((len(x), -1))
x[0,0,0,0] = 4.4444
assert(x[0,0,0,0] == y[0,0])
