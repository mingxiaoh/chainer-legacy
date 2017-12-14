import numpy
import dnn._dnn

x1 = numpy.random.uniform(-1, 1, (3,16,2,4)).astype(numpy.float32)
x2 = numpy.random.uniform(-1, 1, (3,16,2,4)).astype(numpy.float32)
x3 = numpy.random.uniform(-1, 1, (3,16,2,4)).astype(numpy.float32)
x4 = numpy.random.uniform(-1, 1, (3,16,2,4)).astype(numpy.float32)
mx1 = dnn._dnn.mdarray(x1)
mx2 = dnn._dnn.mdarray(x2)
mx3 = dnn._dnn.mdarray(x3)
mx4 = dnn._dnn.mdarray(x4)

x = x1 + x2 + x3 + x4
mx = dnn._dnn.basic_acc_sum((mx1, mx2, mx3, mx4))
# mx = numpy.asarray(mx)
res = numpy.allclose(mx, x, 1e-5, 1e-4, True)
assert(res == True)
