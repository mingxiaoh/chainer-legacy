import numpy
from chainer import testing  # NOQA
from chainer import utils  # NOQA
import dnn._dnn

x1 = numpy.ndarray(shape=(2, 2), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x1)
print(x1)
y = x1 > 0
print(y)
x *= y
# numpy.asarray(x)
# numpy.copyto(x1, x)
# numpy.copyto(x, x1)
