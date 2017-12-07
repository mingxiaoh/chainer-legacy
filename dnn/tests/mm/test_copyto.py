import numpy
#from chainer import testing
#from chainer import utils
import dnn._dnn

x1 = numpy.ndarray(shape=(2,16,2,2), dtype=numpy.float32, order='C')
x2 = numpy.ndarray(shape=(2,16,2,2), dtype=numpy.float32, order='C')
mx1 = dnn._dnn.mdarray(x1)
mx2 = dnn._dnn.mdarray(x2)
numpy.copyto(x2, x1)
dnn._dnn.basic_copyto(mx2, mx1)
t = numpy.asarray(mx2)
numpy.allclose(t, x2, 1e-5, 1e-4, True)


x1 = numpy.ndarray(shape=(2,16,2,2), dtype=numpy.float32, order='C')
x2 = numpy.ndarray(shape=(2,16,2,2), dtype=numpy.float32, order='C')
mx2 = dnn._dnn.mdarray(x2)
numpy.copyto(x2, x1)
dnn._dnn.basic_copyto(mx2, x1)
t = numpy.asarray(mx2)
numpy.allclose(t, x2, 1e-5, 1e-4, True)

