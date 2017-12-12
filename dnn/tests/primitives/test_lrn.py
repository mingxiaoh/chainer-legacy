import numpy
import dnn._dnn

from dnn._dnn import lrn_param_t, LocalResponseNormalization_Py_F32

x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x)

pp = lrn_param_t()
pp.n = 5
pp.k = 2
pp.alpha = 1e-4
pp.beta = .75
pp.algo_kind = dnn._dnn.lrn_param_t.lrn_across_channels

print("fwd")
(y, ws) = LocalResponseNormalization_Py_F32.Forward(x, pp)
print("==============")
(y, ws) = LocalResponseNormalization_Py_F32.Forward(x, pp)

# print ("y =", y)
print ("y.shape=", y.shape)
print ("ws.shape=", ws.shape)
print ("ws.dtype=", ws.dtype)

print("==============")
print("bwd")
gx = LocalResponseNormalization_Py_F32.Backward(x, y, ws, pp)
print("==============")
gx = LocalResponseNormalization_Py_F32.Backward(x, y, ws, pp)
print("gx.shape=", gx.shape)
print("===== Finish backward=========")
