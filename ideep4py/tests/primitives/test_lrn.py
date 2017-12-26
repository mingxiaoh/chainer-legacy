import numpy
import ideep4py._ideep4py

from ideep4py._ideep4py import lrnParam, localResponseNormalization

x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py._ideep4py.mdarray(x)

pp = lrnParam()
pp.n = 5
pp.k = 2
pp.alpha = 1e-4
pp.beta = .75
pp.algo_kind = ideep4py._ideep4py.lrnParam.lrn_across_channels

print("fwd")
(y, ws) = localResponseNormalization.Forward(x, pp)
print("==============")
(y, ws) = localResponseNormalization.Forward(x, pp)

# print ("y =", y)
print("y.shape=", y.shape)
print("ws.shape=", ws.shape)
print("ws.dtype=", ws.dtype)

print("==============")
print("bwd")
gx = localResponseNormalization.Backward(x, y, ws, pp)
print("==============")
gx = localResponseNormalization.Backward(x, y, ws, pp)
print("gx.shape=", gx.shape)
print("===== Finish backward=========")
