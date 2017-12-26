import numpy
import ideep4py
# from ideep4py import linearParam, linear_test
from ideep4py import linearParam, linear

x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)

w = numpy.ndarray(shape=(32, 32), dtype=numpy.float32, order='C')
print("ndarray w", w.shape)
w = ideep4py.mdarray(w)
print("w.dim", w.shape)
b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
b = ideep4py.mdarray(b)

lp = linearParam()
lp.src_d1 = 1
lp.src_d2 = 32
lp.src_ndims = 2
lp.bias_d1 = 32
lp.with_bias = True
print("===============2 dims============")
print("fwd")
y = linear.Forward(x, w, b, lp)
print("================")
y = linear.Forward(x, w, b, lp)
print("================")
y = linear.Forward(x, w, b, lp)

print("================")
print("bwd data")
x = linear.BackwardData(w, y, lp)

weights = linear.BackwardWeights(x, y, lp)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
if lp.with_bias:
    print("gb.shape = ", weights[1].shape)
print("================")
x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
weights = linear.BackwardWeights(x, y, lp)
print("==========4 dims=================")

lp.src_d1 = 1
lp.src_d2 = 32
lp.src_d3 = 224
lp.src_d4 = 224

lp.src_ndims = 4
lp.bias_d1 = 32
lp.with_bias = True
x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)

w = numpy.ndarray(shape=(32, 32, 224, 224), dtype=numpy.float32, order='C')
print("ndarray w", w.shape)
w = ideep4py.mdarray(w)
print("w.dim", w.shape)
b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
b = ideep4py.mdarray(b)

print("fwd")
y = linear.Forward(x, w, b, lp)
print("================")
y = linear.Forward(x, w, b, lp)
print("================")
y = linear.Forward(x, w, b, lp)

print("================")
print("bwd data")
x = linear.BackwardData(w, y, lp)

weights = linear.BackwardWeights(x, y, lp)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
if lp.with_bias:
    print("gb.shape = ", weights[1].shape)
print("================")
x = numpy.ndarray(shape=(1, 32), dtype=numpy.float32, order='C')
x = ideep4py.mdarray(x)
