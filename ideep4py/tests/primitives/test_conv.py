import numpy
import ideep4py._ideep4py

# from ideep4py._ideep4py import convParam, conv_test
from ideep4py._ideep4py import convParam, convolution2D

x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py._ideep4py.mdarray(x)

w = numpy.ndarray(shape=(32, 32, 3, 3), dtype=numpy.float32, order='C')
w = ideep4py._ideep4py.mdarray(w)

b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
b = ideep4py._ideep4py.mdarray(b)

cp = convParam()
cp.src_d1 = 1
cp.src_d2 = 32
cp.src_d3 = 224
cp.src_d4 = 224
cp.weights_d1 = 32
cp.weights_d2 = 32
cp.weights_d3 = 3
cp.weights_d4 = 3
cp.bias_d1 = 32
cp.dst_d1 = 1
cp.dst_d2 = 32
cp.dst_d3 = 224
cp.dst_d4 = 224
cp.sy = cp.sx = 1
cp.pad_lh = cp.pad_lw = cp.pad_rh = cp.pad_rw = 1
cp.with_bias = True

print("fwd")
y = convolution2D.Forward(x, w, b, cp)
print("==============")
y = convolution2D.Forward(x, w, b, cp)
print("==============")
y = convolution2D.Forward(y, w, b, cp)

print("==============")
print("bwd data")
x = convolution2D.BackwardData(w, y, cp)
print("==============")
x = convolution2D.BackwardData(w, y, cp)
print("==============")
x = convolution2D.BackwardData(w, x, cp)

print("==============")
print("bwd weights")
weights = convolution2D.BackwardWeights(x, y, cp)
print("weights=", type(weights))
print("len=", len(weights))
print("gw.shape=", weights[0].shape)
if cp.with_bias:
    print("gb.shape=", weights[1].shape)
print("==============")
x = numpy.ndarray(shape=(1, 32, 224, 224), dtype=numpy.float32, order='C')
x = ideep4py._ideep4py.mdarray(x)
weights = convolution2D.BackwardWeights(x, y, cp)
# print("type=", type(x))
# print("shape=", y.shape)
# print("size=", y.size)
# print("ndim=", y.ndim)
# print("dtype=", y.dtype)
