import numpy
import dnn._dnn

from dnn._dnn import pooling_param_t, Pooling2D_Py_F32

x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x)

pp = pooling_param_t()
pp.src_d1 = 1
pp.src_d2 = 32
pp.src_d3 = 224
pp.src_d4 = 224
pp.dst_d1 = 1
pp.dst_d2 = 32
pp.dst_d3 = 224
pp.dst_d4 = 224
pp.kh = pp.kw = 3
pp.sy = pp.sx = 1
pp.pad_lh = pp.pad_lw = pp.pad_rh = pp.pad_rw = 1
pp.algo_kind = dnn._dnn.pooling_param_t.pooling_avg

print("fwd")
y = Pooling2D_Py_F32.Forward(x, pp)
print("==============")
y = Pooling2D_Py_F32.Forward(x, pp)
print("==============")

pp.algo_kind = dnn._dnn.pooling_param_t.pooling_max
(y, ws) = Pooling2D_Py_F32.Forward(x, pp)
print("==============")
(y, ws) = Pooling2D_Py_F32.Forward(x, pp)

print ("y.shape=", y.shape)
print ("ws.shape=", ws.shape)

print("==============")
#print("bwd data")
#x = Convolution2D_Py_F32.BackwardData(w, y, cp)
#print("==============")
#x = Convolution2D_Py_F32.BackwardData(w, y, cp)
#print("==============")
#x = Convolution2D_Py_F32.BackwardData(w, x, cp)
#
#print("==============")
#print("bwd weights")
#weights = Convolution2D_Py_F32.BackwardWeights(x, y, cp)
#print("weights=", type(weights))
#print("len=", len(weights))
#print("gw.shape=", weights[0].shape)
#if cp.with_bias:
#    print("gb.shape=", weights[1].shape)
#print("==============")
#x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
#x = dnn._dnn.mdarray(x)
#weights = Convolution2D_Py_F32.BackwardWeights(x, y, cp)
#print("type=", type(x))
#print("shape=", y.shape)
#print("size=", y.size)
#print("ndim=", y.ndim)
#print("dtype=", y.dtype)
