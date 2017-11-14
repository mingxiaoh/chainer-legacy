import numpy
import dnn._dnn
#from dnn._dnn import linear_param_t, linear_test
from dnn._dnn import linear_param_t, Linear_Py_F32

x = numpy.ndarray(shape=(1,32,224,224), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x)

w =  numpy.ndarray(shape=(32,32,3,3), dtype=numpy.float32, order='C')
w = dnn._dnn.mdarray(x)

b = numpy.ndarray(shape=(32,), dtype=numpy.float32, order='C')
b = dnn._dnn.mdarray(b)

lp = linear_param_t()
lp.src_d1 = 1
lp.src_d2 = 32
lp.src_d3 = 224
lp.src_d4 = 224
lp.weights_d1 = 32
lp.weights_d2 = 32
lp.weights_d3 = 3
lp.weights_d4 = 3
lp.bias_d1 = 32
lp.dst_d1 = 1
lp.dst_d2 = 32
lp.dst_d3 = 224
lp.dst_d4 = 224
lp.with_bias = True
print("fwd")
y = Linear_Py_F32.Forward(x, w, b, lp)
print("================")
y = Linear_Py_F32.Forward(x, w, b, lp)
print("================")
y = Linear_Py_F32.Forward(x, w, b, lp)

print("================")
print("bwd data")
x = Linear_Py_F32.BackwardData(w, y, lp)

weights = Linear_Py_F32.BackwardWeights(x, y, lp)
print("weights= ", type(weights))
print("len", len(weights))
print("gw.shape", weights[0].shape)
if lp.with_bias:
    print("gb.shape = ", weights[1].shape)
print("================")
x = numpy.ndarray(shape=(1,32,224, 224), dtype=numpy.float32, order='C')
x = dnn._dnn.mdarray(x)
weights = Linear_Py_F32.BackwardWeights(x,y,lp)

