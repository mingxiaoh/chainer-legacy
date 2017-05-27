import numpy as np
from mkldnn.api.support import *
from mkldnn.mdarray import *
import mkldnn.api.memory as m
from mkldnn.chainer.runtime import Engine

e = Engine()
#x = np.ndarray(shape=(2,2,2,2), dtype=np.float32, order='C')
#y = np.random.rand(2,2,2,2).astype(np.float32)

x = np.arange(16).reshape(2,2,2,2).astype(np.float32)
x = x+1
y = np.arange(16).reshape(2,2,2,2).astype(np.float32)
print(x)
print(y)

z = x + y
print(z)

#x = np.random.rand(2,2,2,2).astype(np.float32)
x = mdarray(x, m.memory.nchw, e)
z = x + y
print(type(x))

#y = np.ndarray(shape=(2,2,2,2), dtype=np.float32, order='C')
#y = np.random.rand(2,2,2,2).astype(np.float32)
y = mdarray(y, m.memory.nchw, e)
print(x*1)

z = x + y
print(z * 1)

