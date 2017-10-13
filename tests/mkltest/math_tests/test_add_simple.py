import numpy as np
from chainer.mkld import mdarray
import mkldnn.api.memory as m
from chainer.mkld import Engine
from chainer import variable

e = Engine()

x = np.arange(16).reshape(2, 2, 2, 2).astype(np.float32)
x = x+1
y = np.arange(16).reshape(2, 2, 2, 2).astype(np.float32)


x = mdarray(x, m.memory.nchw, e)
y = mdarray(y, m.memory.nchw, e)

x = variable.Variable(x)
y = variable.Variable(y)

z = x + y

print(z.data * 1)
