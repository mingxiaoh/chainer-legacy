import numpy as np
from mkldnn.support import *
import mkldnn.memory as memory
from mkldnn.mdarray import *

e = engine(engine.cpu, 0)

a = mdarray(memory.dims([2,2,3,4]), memory.memory.f32, memory.memory.nchw, e)
b = np.ndarray([2,2,3,4], dtype=np.float32)

fill = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48], dtype=np.float32)

# Sequence can be sure to initialize a inplace
a_np = np.frombuffer(a, dtype=np.float32)
a_np[...] = fill

a_np = np.array(a)

c = a + b
c1 = a + b

assert (c==c1).any()
