import numpy
from mkldnn.api import memory as m
from mkldnn.chainer.runtime import Engine
from mkldnn.mdarray import mdarray


# Convert ndarray to mdarray
def array(obj, *args):
    if isinstance(obj, mdarray):
        return obj
    elif isinstance(obj, numpy.ndarray):
        # TODO: Do we automatically transfer?

        obj = numpy.ascontiguousarray(obj)
        return mdarray(obj, *args)
    else:
        raise NotImplementedError


def warray(w):
    fmt = None
    if w.ndim == 1:
        fmt = m.memory.x
    elif w.ndim == 2:
        fmt = m.memory.oi
    elif w.ndim == 4:
        fmt = m.memory.oihw
    else:
        raise NotImplementedError

    if w.dtype != numpy.float32:
        raise NotImplementedError

    e = Engine()
    return mdarray(w, fmt, e)
