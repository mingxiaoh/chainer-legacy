from mkldnn.support import *
from mkldnn import reorder as r
from mkldnn import memory as m
from mkldnn.runtime import Stream

import mkldnn
import numpy

def reorder_if_must(usr_m, expect, net_):
    if (usr_m.get_primitive_desc() != expect):
        reorded = m.memory(expect)
        net_.push_back(r.reorder(at(usr_m), reorded))
        return reorded
    else:
        return usr_m

# XXX: move this file to another location
def array(obj, *args):
    if isinstance(obj, mkldnn.mdarray):
        return obj
    elif isinstance(obj, numpy.ndarray):
        # TODO: Do we automatically transfer?

        obj = numpy.ascontiguousarray(obj)
        return mkldnn.mdarray(obj, *args)
    else:
        raise NotImplementedError

class ComputeComplex(object):
    """MKLDNN Compute Complex.

    """
    def __new__(cls, *args, rank=0):
        if hasattr(cls, 'cache'):
            ret = cls.cache.get(rank)

        if ret and ret.matching(*args):
            ret.new = False
        else:
            ret = super(ComputeComplex, cls).__new__(cls)
            ret.new = True
            cls.cache[rank] = ret

        return ret

    def __init__(self):
        if self.new:
            self.dag_ = primitive_list()
            self._hint = None

    def execute_on(self, s = None):
        if s is None:
            # XXX: Refresh everytime
            s = Stream()

        s.submit(self.dag_)
        s.wait()
        return self.outputs

    @property
    def hint(self):
        return self._hint
