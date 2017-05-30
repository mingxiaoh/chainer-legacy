from mkldnn.api.support import *
from mkldnn.api import reorder as r
from mkldnn.api import memory as m
from mkldnn.chainer.runtime import Stream

import mkldnn
import numpy

def reorder_if_must(usr_m, expect, net_):
    if (usr_m.get_primitive_desc() != expect):
        reorded = m.memory(expect)
        net_.push_back(r.reorder(at(usr_m), reorded))
        return reorded
    else:
        return usr_m

def reuse_buffer(d, s):
    if isinstance(s, numpy.ndarray):
        s = numpy.ascontiguousarray(s)
        d.setbuffer(s)

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
    cache_f = {}
    cache_bd = {}
    cache_bw = {}

    cache = { 'f':cache_f, 'bd':cache_bd, 'bw':cache_bw }

    def __new__(cls, *args, **kwargs):
        pos = kwargs.pop('pos')
        assert isinstance(pos, tuple)

        cache = cls.cache[cls.cc_type]
        ret = cache.get(pos)

        if ret and isinstance(ret, cls)  and ret.match(*args, **kwargs):
            ret.new = False
        else:
            ret = super(ComputeComplex, cls).__new__(cls)
            print("Create new CC: ", ret)
            ret.new = True
            cache[pos] = ret

        return ret

    def __init__(self):
        if self.new:
            self.rank = -1
            self.fanout = -1
            self.dag_ = primitive_list()
            self._hint = None

    def execute_on(self, s = None):
        if s is None:
            # XXX: Refresh everytime
            s = Stream()

        s.submit(self.dag_)
        s.wait()
        return self.outputs

    def matching(self, inputs):
        raise NotImplementedError

    @property
    def hint(self):
        return self._hint
