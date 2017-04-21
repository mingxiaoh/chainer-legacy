from mkldnn.support import *
from mkldnn import reorder as r
from mkldnn import memory as m
from mkldnn.runtime import Stream

import mkldnn

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
    else:
        return mkldnn.mdarray(obj, *args)

class ComputeComplex(object):
    """MKLDNN Compute Complex.

    """
    def __init__(self):
        self.net_ = primitive_list()
        self._hint = None

    def execute_on(self, s = None):
        if s is None:
            # XXX: Refresh everytime
            s = Stream()

        s.submit(self.net_)
        s.wait()
        return self.outputs

    @property
    def hint(self):
        return self._hint
