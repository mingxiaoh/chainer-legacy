from mkldnn.support import primitive_list
from mkldnn.support import at
from mkldnn import reorder as r
from mkldnn import memory as m
from mkldnn.runtime import Stream

def reorder_if_must(usr_m, expect, net_):
    if (usr_m.get_primitive_desc() != expect):
        reorded = m.memory(expect)
        net_.push_back(r.reorder(at(usr_m), reorded))
        return reorded
    else:
        return usr_m

class ComputeComplex(object):
    """MKLDNN Compute Complex.

    """
    def __init__(self):
        self.net_ = primitive_list()
        self._hint = None

    def execute_on(self, s = Stream()):
        s.submit(self.net_)
        s.wait()
        return self.output

    @property
    def hint(self):
        return self._hint
