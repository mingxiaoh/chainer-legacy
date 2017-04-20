from mkldnn.support import primitive_list
from mkldnn.runtime import Stream

class ComputeComplex(object):
    """MKLDNN Compute Complex.

    """
    def __init__(self):
        self.net_ = primitive_list()
        self.hint_ = None

    def execute_on(self, s = Stream()):
        s.submit(self.net_)
        s.wait()
        return self.output

    @property
    def hint(self):
        return self.hint
