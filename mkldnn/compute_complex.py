from mkldpy.support import primitive_vector

class ComputeComplex(object):
    """MKLDNN Compute Complex.

    """
    def __init__(self):
        self.net_ = primitive_vector()
        self.hint_ = None

    def execute_on(self, s):
        s.submit(self.net_)
        s.wait()
        return self.output

    @property
    def hint(self):
        return self.hint
