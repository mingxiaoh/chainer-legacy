from chainer import function
from chainer.utils import type_check

from mkldnn.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.support import *
import mkldnn.memory as m
import mkldnn.relu_forward as relu_forward
import mkldnn.relu_backward as relu_backdata
from mkldnn.mdarray import *

def create_backward_desc(d_creator, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]

    return d_creator(*inputs_d)

class ReLUForward(ComputeComplex):
    cc_type = 'f'

    def _create_cc(self, x, e=Engine()):
        mem_pd = x.memory.get_primitive_desc()

        cc_d = relu_forward.desc(forward, mem_pd.desc(), 0.0)
        cc_pd = relu_forward.primitive_desc(cc_d, e)

        y = mdarray(cc_pd.dst_primitive_desc())

        self.dag_.push_back(relu_forward.relu_forward(cc_pd,
                at(x.memory), y.memory))

        self._hint = cc_pd
        self.outputs = y,

    def match(self, *args):
        return True

    def __init__(self, x, pos = (0, 0), e=Engine()):
        assert isinstance(x, mdarray)
        super(ReLUForward, self).__init__()

        if self.new:
            self._create_cc(x, e)

class ReLUBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, x, gy, hint, pos = (0, 0), e=Engine()):
        assert isinstance(gy, mdarray)
        super(ReLUBackward, self).__init__()

        if self.new:
            self._create_cc(x, gy, hint, e)

    def match(self, *args):
        return True

    def _create_cc(self, x, gy, hint, e = Engine()):
        diff_pd = gy.memory.get_primitive_desc()
        mem_pd = x.memory.get_primitive_desc()

        cc_d = relu_backdata.desc(diff_pd, mem_pd, 0.0)
        cc_pd = relu_backdata.primitive_desc(cc_d, e, hint)

        gx = mdarray(cc_pd.diff_src_primitive_desc())

        self.dag_.push_back(relu_backdata.relu_backward(cc_pd,
            at(x.memory), at(gy.memory), gx.memory))

        self.outputs = gx,

class ReLUMKLDNN(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        cc = ReluForward(x,
                pos=(self.rank, self.fanout))

        self.hint = cc.hint

        y, = cc.execute_on()

        return y,

    def backward(self, x, gy):
        cc = LinearBackwardData(x, gy, self.hint,
                pos=(self.rank, self.fanout))

        gx = cc_data.execute_on()

        return gx,
