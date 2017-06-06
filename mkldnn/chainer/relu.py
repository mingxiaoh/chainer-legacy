from chainer import function
from chainer.utils import type_check

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.api.support import *
import mkldnn.api.memory as m
import mkldnn.api.relu_forward as relu_forward
import mkldnn.api.relu_backward as relu_backward
from mkldnn.mdarray import *

def create_backward_desc(d_creator, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]

    return d_creator(*inputs_d)

class ReLUForward(ComputeComplex):
    cc_type = 'f'

    def _create_cc(self, x, e=Engine()):
        #fix bug local variable 'fmt' referenced before assignment
        fmt = m.memory.nchw
        if x.ndim == 2:
            fmt = m.memory.nc
        elif x.ndim == 4:
            fmt = m.memory.nchw

        x = array(x, fmt, e)
        mem_pd = x.memory.get_primitive_desc()

        cc_d = relu_forward.desc(forward, mem_pd.desc(), 0.0)
        cc_pd = relu_forward.primitive_desc(cc_d, e)

        y = mdarray(cc_pd.dst_primitive_desc())

        self.x = x
        self.dag_.push_back(relu_forward.relu_forward(cc_pd,
                at(x.memory), y.memory))

        self._hint = cc_pd
        self.outputs = y,

    def match(self, inputs, *args):
        # TODO: refine it
        x = inputs[0]
        return self.x.shape == x.shape

    def __init__(self, inputs, pos = (0, 0), e=Engine()):
        x = inputs[0]
        # assert isinstance(x, mdarray)
        super(ReLUForward, self).__init__()

        if self.new:
            self._create_cc(x, e)

class ReLUBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, grad_outputs, hint, pos = (0, 0), e=Engine()):
        x = inputs[0]
        gy = grad_outputs[0]

        fmt = m.memory.nchw
        if x.ndim == 2:
            fmt = m.memory.nc

        x = array(x, fmt, e)
        gy = array(gy, fmt, e)

        super(ReLUBackward, self).__init__()

        if self.new:
            self._create_cc(x, gy, hint, e)

    def match(self, inputs, grad_outpus, *args):
        # TODO: refine it
        x = inputs[0]
        gy = grad_outpus[0]
        return self.x.shape == x.shape

    def _create_cc(self, x, gy, hint, e = Engine()):
        diff_pd = gy.memory.get_primitive_desc()
        mem_pd = x.memory.get_primitive_desc()

        cc_d = relu_backward.desc(diff_pd.desc(), mem_pd.desc(), 0.0)
        cc_pd = relu_backward.primitive_desc(cc_d, e, hint)

        gx = mdarray(cc_pd.diff_src_primitive_desc())

        self.dag_.push_back(relu_backward.relu_backward(cc_pd,
            at(x.memory), at(gy.memory), gx.memory))

        self.x = x
        self.gy = gy
        self.outputs = gx,

class ReLUMKLDNN(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        cc = ReLUForward(x,
                pos=(self.rank, self.fanout))

        self.hint = cc.hint

        y, = cc.execute_on()

        return y,

    def backward(self, x, gy):
        cc = ReLUBackward(x, gy, self.hint,
                pos=(self.rank, self.fanout))

        gx, = cc.execute_on()

        return gx,
