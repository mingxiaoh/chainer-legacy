from chainer import function
from chainer.utils import conv
from chainer.utils import type_check

from mkldnn.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.support import *
import mkldnn.memory as m
import mkldnn.convolution_forward as conv_forward
import mkldnn.convolution_backward_data as conv_backdata
import mkldnn.convolution_backward_weights as conv_backweights
from mkldnn.mdarray import *

def create_forward_desc(d_creator, o_expect, inputs, geometry):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]

    return d_creator(forward, convolution_direct,
            *inputs_d, o_expect, *geometry, zero)

def create_backward_desc(d_creator, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]

    return d_creator(*inputs_d)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class ConvolutionForward(ComputeComplex):
    def __init__(self, x, W, b = None, stride=1, pad=0, e=Engine()):
        super(ConvolutionForward, self).__init__()
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        out_c, _, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph)
        assert out_h > 0, 'Height in the output should be positive.'
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw)
        assert out_w > 0, 'Width in the output should be positive.'

        y_d = m.desc((n, out_c, out_h, out_w), m.memory.f32, m.memory.any)

        geometry = (_pair(stride), _pair(pad), _pair(pad))

        # Create primitive_desc from any
        cc_d = create_forward_desc(conv_forward.desc, y_d, (x, W, b), geometry)
        cc_pd = conv_forward.primitive_desc(cc_d, e)

        # Transform inputs
        self.x = array(x, m.memory.nchw, e)
        self.W = array(W, m.memory.oihw, e)
        if b is not None:
            self.b = array(b, m.memory.x, e)

        if b is None:
            y = conv_f_op(cc_pd, self.x, self.W, self.dag_)
        else:
            y = conv_f_op(cc_pd, self.x, self.W, self.b, self.dag_)

        self._hint = cc_pd
        self.outputs = y,

class ConvolutionBackwardData(ComputeComplex):
    def __init__(self, x, W, dummy, gy, hint, e=Engine()):

        # Create primitive descriptor
        cc_d = create_backward_desc(conv_backdata.desc, x, W, gy)
        cc_pd = conv_backdata.primitive_desc(cc_d, e, hint)

        # Transform inputs
        self.gy = array(gy, m.memory.nc, e)
        self.W = array(W, m.memory.oi, e)

        gx = conv_bd_op(cc_pd, self.gy, self.W, self.dag_)

        self.outputs = gx,

class ConvolutionBackwardWeighs(ComputeComplex):
    def __init__(self, x, W, b, gy, hint, e=Engine()):
        super(ConvolutionBackwardWeighs, self).__init__()

        cc_d = create_backward_desc(conv_backweights.desc, x, W, b, gy)
        cc_pd = conv_backweights.primitive_desc(cc_d, e, hint)

        self.gy = array(gy, m.memory.nc, e)
        self.x = array(x, m.memory.nc, e)

        # Prepare outputs mdarray
        gW = mdarray(cc_pd.diff_weights_primitive_desc())
        if b is not None:
            gb = mdarray(cc_pd.diff_bias_primitive_desc())

        if b is not None:
            # XXX: This is ugly, will use swig to do something about it
            # ideal: gW, gb = conv_bwb_op(cc_pd, self.x, self.gy, self.dag_)
            gW = conv_bwb_op(cc_pd, self.x, self.gy, self.dag_)
            gb = gW.extra
        else:
            gW = conv_bw_op(cc_pd, self.x, self.gy)

        if b is not None:
            self.outputs = gW, gb
        else:
            self.outputs = gW,


class Convolution2DFunctionMKLDNN(function.Function):

    def __init__(self, stride=1, pad=0, cover_all=False, deterministic=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all
        self.deterministic = deterministic

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        cc = ConvolutionForward(*inputs, stride = (self.sy, self.sx),
                pad = (self.ph, self.pw))
        self.hint = cc.hint

        y, = cc.execute_on()

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        if len(inputs) == 2:
            inputs += None,

        cc_data = ConvolutionBackwardData(*inputs, *grad_outputs, self.hint)
        cc_weight = ConvolutionBackwardWeighs(*inputs, *grad_outputs, self.hint)

        gx = cc_data.execute_on()
        gW_b = cc_weight.execute_on()

        return gx + gW_b
