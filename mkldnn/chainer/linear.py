from chainer import function
from chainer.utils import type_check

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.api.support import *
import mkldnn.api.memory as m
import mkldnn.api.inner_product_forward as ip_forward
import mkldnn.api.inner_product_backward_data as ip_backdata
import mkldnn.api.inner_product_backward_weights as ip_backweights
from mkldnn.mdarray import *

from mkldnn.api.inner_product_forward import linear_f_op
from mkldnn.api.inner_product_backward_data import linear_bd_op
from mkldnn.api.inner_product_backward_weights import linear_bw_op
from mkldnn.api.inner_product_backward_weights import linear_bwb_op

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

def create_forward_desc(d_creator, o_expect, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]
    x_m = inputs_d[0]
    W_m = inputs_d[1]
    if len(inputs_d) == 3:
        b_m = inputs_d[2]
        return d_creator(forward, x_m, W_m, b_m, o_expect)
    else:
        return d_creator(forward, x_m, W_m, o_expect)

def create_backward_desc(d_creator, *inputs):
    inputs_d = [m.desc(v.shape, m.memory.f32, m.memory.any)
            for v in inputs if v is not None]

    return d_creator(*inputs_d)

class LinearForward(ComputeComplex):
    cc_type = 'f'

    def _create_cc(self, x, W, b, e=Engine()):
        y_d = m.desc((x.shape[0], W.shape[0]), m.memory.f32, m.memory.any)

        # Create primitive_desc from any
        cc_d = create_forward_desc(ip_forward.desc, y_d, x, W, b)
        cc_pd = ip_forward.primitive_desc(cc_d, e)

        # Transform inputs
        self.x = array(x, m.memory.nc, e)
        self.W = array(W, m.memory.oi, e)

        if b is not None:
            self.b = array(b, m.memory.x, e)
            y = linear_f_op(cc_pd, self.x, self.W, self.b, self.dag_)
        else:
            y = linear_f_op(cc_pd, self.x, self.W, self.dag_)

        # Prepare output
        # y = mdarray(cc_pd.dst_primitive_desc())

        # dag = self.dag_

        # # Reorder if must
        # x_m = reorder_if_must(self.x.memory,
        #         cc_pd.src_primitive_desc(), dag)
        # W_m = reorder_if_must(self.W.memory,
        #         cc_pd.weights_primitive_desc(), dag)

        # if b is None:
        #     dag.push_back(ip_forward.inner_product_forward(cc_pd,
        #         at(x_m), at(W_m), y.memory))
        # else:
        #     dag.push_back(ip_forward.inner_product_forward(cc_pd,
        #         at(x_m), at(W_m), at(self.b.memory), y.memory))

        # self.x_m = x_m
        # self.W_m = W_m
        self._hint = cc_pd
        self.outputs = y,

    def _reuse_cc(self, x, W, b, e = Engine()):
        reuse_buffer(self.x, x)
        reuse_buffer(self.W, W)
        if b is not None:
            reuse_buffer(self.b, b)

    def match(self, inputs):
        return len(inputs) == self.argc

    def __init__(self, inputs, pos = (0, 0), e=Engine()):
        super(LinearForward, self).__init__()
        x = _as_mat(inputs[0])
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None
        self.argc = len(inputs)

        if self.new:
            self._create_cc(x, W, b, e)
        else:
            self._reuse_cc(x, W, b, e)

class LinearBackwardData(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, grad_outputs, hint, pos = (0, 0), e=Engine()):
        super(LinearBackwardData, self).__init__()
        W = inputs[1]
        gy = grad_outputs[0]
        self.argc = len(inputs)

        if self.new:
            self.target_ndim = len(inputs[0].shape)
            x = _as_mat(inputs[0])
            self._create_cc(x, W, gy, hint, e)
        else:
            self._reuse_cc(W, gy)

    def match(self, inputs, *args):
        return len(inputs) == self.argc

    def _create_cc(self, x, W, gy, hint, e = Engine()):
        # Create primitive descriptor
        cc_d = create_backward_desc(ip_backdata.desc, x, W, gy)
        cc_pd = ip_backdata.primitive_desc(cc_d, e, hint)

        # Transform inputs
        self.W = array(W, m.memory.oi, e)
        self.gy = array(gy, m.memory.nc, e)

        gx = linear_bd_op(cc_pd, self.gy, self.W, self.dag_, self.target_ndim)

        # # Prepare output mdarray
        # gx = mdarray(cc_pd.diff_src_primitive_desc())

        # dag = self.dag_

        # # Reorder if must
        # gy_m = reorder_if_must(self.gy.memory, cc_pd.diff_dst_primitive_desc(), dag)
        # W_m = reorder_if_must(self.W.memory, cc_pd.weights_primitive_desc(), dag)

        # dag.push_back(ip_backdata.inner_product_backward_data(cc_pd,
        #     at(gy_m), at(W_m), gx.memory))

        # self.gy_m = gy_m
        # self.W_m = W_m

        self.outputs = gx,

    def _reuse_cc(self, W, gy):
        reuse_buffer(self.W, W)
        reuse_buffer(self.gy, gy)

class LinearBackwardWeighs(ComputeComplex):
    cc_type = 'bw'

    def _create_cc(self, x, W, b, gy, hint, e):
        cc_d = create_backward_desc(ip_backweights.desc, x, W, b, gy)
        cc_pd = ip_backweights.primitive_desc(cc_d, e, hint)

        # Transfer inputs to mdarray
        self.x = array(x, m.memory.nc, e)
        self.gy = array(gy, m.memory.nc, e)

        if b is None:
            gW = linear_bw_op(cc_pd, self.x, self.gy, self.dag_)
        else:
            gW = linear_bwb_op(cc_pd, self.x, self.gy, self.dag_)
            gb = gW.extra

        # Prepare outputs mdarray
        # gW = mdarray(cc_pd.diff_weights_primitive_desc())
        # if b is not None:
        #     gb = mdarray(cc_pd.diff_bias_primitive_desc())
        #     self.has_b = True
        # else:
        #     self.has_b = False

        # dag = self.dag_

        # # Reorder if must
        # gy_m = reorder_if_must(self.gy.memory, cc_pd.diff_dst_primitive_desc(), dag)
        # x_m = reorder_if_must(self.x.memory, cc_pd.src_primitive_desc(), dag)

        # if b is not None:
        #     dag.push_back(ip_backweights.inner_product_backward_weights(cc_pd,
        #         at(x_m), at(self.gy.memory), gW.memory, gb.memory))
        # else:
        #     dag.push_back(ip_backweights.inner_product_backward_weights(cc_pd,
        #         at(x_m), at(self.gy.memory), gW.memory))

        # self.x_m = x_m

        if b is None:
            self.outputs = gW,
        else:
            self.outputs = gW, gb

    def _reuse_cc(self, x, gy):
        reuse_buffer(self.x, x)
        reuse_buffer(self.gy, gy)

    def match(self, inputs, *args):
        return len(inputs) == self.argc

    def __init__(self, inputs, grad_outputs, hint, pos, e=Engine()):
        super(LinearBackwardWeighs, self).__init__()
        x = _as_mat(inputs[0])
        gy = grad_outputs[0]
        self.argc = len(inputs)

        if self.new:
            W = inputs[1]
            b = inputs[2] if self.argc == 3 else None

            self._create_cc(x, W, b, gy, hint, e)
        else:
            self._reuse_cc(x, gy)

class LinearFunctionMKLDNN(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        cc = LinearForward(inputs,
                pos=(self.rank, self.fanout))
        self.hint = cc.hint

        y, = cc.execute_on()

        return y,

    def backward(self, inputs, grad_outputs):
        cc_data = LinearBackwardData(inputs, grad_outputs, self.hint,
                pos=(self.rank, self.fanout))
        cc_weight = LinearBackwardWeighs(inputs, grad_outputs, self.hint,
                pos=(self.rank, self.fanout))

        gx = cc_data.execute_on()
        gW_b = cc_weight.execute_on()

        return gx + gW_b

def linearMKLDNN(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
     .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_B, s_1, \
            s_2, ..., s_n)`-shaped float array. Its first dimension
            :math:`(s_B)` is assumed to be the *minibatch dimension*. The
            other dimensions are treated as concatenated one dimension whose
            size must be :math:`(s_1 * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_B, M)`.

    .. seealso:: :class:`~chainer.links.Linear`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype('f')
        >>> W = np.random.uniform(0, 1, (5, 4)).astype('f')
        >>> b = np.random.uniform(0, 1, (5,)).astype('f')
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if b is None:
        return LinearFunctionMKLDNN()(x, W)
    else:
        return LinearFunctionMKLDNN()(x, W, b)
