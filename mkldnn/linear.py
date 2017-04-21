from chainer import function
from chainer.utils import type_check

from mkldnn.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.support import *
import mkldnn.memory as m
import mkldnn.inner_product_forward as ip_forward
import mkldnn.inner_product_backward_data as ip_backdata
import mkldnn.inner_product_backward_weights as ip_backweights
from mkldnn.mdarray import *

class LinearForward(ComputeComplex):
    def __init__(self, x, W, b = None, e=Engine()):
        super(LinearForward, self).__init__()

        y_expect = m.desc(m.dims((x.shape[0], W.shape[0])), m.memory.f32, m.memory.any)
        x_expect = m.desc(m.dims(x.shape), m.memory.f32, m.memory.any)
        W_expect = m.desc(m.dims(W.shape), m.memory.f32, m.memory.any)
        if b is not None:
            b_expect = m.desc(m.dims(b.shape), m.memory.f32, m.memory.any)
            cc_d = ip_forward.desc(forward, x_expect, W_expect, b_expect, y_expect)
        else:
            cc_d = ip_forward.desc(forward, x_expect, W_expect, y_expect)

        cc_pd = ip_forward.primitive_desc(cc_d, e)

        self.x = mdarray(x, m.memory.nc, e)
        self.W = mdarray(W, m.memory.oi, e)
        self.b = mdarray(b, m.memory.x, e) if b is not None else None
        y = mdarray(cc_pd.dst_primitive_desc())

        net = self.net_

        x_m = reorder_if_must(self.x.memory, cc_pd.src_primitive_desc(), net)
        W_m = reorder_if_must(self.W.memory, cc_pd.weights_primitive_desc(), net)

        if b is None:
            net.push_back(ip_forward.inner_product_forward(cc_pd,
                at(x_m), at(W_m), y.memory))
        else:
            net.push_back(ip_forward.inner_product_forward(cc_pd,
                at(x_m), at(W_m), at(self.b.memory), y.memory))

        self.x_m = x_m
        self.W_m = W_m
        self._hint = cc_pd
        self.output = y,


class LinearBackwardData(ComputeComplex):
    def __init__(self, x, W, gy, hint, engine=Engine()):
        super(LinearBackwardData, self).__init__()

class LinearBackwardWeighs(ComputeComplex):
    def __init__(self, x, W, gy, hint, b = None, engine=Engine()):
        super(LinearBackwardWeighs, self).__init__()

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

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
        x = _as_mat(inputs[0])
        W = inputs[1]

        if len(inputs) == 3:
            b = inputs[2]
        else:
            b = None

        cc = LinearForward(x, W, b)
        y, = cc.execute_on()

        # Save forward hint for backward
        self.hint = cc.hint
        return y,

    def backward(sefl, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]

        if len(inputs) == 3:
            b = inputs[2]
        else:
            b = None

        cc_data = LinearBackwardData(w, W, gy, self.hint)
        cc_weight = LinearBackwardWeighs(w, W, b, gy, self.hint)

        if len(inputs) == 3:
            gx, = cc_data.execute_on()
            gW, gb = cc_weight.execute_on()
        else:
            gx, = cc_data.execute_on()
            gW, = cc_weight.execute_on()

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
