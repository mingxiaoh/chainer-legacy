from chainer import function
from chainer.utils import type_check

from mkldnn.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
import mkldnn.memory as m
import mkldnn.inner_product as ip
from mkldnn.mdarray import *

class LinearForward(ComputeComplex):
    def __init__(self, x, W, b = None, engine=Engine()):
        super(LinearForward, self).__init__()

        y_expect = m.desc(m.dims((x.shape[0], W.shape[0])),
                m.memory.f32, m.memory.any)

        self.x = mdarray(x, m.memory.nc, engine)
        self.W = mdarray(W, m.memory.io, engine)

        if b is not None:
            self.b = mdarray(b, m.memory.x, engine)
            cc_pd = ip.desc(y_expect, x.pd, W.pd, b.pd)
        else:
            cc_pd = ip.desc(y_expect, x.pd, W.pd)

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
