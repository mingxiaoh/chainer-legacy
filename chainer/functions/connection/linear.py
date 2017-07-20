from chainer import function
from chainer.utils import type_check

import chainer
from chainer import cuda
from chainer import mkld

import numpy

if mkld.available:
    LinearForward = mkld.linear.LinearForward
    LinearBackwardData = mkld.linear.LinearBackwardData
    LinearBackwardWeighs = mkld.linear.LinearBackwardWeighs


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):

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
        W = _as_mat(inputs[1])
        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = _as_mat(inputs[1])
        gy = grad_outputs[0]

        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


class LinearFunctionMKLDNN(LinearFunction):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim >= 2,
            type_check.prod(x_type.shape[1:]) == type_check.prod(w_type.shape[1:]),
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
        self.W = cc.W

        y, = cc.execute_on()
        y.reset_buf_order()

        return y,

    def backward(self, inputs, grad_outputs):
        cc_data = LinearBackwardData(inputs, grad_outputs, self.hint, self.W,
                                     pos=(self.rank, self.fanout))
        cc_weight = LinearBackwardWeighs(inputs, grad_outputs, self.hint,
                                         pos=(self.rank, self.fanout))

        gx = cc_data.execute_on()
        gx[0].reset_buf_order()
        gW_b = cc_weight.execute_on()
        gW_b[0].reset_buf_order()

        return gx + gW_b


def linear(x, W, b=None):
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
    # XXX: switch the route, work on the critera

    if not isinstance(x, cuda.ndarray) and \
       not isinstance(x.data, cuda.ndarray) and \
       mkld.check_with_mkld((x, W), (2, 4)):
        if b is None:
            return LinearFunctionMKLDNN()(x, W)
        else:
            return LinearFunctionMKLDNN()(x, W, b)
    else:
        if b is None:
            return LinearFunction()(x, W)
        else:
            return LinearFunction()(x, W, b)
