import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ELU(function.Function):

    """Exponential Linear Unit."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        dtype = x[0].dtype.type
        alpha = dtype(self.alpha)
        one = dtype(1)
        y = numpy.where(x[0] < 0, alpha * (numpy.exp(x[0]) - one), x[0])
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : (T)(alpha * (exp(x) - 1))',
            'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        dtype = x[0].dtype.type
        alpha = dtype(self.alpha)
        one = dtype(1)
        gx = numpy.empty_like(x[0])
        numpy.multiply(
            gy, numpy.where(x[0] < 0, alpha * numpy.exp(x[0]), one), gx)
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : (T)(gy * alpha * exp(x))',
            'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function.

    This function is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.

    where :math:`\\alpha` is a parameter.
    See: http://arxiv.org/abs/1511.07289

    Args:
        x (~chainer.Variable): Input variable.
        alpha (float): Parameter :math:`\\alpha`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ELU(alpha=alpha)(x)
