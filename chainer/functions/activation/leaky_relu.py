import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _kern():
    return cuda.elementwise(
        'T cond, T x, T slope', 'T y',
        'y = cond >= 0 ? x : (T)(slope * x)', 'lrelu')


class LeakyReLU(function.Function):

    """Leaky rectifier unit."""

    def __init__(self, slope=0.2):
        self.slope = slope

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        y = numpy.empty_like(x[0])
        numpy.multiply(x[0], numpy.where(x[0] < 0, self.slope, 1), y)
        return y,

    def forward_gpu(self, x):
        y = _kern()(x[0], x[0], self.slope)
        return y,

    def backward_cpu(self, x, gy):
        gx = numpy.empty_like(x[0])
        numpy.multiply(gy[0], numpy.where(x[0] < 0, self.slope, 1), gx)
        return gx,

    def backward_gpu(self, x, gy):
        gx = _kern()(x[0], gy[0], self.slope)
        return gx,


def leaky_relu(x, slope=0.2):
    """Leaky Rectified Linear Unit function.

    This function is expressed as :math:`f(x) = \\max(x, ax)`, where :math:`a`
    is a configurable slope value.

    Args:
        x (~chainer.Variable): Input variable.
        slope (float): Slope value :math:`a`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return LeakyReLU(slope)(x)
