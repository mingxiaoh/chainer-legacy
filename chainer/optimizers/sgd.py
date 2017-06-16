import numpy

from chainer import cuda
from chainer import optimizer
from chainer import mkld

if mkld.mkldnn_enabled:
    mkldnn = mkld.mkldnn


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.scale = numpy.empty((2,), dtype=numpy.float64)
        self.scale[0] = 1
        self.scale[1] = -lr

    def update_one_cpu(self, param, state):
        if mkld.enable_sgdF((param.data, param.grad)):
            input = ()
            for xi in (param.data, param.grad):
                if xi.flags.contiguous is False:
                    input += (xi.copy().astype(xi.dtype),)
                else:
                    input += (xi,)

            mkldnn_sum = mkldnn.Sum_F32()
            ndim = param.data.ndim
            if ndim == 1:
                mkldnn_sum.sum1d_diff(input, param.data, self.scale)
            elif ndim == 2:
                mkldnn_sum.sum2d_diff(input, param.data, self.scale)
            elif ndim == 4:
                mkldnn_sum.sum4d_diff(input, param.data, self.scale)
            else:
                print("Not Implemented dims ", ndim)
        else:
            param.data -= self.lr * param.grad

    def update_one_gpu(self, param, state):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(param.grad, self.lr, param.data)
